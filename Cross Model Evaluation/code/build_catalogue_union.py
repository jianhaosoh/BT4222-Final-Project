"""Build the ordered article metadata union used by shared evaluation."""


import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from evaluate import (catalogue_fingerprints, checked_int64_series, file_provenance,
                      load_articles, utc_times)


def _source_manifest(path: Path) -> dict:
    details = file_provenance(path, hash_file=True)
    details["path"] = path.name
    return details


def _write_manifest_temp(manifest: dict, directory: Path, output_name: str) -> Path:
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output_name}.", suffix=".manifest.tmp", dir=directory,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(manifest, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return temp_path


def _temporary_backup_path(directory: Path, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=".backup", dir=directory)
    os.close(descriptor)
    return Path(name)


def _commit_catalogue_and_manifest(
    temp_catalogue: Path, output: Path, temp_manifest: Path, manifest: Path,
) -> None:
    """Replace outputs with rollback for ordinary exceptions, not process/power loss."""
    had_catalogue, had_manifest = output.exists(), manifest.exists()
    catalogue_backup = manifest_backup = None
    moved_catalogue = moved_manifest = installed_catalogue = installed_manifest = False
    try:
        if had_catalogue:
            catalogue_backup = _temporary_backup_path(output.parent, f".{output.name}.")
        if had_manifest:
            manifest_backup = _temporary_backup_path(manifest.parent, f".{manifest.name}.")
        if had_catalogue:
            os.replace(output, catalogue_backup)
            moved_catalogue = True
        if had_manifest:
            os.replace(manifest, manifest_backup)
            moved_manifest = True
        os.replace(temp_catalogue, output)
        installed_catalogue = True
        os.replace(temp_manifest, manifest)
        installed_manifest = True
    except Exception:
        if installed_manifest and manifest.exists():
            manifest.unlink()
        if moved_manifest and manifest_backup and manifest_backup.exists():
            os.replace(manifest_backup, manifest)
        elif not had_manifest and manifest.exists():
            manifest.unlink()
        if installed_catalogue and output.exists():
            output.unlink()
        if moved_catalogue and catalogue_backup and catalogue_backup.exists():
            os.replace(catalogue_backup, output)
        elif not had_catalogue and output.exists():
            output.unlink()
        for backup, was_moved in ((catalogue_backup, moved_catalogue),
                                  (manifest_backup, moved_manifest)):
            if backup and not was_moved and backup.exists():
                backup.unlink()
        raise
    else:
        if catalogue_backup and catalogue_backup.exists():
            catalogue_backup.unlink()
        if manifest_backup and manifest_backup.exists():
            manifest_backup.unlink()


def read_catalogue(path: Path, *, naive_timezone: str = "UTC",
                   allow_source_duplicates: bool = False) -> pd.DataFrame:
    names = pq.ParquetFile(path).schema_arrow.names
    required = {"article_id", "published_time"}
    if required - set(names):
        raise ValueError(f"{path} is missing {sorted(required - set(names))}")
    columns = [name for name in ("article_id", "published_time", "topics", "category") if name in names]
    frame = pd.read_parquet(path, columns=columns)
    frame["article_id"] = checked_int64_series(frame.article_id, f"{path}.article_id")
    duplicate_rows = int(frame.article_id.duplicated().sum())
    if duplicate_rows and not allow_source_duplicates:
        examples = frame.loc[frame.article_id.duplicated(keep=False), "article_id"].drop_duplicates().head(5).tolist()
        raise ValueError(
            f"{path} contains {duplicate_rows} duplicate article_id rows (examples: {examples}); "
            "clean the source or pass --allow-source-duplicates to keep the first row"
        )
    if "category" in frame:
        frame["category"] = frame["category"].astype("string")
    raw_times = frame.published_time
    frame["published_time"] = utc_times(
        raw_times, naive_timezone=naive_timezone, errors="coerce"
    )
    malformed = raw_times.notna() & frame.published_time.isna()
    if malformed.any():
        raise ValueError(f"{path} contains {int(malformed.sum())} unparseable published_time values")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-val", required=True, type=Path,
                        help="First-priority train/validation catalogue snapshot")
    parser.add_argument("--test", required=True, type=Path,
                        help="Second-priority test catalogue snapshot")
    parser.add_argument("--text-flags", nargs="*", type=Path, default=[],
                        help="Optional ordered embedding metadata Parquets with article_id, has_text_embedding")
    parser.add_argument("--image-flags", nargs="*", type=Path, default=[],
                        help="Optional ordered embedding metadata Parquets with article_id, has_image_embedding")
    parser.add_argument("--naive-timezone", default="UTC",
                        help="Timezone to assign to timezone-naive source timestamps (default: UTC)")
    parser.add_argument("--allow-source-duplicates", action="store_true",
                        help="Allow duplicate IDs within any source; first row wins and duplicate counts are recorded")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source_paths = [args.train_val, args.test, *args.text_flags, *args.image_flags]
    output_resolved = args.output.resolve()
    if any(source.resolve() == output_resolved for source in source_paths):
        parser.error("--output must not overwrite any catalogue or embedding-flag source")
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    protected_paths = [*source_paths, args.output]
    if manifest_path.resolve() in {path.resolve() for path in protected_paths}:
        parser.error("--output manifest path must not overwrite an input source or the catalogue output")
    source_provenance_before = {
        "train_val": _source_manifest(args.train_val),
        "test": _source_manifest(args.test),
        "text_flags": [_source_manifest(path) for path in args.text_flags],
        "image_flags": [_source_manifest(path) for path in args.image_flags],
    }
    first = read_catalogue(args.train_val, naive_timezone=args.naive_timezone,
                           allow_source_duplicates=args.allow_source_duplicates)
    second = read_catalogue(args.test, naive_timezone=args.naive_timezone,
                            allow_source_duplicates=args.allow_source_duplicates)
    # First snapshot wins, including null fields, exactly as the project policy specifies.
    union = pd.concat([first, second], ignore_index=True, sort=False)
    before = len(union)
    first_ids, second_ids = set(first.article_id), set(second.article_id)
    union = union.drop_duplicates("article_id", keep="first").reset_index(drop=True)
    union["_catalogue_row_order"] = np.arange(len(union), dtype=np.int64)
    flag_source_diagnostics = {"has_text_embedding": [], "has_image_embedding": []}
    catalogue_ids = set(union.article_id)
    for paths, flag in ((args.text_flags, "has_text_embedding"),
                        (args.image_flags, "has_image_embedding")):
        if paths:
            pieces = []
            for path in paths:
                names = pq.ParquetFile(path).schema_arrow.names
                if not {"article_id", flag}.issubset(names):
                    raise ValueError(f"{path} must contain article_id and {flag}")
                piece = pd.read_parquet(path, columns=["article_id", flag])
                piece["article_id"] = checked_int64_series(piece.article_id, f"{path}.article_id")
                duplicate_count = int(piece.article_id.duplicated().sum())
                if duplicate_count and not args.allow_source_duplicates:
                    raise ValueError(
                        f"{path} contains {duplicate_count} duplicate article_id rows; "
                        "clean the source or pass --allow-source-duplicates"
                    )
                unknown_ids = set(piece.article_id) - catalogue_ids
                flag_source_diagnostics[flag].append({
                    "file": path.name,
                    "duplicate_rows": duplicate_count,
                    "flag_rows_for_unknown_article_ids": int(piece.article_id.isin(unknown_ids).sum()),
                    "unknown_article_id_count": int(len(unknown_ids)),
                    "unknown_article_id_examples": sorted(unknown_ids)[:10],
                })
                if not piece[flag].dropna().isin([0, 1, False, True]).all():
                    raise ValueError(f"{path}.{flag} must contain only true, false, or null")
                pieces.append(piece)
            flags = pd.concat(pieces, ignore_index=True, sort=False)
            flags = flags.drop_duplicates("article_id", keep="first")
            union = union.drop(columns=[flag], errors="ignore")
            union = union.merge(flags, on="article_id", how="left", validate="one_to_one")
            union = union.sort_values("_catalogue_row_order", kind="mergesort").reset_index(drop=True)
            union[flag] = union[flag].astype("boolean")
    union = union.drop(columns="_catalogue_row_order")
    if union.article_id.isna().any() or union.article_id.duplicated().any():
        raise ValueError("Catalogue union must have one non-null row per article_id")
    id_fingerprint, content_fingerprint = catalogue_fingerprints(union)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{args.output.stem}.", suffix=args.output.suffix,
        dir=args.output.parent,
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    temp_manifest_path = None
    try:
        union.to_parquet(temp_path, index=False, compression="zstd")
        evaluator_catalogue = load_articles(temp_path, naive_timezone=args.naive_timezone)
        written_id_fingerprint, written_content_fingerprint = catalogue_fingerprints(evaluator_catalogue)
        if (id_fingerprint, content_fingerprint) != (
            written_id_fingerprint, written_content_fingerprint
        ):
            raise ValueError(
                "Catalogue fingerprints changed after Parquet write/read using the evaluator loader"
            )
        id_fingerprint, content_fingerprint = written_id_fingerprint, written_content_fingerprint
        source_provenance_after = {
            "train_val": _source_manifest(args.train_val),
            "test": _source_manifest(args.test),
            "text_flags": [_source_manifest(path) for path in args.text_flags],
            "image_flags": [_source_manifest(path) for path in args.image_flags],
        }
        if source_provenance_before != source_provenance_after:
            raise RuntimeError("An input source changed during catalogue construction; outputs were not committed")
        manifest = {
               "builder_version": 9,
               "sources": source_provenance_before,
               "source_provenance_verified_unchanged_during_build": True,
               "precedence": "train_val then test; first row kept including null fields; embedding flag source lists are ordered and first row per article wins",
               "train_val_rows": int(len(first)), "test_rows": int(len(second)),
               "train_val_unique_ids": int(first.article_id.nunique()),
               "test_unique_ids": int(second.article_id.nunique()),
               "train_val_duplicate_rows": int(first.article_id.duplicated().sum()),
               "test_duplicate_rows": int(second.article_id.duplicated().sum()),
               "embedding_flag_source_diagnostics": flag_source_diagnostics,
               "overlap_ids": int(len(set(first.article_id) & set(second.article_id))),
               "retained_from_train_val": int(len(first_ids)),
               "retained_from_test": int(len(second_ids - first_ids)),
               "union_rows": int(len(union)), "duplicate_ids_removed": int(before - len(union)),
               "catalogue_id_fingerprint_sha256": id_fingerprint,
               "catalogue_content_fingerprint_sha256": content_fingerprint,
               "fingerprint_scope": "logical normalized table; ID hash follows preserved catalogue row order, metadata content hash is canonicalized by article_id; not Parquet file-byte hashes",
               "fingerprint_roundtrip_verified": True,
               "naive_timestamp_timezone": args.naive_timezone,
               "row_order": "first occurrence in train_val source, followed by first occurrences from test; not chronological or a ranking",
               "output_file": args.output.name}
        temp_manifest_path = _write_manifest_temp(manifest, args.output.parent, args.output.name)
        _commit_catalogue_and_manifest(temp_path, args.output, temp_manifest_path, manifest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
        if temp_manifest_path and temp_manifest_path.exists():
            temp_manifest_path.unlink()
    print({"train_val_rows": len(first), "test_rows": len(second),
           "union_rows": len(union), "duplicate_ids_removed": before - len(union),
           "overlap_ids": manifest["overlap_ids"],
           "catalogue_id_fingerprint_sha256": id_fingerprint,
           "catalogue_content_fingerprint_sha256": content_fingerprint,
           "text_flag_sources": [str(p) for p in args.text_flags],
           "image_flag_sources": [str(p) for p in args.image_flags],
           "output": str(args.output), "manifest": str(manifest_path)})


if __name__ == "__main__":
    main()

