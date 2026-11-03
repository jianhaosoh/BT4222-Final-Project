"""Data preparation, metrics and export helpers for experiment 2.1_lstm.

This module deliberately does not import PyTorch, so its logic can be checked
without a GPU runtime. The notebook ``experiment.ipynb`` calls these helpers;
``lstm_model.py`` holds the network and training loop.

Conventions
-----------
* A candidate row is one (user_id, impression_id, candidate_article_id) from the
  shared feature table. No candidate is added or removed.
* An impression ("behaviour") is the composite key (user_id, impression_id).
* History sequences come from each split's own fixed pre-period history file.
  Every retained event must be strictly earlier than that split's earliest
  impression; later in-period behaviour is never appended.
* Article index 0 is padding. Every other article (catalogue or not) gets an
  index >= 1; articles absent from the catalogue keep explicit missing flags.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SPLITS = ("train", "validation", "test")
EXPORT_SCHEMA_VERSION = 3

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# Candidate locations relative to DATA_ROOT, in precedence order. The first
# entry follows Experiments/README.md; later entries are the layouts actually
# observed in the shared folder / mini sample. Resolution never invents data:
# if no candidate exists a FileNotFoundError lists every path tried.
PATH_CANDIDATES: dict[str, list[str]] = {
    "features_train": ["feature_engineering/train_features.parquet"],
    "features_validation": ["feature_engineering/validation_features.parquet"],
    "features_test": ["feature_engineering/test_features.parquet"],
    "feature_vocabulary": ["feature_engineering/feature_vocabulary.json"],
    "history_train": ["train_val/train/history.parquet"],
    "history_validation": ["train_val/validation/history.parquet"],
    "history_test": ["test/history.parquet"],
    "behaviors_train": ["train_val/train/behaviors.parquet",
                        "train_val/train/behaviours.parquet"],
    "behaviors_validation": ["train_val/validation/behaviors.parquet",
                             "train_val/validation/behaviours.parquet"],
    "behaviors_test": ["test/behaviors.parquet", "test/behaviours.parquet"],
    "articles_train_val": ["train_val/articles.parquet"],
    "articles_test": ["test/articles.parquet"],
    "text_train_val": ["embeddings/train_val/text_embeddings.parquet",
                       "train_val/article embeddings/text_embeddings.parquet"],
    "image_train_val": ["embeddings/train_val/image_embeddings.parquet",
                        "train_val/article embeddings/image_embeddings.parquet"],
    "text_test": ["embeddings/test/text_embeddings.parquet",
                  "test/article embeddings/text_embeddings.parquet"],
    "image_test": ["embeddings/test/image_embeddings.parquet",
                   "test/article embeddings/image_embeddings.parquet"],
    "embedding_metadata_train_val": ["embeddings/train_val/metadata.json",
                                     "train_val/article embeddings/metadata.json"],
    "embedding_metadata_test": ["embeddings/test/metadata.json",
                                "test/article embeddings/metadata.json"],
}
OPTIONAL_PATHS = {
    "behaviors_train", "behaviors_validation", "behaviors_test",
    "embedding_metadata_train_val", "embedding_metadata_test",
}
# loader_paths.json (supplied with the mini sample) keys -> our keys.
LOADER_PATH_KEYS = {
    "train_val_text_embeddings": "text_train_val",
    "train_val_image_embeddings": "image_train_val",
    "test_text_embeddings": "text_test",
    "test_image_embeddings": "image_test",
    "validation_behaviors": "behaviors_validation",
}


def resolve_paths(data_root: Path, overrides: dict | None = None,
                  required: Iterable[str] | None = None) -> tuple[dict, list]:
    """Resolve every input explicitly.

    Precedence: ``overrides`` (absolute or DATA_ROOT-relative), then
    ``loader_paths.json`` in DATA_ROOT if present, then PATH_CANDIDATES.
    Returns ({key: Path or None}, notes). Raises if a required input is missing.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT is not a directory: {data_root}")
    overrides = dict(overrides or {})
    notes = []
    loader_map = data_root / "loader_paths.json"
    loader_entries = {}
    if loader_map.is_file():
        raw = json.loads(loader_map.read_text())
        for source_key, key in LOADER_PATH_KEYS.items():
            if raw.get(source_key):
                loader_entries[key] = raw[source_key]
        notes.append(f"loader_paths.json found: {sorted(loader_entries)}")

    required = set(PATH_CANDIDATES) - OPTIONAL_PATHS if required is None else set(required)
    resolved, missing = {}, {}
    for key, candidates in PATH_CANDIDATES.items():
        tried = []
        ordered = []
        if key in overrides:
            ordered.append(overrides[key])
        if key in loader_entries:
            ordered.append(loader_entries[key])
        ordered.extend(candidates)
        found = None
        for relative in ordered:
            path = Path(relative)
            path = path if path.is_absolute() else data_root / path
            tried.append(str(path))
            if path.is_file():
                if found is None:
                    found = path
                elif path.resolve() != found.resolve():
                    notes.append(f"{key}: using {found}; also present {path}")
        resolved[key] = found
        if found is None and key in required:
            missing[key] = tried
    if missing:
        detail = "\n".join(f"  {k}: tried {v}" for k, v in missing.items())
        raise FileNotFoundError(f"Required inputs not found:\n{detail}")
    return resolved, notes


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def describe_inputs(paths: dict, hash_files: bool = True) -> dict:
    out = {}
    for key, path in paths.items():
        if path is None:
            out[key] = None
            continue
        stat = Path(path).stat()
        out[key] = {
            "path": str(path), "bytes": stat.st_size,
            "modified_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
            "sha256": sha256_file(path) if hash_files else None,
        }
    return out


# --------------------------------------------------------------------------
# Candidate rows from the shared feature table
# --------------------------------------------------------------------------

KEY_COLUMNS = ["user_id", "impression_id", "candidate_article_id"]
CONTEXT_BOOL_COLUMNS = ["is_sso_user", "is_subscriber", "is_front_page"]
BASE_FEATURE_COLUMNS = (
    KEY_COLUMNS + ["impression_time", "clicked", "hour", "day_of_week",
                   "article_age_h", "article_age_missing"]
    + CONTEXT_BOOL_COLUMNS
)


def composite_key(user_id, impression_id) -> np.ndarray:
    user = np.asarray(user_id, dtype=np.int64)
    impression = np.asarray(impression_id, dtype=np.int64)
    if (user < 0).any() or (impression < 0).any() or (user >= 2**31).any() \
            or (impression >= 2**32).any():
        raise ValueError("IDs outside the range supported by the composite key")
    return (user << 32) | impression


def load_candidate_table(path: Path, split: str) -> tuple[pd.DataFrame, dict]:
    """Read keys, labels and the small set of shared context columns.

    Rows are reordered so each impression is contiguous (first-appearance
    order), keeping the original within-impression (in-view) order.
    """
    schema = pq.ParquetFile(path).schema_arrow
    names = schema.names
    device_columns = [c for c in names if c.startswith("ohe_device_type_")]
    needed = BASE_FEATURE_COLUMNS + device_columns
    absent = sorted(set(needed) - set(names))
    if absent:
        raise ValueError(f"{path} lacks columns {absent}")
    frame = pd.read_parquet(path, columns=needed)
    n_rows = len(frame)
    if frame[KEY_COLUMNS + ["impression_time"]].isna().any().any():
        raise ValueError(f"{split}: null keys or impression_time")
    for column in KEY_COLUMNS:
        frame[column] = frame[column].astype("int64")
    dup = frame.duplicated(KEY_COLUMNS)
    if dup.any():
        raise ValueError(f"{split}: {int(dup.sum())} duplicate candidate keys; "
                         "investigate the source instead of deduplicating")

    key = composite_key(frame["user_id"], frame["impression_id"])
    codes, uniques = pd.factorize(key, sort=False)
    order = np.argsort(codes, kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    codes = codes[order]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    offsets = np.r_[starts, len(frame)].astype(np.int64)
    if len(starts) != len(uniques):
        raise AssertionError("impressions are not contiguous after sorting")

    time_ns = frame["impression_time"].astype("datetime64[us]").astype("int64")
    per_imp_min = np.minimum.reduceat(time_ns.to_numpy(), starts)
    per_imp_max = np.maximum.reduceat(time_ns.to_numpy(), starts)
    if (per_imp_min != per_imp_max).any():
        raise ValueError(f"{split}: impression_time varies within an impression")

    labels = frame["clicked"]
    info = {"split": split, "candidate_rows": n_rows, "impressions": len(starts),
            "users": int(frame["user_id"].nunique()),
            "device_columns": device_columns}
    if split == "test":
        if labels.notna().any():
            info["note"] = "test has non-null labels; they are ignored"
        frame["clicked"] = np.nan
        info["labelled"] = False
    else:
        if labels.isna().any():
            raise ValueError(f"{split}: missing click labels")
        if not labels.isin([0, 1]).all():
            raise ValueError(f"{split}: labels outside {{0,1}}")
        frame["clicked"] = labels.astype("float32")
        y = frame["clicked"].to_numpy()
        pos = np.add.reduceat(y, starts)
        size = np.diff(offsets)
        info.update({
            "labelled": True, "positive_rate": float(y.mean()),
            "impressions_without_click": int((pos == 0).sum()),
            "impressions_all_clicked": int((pos == size).sum()),
            "impressions_with_both_classes": int(((pos > 0) & (pos < size)).sum()),
        })
    slate = np.diff(offsets)
    info["slate_size_min_median_max"] = [int(slate.min()), float(np.median(slate)),
                                         int(slate.max())]
    reused = frame.drop_duplicates(["user_id", "impression_id"])
    info["impression_ids_reused_across_users"] = int(
        reused["impression_id"].duplicated().sum())
    return frame, {"offsets": offsets, **info}


def cross_check_behaviours(frame: pd.DataFrame, behaviours_path: Path,
                           split: str) -> dict:
    """Confirm the feature rows equal the exploded in-view slates (and labels)."""
    names = pq.ParquetFile(behaviours_path).schema_arrow.names
    columns = ["user_id", "impression_id", "article_ids_inview"]
    has_clicks = "article_ids_clicked" in names and split != "test"
    if has_clicks:
        columns.append("article_ids_clicked")
    table = pq.read_table(behaviours_path, columns=columns)
    inview = table["article_ids_inview"].combine_chunks()
    lengths = pc.list_value_length(inview).to_numpy(zero_copy_only=False)
    exploded = pd.DataFrame({
        "user_id": np.repeat(table["user_id"].to_numpy().astype(np.int64), lengths),
        "impression_id": np.repeat(table["impression_id"].to_numpy().astype(np.int64),
                                   lengths),
        "candidate_article_id": pc.list_flatten(inview).to_numpy().astype(np.int64),
    })
    result = {"behaviour_rows": table.num_rows, "exploded_candidates": len(exploded)}
    if has_clicks:
        clicked = table["article_ids_clicked"].combine_chunks()
        click_frame = pd.DataFrame({
            "user_id": np.repeat(table["user_id"].to_numpy().astype(np.int64),
                                 pc.list_value_length(clicked).to_numpy(zero_copy_only=False)),
            "impression_id": np.repeat(
                table["impression_id"].to_numpy().astype(np.int64),
                pc.list_value_length(clicked).to_numpy(zero_copy_only=False)),
            "candidate_article_id": pc.list_flatten(clicked).to_numpy().astype(np.int64),
            "_click": 1.0,
        }).drop_duplicates(KEY_COLUMNS)
        exploded = exploded.merge(click_frame, on=KEY_COLUMNS, how="left")
        exploded["_click"] = exploded["_click"].fillna(0.0)
    merged = frame[KEY_COLUMNS + ["clicked"]].merge(
        exploded, on=KEY_COLUMNS, how="outer", indicator=True)
    result["feature_rows_not_in_behaviours"] = int((merged["_merge"] == "left_only").sum())
    result["behaviour_candidates_not_in_features"] = int(
        (merged["_merge"] == "right_only").sum())
    if has_clicks:
        both = merged[merged["_merge"] == "both"]
        result["label_mismatches"] = int((both["clicked"] != both["_click"]).sum())
    return result


def select_development_sample(frame: pd.DataFrame, offsets: np.ndarray,
                              user_fraction: float | None, seed: int,
                              max_users: int | None = None):
    """Keep whole users (and so whole impressions) by a seeded hash rule."""
    if user_fraction is None and max_users is None:
        return frame, offsets, None
    users = np.sort(frame["user_id"].unique())
    rng = np.random.default_rng(seed)
    priority = rng.random(len(users))
    order = users[np.argsort(priority, kind="stable")]
    count = len(users)
    if user_fraction is not None:
        count = max(1, int(round(len(users) * user_fraction)))
    if max_users is not None:
        count = min(count, max_users)
    keep_users = set(order[:count].tolist())
    keep_rows = frame["user_id"].isin(keep_users).to_numpy()
    sizes = np.diff(offsets)
    keep_imp = keep_rows[offsets[:-1]]
    new = frame.loc[keep_rows].reset_index(drop=True)
    new_offsets = np.r_[0, np.cumsum(sizes[keep_imp])].astype(np.int64)
    rule = {"rule": "whole users chosen by seeded random priority",
            "seed": seed, "user_fraction": user_fraction, "max_users": max_users,
            "users_kept": count, "users_total": len(users),
            "impressions_kept": int(keep_imp.sum()), "candidate_rows_kept": len(new)}
    return new, new_offsets, rule


# --------------------------------------------------------------------------
# Chronological history sequences
# --------------------------------------------------------------------------

HISTORY_LIST_COLUMNS = ["article_id_fixed", "impression_time_fixed",
                        "read_time_fixed", "scroll_percentage_fixed"]


@dataclass
class HistoryArrays:
    """Most recent ``max_len`` fixed-history events per user, oldest first.

    Row ``len(user_ids)`` is an extra empty row for behaviour users without a
    history row. Padding positions have article_id 0 / time 0 / NaN values.
    """
    user_ids: np.ndarray            # [U] int64, sorted
    lengths: np.ndarray             # [U+1] int32 retained events
    full_lengths: np.ndarray        # [U+1] int32 valid events before truncation
    article_ids: np.ndarray         # [U+1, L] int64 original IDs
    times_s: np.ndarray             # [U+1, L] int64 epoch seconds
    read_time: np.ndarray           # [U+1, L] float32 seconds (NaN missing)
    scroll: np.ndarray              # [U+1, L] float32 percentage (NaN missing)
    stats: dict = field(default_factory=dict)

    def rows_for(self, user_ids: np.ndarray) -> np.ndarray:
        position = np.searchsorted(self.user_ids, user_ids)
        position = np.clip(position, 0, max(len(self.user_ids) - 1, 0))
        found = (len(self.user_ids) > 0) & (self.user_ids[position] == user_ids)
        return np.where(found, position, len(self.user_ids)).astype(np.int64)


def _flat_list(table: pa.Table, column: str):
    array = table[column].combine_chunks()
    return array, pc.list_value_length(array).fill_null(0).to_numpy(zero_copy_only=False)


def build_history(path: Path, split_start: pd.Timestamp, max_len: int,
                  users_needed: np.ndarray | None = None) -> HistoryArrays:
    table = pq.read_table(path, columns=["user_id"] + HISTORY_LIST_COLUMNS)
    user_ids = table["user_id"].to_numpy().astype(np.int64)
    if len(np.unique(user_ids)) != len(user_ids):
        raise ValueError(f"{path}: duplicate user rows in history")
    arrays, lengths = {}, None
    for column in HISTORY_LIST_COLUMNS:
        array, column_lengths = _flat_list(table, column)
        if lengths is None:
            lengths = column_lengths
        elif not np.array_equal(lengths, column_lengths):
            raise ValueError(f"{path}: {column} not aligned with article_id_fixed")
        arrays[column] = pc.list_flatten(array)
    parent = np.repeat(np.arange(len(user_ids)), lengths)
    art = arrays["article_id_fixed"]
    times = arrays["impression_time_fixed"].cast(pa.timestamp("us"))
    valid = (pc.is_valid(art).to_numpy(zero_copy_only=False)
             & pc.is_valid(times).to_numpy(zero_copy_only=False))
    art_np = art.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    time_us = times.cast(pa.int64()).fill_null(0).to_numpy(zero_copy_only=False)
    read = arrays["read_time_fixed"].cast(pa.float32()).to_numpy(zero_copy_only=False)
    scroll = arrays["scroll_percentage_fixed"].cast(pa.float32()).to_numpy(
        zero_copy_only=False)
    read = np.where(np.isfinite(read) & (read >= 0), read, np.nan).astype(np.float32)
    scroll = np.where(np.isfinite(scroll) & (scroll >= 0) & (scroll <= 100),
                      scroll, np.nan).astype(np.float32)

    start_us = int(pd.Timestamp(split_start).value // 1000)
    late = valid & (time_us >= start_us)
    if late.any():
        raise ValueError(f"{path}: {int(late.sum())} history events at/after the "
                         f"split start {split_start}; fixed history must precede it")
    stats = {"history_users": len(user_ids), "events_total": int(len(valid)),
             "events_dropped_null_id_or_time": int((~valid).sum()),
             "read_time_missing": int(np.isnan(read[valid]).sum()),
             "scroll_missing": int(np.isnan(scroll[valid]).sum()),
             "max_event_time": (pd.Timestamp(int(time_us[valid].max()), unit="us")
                                .isoformat() if valid.any() else None),
             "split_start": pd.Timestamp(split_start).isoformat()}

    keep = np.flatnonzero(valid)
    position = np.arange(len(valid))
    # Chronological sort per user; original list position breaks timestamp ties.
    order = keep[np.lexsort((position[keep], time_us[keep], parent[keep]))]
    p_sorted = parent[order]
    n_valid = np.bincount(p_sorted, minlength=len(user_ids)).astype(np.int32)
    first = np.r_[0, np.cumsum(n_valid)[:-1]]
    rank_in_user = np.arange(len(order)) - first[p_sorted]
    from_end = n_valid[p_sorted] - 1 - rank_in_user
    retained = from_end < max_len
    order, p_sorted = order[retained], p_sorted[retained]
    kept = np.minimum(n_valid, max_len).astype(np.int32)
    column_index = (rank_in_user[retained]
                    - (n_valid[p_sorted] - kept[p_sorted]))

    sort_users = np.argsort(user_ids, kind="stable")
    new_row = np.empty_like(sort_users)
    new_row[sort_users] = np.arange(len(user_ids))
    n_rows = len(user_ids) + 1
    shape = (n_rows, max_len)
    out_art = np.zeros(shape, np.int64)
    out_time = np.zeros(shape, np.int64)
    out_read = np.full(shape, np.nan, np.float32)
    out_scroll = np.full(shape, np.nan, np.float32)
    rows = new_row[p_sorted]
    out_art[rows, column_index] = art_np[order]
    out_time[rows, column_index] = time_us[order] // 1_000_000
    out_read[rows, column_index] = read[order]
    out_scroll[rows, column_index] = scroll[order]
    lengths_out = np.zeros(n_rows, np.int32)
    lengths_out[new_row] = kept
    full_out = np.zeros(n_rows, np.int32)
    full_out[new_row] = n_valid
    hist = HistoryArrays(np.sort(user_ids), lengths_out, full_out, out_art,
                         out_time, out_read, out_scroll, stats)
    stats.update({
        "max_len": max_len,
        "users_truncated": int((n_valid > max_len).sum()),
        "users_empty": int((n_valid == 0).sum()),
        "length_quantiles_before_truncation": {
            q: float(np.quantile(n_valid, q)) if len(n_valid) else None
            for q in (0.0, 0.5, 0.9, 1.0)},
    })
    if users_needed is not None:
        rows_needed = hist.rows_for(np.unique(users_needed))
        stats["behaviour_users_without_history_row"] = int(
            (rows_needed == len(user_ids)).sum())
    return hist


# --------------------------------------------------------------------------
# Article catalogue, vocabularies and frozen vectors
# --------------------------------------------------------------------------

def normalise_token(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def load_catalogue(paths: list[Path], batch_size: int = 20_000) -> tuple[pd.DataFrame, dict]:
    """Union of article catalogues; the first catalogue wins on overlap.

    Also returns each catalogue's own ID set (the per-split eligibility
    snapshots). Body text is read in batches only to count words. Future
    article totals (total_inviews, total_pageviews, total_read_time) are never
    read.
    """
    needed = ["article_id", "category", "article_type", "premium",
              "published_time", "topics", "sentiment_score"]
    frames, info, snapshot_ids = [], {}, {}
    for path in paths:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        absent = sorted(set(needed + ["body"]) - set(names) - {"sentiment_score"})
        if absent:
            raise ValueError(f"{path} lacks {absent}")
        columns = [c for c in needed if c in names]
        part = pd.read_parquet(path, columns=columns)
        part["article_id"] = part["article_id"].astype("int64")
        words = []
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["body"]):
            words.extend(len(t.split()) if isinstance(t, str) and t.strip() else np.nan
                         for t in batch.column(0).to_pylist())
        part["body_word_count"] = np.asarray(words, dtype="float64")
        if part["article_id"].duplicated().any():
            raise ValueError(f"{path}: duplicate article IDs")
        snapshot_ids[str(path)] = np.sort(part["article_id"].to_numpy())
        info[str(path)] = {"articles": len(part),
                           "published_time_missing": int(part["published_time"].isna().sum())}
        frames.append(part)
    catalogue = pd.concat(frames, ignore_index=True)
    before = len(catalogue)
    catalogue = catalogue.drop_duplicates("article_id", keep="first")
    info["overlapping_ids_first_kept"] = before - len(catalogue)
    catalogue = catalogue.set_index("article_id", verify_integrity=True)
    catalogue["published_time"] = pd.to_datetime(catalogue["published_time"])
    return catalogue, {"info": info, "snapshot_ids": snapshot_ids}


FIXED_SIZE_LIST_ERROR = "Expected all lists to be of size"


def _duckdb_read(path: Path, columns: list[str] | None = None) -> pa.Table:
    """Read a Parquet file with DuckDB's own Parquet reader.

    DuckDB decodes Parquet LIST columns from the Parquet levels rather than the
    stored Arrow fixed_size_list hint, so it reads nullable fixed-size vector
    columns that pyarrow 18-20 reject ("Expected all lists to be of size=...").
    Values are returned unchanged as float32 lists; nulls stay null.
    """
    import duckdb
    select = ", ".join('"' + c.replace('"', '""') + '"' for c in columns) if columns else "*"
    escaped = str(path).replace("'", "''")
    with duckdb.connect() as con:
        relation = con.sql(f"SELECT {select} FROM read_parquet('{escaped}')")
        # DuckDB 1.x and 2.x both provide to_arrow_table on relations.
        if hasattr(relation, "to_arrow_table"):
            table = relation.to_arrow_table()
        else:  # pragma: no cover - older DuckDB
            table = relation.fetch_arrow_table()
    if isinstance(table, pa.RecordBatchReader):  # pragma: no cover
        table = table.read_all()
    return table


def read_vector_table(path: Path, columns: list[str] | None = None) -> tuple[pa.Table, str]:
    """pyarrow first; DuckDB only for the known nullable fixed-size-list failure."""
    try:
        return pq.read_table(path, columns=columns), f"pyarrow {pa.__version__}"
    except pa.ArrowInvalid as error:
        if FIXED_SIZE_LIST_ERROR not in str(error):
            raise RuntimeError(f"Could not read {path}: {error!r}") from error
        try:
            import duckdb
        except ImportError as missing:
            raise RuntimeError(
                f"{path}: pyarrow {pa.__version__} cannot decode its nullable fixed-size "
                "vector column, and DuckDB is not installed for the fallback reader. "
                "Install duckdb (pip install duckdb) and rerun; do not substitute zeros."
            ) from missing
        try:
            table = _duckdb_read(path, columns)
        except Exception as fallback_error:
            raise RuntimeError(
                f"{path}: pyarrow failed ({error}) and DuckDB also failed "
                f"({fallback_error!r}). The file exists, so this is a decoding problem, "
                "not missing vectors; inspect the file rather than dropping rows."
            ) from fallback_error
        return table, (f"duckdb {duckdb.__version__} (pyarrow {pa.__version__} raised: "
                       f"{str(error)[:80]})")


def decode_vectors(path: Path, vector_column: str, flag_column: str,
                   expected_dim: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Actually decode a vector column (list, large_list or fixed_size_list).

    Returns (article_ids [n], matrix [n_available, dim] float32 in the order of
    ids[available], available mask [n]). Unreadable files raise with a
    diagnosis; legitimate nulls are reported, never filled here.
    """
    table, reader = read_vector_table(path, ["article_id", vector_column, flag_column])
    ids, matrix, available, info = decode_vectors_from_table(
        table, vector_column, flag_column, expected_dim, str(path))
    info.update({"storage_type": str(pq.ParquetFile(path).schema_arrow.field(vector_column).type),
                 "reader": reader})
    return ids, matrix, available, info


def decode_vectors_from_table(table: pa.Table, vector_column: str, flag_column: str,
                              expected_dim: int, path: str):
    ids = table["article_id"].to_numpy().astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate article_id rows")
    vectors = table[vector_column].combine_chunks()
    if not (pa.types.is_list(vectors.type) or pa.types.is_large_list(vectors.type)
            or pa.types.is_fixed_size_list(vectors.type)):
        raise TypeError(f"{path}: {vector_column} has type {vectors.type}")
    available = pc.is_valid(vectors).to_numpy(zero_copy_only=False)
    if not pa.types.is_fixed_size_list(vectors.type):
        # Some writers store a missing vector as an empty list rather than null.
        empty = available & (pc.list_value_length(vectors).fill_null(0)
                             .to_numpy(zero_copy_only=False) == 0)
        available = available & ~empty
    present = vectors.filter(pa.array(available))
    lengths = pc.list_value_length(present).to_numpy(zero_copy_only=False)
    if len(lengths) and not (lengths == expected_dim).all():
        bad = np.unique(lengths[lengths != expected_dim])
        raise ValueError(f"{path}: vector lengths {bad[:5]} != {expected_dim}")
    flat = pc.list_flatten(present)
    if flat.null_count:
        raise ValueError(f"{path}: null values inside vectors")
    matrix = flat.to_numpy(zero_copy_only=False).astype(np.float32).reshape(
        -1, expected_dim)
    finite = np.isfinite(matrix).all(axis=1)
    if not finite.all():
        raise ValueError(f"{path}: {int((~finite).sum())} vectors with non-finite values")
    flags = table[flag_column].to_numpy(zero_copy_only=False)
    flags = np.asarray(pd.Series(flags).fillna(False), dtype=bool)
    if not np.array_equal(flags, available):
        raise ValueError(f"{path}: {flag_column} disagrees with vector null-ness "
                         f"({int((flags != available).sum())} rows)")
    norms = np.linalg.norm(matrix, axis=1) if len(matrix) else np.array([np.nan])
    info = {"path": str(path), "rows": len(ids), "available": int(available.sum()),
            "missing": int((~available).sum()), "dim": expected_dim,
            "norm_min_median_max": [float(norms.min()), float(np.median(norms)),
                                    float(norms.max())]}
    return ids, matrix, available, info


def write_compatible_vectors(source: Path, destination: Path, vector_column: str,
                             flag_column: str, expected_dim: int) -> dict:
    """Write a copy whose vectors are nullable variable-length float32 lists.

    Every column, row order, article ID, null and float value is preserved; the
    copy is re-read with pyarrow and compared exactly before it is accepted.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("write the compatible copy to a different path")
    table, reader = read_vector_table(source)
    column = table[vector_column].combine_chunks()
    # list_flatten emits exactly the values of non-null entries, in order, so
    # offsets rebuilt from the per-entry lengths (0 for nulls) line up with them.
    values = pc.list_flatten(column).cast(pa.float32())
    lengths = pc.list_value_length(column).fill_null(0).to_numpy(zero_copy_only=False)
    offsets = pa.array(np.r_[0, np.cumsum(lengths)].astype(np.int64))
    if offsets[-1].as_py() != len(values):
        raise AssertionError(f"{source}: flattened values do not match list lengths")
    rebuilt = pa.ListArray.from_arrays(offsets.cast(pa.int32()), values,
                                       type=pa.list_(pa.float32()), mask=pc.is_null(column))
    out = table.set_column(table.schema.get_field_index(vector_column),
                           pa.field(vector_column, pa.list_(pa.float32())), rebuilt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.parquet")
    pq.write_table(out, temporary)
    check = pq.read_table(temporary)
    for name in table.column_names:
        if name == vector_column:
            continue
        if not check[name].equals(table[name]):
            temporary.unlink(missing_ok=True)
            raise AssertionError(f"column {name} changed in the compatible copy")
    a_ids, a_mat, a_ok, _ = decode_vectors_from_table(table, vector_column, flag_column,
                                                       expected_dim, str(source))
    b_ids, b_mat, b_ok, _ = decode_vectors_from_table(check, vector_column, flag_column,
                                                       expected_dim, str(temporary))
    if not (np.array_equal(a_ids, b_ids) and np.array_equal(a_ok, b_ok)
            and np.array_equal(a_mat, b_mat)):
        temporary.unlink(missing_ok=True)
        raise AssertionError("vectors changed in the compatible copy")
    os.replace(temporary, destination)
    return {"source": str(source), "source_sha256": sha256_file(source),
            "source_reader": reader, "destination": str(destination),
            "destination_sha256": sha256_file(destination), "rows": len(a_ids),
            "available": int(a_ok.sum()), "dim": expected_dim,
            "verified_exact": True, "pyarrow": pa.__version__}


def union_vectors(sources: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                  check_overlap: bool = True) -> tuple[dict, dict]:
    """First source wins (train/validation lookup first, then test)."""
    lookup: dict[int, np.ndarray] = {}
    known: set[int] = set()
    mismatches = overlaps = 0
    for ids, matrix, available in sources:
        rows = np.cumsum(available) - 1
        for i, article_id in enumerate(ids.tolist()):
            vector = matrix[rows[i]] if available[i] else None
            if article_id in known:
                overlaps += 1
                if check_overlap and vector is not None and article_id in lookup:
                    if not np.allclose(lookup[article_id], vector, atol=1e-6):
                        mismatches += 1
                continue
            known.add(article_id)
            if vector is not None:
                lookup[article_id] = vector
    return lookup, {"ids_known": len(known), "ids_with_vector": len(lookup),
                    "overlapping_ids": overlaps, "overlap_vector_mismatches": mismatches,
                    "known_ids": known}


@dataclass
class ArticleTable:
    """Index-aligned article inputs. Row 0 is padding.

    Frozen vectors are stored compactly: ``text_row``/``image_row`` point into
    ``text``/``image`` matrices whose row 0 is a zero placeholder used only
    together with the ``has_text``/``has_image`` = False flags.
    """
    article_ids: np.ndarray        # [A] original IDs, row 0 = -1
    category: np.ndarray           # [A] int64: 0 pad, 1 missing, 2 unknown, 3+ vocab
    article_type: np.ndarray       # [A] int64, same convention
    topics: np.ndarray             # [A, T] int64: 0 pad, 1 unknown, 2+ vocab
    numeric: np.ndarray            # [A, F] float32 standardised + flags
    numeric_names: list
    published_s: np.ndarray        # [A] int64 epoch seconds (0 = missing)
    published_missing: np.ndarray  # [A] bool
    text_row: np.ndarray           # [A] int64 row in ``text`` (0 = none)
    text: np.ndarray               # [1 + n_text, Dt] float32
    has_text: np.ndarray           # [A] bool
    image_row: np.ndarray          # [A] int64 row in ``image`` (0 = none)
    image: np.ndarray              # [1 + n_image, Di] float32
    has_image: np.ndarray          # [A] bool
    id_bucket: np.ndarray          # [A] int64 learned-ID index (0 = OOV / off)

    def rows_for(self, article_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(article_ids, dtype=np.int64)
        # article_ids is [-1, sorted IDs...], so a binary search maps ID -> row.
        pos = np.searchsorted(self.article_ids, ids)
        pos = np.clip(pos, 0, len(self.article_ids) - 1)
        return np.where(self.article_ids[pos] == ids, pos, 0).astype(np.int64)


def fit_article_stats(catalogue: pd.DataFrame, train_article_ids: np.ndarray) -> dict:
    """Standardisation stats from articles referenced by train only."""
    present = catalogue.index.intersection(pd.Index(np.unique(train_article_ids)))
    sub = catalogue.loc[present]
    stats = {}
    for name, values in {
        "log_body_words": np.log1p(sub["body_word_count"].astype("float64")),
        "sentiment": (sub["sentiment_score"].astype("float64")
                      if "sentiment_score" in sub else pd.Series(dtype="float64")),
    }.items():
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        mean = float(values.mean()) if len(values) else 0.0
        std = float(values.std()) if len(values) > 1 else 1.0
        stats[name] = {"mean": mean, "std": std if std > 1e-6 else 1.0}
    stats["fit_articles"] = int(len(sub))
    return stats


NUMERIC_NAMES = ["log_body_words_z", "body_words_missing", "sentiment_z",
                 "sentiment_missing", "premium", "premium_missing", "in_catalogue",
                 "has_text_vector", "has_image_vector"]


def build_article_table(catalogue: pd.DataFrame, all_ids: np.ndarray, vocab: dict,
                        article_stats: dict, text_lookup: dict, image_lookup: dict,
                        text_dim: int, image_dim: int, max_topics: int = 8,
                        id_vocab: dict | None = None) -> tuple[ArticleTable, dict]:
    """One row per article in ``all_ids`` (catalogue articles and any referenced
    ID). IDs absent from the catalogue are kept with missing metadata flags."""
    ids = np.unique(np.asarray(all_ids, dtype=np.int64))
    ids = ids[ids > 0]
    n = len(ids) + 1
    category_vocab = {normalise_token(v): i + 3 for i, v in enumerate(vocab["category"])}
    type_vocab = {normalise_token(v): i + 3 for i, v in enumerate(vocab["article_type"])}
    topic_vocab = {str(v): i + 2 for i, v in enumerate(vocab["topics"])}
    found = np.isin(ids, catalogue.index.to_numpy())
    sub = catalogue.reindex(ids)

    category = np.zeros(n, np.int64)
    article_type = np.zeros(n, np.int64)
    category[1:] = [1 if c is None or (isinstance(c, float) and math.isnan(c)) or pd.isna(c)
                    else category_vocab.get(normalise_token(c), 2) for c in sub["category"]]
    article_type[1:] = [1 if t is None or (isinstance(t, float) and math.isnan(t))
                        else type_vocab.get(str(t), 2) for t in sub["article_type"]]
    topics = np.zeros((n, max_topics), np.int64)
    for row, tops in enumerate(sub["topics"], start=1):
        if isinstance(tops, (list, tuple, np.ndarray)):
            unique_topics = list(dict.fromkeys(str(t) for t in tops if t is not None))
            for j, topic in enumerate(unique_topics[:max_topics]):
                topics[row, j] = topic_vocab.get(topic, 1)
    published = sub["published_time"]
    ok = published.notna().to_numpy()
    published_s = np.zeros(n, np.int64)
    published_s[1:][ok] = (published[ok].astype("datetime64[us]").astype("int64")
                           .to_numpy() // 1_000_000)
    published_missing = np.ones(n, bool)
    published_missing[1:] = ~ok

    def standardise(values, key):
        missing = ~np.isfinite(values)
        scaled = (values - article_stats[key]["mean"]) / article_stats[key]["std"]
        return np.where(missing, 0.0, scaled), missing.astype(float)

    log_words = np.r_[np.nan, np.log1p(sub["body_word_count"].astype("float64").to_numpy())]
    sentiment = np.r_[np.nan, sub["sentiment_score"].astype("float64").to_numpy()
                      if "sentiment_score" in sub else np.full(n - 1, np.nan)]
    premium = np.r_[np.nan, sub["premium"].map(
        lambda v: np.nan if v is None or pd.isna(v) else float(bool(v))).to_numpy(dtype=float)]

    def compact(lookup, dim):
        rows = np.zeros(n, np.int64)
        present = [i for i, a in enumerate(ids.tolist(), start=1) if a in lookup]
        matrix = np.zeros((len(present) + 1, dim), np.float32)
        for k, i in enumerate(present, start=1):
            matrix[k] = lookup[int(ids[i - 1])]
            rows[i] = k
        return rows, matrix, rows > 0

    text_row, text, has_text = compact(text_lookup, text_dim)
    image_row, image, has_image = compact(image_lookup, image_dim)
    in_catalogue = np.r_[False, found]
    words, words_missing = standardise(log_words, "log_body_words")
    sent, sent_missing = standardise(sentiment, "sentiment")
    prem_missing = ~np.isfinite(premium)
    numeric = np.column_stack([
        words, words_missing, sent, sent_missing,
        np.where(prem_missing, 0.0, premium), prem_missing.astype(float),
        in_catalogue.astype(float), has_text.astype(float), has_image.astype(float),
    ]).astype(np.float32)
    numeric[0] = 0.0
    id_bucket = np.zeros(n, np.int64)
    if id_vocab:
        id_bucket[1:] = [id_vocab.get(a, 0) for a in ids.tolist()]
    table = ArticleTable(np.r_[-1, ids], category, article_type, topics, numeric,
                         list(NUMERIC_NAMES), published_s, published_missing,
                         text_row, text, has_text, image_row, image, has_image, id_bucket)
    info = {"articles": len(ids), "not_in_catalogue": int((~found).sum()),
            "category_unknown": int((category == 2).sum()),
            "category_missing": int((category[1:] == 1).sum()),
            "type_unknown": int((article_type == 2).sum()),
            "published_time_missing": int(published_missing[1:].sum()),
            "with_text_vector": int(has_text.sum()),
            "with_image_vector": int(has_image.sum())}
    return table, info


def fit_id_vocab(train_article_ids: np.ndarray, min_count: int) -> dict:
    """Optional learned article-ID buckets, fitted on train references only."""
    ids, counts = np.unique(np.asarray(train_article_ids, dtype=np.int64),
                            return_counts=True)
    keep = ids[(counts >= min_count) & (ids > 0)]
    return {int(a): i + 1 for i, a in enumerate(keep.tolist())}


# --------------------------------------------------------------------------
# Model-ready arrays for one split
# --------------------------------------------------------------------------

def _z(values, stats):
    return (values - stats["mean"]) / stats["std"]


def candidate_age_hours(published_s: np.ndarray, published_missing: np.ndarray,
                        request_time_s: np.ndarray) -> np.ndarray:
    """Article age at the request in hours; NaN if unknown or published later.

    Same rule as the shared Step 3 ``article_age_h``, but computed from the
    catalogue so it is constructible for any article, inside or outside the
    recorded slate.
    """
    age = (np.asarray(request_time_s, dtype=np.int64)
           - np.asarray(published_s, dtype=np.int64)) / 3600.0
    return np.where(np.asarray(published_missing) | (age < 0), np.nan, age)


def candidate_features(age_h: np.ndarray, stats: dict) -> np.ndarray:
    """[log_age_h_z, age_missing]. Pool-independent by design (no slate size or
    within-slate rank, which a catalogue request does not have)."""
    missing = ~np.isfinite(age_h)
    log_age = np.log1p(np.clip(np.where(missing, 0.0, age_h), 0, None))
    return np.column_stack([np.where(missing, 0.0, _z(log_age, stats["log_candidate_age_h"])),
                            missing.astype(float)]).astype(np.float32)


CANDIDATE_FEATURE_NAMES = ["log_age_h_z", "age_missing"]


def fit_numeric_stats(frame: pd.DataFrame, hist: HistoryArrays,
                      impression_offsets: np.ndarray, articles: ArticleTable) -> dict:
    """Train-only standardisation constants for candidate, context and history."""
    def moments(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            return {"mean": 0.0, "std": 1.0}
        std = float(values.std())
        return {"mean": float(values.mean()), "std": std if std > 1e-6 else 1.0}

    starts = impression_offsets[:-1]
    imp_time = (frame["impression_time"].astype("datetime64[us]").astype("int64")
                .to_numpy()[starts] // 1_000_000)
    rows_c = articles.rows_for(frame["candidate_article_id"].to_numpy())
    age = candidate_age_hours(articles.published_s[rows_c], articles.published_missing[rows_c],
                              np.repeat(imp_time, np.diff(impression_offsets)))
    users = frame["user_id"].to_numpy()[starts]
    rows = hist.rows_for(users)
    mask = np.arange(hist.times_s.shape[1])[None, :] < hist.lengths[rows][:, None]
    gaps = (imp_time[:, None] - hist.times_s[rows]) / 3600.0
    hmask = np.arange(hist.times_s.shape[1])[None, :] < hist.lengths[:, None]
    prev_gap = np.diff(hist.times_s, axis=1, prepend=0) / 3600.0
    prev_mask = hmask & (np.arange(hist.times_s.shape[1])[None, :] > 0)
    return {
        "log_candidate_age_h": moments(np.log1p(np.clip(age, 0, None))),
        "log_history_length": moments(np.log1p(hist.full_lengths[rows])),
        "log_gap_to_impression_h": moments(np.log1p(np.clip(gaps[mask], 0, None))),
        "log_read_time": moments(np.log1p(hist.read_time[hmask])),
        "scroll_fraction": moments(hist.scroll[hmask] / 100.0),
        "log_gap_prev_h": moments(np.log1p(np.clip(prev_gap[prev_mask], 0, None))),
        "fit_partition": "train (after any development sampling)",
    }


@dataclass
class SplitArrays:
    split: str
    user_id: np.ndarray            # [I] per impression (request)
    impression_id: np.ndarray      # [I]
    imp_time_s: np.ndarray         # [I] int64
    imp_user_row: np.ndarray       # [I] row into history arrays
    imp_context: np.ndarray        # [I, C] float32 (request context, pool-independent)
    offsets: np.ndarray            # [I+1] recorded-slate offsets
    cand_article_id: np.ndarray    # [N] original
    cand_article_row: np.ndarray   # [N] article table row
    cand_features: np.ndarray      # [N, K] float32
    labels: np.ndarray             # [N] float32 (NaN for test)
    hist_article_row: np.ndarray   # [U+1, L]
    hist_times_s: np.ndarray       # [U+1, L]
    hist_features: np.ndarray      # [U+1, L, F] float32
    hist_lengths: np.ndarray       # [U+1]
    hist_full_lengths: np.ndarray  # [U+1]
    context_names: list
    candidate_names: list
    event_names: list

    @property
    def n_impressions(self):
        return len(self.user_id)

    @property
    def n_candidates(self):
        return len(self.cand_article_id)


def build_split_arrays(split: str, frame: pd.DataFrame, offsets: np.ndarray,
                       hist: HistoryArrays, articles: ArticleTable, stats: dict,
                       device_columns: list) -> SplitArrays:
    starts = offsets[:-1]
    first = frame.iloc[starts]
    imp_time = (first["impression_time"].astype("datetime64[us]").astype("int64")
                .to_numpy() // 1_000_000)
    user_rows = hist.rows_for(first["user_id"].to_numpy())

    # Request context: known before recommending and independent of the article
    # pool (the recorded slate size is deliberately not used).
    hour = first["hour"].to_numpy(dtype="float64")
    dow = first["day_of_week"].to_numpy(dtype="int64")
    ctx = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]
    names = ["hour_sin", "hour_cos"]
    for d in range(7):
        ctx.append((dow == d).astype(float))
        names.append(f"dow_{d}")
    for column in CONTEXT_BOOL_COLUMNS:
        values = first[column]
        missing = values.isna().to_numpy()
        ctx.append(np.where(missing, 0.0, values.fillna(False).astype(bool)
                            .to_numpy(dtype=float)))
        ctx.append(missing.astype(float))
        names += [column, f"{column}_missing"]
    for column in device_columns:
        ctx.append(first[column].to_numpy(dtype="float64"))
        names.append(column)
    full_len = hist.full_lengths[user_rows]
    ctx.append(_z(np.log1p(full_len), stats["log_history_length"]))
    ctx.append((full_len == 0).astype(float))
    names += ["log_history_length_z", "history_empty"]
    imp_context = np.column_stack(ctx).astype(np.float32)

    article_rows = articles.rows_for(frame["candidate_article_id"].to_numpy())
    if (article_rows == 0).any():
        raise AssertionError("candidate missing from article table")
    age = candidate_age_hours(articles.published_s[article_rows],
                              articles.published_missing[article_rows],
                              np.repeat(imp_time, np.diff(offsets)))
    cand = candidate_features(age, stats)

    # History event features (per user, impression-independent).
    hist_rows = articles.rows_for(hist.article_ids)
    length = hist.lengths
    mask = np.arange(hist.article_ids.shape[1])[None, :] < length[:, None]
    hist_rows = np.where(mask, hist_rows, 0)
    if (hist_rows[mask] == 0).any():
        raise AssertionError("history article missing from article table")
    read_missing = ~np.isfinite(hist.read_time)
    scroll_missing = ~np.isfinite(hist.scroll)
    read_z = _z(np.log1p(np.where(read_missing, 0, hist.read_time)), stats["log_read_time"])
    scroll_z = _z(np.where(scroll_missing, 0, hist.scroll) / 100.0, stats["scroll_fraction"])
    prev = np.diff(hist.times_s, axis=1, prepend=0) / 3600.0
    prev_missing = ~mask | (np.arange(mask.shape[1])[None, :] == 0)
    prev_z = np.where(prev_missing, 0.0,
                      _z(np.log1p(np.clip(prev, 0, None)), stats["log_gap_prev_h"]))
    pub = articles.published_s[hist_rows]
    pub_missing = articles.published_missing[hist_rows]
    age_read_h = (hist.times_s - pub) / 3600.0
    age_read_missing = pub_missing | ~mask | (age_read_h < 0)
    age_read_z = np.where(age_read_missing, 0.0,
                          _z(np.log1p(np.clip(age_read_h, 0, None)),
                             stats["log_candidate_age_h"]))
    events = np.stack([
        np.where(read_missing, 0.0, read_z), read_missing.astype(float),
        np.where(scroll_missing, 0.0, scroll_z), scroll_missing.astype(float),
        prev_z, prev_missing.astype(float), age_read_z, age_read_missing.astype(float),
    ], axis=-1).astype(np.float32)
    events[~mask] = 0.0
    event_names = ["log_read_time_z", "read_time_missing", "scroll_z",
                   "scroll_missing", "log_gap_prev_h_z", "gap_prev_missing",
                   "log_article_age_at_read_z", "age_at_read_missing"]

    gaps = imp_time[:, None] - hist.times_s[user_rows]
    gap_mask = np.arange(gaps.shape[1])[None, :] < hist.lengths[user_rows][:, None]
    if (gaps[gap_mask] <= 0).any():
        raise AssertionError(f"{split}: a history event is not before its impression")

    return SplitArrays(
        split=split,
        user_id=first["user_id"].to_numpy(dtype=np.int64),
        impression_id=first["impression_id"].to_numpy(dtype=np.int64),
        imp_time_s=imp_time, imp_user_row=user_rows, imp_context=imp_context,
        offsets=offsets.astype(np.int64),
        cand_article_id=frame["candidate_article_id"].to_numpy(dtype=np.int64),
        cand_article_row=article_rows, cand_features=cand,
        labels=frame["clicked"].to_numpy(dtype=np.float32, na_value=np.nan),
        hist_article_row=hist_rows, hist_times_s=hist.times_s,
        hist_features=events, hist_lengths=hist.lengths,
        hist_full_lengths=hist.full_lengths,
        context_names=names, candidate_names=list(CANDIDATE_FEATURE_NAMES),
        event_names=event_names)


def permute_history(arrays: SplitArrays, seed: int) -> SplitArrays:
    """Order-only control: permute each user's retained events with a fixed seed.

    Run this *after* build_split_arrays, so every event keeps exactly the
    features computed in true chronological order (including its gap to the
    previous event, read time, scroll, age at read and timestamp, hence its gap
    to the impression). The ordered and shuffled variants therefore receive the
    same per-event information; only the sequence order seen by the LSTM differs.
    """
    rng = np.random.default_rng(seed)
    rows, max_len = arrays.hist_article_row.shape
    keys = rng.random((rows, max_len))
    keys[np.arange(max_len)[None, :] >= arrays.hist_lengths[:, None]] = np.inf
    perm = np.argsort(keys, axis=1, kind="stable")
    take = lambda a: np.take_along_axis(a, perm if a.ndim == 2 else perm[..., None], axis=1)
    return SplitArrays(**{**arrays.__dict__,
                          "hist_article_row": take(arrays.hist_article_row),
                          "hist_times_s": take(arrays.hist_times_s),
                          "hist_features": take(arrays.hist_features)})


def coverage_report(split: SplitArrays, articles: ArticleTable) -> dict:
    cand = split.cand_article_row
    mask = np.arange(split.hist_article_row.shape[1])[None, :] < split.hist_lengths[:, None]
    hist = split.hist_article_row[mask]
    out = {}
    for name, rows in (("candidates", cand), ("history_events", hist)):
        out[name] = {
            "rows": int(len(rows)),
            "in_catalogue": float(articles.numeric[rows, 6].mean()) if len(rows) else None,
            "text_vector": float(articles.has_text[rows].mean()) if len(rows) else None,
            "image_vector": float(articles.has_image[rows].mean()) if len(rows) else None,
            "category_known": float((articles.category[rows] >= 3).mean()) if len(rows) else None,
        }
    return out


# --------------------------------------------------------------------------
# Catalogue eligibility (prediction pool for recommendation requests)
# --------------------------------------------------------------------------

@dataclass
class EligibilityPolicy:
    """Versioned rule for which articles a request may be recommended.

    An article is eligible for a request at time t when it is in the split's
    catalogue snapshot, has a known publication time, and was published at or
    before t (never a future article). ``max_age_hours`` is an optional,
    explicitly declared freshness limit; None means no age limit.
    """
    snapshot_by_split: dict
    max_age_hours: float | None = None
    require_published_time: bool = True
    version: str = "time_eligible_catalogue_v1"

    def to_json(self) -> dict:
        return {"version": self.version, "prediction_pool": "time_eligible_catalogue",
                "snapshot_by_split": dict(self.snapshot_by_split),
                "published_at_or_before_request": True,
                "missing_published_time": "ineligible (availability cannot be verified)",
                "require_published_time": self.require_published_time,
                "max_age_hours": self.max_age_hours}


@dataclass
class CatalogueIndex:
    """A split's eligible-article snapshot, sorted by (published time, article ID),
    so the articles eligible at time t form one contiguous slice."""
    split: str
    snapshot: str
    rows: np.ndarray            # [P] article-table rows, sorted
    published_s: np.ndarray     # [P] sorted publication times
    article_ids: np.ndarray     # [P] original IDs in the same order
    in_snapshot: np.ndarray     # [A] bool over the article table
    snapshot_size: int
    excluded_missing_published: int
    cum_text: np.ndarray        # [P+1] prefix counts of text-vector availability
    cum_image: np.ndarray       # [P+1]

    def bounds(self, request_time_s, max_age_hours=None):
        t = np.asarray(request_time_s, dtype=np.int64)
        hi = np.searchsorted(self.published_s, t, side="right")
        if max_age_hours is None:
            lo = np.zeros_like(hi)
        else:
            lo = np.searchsorted(self.published_s, t - int(round(max_age_hours * 3600)),
                                 side="left")
        return lo.astype(np.int64), hi.astype(np.int64)


def build_catalogue_index(split: str, snapshot: str, snapshot_ids: np.ndarray,
                          articles: ArticleTable) -> CatalogueIndex:
    rows = articles.rows_for(snapshot_ids)
    if (rows == 0).any():
        raise AssertionError(f"{snapshot}: catalogue IDs missing from the article table")
    missing = articles.published_missing[rows]
    rows = rows[~missing]
    order = np.lexsort((articles.article_ids[rows], articles.published_s[rows]))
    rows = rows[order]
    in_snapshot = np.zeros(len(articles.article_ids), bool)
    in_snapshot[articles.rows_for(snapshot_ids)] = True
    return CatalogueIndex(
        split=split, snapshot=snapshot, rows=rows, published_s=articles.published_s[rows],
        article_ids=articles.article_ids[rows], in_snapshot=in_snapshot,
        snapshot_size=int(len(snapshot_ids)), excluded_missing_published=int(missing.sum()),
        cum_text=np.r_[0, np.cumsum(articles.has_text[rows])],
        cum_image=np.r_[0, np.cumsum(articles.has_image[rows])])


def eligibility_summary(arrays: SplitArrays, index: CatalogueIndex,
                        max_age_hours=None, bytes_per_score_row: float | None = None,
                        requests=None) -> dict:
    times = arrays.imp_time_s if requests is None else arrays.imp_time_s[np.asarray(requests)]
    lo, hi = index.bounds(times, max_age_hours)
    count = hi - lo
    pairs = int(count.sum())
    text_pairs = int((index.cum_text[hi] - index.cum_text[lo]).sum())
    image_pairs = int((index.cum_image[hi] - index.cum_image[lo]).sum())
    out = {
        "split": arrays.split, "snapshot": index.snapshot,
        "snapshot_articles": index.snapshot_size,
        "snapshot_excluded_missing_published_time": index.excluded_missing_published,
        "snapshot_with_text_vector": int(index.cum_text[-1]),
        "snapshot_with_image_vector": int(index.cum_image[-1]),
        "requests": int(len(times)), "eligible_pairs": pairs,
        "requests_in_split": int(arrays.n_impressions),
        "empty_pools": int((count == 0).sum()),
        "pool_size_min_median_max": ([int(count.min()), float(np.median(count)),
                                      int(count.max())] if len(count) else None),
        "pair_share_with_text_vector": text_pairs / pairs if pairs else None,
        "pair_share_with_image_vector": image_pairs / pairs if pairs else None,
        "max_age_hours": max_age_hours,
    }
    if bytes_per_score_row:
        out["estimated_full_score_csv_gb"] = pairs * bytes_per_score_row / 1e9
    return out


CLICK_EXCLUSION_REASONS = ("not_in_catalogue_snapshot", "published_time_missing",
                           "published_after_request", "older_than_max_age")


def clicked_eligibility(arrays: SplitArrays, index: CatalogueIndex,
                        articles: ArticleTable, max_age_hours=None) -> dict:
    """Classify every recorded clicked article as eligible or excluded (with reason)."""
    if np.isnan(arrays.labels).all():
        return {"labelled": False}
    sizes = np.diff(arrays.offsets)
    imp = np.repeat(np.arange(arrays.n_impressions), sizes)
    clicked = arrays.labels == 1
    rows = arrays.cand_article_row[clicked]
    t = arrays.imp_time_s[imp[clicked]]
    reason = np.full(len(rows), "", dtype=object)
    reason[~index.in_snapshot[rows]] = "not_in_catalogue_snapshot"
    pub_missing = articles.published_missing[rows]
    reason[(reason == "") & pub_missing] = "published_time_missing"
    pub = articles.published_s[rows]
    reason[(reason == "") & (pub > t)] = "published_after_request"
    if max_age_hours is not None:
        reason[(reason == "") & (pub < t - int(round(max_age_hours * 3600)))] = "older_than_max_age"
    eligible = reason == ""
    return {"labelled": True, "impression": imp[clicked], "article_id": arrays.cand_article_id[clicked],
            "eligible": eligible, "reason": reason,
            "summary": {"clicked_articles": int(len(rows)),
                        "eligible": int(eligible.sum()),
                        "eligibility_coverage": float(eligible.mean()) if len(rows) else None,
                        **{r: int((reason == r).sum()) for r in CLICK_EXCLUSION_REASONS}}}


def clicks_by_request(clicks: dict) -> dict:
    """impression index -> (clicked article IDs, eligible flags)."""
    if not clicks.get("labelled"):
        return {}
    frame = pd.DataFrame({"imp": clicks["impression"], "aid": clicks["article_id"],
                          "eligible": clicks["eligible"]})
    return {int(i): (g["aid"].to_numpy(), g["eligible"].to_numpy())
            for i, g in frame.groupby("imp", sort=False)}


def _ideal_dcg(n_relevant: int, k: int) -> float:
    return float(sum(1.0 / math.log2(r + 1) for r in range(1, min(n_relevant, k) + 1)))


def click_hits(click_lookup: dict, i: int, shortlist: np.ndarray, ks=(100,)):
    """Per-request observed-click retrieval bookkeeping; None without known clicks.

    ``shortlist`` must be in the exported order (descending score, ties by
    ascending article ID). For each K: hits@K (clicked IDs in the top K),
    DCG@K with gain 1 for an original known click and 0 observed gain otherwise,
    and the ideal DCG over **all** original known clicks (including clicks
    excluded by eligibility or missed by the shortlist), capped at K.
    """
    entry = click_lookup.get(int(i))
    if entry is None:
        return None
    aids, eligible = entry
    shortlist = np.asarray(shortlist, dtype=np.int64)
    gains = np.isin(shortlist, aids).astype(float)
    discounts = 1.0 / np.log2(np.arange(2, len(shortlist) + 2))
    out = {"n": int(len(aids)), "eligible": int(eligible.sum())}
    for k in ks:
        out[f"hits@{k}"] = int(np.isin(aids, shortlist[:k]).sum())
        out[f"dcg@{k}"] = float((gains[:k] * discounts[:k]).sum())
        out[f"idcg@{k}"] = _ideal_dcg(len(aids), k)
    return out


def recall_from_hits(hits: dict, clicks: dict, ks=(100,), requests_scored: int | None = None) -> dict:
    """Macro-averaged catalogue metrics over scored requests with known clicks.

    * known_click_recall@K: hits@K / all original known clicks (clicks rejected
      by eligibility stay in the denominator; an empty pool scores 0).
    * observed_click_ndcg@K: DCG@K / ideal DCG over all original known clicks.
    * eligible_only_recall@K: hits@K / eligible clicks, over requests with >= 1.
    Unshown articles have unknown outcomes; zero observed gain is bookkeeping,
    not a negative label.
    """
    if not clicks.get("labelled"):
        return {"labelled": False}
    frame = pd.DataFrame.from_dict(hits, orient="index")
    result = {"requests_with_known_clicks_evaluated": int(len(frame)),
              "requests_without_known_clicks_scored": (int(requests_scored - len(frame))
                                                       if requests_scored is not None else None),
              "known_clicks_evaluated": int(frame["n"].sum()) if len(frame) else 0,
              "known_clicks_eligible": int(frame["eligible"].sum()) if len(frame) else 0,
              "split_click_eligibility": clicks["summary"]}
    for k in ks:
        if frame.empty:
            result.update({f"known_click_recall@{k}": None, f"observed_click_ndcg@{k}": None,
                           f"eligible_only_recall@{k}": None})
            continue
        result[f"known_click_recall@{k}"] = float((frame[f"hits@{k}"] / frame["n"]).mean())
        result[f"observed_click_ndcg@{k}"] = float((frame[f"dcg@{k}"] / frame[f"idcg@{k}"]).mean())
        elig = frame[frame["eligible"] > 0]
        result[f"eligible_only_recall@{k}"] = (float((elig[f"hits@{k}"] / elig["eligible"]).mean())
                                               if len(elig) else None)
        result[f"eligible_only_requests@{k}"] = int(len(elig))
    return result


def per_request_metrics(hits: dict, requests, k: int = 100) -> dict:
    """Recall@K and observed-click nDCG@K per request (NaN without known clicks),
    aligned to ``requests`` for paired comparisons."""
    recall = np.full(len(requests), np.nan)
    ndcg = np.full(len(requests), np.nan)
    for j, i in enumerate(requests):
        h = hits.get(int(i))
        if h is not None:
            recall[j] = h[f"hits@{k}"] / h["n"]
            ndcg[j] = h[f"dcg@{k}"] / h[f"idcg@{k}"]
    return {"recall": recall, "ndcg": ndcg}


def known_click_recall(shortlists: dict, clicks: dict, ks=(100,)) -> dict:
    """Convenience wrapper over stored shortlists (index -> ordered IDs)."""
    lookup = clicks_by_request(clicks)
    hits = {i: h for i, top in shortlists.items()
            if (h := click_hits(lookup, i, np.asarray(top), ks)) is not None}
    return recall_from_hits(hits, clicks, ks, requests_scored=len(shortlists))


def top_k_reference(scores: np.ndarray, article_ids: np.ndarray, k: int) -> np.ndarray:
    """Reference ordering: descending score, ties by ascending article ID."""
    order = np.lexsort((article_ids, -np.asarray(scores, dtype=np.float64)))
    return np.asarray(article_ids)[order[:k]]


class ShortlistWriter:
    """Streams schema-v3 catalogue outputs for one split.

    * rankings/{split}/submission_top_100.csv - one row per request
    * predictions/{split}/eligibility_manifest.csv - pool size per request
    * full catalogue scores (every scored request, or a declared score sample)
    """

    def __init__(self, run_dir: Path, split: str, shortlist_size: int,
                 full_scores: bool, flush_every: int = 5000, sampled_requests: bool = False):
        """``sampled_requests``: only a declared subset of the split's requests is
        scored (e.g. a smoke run); every file then lives under a
        ``sampled_requests/`` folder so it can never pass as a complete export."""
        self.split, self.k = split, shortlist_size
        sub = "sampled_requests" if sampled_requests else ""
        self.rank_path = (Path(run_dir) / "rankings" / split / sub /
                          f"submission_top_{shortlist_size}.csv")
        base = Path(run_dir) / "predictions" / split / sub
        self.manifest_path = base / "eligibility_manifest.csv"
        # Full scores for every scored request, or only for a declared score sample.
        name = "submission_click_probabilities.csv"
        self.score_path = base / name if full_scores else base / "score_sample" / name
        for path in (self.rank_path, self.manifest_path, self.score_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.unlink(missing_ok=True)
        self.flush_every = flush_every
        self._ranks, self._manifest = [], []
        self._score_header = True
        self.score_rows = 0
        self.requests = 0

    def add(self, behaviour_id: str, request_time: str, eligible_count: int,
            shortlist_ids, scores=None, article_ids=None):
        if len(shortlist_ids) != min(self.k, eligible_count):
            raise AssertionError(f"{behaviour_id}: shortlist length {len(shortlist_ids)}")
        if len(set(shortlist_ids)) != len(shortlist_ids):
            raise AssertionError(f"{behaviour_id}: duplicate shortlist IDs")
        self._ranks.append((behaviour_id, json.dumps([int(a) for a in shortlist_ids])))
        self._manifest.append((behaviour_id, request_time, int(eligible_count),
                               int(len(shortlist_ids)), scores is not None))
        if scores is not None:
            frame = pd.DataFrame({"behaviour_id": behaviour_id,
                                  "article_id": np.asarray(article_ids, dtype=np.int64),
                                  "score": np.asarray(scores, dtype=np.float64)})
            frame["click_probability"] = 1.0 / (1.0 + np.exp(-frame["score"].to_numpy()))
            frame = frame.sort_values("article_id", kind="stable")
            frame.to_csv(self.score_path, mode="a", header=self._score_header, index=False)
            self._score_header = False
            self.score_rows += len(frame)
        self.requests += 1
        if len(self._ranks) >= self.flush_every:
            self.flush()

    def flush(self):
        if self._ranks:
            pd.DataFrame(self._ranks, columns=["behaviour_id", "article_ids"]).to_csv(
                self.rank_path, mode="a", header=not self.rank_path.exists(), index=False)
            pd.DataFrame(self._manifest, columns=["behaviour_id", "request_time",
                                                  "eligible_count", "shortlist_length",
                                                  "scores_exported"]).to_csv(
                self.manifest_path, mode="a", header=not self.manifest_path.exists(), index=False)
            self._ranks, self._manifest = [], []

    def close(self) -> dict:
        self.flush()
        if not self.rank_path.exists():   # zero requests: still write headers
            pd.DataFrame(columns=["behaviour_id", "article_ids"]).to_csv(self.rank_path, index=False)
        return {"rankings": str(self.rank_path), "manifest": str(self.manifest_path),
                "scores": str(self.score_path) if self.score_rows else None,
                "requests": self.requests, "score_rows": self.score_rows}


def validate_catalogue_exports(split_arrays: SplitArrays, index: CatalogueIndex,
                               outputs: dict, shortlist_size: int,
                               max_age_hours=None, max_score_requests: int = 200,
                               requests=None) -> dict:
    """Re-read the written files and check them against the eligibility rule and,
    for requests with exported scores, against the full score rows. ``requests``
    restricts the expected behaviours to a declared request subset."""
    ranks = pd.read_csv(outputs["rankings"], dtype={"behaviour_id": str})
    manifest = pd.read_csv(outputs["manifest"], dtype={"behaviour_id": str})
    keep = (np.arange(split_arrays.n_impressions) if requests is None
            else np.sort(np.asarray(requests, dtype=np.int64)))
    expected = behaviour_ids(split_arrays.user_id[keep], split_arrays.impression_id[keep])
    if ranks["behaviour_id"].duplicated().any():
        raise AssertionError("duplicate behaviours in the shortlist file")
    if set(ranks["behaviour_id"]) != set(expected):
        raise AssertionError("shortlist behaviours differ from the split's requests")
    lo, hi = index.bounds(split_arrays.imp_time_s[keep], max_age_hours)
    expected_counts = pd.Series(hi - lo, index=expected)
    got_counts = manifest.set_index("behaviour_id")["eligible_count"]
    if not got_counts.reindex(expected).eq(expected_counts).all():
        raise AssertionError("eligible counts differ from the eligibility rule")
    lists = ranks.set_index("behaviour_id")["article_ids"].map(json.loads)
    lengths = lists.map(len)
    if not lengths.reindex(expected).eq(np.minimum(shortlist_size, expected_counts)).all():
        raise AssertionError("shortlist length != min(K, eligible count)")
    # Membership: every listed article is eligible for that request.
    position = {b: i for i, b in enumerate(expected)}
    pub_by_id = dict(zip(index.article_ids.tolist(), index.published_s.tolist()))
    bad = 0
    for behaviour_id, ids in lists.items():
        i = keep[position[behaviour_id]]
        t = split_arrays.imp_time_s[i]
        floor = -np.inf if max_age_hours is None else t - max_age_hours * 3600
        for a in ids:
            p = pub_by_id.get(a)
            if p is None or p > t or p < floor:
                bad += 1
    if bad:
        raise AssertionError(f"{bad} shortlisted articles are not eligible for their request")
    checked_against_scores = 0
    if outputs.get("scores"):
        # Stream the score file and fully check up to ``max_score_requests``
        # requests (all rows of each), so multi-GB exports never load at once.
        exported = manifest.loc[manifest["scores_exported"].astype(bool), "behaviour_id"]
        selected = set(exported.head(max_score_requests).tolist())
        groups = {}
        for chunk in pd.read_csv(outputs["scores"], dtype={"behaviour_id": str},
                                 float_precision="round_trip", chunksize=2_000_000):
            p = chunk["click_probability"].to_numpy()
            if not (np.isfinite(chunk["score"]).all() and np.isfinite(p).all()
                    and (p >= 0).all() and (p <= 1).all()):
                raise AssertionError("non-finite score or probability outside [0, 1]")
            unknown = set(chunk["behaviour_id"].unique()) - set(exported)
            if unknown:
                raise AssertionError(f"scores for requests not marked as exported: {list(unknown)[:3]}")
            part = chunk[chunk["behaviour_id"].isin(selected)]
            for behaviour_id, group in part.groupby("behaviour_id", sort=False):
                groups.setdefault(behaviour_id, []).append(group)
        for behaviour_id in selected:
            group = pd.concat(groups.get(behaviour_id, [pd.DataFrame(columns=["article_id", "score"])]))
            if group["article_id"].duplicated().any():
                raise AssertionError(f"{behaviour_id}: duplicate article rows")
            if len(group) != expected_counts[behaviour_id]:
                raise AssertionError(f"{behaviour_id}: score rows != eligible count")
            reference = top_k_reference(group["score"].to_numpy(dtype=float),
                                        group["article_id"].to_numpy(dtype=np.int64),
                                        shortlist_size)
            if list(reference) != lists[behaviour_id]:
                raise AssertionError(f"{behaviour_id}: shortlist differs from its scores")
            checked_against_scores += 1
    return {"requests": int(len(ranks)), "eligible_pairs": int(expected_counts.sum()),
            "empty_pools": int((expected_counts == 0).sum()),
            "shortlists_checked_against_full_scores": checked_against_scores,
            "membership_and_length_checked": True}


# --------------------------------------------------------------------------
# Metrics: per-impression AUC and nDCG@k with fair tie handling
# --------------------------------------------------------------------------

def impression_metrics(offsets: np.ndarray, labels: np.ndarray, scores: np.ndarray,
                       k: int = 5, return_per_impression: bool = False) -> dict:
    """Equal-weight mean over impressions (composite key groups = offsets).

    AUC: Mann-Whitney with average ranks (ties get half credit); impressions
    without both classes are excluded. nDCG@k: cutoff min(k, slate); tied
    scores receive expected credit (mean label within the tie group at each
    position); impressions without a positive are excluded.
    """
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("non-finite scores")
    if np.isnan(labels).any():
        raise ValueError("metrics need labels; this split is unlabelled")
    sizes = np.diff(offsets)
    group = np.repeat(np.arange(len(sizes)), sizes)
    data = pd.DataFrame({"g": group, "y": labels, "s": scores})
    positives = np.bincount(group, weights=labels, minlength=len(sizes))
    negatives = sizes - positives
    ranks = data.groupby("g")["s"].rank(method="average").to_numpy()
    rank_sum = np.bincount(group, weights=ranks * labels, minlength=len(sizes))
    auc_ok = (positives > 0) & (negatives > 0)
    auc = np.full(len(sizes), np.nan)
    auc[auc_ok] = ((rank_sum[auc_ok] - positives[auc_ok] * (positives[auc_ok] + 1) / 2)
                   / (positives[auc_ok] * negatives[auc_ok]))

    data["tie_y"] = data.groupby(["g", "s"])["y"].transform("mean")
    ranked = data.sort_values(["g", "s"], ascending=[True, False], kind="stable")
    position = ranked.groupby("g").cumcount().to_numpy() + 1
    top = position <= k
    discount = 1.0 / np.log2(position + 1)
    dcg = np.bincount(ranked["g"].to_numpy(), weights=ranked["tie_y"].to_numpy()
                      * discount * top, minlength=len(sizes))
    ideal_prefix = np.r_[0.0, np.cumsum(1.0 / np.log2(np.arange(1, k + 1) + 1))]
    ideal = ideal_prefix[np.minimum(positives, np.minimum(k, sizes)).astype(int)]
    ndcg_ok = positives > 0
    ndcg = np.full(len(sizes), np.nan)
    ndcg[ndcg_ok] = dcg[ndcg_ok] / ideal[ndcg_ok]
    result = {
        "mean_impression_auc": float(np.nanmean(auc)) if auc_ok.any() else None,
        f"mean_impression_ndcg@{k}": float(np.nanmean(ndcg)) if ndcg_ok.any() else None,
        "candidate_rows": int(len(labels)), "impressions": int(len(sizes)),
        "auc_eligible_impressions": int(auc_ok.sum()),
        "auc_excluded_single_class": int((~auc_ok).sum()),
        f"ndcg@{k}_eligible_impressions": int(ndcg_ok.sum()),
        f"ndcg@{k}_excluded_no_positive": int((~ndcg_ok).sum()),
    }
    if return_per_impression:
        result["_per_impression"] = {"auc": auc, f"ndcg@{k}": ndcg}
    return result


METRIC_DEFINITIONS = {
    "mean_impression_auc": (
        "Per (user_id, impression_id) slate: Mann-Whitney AUC of clicked vs "
        "unclicked candidates with average ranks (tied scores get half credit); "
        "equal-weight mean over impressions with both label classes. Primary "
        "metric, maximised for early stopping, checkpoint and tuning selection."),
    "mean_impression_ndcg@5": (
        "Per slate: binary-gain DCG over the top min(5, slate size) positions, tied "
        "scores given expected credit (mean label of the tie group at each position), "
        "divided by the ideal DCG; equal-weight mean over impressions with at least one "
        "click. Secondary metric."),
}


def user_cluster_bootstrap(users: np.ndarray, a: np.ndarray, b: np.ndarray,
                           n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired difference mean(a - b) over impressions eligible in both, with a
    95% interval from resampling users (impressions of a user are correlated)."""
    ok = np.isfinite(a) & np.isfinite(b)
    users, diff = np.asarray(users)[ok], (a - b)[ok]
    codes, uniques = pd.factorize(users)
    sums = np.bincount(codes, weights=diff)
    counts = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    chunk = max(1, int(2e7 // max(len(uniques), 1)))
    for start in range(0, n_boot, chunk):
        stop = min(n_boot, start + chunk)
        draws = rng.integers(0, len(uniques), size=(stop - start, len(uniques)))
        boot[start:stop] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"mean_difference": float(diff.mean()) if len(diff) else None,
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
            if len(diff) else None,
            "impressions": int(ok.sum()), "users": int(len(uniques)),
            "n_boot": n_boot, "seed": seed}


# --------------------------------------------------------------------------
# Exports (schema version 2)
# --------------------------------------------------------------------------

def behaviour_ids(user_id: np.ndarray, impression_id: np.ndarray) -> np.ndarray:
    return (pd.Series(np.asarray(user_id, dtype=np.int64)).astype(str) + ":"
            + pd.Series(np.asarray(impression_id, dtype=np.int64)).astype(str)).to_numpy()


def score_frame(split: SplitArrays, scores: np.ndarray,
                probabilities: np.ndarray | None = None) -> pd.DataFrame:
    sizes = np.diff(split.offsets)
    bid = np.repeat(behaviour_ids(split.user_id, split.impression_id), sizes)
    frame = pd.DataFrame({"behaviour_id": bid,
                          "article_id": split.cand_article_id.astype(np.int64),
                          "score": np.asarray(scores, dtype=np.float64)})
    if probabilities is not None:
        frame["click_probability"] = np.asarray(probabilities, dtype=np.float64)
    return frame.sort_values(["behaviour_id", "article_id"], kind="stable",
                             ignore_index=True)


def ranking_frame(scores: pd.DataFrame) -> pd.DataFrame:
    ordered = scores.sort_values(["behaviour_id", "score", "article_id"],
                                 ascending=[True, False, True], kind="stable")
    lists = ordered.groupby("behaviour_id", sort=True)["article_id"].agg(list)
    return pd.DataFrame({"behaviour_id": lists.index,
                         "article_ids": [json.dumps([int(a) for a in v]) for v in lists]})


def validate_exports(split: SplitArrays, scores: pd.DataFrame,
                     rankings: pd.DataFrame) -> dict:
    expected = score_frame(split, np.zeros(split.n_candidates))[["behaviour_id", "article_id"]]
    if list(scores.columns[:3]) != ["behaviour_id", "article_id", "score"]:
        raise AssertionError(f"score columns {list(scores.columns)}")
    if scores.duplicated(["behaviour_id", "article_id"]).any():
        raise AssertionError("duplicate candidate keys in scores")
    if len(scores) != len(expected) or not (
            scores[["behaviour_id", "article_id"]].reset_index(drop=True)
            .equals(expected.reset_index(drop=True))):
        raise AssertionError("score keys differ from the source candidate keys")
    if not np.isfinite(scores["score"].to_numpy()).all():
        raise AssertionError("non-finite scores")
    if "click_probability" in scores:
        p = scores["click_probability"].to_numpy()
        if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
            raise AssertionError("click_probability outside [0, 1]")
    if rankings["behaviour_id"].duplicated().any():
        raise AssertionError("duplicate behaviour rows in rankings")
    groups = scores.groupby("behaviour_id", sort=True)
    if len(rankings) != groups.ngroups:
        raise AssertionError("ranking rows != behaviours")
    lookup = dict(zip(rankings["behaviour_id"], rankings["article_ids"]))
    reference = ranking_frame(scores)
    for behaviour_id, encoded in zip(reference["behaviour_id"], reference["article_ids"]):
        if lookup.get(behaviour_id) != encoded:
            raise AssertionError(f"ranking mismatch for {behaviour_id}")
    sizes = groups.size()
    decoded_sizes = rankings.set_index("behaviour_id")["article_ids"].map(
        lambda s: len(json.loads(s)))
    if not decoded_sizes.reindex(sizes.index).eq(sizes).all():
        raise AssertionError("ranking length differs from slate size")
    return {"score_rows": len(scores), "behaviours": len(rankings),
            "complete_slate_coverage": True,
            "score_min_max": [float(scores["score"].min()), float(scores["score"].max())]}


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp",
                                     delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_finite(obj), indent=2, default=_json_default,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def _finite(value):
    """Replace NaN/inf with None so the JSON is strict."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def file_manifest(root: Path, hash_files: bool = True) -> dict:
    root = Path(root)
    entries = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        entries[relative] = {"bytes": path.stat().st_size,
                             "sha256": sha256_file(path) if hash_files else None}
    return entries


def _copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def persist_directory(source: Path, destination: Path) -> dict:
    """Copy a run folder to persistent storage without ever overwriting a good
    file in place: each changed file goes to a temporary name and is renamed;
    files whose size and hash already match are skipped; manifest.json last.
    Then every file is verified against the local hashes."""
    source, destination = Path(source), Path(destination)
    local = file_manifest(source)
    copied = skipped = 0
    for relative, entry in local.items():
        target = destination / relative
        if (target.is_file() and target.stat().st_size == entry["bytes"]
                and sha256_file(target) == entry["sha256"]):
            skipped += 1
            continue
        _copy_atomic(source / relative, target)
        copied += 1
    if (source / "manifest.json").is_file():
        _copy_atomic(source / "manifest.json", destination / "manifest.json")
    remote = {k: {"bytes": (destination / k).stat().st_size,
                  "sha256": sha256_file(destination / k)}
              for k in local if (destination / k).is_file()}
    missing = sorted(set(local) - set(remote))
    different = sorted(k for k in local if k in remote and local[k] != remote[k])
    return {"destination": str(destination), "files": len(local), "copied": copied,
            "already_identical": skipped, "missing": missing, "different": different,
            "verified": not missing and not different, "manifest": local}
