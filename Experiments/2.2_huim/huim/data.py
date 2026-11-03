"""Load and check the shared Drive inputs used by the HUIM Stage 1 experiment.

Nothing here modifies the shared source folder. Paths are resolved from a
configurable data root plus an optional ``loader_paths.json``-style override map.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .io_utils import file_record, read_json

SPLITS = ("train", "validation", "test")
DEVELOPMENT_SPLITS = ("train", "validation")

# Article columns used by the model. Totals (inviews/pageviews/read time) are
# first-seven-day aggregates and last_modified_time is later than the impression:
# both are excluded as leakage (Interim Report 2.1.4).
ARTICLE_COLUMNS = [
    "article_id", "published_time", "premium", "category", "subcategory",
    "article_type", "topics", "sentiment_label", "body",
]
EXCLUDED_ARTICLE_COLUMNS = {
    "total_inviews": "future aggregate over the release period",
    "total_pageviews": "future aggregate over the release period",
    "total_read_time": "future aggregate over the release period",
    "last_modified_time": "later than the impression for most rows",
}
BEHAVIOUR_COLUMNS_REQUIRED = ["user_id", "impression_id", "impression_time", "article_ids_inview"]
BEHAVIOUR_COLUMNS_OPTIONAL = ["device_type", "is_sso_user", "is_subscriber", "article_ids_clicked",
                              "is_beyond_accuracy", "article_id"]
# Current-page/next-page outcomes are post-impression and never loaded as inputs.
EXCLUDED_BEHAVIOUR_COLUMNS = {
    "read_time": "current-page dwell outcome (post-impression)",
    "scroll_percentage": "current-page scroll outcome (post-impression)",
    "next_read_time": "subsequent-article outcome",
    "next_scroll_percentage": "subsequent-article outcome",
    "article_id": "current page: absent from official test behaviours, so not usable as a model input",
    "gender": "mostly null; not used in this compact model",
    "postcode": "mostly null; not used in this compact model",
    "age": "mostly null; not used in this compact model",
    "session_id": "identifier, not a predictor",
}
HISTORY_COLUMNS = ["user_id", "article_id_fixed", "impression_time_fixed",
                   "read_time_fixed", "scroll_percentage_fixed"]


@dataclass
class DataPaths:
    root: Path
    behaviours: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    history_profiles: dict = field(default_factory=dict)
    vocabulary: Path | None = None
    articles: dict = field(default_factory=dict)          # catalogue name -> path
    text_vectors: dict = field(default_factory=dict)      # source name -> path
    image_vectors: dict = field(default_factory=dict)
    vector_metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        def conv(v):
            if isinstance(v, dict):
                return {k: str(x) if x is not None else None for k, x in v.items()}
            return str(v) if v is not None else None
        return {k: conv(v) for k, v in self.__dict__.items()}


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path is not None and Path(path).is_file():
            return Path(path)
    return None


def resolve_paths(root: Path, overrides: dict | None = None) -> DataPaths:
    """Resolve inputs under ``root``; ``overrides`` maps keys (as in the mini's
    loader_paths.json) to paths relative to root. Raises with every missing path."""
    root = Path(root)
    overrides = dict(overrides or {})
    loader_json = root / "loader_paths.json"
    if loader_json.is_file():
        for key, value in read_json(loader_json).items():
            if key not in ("note", "data_root"):
                overrides.setdefault(key, value)

    def over(key):
        return root / overrides[key] if key in overrides else None

    paths = DataPaths(root=root)
    tv, te = root / "train_val", root / "test"
    paths.behaviours = {
        "train": _first_existing([over("train_behaviors"), tv / "train/behaviors.parquet",
                                  tv / "train/behaviours.parquet"]),
        "validation": _first_existing([over("validation_behaviors"), tv / "validation/behaviours.parquet",
                                       tv / "validation/behaviors.parquet"]),
        "test": _first_existing([over("test_behaviors"), te / "behaviors.parquet", te / "behaviours.parquet"]),
    }
    paths.history = {
        "train": _first_existing([tv / "train/history.parquet"]),
        "validation": _first_existing([tv / "validation/history.parquet"]),
        "test": _first_existing([te / "history.parquet"]),
    }
    fe = root / "feature_engineering"
    paths.features = {s: _first_existing([fe / f"{s}_features.parquet"]) for s in SPLITS}
    paths.history_profiles = {s: _first_existing([fe / f"{s}_history_profiles.parquet"]) for s in SPLITS}
    paths.vocabulary = _first_existing([fe / "feature_vocabulary.json"])
    paths.articles = {"train_val": _first_existing([tv / "articles.parquet"]),
                      "test": _first_existing([te / "articles.parquet"])}
    for source, base in (("train_val", tv), ("test", te)):
        candidates_text = [over(f"{source}_text_embeddings"), base / "article embeddings/text_embeddings.parquet",
                           root / f"embeddings/{source}/text_embeddings.parquet"]
        candidates_image = [over(f"{source}_image_embeddings"), base / "article embeddings/image_embeddings.parquet",
                            root / f"embeddings/{source}/image_embeddings.parquet"]
        paths.text_vectors[source] = _first_existing(candidates_text)
        paths.image_vectors[source] = _first_existing(candidates_image)
        text_path = paths.text_vectors[source]
        paths.vector_metadata[source] = _first_existing([text_path.parent / "metadata.json"]) if text_path else None

    required = {
        **{f"behaviours[{k}]": v for k, v in paths.behaviours.items()},
        **{f"history[{k}]": v for k, v in paths.history.items()},
        **{f"features[{k}]": v for k, v in paths.features.items()},
        "vocabulary": paths.vocabulary,
        **{f"articles[{k}]": v for k, v in paths.articles.items()},
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise FileNotFoundError(f"Missing required inputs under {root}: {missing}")
    return paths


def input_manifest(paths: DataPaths, hash_limit_bytes: int | None = 2 * 1024 ** 3) -> dict:
    """File sizes/mtimes/SHA-256 plus Parquet row counts and schemas."""
    manifest = {}
    flat = {}
    for key, value in paths.__dict__.items():
        if isinstance(value, dict):
            for sub, path in value.items():
                if path is not None:
                    flat[f"{key}.{sub}"] = Path(path)
        elif isinstance(value, Path) and key != "root":
            flat[key] = value
    for name, path in flat.items():
        record = file_record(path, hash_limit_bytes)
        if path.suffix == ".parquet":
            meta = pq.ParquetFile(path)
            record["rows"] = meta.metadata.num_rows
            record["columns"] = [f"{f.name}: {f.type}" for f in meta.schema_arrow]
        manifest[name] = record
    return manifest


# ---------------------------------------------------------------- articles ---

def load_articles(paths: DataPaths, catalogue_names: list[str]) -> tuple[pd.DataFrame, dict]:
    """Union of the named article catalogues. Overlapping IDs keep the first
    catalogue's row (as Step 3 does); differences in published_time are reported.

    Returns the union frame (one row per article_id, with `_source`) and a dict
    of per-catalogue sorted ID arrays (the catalogue membership)."""
    frames, members, report = [], {}, {}
    for name in catalogue_names:
        path = paths.articles[name]
        available = set(pq.ParquetFile(path).schema_arrow.names)
        missing = set(ARTICLE_COLUMNS) - available
        if missing:
            raise ValueError(f"{path} lacks article columns {sorted(missing)}")
        frame = pd.read_parquet(path, columns=ARTICLE_COLUMNS)
        if frame.article_id.isna().any() or frame.article_id.duplicated().any():
            raise ValueError(f"{path}: null or duplicate article_id")
        frame["article_id"] = frame.article_id.astype("int64")
        frame["_source"] = name
        members[name] = np.sort(frame.article_id.to_numpy())
        report[name] = {"rows": len(frame), "published_time_missing": int(frame.published_time.isna().sum()),
                        "published_min": str(frame.published_time.min()),
                        "published_max": str(frame.published_time.max())}
        frames.append(frame)
    union = pd.concat(frames, ignore_index=True)
    if len(frames) > 1:
        first, second = frames[0].set_index("article_id"), frames[1].set_index("article_id")
        common = first.index.intersection(second.index)
        a = first.loc[common, "published_time"]
        b = second.loc[common, "published_time"]
        diff_pub = ~((a == b) | (a.isna() & b.isna()))
        report["overlap"] = {"ids_in_both": int(len(common)), "published_time_differs": int(diff_pub.sum()),
                             "policy": f"overlapping IDs use the '{catalogue_names[0]}' row (metadata and "
                                       "published_time) in every catalogue, as Step 3 does"}
    union = union.drop_duplicates("article_id", keep="first").sort_values("article_id").reset_index(drop=True)
    # Word count of the snapshot body; body text itself is not a model input.
    union["body_word_count"] = union["body"].map(
        lambda v: len(v.split()) if isinstance(v, str) and v.strip() else np.nan)
    union = union.drop(columns="body")
    report["union_rows"] = len(union)
    return union, {"members": members, "report": report}


# ------------------------------------------------------------------ vectors --

def _decode_vector_column(table: pa.Table, column: str, flag: str, expected_dim: int | None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Decode a (fixed- or variable-length, nullable) list<float> column.

    Returns (ids, matrix of available vectors float32, diagnostics). Rows whose
    vector is null, flagged unavailable, wrong length or non-finite are reported
    separately; no vector is invented."""
    ids = table["article_id"].to_numpy().astype("int64")
    table = as_arrow_table(table)
    col = table[column].combine_chunks()
    if pa.types.is_fixed_size_list(col.type) or pa.types.is_large_list(col.type):
        col = col.cast(pa.list_(col.type.value_type))
    if not pa.types.is_floating(col.type.value_type):
        raise TypeError(f"{column} has non-float values: {col.type}")
    lengths = pc.list_value_length(col).to_numpy(zero_copy_only=False)
    lengths = np.where(np.isnan(lengths.astype("float64")), -1, lengths).astype("int64") \
        if lengths.dtype.kind == "f" else np.asarray(lengths, dtype="float64")
    valid = col.is_valid().to_numpy(zero_copy_only=False)
    lengths = np.where(valid, np.nan_to_num(np.asarray(lengths, dtype="float64"), nan=-1), -1).astype("int64")
    if flag in table.column_names:
        raw_flags = pd.Series(table[flag].to_numpy(zero_copy_only=False))
        flag_null = raw_flags.isna().to_numpy()
        flag_values = np.asarray(raw_flags.fillna(False).astype(bool))
    else:
        flag_null = np.zeros(len(ids), bool)
        flag_values = valid.copy()
    observed = lengths[lengths > 0]
    dim = int(expected_dim) if expected_dim else (int(np.bincount(observed).argmax()) if observed.size else 0)
    usable = valid & flag_values & (lengths == dim)
    flat = pc.list_flatten(col.filter(pa.array(usable))).to_numpy(zero_copy_only=False).astype("float32")
    matrix = flat.reshape(-1, dim) if dim else np.zeros((0, 0), "float32")
    finite = np.isfinite(matrix).all(axis=1) if len(matrix) else np.zeros(0, bool)
    kept_ids = ids[usable][finite]
    diagnostics = {
        "rows": int(len(ids)), "dimension": dim,
        "null_vectors": int((~valid).sum()),
        "flag_false": int((~flag_values).sum()),
        "flag_true_but_null": int((flag_values & ~valid).sum()),
        "flag_false_but_vector_present": int((~flag_values & valid).sum()),
        "flag_null": int(flag_null.sum()),
        "wrong_length": int((valid & (lengths != dim)).sum()),
        "non_finite": int((~finite).sum()),
        "usable": int(len(kept_ids)),
        "duplicate_ids": int(pd.Series(ids).duplicated().sum()),
    }
    diagnostics["legitimately_missing"] = int((~valid & ~flag_values).sum())
    diagnostics["all_row_ids"] = ids  # popped by load_vectors; source-row precedence uses every row
    return kept_ids, matrix[finite], diagnostics


def vector_problems(diag: dict) -> list[str]:
    """Malformed or inconsistent vector records. These are file problems, never
    legitimate missingness: the run stops instead of treating them as missing."""
    checks = {"wrong_length": "vectors with the wrong dimension", "non_finite": "non-finite vectors",
              "flag_true_but_null": "flag says present but vector is null",
              "flag_false_but_vector_present": "flag says missing but a vector is present",
              "flag_null": "null availability flags", "duplicate_ids": "duplicate article IDs within a source"}
    return [f"{diag[k]} {msg}" for k, msg in checks.items() if diag.get(k)]


def as_arrow_table(obj) -> pa.Table:
    """Coerce reader results to a concrete pyarrow.Table.

    DuckDB >= 1.4 returns a pyarrow.RecordBatchReader from ``.arrow()``; older
    versions return a Table. Polars returns a Table. Anything else is an error,
    never silently converted."""
    if isinstance(obj, pa.Table):
        return obj
    if isinstance(obj, pa.RecordBatchReader):
        return obj.read_all()
    if isinstance(obj, pa.RecordBatch):
        return pa.Table.from_batches([obj])
    if hasattr(obj, "read_all"):  # reader-like objects from other bindings
        return as_arrow_table(obj.read_all())
    raise TypeError(f"Unsupported Arrow result type: {type(obj).__name__}")


def _duckdb_table(path: Path, columns: list[str], column: str) -> tuple[pa.Table, str]:
    import duckdb

    def q(name):
        return '"' + name.replace('"', '""') + '"'

    select = ", ".join(f"CAST({q(c)} AS FLOAT[]) AS {q(c)}" if c == column else q(c) for c in columns)
    con = duckdb.connect()
    try:
        result = con.execute(f"SELECT {select} FROM read_parquet(?)", [str(path)])
        raw = result.arrow()  # DuckDB >= 1.4: RecordBatchReader; older: Table
        return as_arrow_table(raw), type(raw).__name__
    finally:
        con.close()


def _polars_table(path: Path, columns: list[str], column: str) -> tuple[pa.Table, str]:
    import polars as pl

    frame = pl.read_parquet(path, columns=columns)
    frame = frame.with_columns(pl.col(column).cast(pl.List(pl.Float32)))
    raw = frame.to_arrow()
    return as_arrow_table(raw), type(raw).__name__


def read_vector_table(path: Path, column: str, flag: str, readers=("pyarrow", "duckdb", "polars")) -> tuple[pa.Table, dict]:
    """Read article_id + vector (+flag) as a concrete pyarrow.Table.

    PyArrow first; if it cannot decode the column (e.g. a nullable fixed-size
    list on some versions), retry with DuckDB or Polars when installed. Every
    backend result is coerced to a Table *here*, so downstream decoding never
    sees a RecordBatchReader. Values, dimensions and null masks are preserved
    (casts only change the list container, FLOAT stays float32). The original
    error, the reader used and its raw return type are recorded; if every reader
    fails, raise: an unreadable file is not legitimate missingness."""
    path = Path(path)
    source_schema = pq.ParquetFile(path).schema_arrow
    columns = ["article_id", column] + ([flag] if flag in source_schema.names else [])
    errors = {}
    def _pyarrow():
        raw = pq.read_table(path, columns=columns)
        return as_arrow_table(raw), type(raw).__name__

    backends = {"pyarrow": _pyarrow,
                "duckdb": lambda: _duckdb_table(path, columns, column),
                "polars": lambda: _polars_table(path, columns, column)}
    for name in readers:
        try:
            table, raw_type = backends[name]()
            if not isinstance(table, pa.Table):
                raise TypeError(f"{name} backend returned {type(table).__name__}, not a Table")
            info = {"reader": name, "raw_return_type": raw_type,
                    "source_vector_type": str(source_schema.field(column).type)}
            if errors:
                info["earlier_reader_errors"] = dict(errors)
            return table, info
        except Exception as error:
            errors[name] = repr(error)
    raise RuntimeError(f"Could not decode vectors at {path}; reader errors: {errors}. "
                       "This is a file-loading failure, not legitimate missingness.")


def vector_file_check(paths: DataPaths, expected_dims: dict) -> dict:
    """Independent preflight of every original vector file: reader used, raw
    return type, rows, dimension, null/flag/length/finite counts. Raises if a
    file cannot be decoded by any reader (never treated as missing vectors)."""
    out = {}
    for kind, table_paths, column, flag in (("text", paths.text_vectors, "text_embedding", "has_text_embedding"),
                                            ("image", paths.image_vectors, "image_embedding", "has_image_embedding")):
        for source, path in table_paths.items():
            key = f"{kind}.{source}"
            if path is None:
                out[key] = {"status": "file_not_found"}
                continue
            table, info = read_vector_table(path, column, flag)
            _, matrix, diag = _decode_vector_column(table, column, flag, expected_dims.get(kind))
            diag.pop("all_row_ids")
            problems = vector_problems(diag)
            out[key] = {"path": str(path), **info, **diag, "problems": problems,
                        "passed": bool(diag["dimension"] == expected_dims.get(kind) and not problems)}
    return out


def load_vectors(paths: DataPaths, kind: str, source_order: Iterable[str], expected_dim: int | None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Combine vector sources in order (policy published_at_or_before_request_v1).

    Precedence is by SOURCE RECORD: the first source that has a row for an
    article ID wins, even when that row's vector is legitimately missing; a
    later snapshot never backfills it. A file that cannot be decoded, or that
    contains malformed/inconsistent records, raises; legitimate missing vectors
    stay missing (reported), never zero-filled."""
    table_paths = paths.text_vectors if kind == "text" else paths.image_vectors
    column, flag = (("text_embedding", "has_text_embedding") if kind == "text"
                    else ("image_embedding", "has_image_embedding"))
    kept_ids, kept_vectors, report = [], [], {}
    claimed = np.zeros(0, dtype=np.int64)  # every article ID with a row in an earlier source
    for source in source_order:
        path = table_paths.get(source)
        if path is None:
            report[source] = {"status": "file_not_found"}
            continue
        table, reader = read_vector_table(path, column, flag)
        ids, matrix, diag = _decode_vector_column(table, column, flag, expected_dim)
        row_ids = diag.pop("all_row_ids")
        diag["reader"] = reader
        problems = vector_problems(diag)
        if problems:
            raise ValueError(f"{kind} vectors at {path}: {problems}. These are file problems, not missingness.")
        earlier_row = np.isin(row_ids, claimed)
        earlier_usable = np.isin(ids, claimed)
        diag["rows_overridden_by_earlier_source"] = int(earlier_row.sum())
        diag["usable_vectors_overridden_by_earlier_source"] = int(earlier_usable.sum())
        # later vectors for IDs whose earlier record was legitimately missing: NOT used
        earlier_missing = np.setdiff1d(claimed, np.concatenate(kept_ids) if kept_ids else np.zeros(0, np.int64))
        diag["later_vectors_not_backfilled_for_earlier_missing"] = int(np.isin(ids, earlier_missing).sum())
        if len(matrix) and earlier_usable.any() and kept_ids:
            prior_ids = np.concatenate(kept_ids)
            prior_vec = np.concatenate(kept_vectors)
            # Overlap of usable vectors in both sources (IDs whose earlier record is missing are not in prior_ids).
            _, ia, ib = np.intersect1d(prior_ids, ids, assume_unique=True, return_indices=True)
            if len(ia):
                diag["overlap_max_abs_diff"] = float(np.abs(prior_vec[ia] - matrix[ib]).max())
        new = ~earlier_usable
        kept_ids.append(ids[new])
        kept_vectors.append(matrix[new])
        claimed = np.union1d(claimed, row_ids)
        diag["added_usable_vectors"] = int(new.sum())
        diag["path"] = str(path)
        report[source] = diag
    dims = {v.shape[1] for v in kept_vectors if len(v)}
    if len(dims) > 1:
        raise ValueError(f"{kind} vector sources disagree on dimension: {dims}")
    ids = np.concatenate(kept_ids) if kept_ids else np.zeros(0, "int64")
    vectors = np.concatenate(kept_vectors) if kept_vectors else np.zeros((0, expected_dim or 0), "float32")
    order = np.argsort(ids)
    report["precedence"] = "first source record per article_id wins, including legitimately missing vectors"
    report["source_order"] = list(source_order)
    report["article_ids_with_a_record"] = int(len(claimed))
    report["usable_vectors"] = int(len(ids))
    return ids[order], vectors[order], report


# --------------------------------------------------------------- behaviours --

def load_behaviours(path: Path, split: str) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(BEHAVIOUR_COLUMNS_REQUIRED) - available
    if missing:
        raise ValueError(f"{path} lacks {sorted(missing)}")
    if "article_ids_clicked" not in available:
        raise ValueError(f"{split}: expected labelled project behaviour data")
    columns = BEHAVIOUR_COLUMNS_REQUIRED + [c for c in BEHAVIOUR_COLUMNS_OPTIONAL
                if c in available and (c != "article_ids_clicked" or split in DEVELOPMENT_SPLITS)]
    frame = pd.read_parquet(path, columns=columns)
    if frame[["user_id", "impression_id"]].isna().any().any():
        raise ValueError(f"{split}: null user_id/impression_id")
    frame["user_id"] = frame.user_id.astype("int64")
    frame["impression_id"] = frame.impression_id.astype("int64")
    if frame.duplicated(["user_id", "impression_id"]).any():
        raise ValueError(f"{split}: duplicate (user_id, impression_id) request keys; investigate before use")
    if frame.impression_time.isna().any():
        raise ValueError(f"{split}: requests without impression_time")
    frame["behaviour_id"] = frame.user_id.astype(str) + ":" + frame.impression_id.astype(str)
    frame["article_ids_inview"] = frame.article_ids_inview.map(lambda v: np.asarray(v, dtype="int64"))
    if (frame.article_ids_inview.map(len) == 0).any():
        raise ValueError(f"{split}: empty recorded slate")
    if frame.article_ids_inview.map(lambda a: len(a) != len(np.unique(a))).any():
        raise ValueError(f"{split}: duplicate article within a recorded slate")
    if split in DEVELOPMENT_SPLITS:
        if "article_ids_clicked" not in frame:
            raise ValueError(f"{split}: no click labels")
        frame["article_ids_clicked"] = frame.article_ids_clicked.map(lambda v: np.asarray(v, dtype="int64"))
        bad = [not set(c.tolist()) <= set(i.tolist()) for c, i in zip(frame.article_ids_clicked, frame.article_ids_inview)]
        if any(bad):
            raise ValueError(f"{split}: clicked article outside recorded slate in {sum(bad)} rows")
    else:
        frame["article_ids_clicked"] = None  # holdout labels remain on disk for final evaluation only
    # Deterministic request order used by every export.
    return frame.sort_values(["user_id", "impression_id"], kind="stable").reset_index(drop=True)


def load_recorded_pairs(feature_path: Path, split: str, behaviours: pd.DataFrame) -> pd.DataFrame:
    """Read only keys + label from the shared, already-exploded feature table and
    check it is exactly the recorded slates (no re-explosion, no added negatives)."""
    cols = ["user_id", "impression_id", "candidate_article_id", "clicked"]
    pairs = pd.read_parquet(feature_path, columns=cols)
    for c in cols[:3]:
        pairs[c] = pairs[c].astype("int64")
    if pairs.duplicated(cols[:3]).any():
        raise ValueError(f"{split}: duplicate candidate keys in the shared feature table")
    exploded = behaviours[["user_id", "impression_id", "article_ids_inview"]].explode("article_ids_inview")
    exploded = exploded.rename(columns={"article_ids_inview": "candidate_article_id"})
    exploded["candidate_article_id"] = exploded.candidate_article_id.astype("int64")
    if len(exploded) != len(pairs):
        raise ValueError(f"{split}: feature table has {len(pairs)} rows; behaviours imply {len(exploded)}")
    check = pairs.merge(exploded, on=cols[:3], how="outer", indicator=True)
    if not (check["_merge"] == "both").all():
        raise ValueError(f"{split}: feature-table keys differ from recorded slates")
    if split in DEVELOPMENT_SPLITS:
        if pairs.clicked.isna().any():
            raise ValueError(f"{split}: missing click labels")
        clicks = behaviours[["user_id", "impression_id", "article_ids_clicked"]].explode("article_ids_clicked")
        clicks = set(zip(clicks.user_id, clicks.impression_id, clicks.article_ids_clicked.astype("int64")))
        derived = np.array([k in clicks for k in zip(pairs.user_id, pairs.impression_id, pairs.candidate_article_id)])
        if not np.array_equal(derived, pairs.clicked.astype(int).to_numpy() == 1):
            raise ValueError(f"{split}: feature-table labels disagree with article_ids_clicked")
        pairs["clicked"] = pairs.clicked.astype("int8")
    else:
        if pairs.clicked.isna().any() or not pairs.clicked.isin([0, 1]).all():
            raise ValueError("Expected complete binary labels in the project holdout feature file.")
        pairs["clicked"] = np.nan  # reserved ground truth, masked during development
    return pairs


def load_history(path: Path, split_start: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Per-user aligned fixed-history events strictly before the split start.

    Events at/after the cutoff or with missing time are excluded and counted.
    Later behaviour clicks are never appended."""
    frame = pd.read_parquet(path, columns=HISTORY_COLUMNS)
    if frame.user_id.duplicated().any():
        raise ValueError(f"{path}: duplicate history users")
    lens = {c: frame[c].map(len) for c in HISTORY_COLUMNS[1:]}
    misaligned = (lens["article_id_fixed"] != lens["impression_time_fixed"]) | \
                 (lens["article_id_fixed"] != lens["read_time_fixed"]) | \
                 (lens["article_id_fixed"] != lens["scroll_percentage_fixed"])
    if misaligned.any():
        raise ValueError(f"{path}: misaligned fixed-history arrays for {int(misaligned.sum())} users")
    counts = lens["article_id_fixed"].to_numpy()
    events = pd.DataFrame({
        "user_id": np.repeat(frame.user_id.astype("int64").to_numpy(), counts),
        "article_id": np.concatenate([np.asarray(v, "int64") for v in frame.article_id_fixed]) if counts.sum() else [],
        "time": np.concatenate([np.asarray(v, "datetime64[us]") for v in frame.impression_time_fixed]) if counts.sum() else [],
        "read_time": np.concatenate([np.asarray(v, "float64") for v in frame.read_time_fixed]) if counts.sum() else [],
        "scroll": np.concatenate([np.asarray(v, "float64") for v in frame.scroll_percentage_fixed]) if counts.sum() else [],
    })
    events["event_index"] = np.concatenate([np.arange(c) for c in counts]) if counts.sum() else []
    total = len(events)
    keep = events.time.notna() & (events.time < np.datetime64(split_start))
    excluded = int((~keep).sum())
    events = events.loc[keep].copy()
    # Invalid numeric values become missing (not zero), matching Step 3's valid_number.
    events.loc[~np.isfinite(events.read_time) | (events.read_time < 0), "read_time"] = np.nan
    events.loc[~np.isfinite(events.scroll) | (events.scroll < 0) | (events.scroll > 100), "scroll"] = np.nan
    events = events.sort_values(["user_id", "time", "event_index"], kind="stable").reset_index(drop=True)
    report = {"users": int(len(frame)), "events_total": int(total), "events_kept": int(len(events)),
              "events_excluded_missing_time_or_at_after_cutoff": excluded,
              "cutoff_exclusive": str(split_start),
              "max_event_time": str(events.time.max()) if len(events) else None,
              "users_with_empty_history_after_cutoff": int(len(set(frame.user_id) - set(events.user_id)))}
    users = np.sort(frame.user_id.astype("int64").to_numpy())
    return events, {"report": report, "users": users}
