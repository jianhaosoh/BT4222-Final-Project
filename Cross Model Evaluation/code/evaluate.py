"""Shared evaluation for schema-v4 catalogue and recorded-slate exports.

Labels are candidate-level rows from the agreed split feature file. Rankings are
the model's genuine catalogue shortlist, not the logged slate. Unshown articles
remain unknown; retrieval metrics measure recovery of observed clicks only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
import warnings
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from .similarity import normalize_category, pairwise_shared_diversity_by_cutoff
except ImportError:  # Allow direct execution from this folder.
    from similarity import normalize_category, pairwise_shared_diversity_by_cutoff


def _slate_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return math.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _slate_ndcg(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores, ordered_labels = scores[order], labels[order].astype(float)
    dcg, start = 0.0, 0
    while start < min(len(order), k):
        end = start + 1
        while end < len(order) and ordered_scores[end] == ordered_scores[start]:
            end += 1
        stop = min(end, k)
        mean_gain = float(ordered_labels[start:end].mean())
        dcg += mean_gain * float((1.0 / np.log2(np.arange(start + 2, stop + 2))).sum())
        start = end
    positives = int(labels.sum())
    ideal = float((1.0 / np.log2(np.arange(2, min(positives, k) + 2))).sum())
    return dcg / ideal if ideal else math.nan


def per_impression_metrics(
    frame: pd.DataFrame, *, impression_col: str = "impression_id",
    user_col: str | None = "user_id", label_col: str = "clicked",
    score_col: str = "score", ks: Iterable[int] = (5,),
) -> pd.DataFrame:
    """Tie-averaged AUC/nDCG for the recorded-slate diagnostic."""
    group_cols = ([user_col] if user_col else []) + [impression_col]
    required = {*group_cols, label_col, score_col}
    if required - set(frame):
        raise ValueError(f"Missing metric columns: {sorted(required - set(frame))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Metric input contains null keys, labels, or scores")
    if not frame[label_col].isin([0, 1]).all():
        raise ValueError("Labels must be binary 0/1")
    if not np.isfinite(pd.to_numeric(frame[score_col], errors="raise").to_numpy(float)).all():
        raise ValueError("Scores must be finite")
    rows = []
    for key, group in frame.groupby(group_cols, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        y = group[label_col].to_numpy(dtype=np.int8)
        scores = group[score_col].to_numpy(dtype=float)
        row = dict(zip(group_cols, key))
        row.update({"auc": _slate_auc(y, scores), "candidate_rows": int(len(group)),
                    "positive_candidates": int(y.sum())})
        for k in ks:
            row[f"ndcg@{int(k)}"] = _slate_ndcg(y, scores, int(k)) if y.any() else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_per_impression(per: pd.DataFrame, *, group_columns: Iterable[str] = ()) -> pd.DataFrame:
    groups = list(group_columns)
    frames = per.groupby(groups, dropna=False, sort=False) if groups else [(None, per)]
    rows = []
    for key, frame in frames:
        row = {}
        if groups:
            values = key if isinstance(key, tuple) else (key,)
            row.update(zip(groups, values))
        row.update({"impressions": int(len(frame)), "candidate_rows": int(frame.candidate_rows.sum()),
                    "positive_candidates": int(frame.positive_candidates.sum()),
                    "auc_impressions_excluded_single_class": int(frame.auc.isna().sum()),
                    "mean_impression_auc": float(frame.auc.mean()) if frame.auc.notna().any() else None})
        for column in [c for c in frame if c.startswith("ndcg@")]:
            row[column] = float(frame[column].mean()) if frame[column].notna().any() else None
            row[f"{column}_impressions_excluded_no_positive"] = int(frame[column].isna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_shared_segments(
    per_request: pd.DataFrame, segments: pd.DataFrame, segment_columns: Sequence[str],
    *, min_requests: int = 100,
) -> pd.DataFrame:
    """Attach request-level segments and compute macro summaries."""
    if "behaviour_id" not in per_request:
        raise ValueError("Per-request metrics need behaviour_id")
    if (per_request.behaviour_id.isna().any()
            or per_request.behaviour_id.astype(str).duplicated().any()):
        raise ValueError("Per-request metrics must contain one non-null row per behaviour_id")
    if "behaviour_id" not in segments:
        raise ValueError("Segment table needs behaviour_id")
    missing = set(segment_columns) - set(segments.columns)
    if missing:
        raise ValueError(f"Segment table missing columns: {sorted(missing)}")
    if segments.behaviour_id.isna().any() or segments.behaviour_id.astype(str).duplicated().any():
        raise ValueError("Segment table must contain one non-null row per behaviour_id")
    request_table = per_request.copy()
    segment_table = segments[["behaviour_id", *segment_columns]].copy()
    request_table["behaviour_id"] = request_table.behaviour_id.astype(str)
    segment_table["behaviour_id"] = segment_table.behaviour_id.astype(str)
    attached = request_table.merge(segment_table,
                                 on="behaviour_id", how="left", validate="one_to_one",
                                 indicator="_segment_merge")
    matched_mask = attached["_segment_merge"].eq("both")
    attached = attached.drop(columns="_segment_merge")
    if not matched_mask.any():
        raise ValueError("No evaluated requests matched the segment table")
    prefixes = ("known_click_recall@", "observed_click_ndcg@", "eligible_only_recall@",
                "median_article_age_h@", "unknown_age_share@", "age_le_24h_share@",
                "age_24_72h_share@", "age_gt_72h_share@", "topic_coverage@",
                "topic_article_coverage@", "intra_list_diversity@", "has_text_embedding_share@",
                "shared_intra_list_diversity@", "distinct_categories@",
                "has_image_embedding_share@", "has_text_embedding_missing_share@",
                "has_image_embedding_missing_share@", "has_text_embedding_unknown_share@",
                "has_image_embedding_unknown_share@", "observed_continuation_recall@",
                "continuation_observation_coverage@", "continuation_rate_observed_topk@",
                "session_continuity_proxy@", "continuity_proxy_coverage@")
    metric_columns = [c for c in attached if c.startswith(prefixes)]
    summaries = []
    for key, group in attached.loc[matched_mask].groupby(list(segment_columns), dropna=False, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        row = {}
        for column, value in zip(segment_columns, values):
            row[column] = None if pd.isna(value) else value.item() if hasattr(value, "item") else value
        row["requests"] = int(len(group))
        row["users"] = int(group.user_id.nunique()) if "user_id" in group else None
        row["below_min_requests"] = len(group) < min_requests
        for metric in metric_columns:
            row[metric] = float(group[metric].mean()) if group[metric].notna().any() else None
        if row["below_min_requests"]:
            warnings.warn(f"Segment {values} contains {len(group)} requests (<{min_requests}); interpret cautiously")
        summaries.append(row)
    attached.attrs["segment_summaries"] = summaries
    attached.attrs["segment_matching"] = {
        "evaluated_requests": int(len(attached)), "matched_requests": int(matched_mask.sum()),
        "unmatched_requests": int((~matched_mask).sum()),
        "matching_coverage": float(matched_mask.mean()) if len(attached) else None,
    }
    return attached


def _validate_request_ranking_scope(
    per_request: pd.DataFrame, rankings: pd.DataFrame,
) -> tuple[set[str], dict[str, list[int]]]:
    """Require one ranking for every evaluated request, with no extras."""
    for name, frame in (("Per-request metrics", per_request), ("Rankings", rankings)):
        if "behaviour_id" not in frame:
            raise ValueError(f"{name} need behaviour_id")
        if frame.behaviour_id.isna().any() or frame.behaviour_id.astype(str).duplicated().any():
            raise ValueError(f"{name} must contain one non-null row per behaviour_id")
    if "article_ids" not in rankings:
        raise ValueError("Rankings need article_ids")
    normalized_requests = _normalize_behaviour_ids(per_request.behaviour_id)
    normalized_rankings = _normalize_behaviour_ids(rankings.behaviour_id)
    request_ids = set(normalized_requests.astype(str))
    ranking_ids = set(normalized_rankings.astype(str))
    missing = request_ids - ranking_ids
    extra = ranking_ids - request_ids
    if missing or extra:
        raise ValueError(
            "Ranking/evaluation request scope mismatch: "
            f"{len(missing)} evaluated requests lack rankings, "
            f"{len(extra)} ranking requests are outside evaluation"
        )
    ranking_map = dict(zip(normalized_rankings.astype(str), rankings.article_ids))
    return request_ids, ranking_map


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, usecols=columns)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    raise ValueError(f"Unsupported file type: {path.suffix}; use CSV or Parquet")


def table_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=0).columns.tolist()
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).schema_arrow.names
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _behavior_id(user_id: Any, impression_id: Any) -> str:
    return f"{int(user_id)}:{int(impression_id)}"


def _normalize_behaviour_ids(values: Any, column: str = "behaviour_id") -> pd.Series:
    """Validate user:impression keys and return their canonical int64 form."""
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    if series.isna().any():
        raise ValueError(f"{column} contains null values")
    user_ids, impression_ids = [], []
    for value in series:
        if not isinstance(value, str) or value.count(":") != 1:
            raise ValueError(f"{column} must use the user_id:impression_id string format")
        user, impression = value.split(":")
        if not user.lstrip("+-").isdigit() or not impression.lstrip("+-").isdigit():
            raise ValueError(f"{column} must contain integer user and impression IDs")
        user_ids.append(user)
        impression_ids.append(impression)
    users = checked_int64_series(user_ids, f"{column} user_id")
    impressions = checked_int64_series(impression_ids, f"{column} impression_id")
    normalized = pd.Series(
        [_behavior_id(user, impression) for user, impression in zip(users, impressions)],
        index=series.index, dtype="string",
    )
    if normalized.duplicated().any():
        raise ValueError(f"{column} contains duplicate request IDs after normalization")
    return normalized


def checked_int64_series(values: Any, column: str) -> pd.Series:
    """Convert integer IDs without lossy float coercion; reject nulls and overflow."""
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    nulls = series.isna()
    if nulls.any():
        raise ValueError(f"{column} contains {int(nulls.sum())} null values")
    minimum, maximum = -(1 << 63), (1 << 63) - 1
    safe_float = 1 << 53
    result = []
    for value in series:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{column} contains a boolean ID: {value!r}")
        if isinstance(value, numbers.Integral):
            parsed = int(value)
        elif isinstance(value, numbers.Real):
            numeric = float(value)
            if not math.isfinite(numeric) or not numeric.is_integer() or abs(numeric) > safe_float:
                raise ValueError(f"{column} contains an unsafe or non-integer value: {value!r}")
            parsed = int(numeric)
        else:
            try:
                decimal_value = Decimal(str(value).strip())
            except (InvalidOperation, ValueError) as exc:
                raise ValueError(f"{column} contains a non-integer value: {value!r}") from exc
            if not decimal_value.is_finite() or decimal_value != decimal_value.to_integral_value():
                raise ValueError(f"{column} contains a non-integer value: {value!r}")
            parsed = int(decimal_value)
        if parsed < minimum or parsed > maximum:
            raise ValueError(f"{column} value is outside int64 range: {parsed}")
        result.append(parsed)
    return pd.Series(result, index=series.index, dtype="int64")


def _normalize_boolean_flag(series: pd.Series, column: str) -> pd.Series:
    valid = series.dropna().isin([0, 1, False, True])
    if not valid.all():
        invalid = series.loc[series.notna() & ~series.isin([0, 1, False, True])]
        raise ValueError(
            f"{column} must contain only true, false, 0, 1, or null; "
            f"invalid examples: {invalid.head(5).tolist()}"
        )
    return series.map(lambda value: pd.NA if pd.isna(value) else bool(value)).astype("boolean")


def utc_times(values: pd.Series, *, naive_timezone: str = "UTC", errors: str = "coerce") -> pd.Series:
    """Normalize naive or mixed-offset timestamp values to UTC without assuming offsets match."""
    try:
        parsed = pd.to_datetime(values, errors=errors)
    except (TypeError, ValueError):
        parsed = None
    if parsed is not None and isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert("UTC")
    if parsed is not None and pd.api.types.is_datetime64_dtype(parsed.dtype):
        localized = parsed.dt.tz_localize(
            naive_timezone, ambiguous="NaT", nonexistent="NaT"
        ).dt.tz_convert("UTC")
        if errors == "raise" and (values.notna() & localized.isna()).any():
            raise ValueError("Timestamp contains an ambiguous or nonexistent local time")
        return localized

    # Object dtype can occur when strings carry different UTC offsets. Parse each
    # value independently, localizing only naive values and converting aware ones.
    normalized = []
    for value in values:
        if pd.isna(value):
            normalized.append(pd.NaT)
            continue
        try:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                normalized.append(pd.NaT)
                continue
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(
                    naive_timezone, ambiguous="NaT", nonexistent="NaT"
                )
            else:
                timestamp = timestamp.tz_convert("UTC")
            if pd.isna(timestamp) and errors == "raise":
                raise ValueError(f"Ambiguous/nonexistent local timestamp: {value!r}")
            normalized.append(timestamp)
        except (TypeError, ValueError, OverflowError) as exc:
            if errors == "raise":
                raise ValueError(f"Could not parse timestamp {value!r}") from exc
            normalized.append(pd.NaT)
    return pd.Series(pd.to_datetime(normalized, utc=True, errors=errors), index=values.index)


def catalogue_fingerprints(articles: pd.DataFrame) -> tuple[str, str]:
    """Fingerprint IDs in preserved catalogue order and metadata canonically."""
    frame = articles.reset_index(drop=True).copy()
    if "article_id" not in frame:
        raise ValueError("Catalogue fingerprint requires article_id")
    frame["article_id"] = checked_int64_series(frame.article_id, "catalogue article_id")
    id_hash = hashlib.sha256(frame.article_id.to_numpy(dtype="<i8").tobytes()).hexdigest()
    # Metadata identity remains independent of physical row order.
    frame = frame.sort_values("article_id", kind="mergesort").reset_index(drop=True)
    fields = ["article_id"] + [c for c in (
        "published_time", "topics", "category", "has_text_embedding", "has_image_embedding"
    ) if c in frame]
    canonical = frame[fields].copy()
    if "published_time" in canonical:
        dates = pd.to_datetime(canonical.published_time, utc=True, errors="coerce")
        canonical["published_time"] = dates.map(lambda x: x.isoformat() if pd.notna(x) else None)
    if "topics" in canonical:
        canonical["topics"] = canonical.topics.map(_canonical_topics_json)
    if "category" in canonical:
        canonical["category"] = canonical.category.map(lambda x: None if pd.isna(x) else str(x))
    for col in ("has_text_embedding", "has_image_embedding"):
        if col in canonical:
            canonical[col] = canonical[col].map(lambda x: None if pd.isna(x) else bool(x))
    content_hash = hashlib.sha256(canonical.to_json(
        orient="records", force_ascii=False, date_format="iso").encode("utf-8")).hexdigest()
    return id_hash, content_hash


def _canonical_topics_json(value: Any) -> str | None:
    """Serialize list-like topic metadata identically across Arrow/Pandas round trips."""
    def normalize(item: Any) -> Any:
        if item is None or item is pd.NA or item is pd.NaT:
            return None
        if isinstance(item, np.ndarray):
            return [normalize(value) for value in item.tolist()]
        if isinstance(item, np.generic):
            return normalize(item.item())
        if isinstance(item, (pd.Timestamp,)):
            return item.isoformat()
        if isinstance(item, float) and math.isnan(item):
            return None
        if isinstance(item, dict):
            return {str(key): normalize(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, (list, tuple)):
            return [normalize(value) for value in item]
        if isinstance(item, set):
            values = [normalize(value) for value in item]
            return sorted(values, key=lambda value: json.dumps(value, sort_keys=True, default=str))
        if isinstance(item, (str, int, float, bool)):
            return item
        return str(item)

    normalized = normalize(value)
    return None if normalized is None else json.dumps(
        normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def _parse_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"Expected a list of article IDs, got {type(value).__name__}")
    ids = checked_int64_series(value, "ranking article IDs").tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("A ranking contains duplicate article IDs")
    return ids


def _topic_set(value: Any) -> set[str]:
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return set()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                value = parsed
            else:
                value = [value]
        except json.JSONDecodeError:
            value = [value]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    result = set()
    for item in value:
        if item is None or item is pd.NA or (isinstance(item, float) and math.isnan(item)):
            continue
        if isinstance(item, dict):
            item = item.get("topic", item.get("topic_id", item.get("id", json.dumps(item, sort_keys=True))))
            if item is None or item is pd.NA:
                continue
        result.add(str(item))
    return result


def file_provenance(path: Path, *, hash_file: bool = False) -> dict:
    stat = path.stat()
    result = {"path": str(path), "bytes": int(stat.st_size), "modified": stat.st_mtime}
    if hash_file:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            original_state = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            opened_state = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            if opened_state != original_state:
                raise RuntimeError(f"{path} changed before its provenance hash was computed")
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
            after = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise RuntimeError(f"{path} changed while its provenance hash was being computed")
        current_state = path.stat()
        if (current_state.st_dev, current_state.st_ino, current_state.st_size, current_state.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise RuntimeError(f"{path} was replaced while its provenance hash was being computed")
        result["sha256"] = digest.hexdigest()
    return result


def _json_default(value: Any):
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if value is pd.NA:
        return None
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def load_labels(path: Path, *, naive_timezone: str = "UTC") -> tuple[pd.DataFrame, dict[str, set[int]]]:
    """Load candidate labels and collapse them to request-level known clicks."""
    required = ["user_id", "impression_id", "candidate_article_id", "clicked", "impression_time"]
    frame = read_table(path, required)
    if frame[required].isna().any().any():
        raise ValueError("Label features contain null IDs, labels, or impression_time")
    if not frame["clicked"].isin([0, 1, False, True]).all():
        raise ValueError("clicked must contain binary 0/1 labels")
    for column in ("user_id", "impression_id", "candidate_article_id"):
        frame[column] = checked_int64_series(frame[column], column)
    if frame.duplicated(["user_id", "impression_id", "candidate_article_id"]).any():
        raise ValueError("Label features contain duplicate request/article keys")
    frame["impression_time"] = utc_times(
        frame["impression_time"], naive_timezone=naive_timezone, errors="raise"
    )
    frame["behaviour_id"] = [
        _behavior_id(u, i) for u, i in zip(frame.user_id, frame.impression_id)
    ]
    request = frame.groupby("behaviour_id", sort=False).agg(
        user_id=("user_id", "first"), impression_id=("impression_id", "first"),
        impression_time=("impression_time", "first"),
        logged_candidate_count=("candidate_article_id", "size"),
        time_count=("impression_time", "nunique"),
    ).reset_index()
    if (request.time_count != 1).any():
        raise ValueError("impression_time varies within a request")
    clicked = frame.loc[frame.clicked.astype(bool)].groupby("behaviour_id")[
        "candidate_article_id"
    ].agg(lambda x: set(x.astype(int)))
    request["clicked_ids"] = request.behaviour_id.map(clicked).map(lambda x: x if isinstance(x, set) else set())
    request = request.drop(columns="time_count")
    click_map = dict(zip(request.behaviour_id, request.clicked_ids))
    return request, click_map


def load_articles(path: Path, *, naive_timezone: str = "UTC",
                  category_fallback_for_topics: bool = False) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq
        available = pq.ParquetFile(path).schema_arrow.names
    else:
        available = list(pd.read_csv(path, nrows=0).columns)
    if "article_id" not in available or "published_time" not in available:
        raise ValueError("Article union must contain article_id and published_time")
    wanted = [c for c in ("article_id", "published_time", "topics", "category",
                           "has_text_embedding", "has_image_embedding") if c in available]
    articles = read_table(path, wanted)
    articles["article_id"] = checked_int64_series(articles.article_id, "article catalogue article_id")
    if articles.article_id.duplicated().any():
        raise ValueError("Article union must have one row per article_id (apply source precedence first)")
    articles["published_time"] = utc_times(
        articles.published_time, naive_timezone=naive_timezone, errors="coerce"
    )
    topics = articles["topics"] if "topics" in articles else pd.Series([None] * len(articles), index=articles.index)
    if category_fallback_for_topics and "category" in articles:
        topics = pd.Series(
            [topic if _topic_set(topic) else category
             for topic, category in zip(topics, articles.category)],
            index=articles.index,
        )
    articles["_topic_set"] = topics.map(_topic_set)
    if "has_text_embedding" in articles:
        articles["has_text_embedding"] = _normalize_boolean_flag(
            articles["has_text_embedding"], "has_text_embedding"
        )
    if "has_image_embedding" in articles:
        articles["has_image_embedding"] = _normalize_boolean_flag(
            articles["has_image_embedding"], "has_image_embedding"
        )
    return articles.set_index("article_id", drop=False)


def load_rankings(path: Path) -> pd.DataFrame:
    frame = read_table(path)
    if not {"behaviour_id", "article_ids"}.issubset(frame.columns):
        raise ValueError("Ranking file needs behaviour_id and article_ids columns")
    frame["behaviour_id"] = _normalize_behaviour_ids(frame.behaviour_id)
    frame["article_ids"] = frame.article_ids.map(_parse_ids)
    return frame[["behaviour_id", "article_ids"]]


def validate_catalogue_scores(path: Path, rankings: pd.DataFrame) -> dict:
    """Verify top-K score rows agree exactly with the saved ordered shortlist."""
    scores = read_table(path, ["behaviour_id", "article_id", "rank", "score"])
    required = {"behaviour_id", "article_id", "rank", "score"}
    if required - set(scores.columns):
        raise ValueError(f"Catalogue score file missing columns: {sorted(required - set(scores.columns))}")
    if scores[list(required)].isna().any().any() or scores.duplicated(["behaviour_id", "article_id"]).any():
        raise ValueError("Catalogue score file has null fields or duplicate request/article keys")
    for column in ("article_id", "rank"):
        scores[column] = checked_int64_series(scores[column], f"catalogue score {column}")
    scores["score"] = pd.to_numeric(scores.score, errors="raise")
    if not np.isfinite(scores.score.to_numpy(dtype=float)).all():
        raise ValueError("Catalogue scores must be finite")
    ranking_map = dict(zip(rankings.behaviour_id, rankings.article_ids))
    if set(scores.behaviour_id) - set(ranking_map):
        raise ValueError("Catalogue score rows include requests absent from ranking file")
    compared = 0
    for behaviour_id, group in scores.groupby("behaviour_id", sort=False):
        ordered = group.sort_values("rank")
        expected_ids = ranking_map[behaviour_id]
        ranks = ordered["rank"].tolist()
        ids = ordered.article_id.astype(int).tolist()
        if ranks != list(range(1, len(ids) + 1)):
            raise ValueError(f"{behaviour_id}: catalogue scores must have contiguous 1-based ranks")
        if ids != expected_ids:
            raise ValueError(f"{behaviour_id}: catalogue scores and rankings disagree on IDs/order")
        # Schema-v4 tie policy: score descending, then original article ID ascending.
        pairs = list(zip(ordered.score.astype(float), ids))
        if pairs != sorted(pairs, key=lambda pair: (-pair[0], pair[1])):
            raise ValueError(f"{behaviour_id}: score order violates descending score/article-ID tie policy")
        compared += len(ids)
    expected_rows = sum(len(ids) for ids in rankings.article_ids)
    if compared != expected_rows:
        raise ValueError(f"Catalogue score row count {compared} != ranking membership count {expected_rows}")
    return {"score_rows": int(compared), "ranking_membership_match": True,
            "request_count_with_scores": int(scores.behaviour_id.nunique())}


def _dcg(relevant_ranks: list[int]) -> float:
    return float(sum(1.0 / np.log2(rank + 1) for rank in relevant_ranks))


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else float("nan")


def _pairwise_diversity_by_cutoff(
    topic_sets: Sequence[set[str]], cutoffs: Sequence[int],
) -> dict[int, float]:
    """Compute exact mean pairwise Jaccard distance for several list cutoffs.

    Articles without known topics are omitted, matching the evaluator's previous
    definition. A shortlist-local vocabulary keeps the incidence matrix small.
    """
    cutoffs = sorted(set(int(k) for k in cutoffs))
    nonempty: list[set[str]] = []
    prefix_counts: dict[int, int] = {}
    for position, topics in enumerate(topic_sets, start=1):
        if topics:
            nonempty.append(topics)
        if position in cutoffs:
            prefix_counts[position] = len(nonempty)
    n = len(nonempty)
    if n < 2:
        return {k: float("nan") for k in cutoffs}

    vocabulary = {topic: index for index, topic in enumerate(
        sorted(set().union(*nonempty))
    )}
    incidence = np.zeros((n, len(vocabulary)), dtype=np.float64)
    sizes = np.empty(n, dtype=np.float64)
    for row, topics in enumerate(nonempty):
        columns = [vocabulary[topic] for topic in topics]
        incidence[row, columns] = 1.0
        sizes[row] = len(columns)

    intersections = incidence @ incidence.T
    unions = sizes[:, None] + sizes[None, :] - intersections
    upper_i, upper_j = np.triu_indices(n, k=1)
    distances = 1.0 - intersections[upper_i, upper_j] / unions[upper_i, upper_j]
    output: dict[int, float] = {}
    for cutoff in cutoffs:
        available = min(cutoff, len(topic_sets))
        count = prefix_counts.get(available)
        if count is None:
            count = sum(bool(topics) for topics in topic_sets[:available])
        if count < 2:
            output[cutoff] = float("nan")
        else:
            mask = (upper_i < count) & (upper_j < count)
            output[cutoff] = float(distances[mask].mean())
    return output


def evaluate_rankings(
    request_frame: pd.DataFrame,
    click_map: dict[str, set[int]],
    rankings: pd.DataFrame,
    articles: pd.DataFrame,
    *,
    export_k: int = 200,
    ks: tuple[int, ...] = (100, 200),
    allow_subset: bool = False,
    include_top_items: bool = True,
    similarity_constants: dict[str, Any] | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate catalogue retrieval and beyond-accuracy properties of a ranking."""
    if export_k < 1 or any(k < 1 for k in ks):
        raise ValueError("export_k and metric cutoffs must be positive")
    if not ks:
        raise ValueError("At least one metric cutoff is required")
    if any(k > export_k for k in ks):
        raise ValueError(f"Every metric cutoff must be <= export_k ({export_k})")
    request_frame = request_frame.copy()
    rankings = rankings.copy()
    request_frame["behaviour_id"] = _normalize_behaviour_ids(request_frame.behaviour_id)
    rankings["behaviour_id"] = _normalize_behaviour_ids(rankings.behaviour_id)
    expected = set(request_frame.behaviour_id.astype(str))
    received = set(rankings.behaviour_id.astype(str))
    unexpected, missing = received - expected, expected - received
    if unexpected or (missing and not allow_subset):
        raise ValueError(f"Ranking request coverage mismatch: {len(missing)} missing, {len(unexpected)} unexpected")
    if not received:
        raise ValueError("Ranking file contains no requests")
    req = request_frame.set_index("behaviour_id")
    # Normalize to nanoseconds before converting to integer timestamps. Pandas
    # 3.x preserves Parquet timestamp units (often microseconds), while
    # Timestamp.value/unit behavior can therefore differ across inputs.
    pub_known = (
        articles.published_time.dropna().sort_values()
        .astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    )
    records: list[dict] = []
    top_records: list[dict] = []
    ranking_map = {
        str(behaviour_id): article_ids
        for behaviour_id, article_ids in zip(rankings.behaviour_id, rankings.article_ids)
    }

    selected_requests = request_frame.loc[request_frame.behaviour_id.astype(str).isin(received)]
    total_requests = len(selected_requests)
    for request_number, behaviour_id in enumerate(selected_requests.behaviour_id, start=1):
        row = req.loc[behaviour_id]
        ids = ranking_map[behaviour_id]
        clicked = click_map[behaviour_id]
        imp_time = row.impression_time
        impression_time_ns = pd.Timestamp(imp_time).as_unit("ns").value
        eligible_count = int(np.searchsorted(pub_known, impression_time_ns, side="right"))
        expected_length = min(export_k, eligible_count)
        if len(ids) != expected_length:
            raise ValueError(
                f"{behaviour_id}: ranking has {len(ids)} IDs; expected {expected_length} "
                f"from export_k={export_k} and {eligible_count} eligible catalogue articles"
            )
        missing_meta = [a for a in ids if a not in articles.index]
        if missing_meta:
            raise ValueError(f"{behaviour_id}: ranked articles absent from catalogue, e.g. {missing_meta[:5]}")
        ranked_meta = articles.loc[ids] if ids else articles.iloc[0:0]
        invalid = ranked_meta.published_time.isna() | (ranked_meta.published_time > imp_time)
        if invalid.any():
            raise ValueError(f"{behaviour_id}: shortlist includes ineligible or undated articles")

        clicked_eligible = set()
        excluded_unknown, excluded_future, excluded_catalogue = 0, 0, 0
        for article_id in clicked:
            if article_id not in articles.index:
                excluded_catalogue += 1
            else:
                pub = articles.at[article_id, "published_time"]
                if pd.isna(pub):
                    excluded_unknown += 1
                elif pub > imp_time:
                    excluded_future += 1
                else:
                    clicked_eligible.add(article_id)

        record = {
            "behaviour_id": behaviour_id,
            "user_id": int(row.user_id),
            "impression_id": int(row.impression_id),
            "impression_time": imp_time,
            "logged_candidate_count": int(row.logged_candidate_count),
            "eligible_catalogue_count": eligible_count,
            "shortlist_length": len(ids),
            "n_clicked": len(clicked),
            "n_clicked_eligible": len(clicked_eligible),
            "n_clicked_missing_from_catalogue": excluded_catalogue,
            "n_clicked_missing_published_time": excluded_unknown,
            "n_clicked_future_publication": excluded_future,
        }
        for k in sorted(set(ks)):
            top = ids[:k]
            hits = len(clicked.intersection(top))
            ranks_hit = [i for i, aid in enumerate(top, start=1) if aid in clicked]
            ideal_dcg = _dcg(list(range(1, min(len(clicked), k) + 1)))
            observed_dcg = _dcg(ranks_hit)
            record[f"hits@{k}"] = hits
            record[f"known_click_recall@{k}"] = hits / len(clicked) if clicked else np.nan
            record[f"observed_click_ndcg@{k}"] = observed_dcg / ideal_dcg if ideal_dcg else np.nan
            record[f"eligible_only_recall@{k}"] = (
                len(clicked_eligible.intersection(top)) / len(clicked_eligible)
                if clicked_eligible else np.nan
            )
            for flag in ("has_text_embedding", "has_image_embedding"):
                if flag not in articles:
                    continue
                by_status = {"present": set(), "missing": set(), "unknown": set()}
                for article_id in clicked:
                    if article_id not in articles.index or pd.isna(articles.at[article_id, flag]):
                        by_status["unknown"].add(article_id)
                    elif bool(articles.at[article_id, flag]):
                        by_status["present"].add(article_id)
                    else:
                        by_status["missing"].add(article_id)
                for status, click_ids in by_status.items():
                    hit_count = len(click_ids.intersection(top))
                    record[f"{flag}_{status}_clicks@{k}"] = len(click_ids)
                    record[f"{flag}_{status}_hits@{k}"] = hit_count
                    record[f"{flag}_{status}_recall@{k}"] = (
                        hit_count / len(click_ids) if click_ids else np.nan
                    )

        # Freshness distribution and available embedding coverage by ranked slots.
        age_hours = ((imp_time - ranked_meta.published_time).dt.total_seconds() / 3600).to_numpy(float)
        n_ranked = len(ids)
        max_k = min(max(ks), len(ranked_meta))
        diversity_by_cutoff = _pairwise_diversity_by_cutoff(
            ranked_meta.iloc[:max_k]._topic_set.tolist(), ks
        )
        shared_diversity_by_cutoff = None
        if similarity_constants is not None:
            shared_diversity_by_cutoff = pairwise_shared_diversity_by_cutoff(
                ranked_meta.iloc[:max_k]._topic_set.tolist(),
                [normalize_category(x) for x in ranked_meta.iloc[:max_k].get(
                    "category", pd.Series([None] * max_k, index=ranked_meta.iloc[:max_k].index)
                )],
                ks, similarity_constants,
            )
        for k in sorted(set(ks)):
            ages = age_hours[:k]
            finite = np.isfinite(ages)
            denom = len(ages)
            record[f"median_article_age_h@{k}"] = float(np.median(ages[finite])) if finite.any() else np.nan
            record[f"unknown_age_share@{k}"] = float((~finite).mean()) if denom else np.nan
            record[f"age_le_24h_share@{k}"] = float((ages[finite] <= 24).sum() / denom) if denom else np.nan
            record[f"age_24_72h_share@{k}"] = float(((ages[finite] > 24) & (ages[finite] <= 72)).sum() / denom) if denom else np.nan
            record[f"age_gt_72h_share@{k}"] = float((ages[finite] > 72).sum() / denom) if denom else np.nan
            topic_lists = ranked_meta.iloc[:k]._topic_set.tolist()
            known_topic_lists = [x for x in topic_lists if x]
            record[f"topic_coverage@{k}"] = len(set().union(*known_topic_lists)) if known_topic_lists else 0
            record[f"topic_article_coverage@{k}"] = len(known_topic_lists) / len(topic_lists) if topic_lists else np.nan
            record[f"intra_list_diversity@{k}"] = diversity_by_cutoff[k]
            if shared_diversity_by_cutoff is not None:
                record[f"shared_intra_list_diversity@{k}"] = shared_diversity_by_cutoff[k]
            if "category" in ranked_meta:
                known_categories = [normalize_category(x) for x in ranked_meta.iloc[:k].category]
                record[f"distinct_categories@{k}"] = len({x for x in known_categories if x is not None})
            else:
                record[f"distinct_categories@{k}"] = 0
            for flag in ("has_text_embedding", "has_image_embedding"):
                if flag in ranked_meta:
                    values = ranked_meta.iloc[:k][flag]
                    record[f"{flag}_share@{k}"] = float((values == True).fillna(False).sum() / denom) if denom else np.nan
                    record[f"{flag}_missing_share@{k}"] = float((values == False).fillna(False).sum() / denom) if denom else np.nan
                    record[f"{flag}_unknown_share@{k}"] = float(values.isna().sum() / denom) if denom else np.nan
        records.append(record)
        if include_top_items:
            for rank, article_id in enumerate(ids, start=1):
                top_records.append({
                    "behaviour_id": behaviour_id, "user_id": int(row.user_id),
                    "impression_id": int(row.impression_id), "rank": rank,
                    "article_id": article_id, "is_known_clicked": article_id in clicked,
                    "article_age_h": float(age_hours[rank - 1]) if np.isfinite(age_hours[rank - 1]) else np.nan,
                })
        if request_number % 1000 == 0 or request_number == total_requests:
            print(f"Evaluated {request_number:,}/{total_requests:,} requests", flush=True)

    per_request = pd.DataFrame.from_records(records)
    kmetrics = [c for c in per_request if c.startswith(("known_click_recall@", "observed_click_ndcg@", "eligible_only_recall@"))]
    summary = {"requests": int(len(per_request)),
               "label_requests_available": int(len(request_frame)),
               "requests_without_export": int(len(missing)),
               "requests_with_known_clicks": int((per_request.n_clicked > 0).sum()),
               "requests_without_known_clicks": int((per_request.n_clicked == 0).sum()),
               "users": int(per_request.user_id.nunique()),
               "logged_candidate_rows": int(per_request.logged_candidate_count.sum()),
               "ranked_catalogue_rows": int(per_request.shortlist_length.sum()),
               "shortlist_length_min_median_max": [int(per_request.shortlist_length.min()),
                   float(per_request.shortlist_length.median()), int(per_request.shortlist_length.max())],
               "empty_pool_requests": int((per_request.eligible_catalogue_count == 0).sum()),
               "known_clicks": int(per_request.n_clicked.sum()),
               "eligible_known_clicks": int(per_request.n_clicked_eligible.sum()),
               "eligibility_coverage_of_known_clicks": (
                   float(per_request.n_clicked_eligible.sum() / per_request.n_clicked.sum())
                   if per_request.n_clicked.sum() else None),
               "click_exclusions": {
                   "not_in_catalogue": int(per_request.n_clicked_missing_from_catalogue.sum()),
                   "missing_publication_time": int(per_request.n_clicked_missing_published_time.sum()),
                   "published_after_request": int(per_request.n_clicked_future_publication.sum()),
               }}
    for metric in kmetrics:
        summary[metric] = float(per_request[metric].mean()) if per_request[metric].notna().any() else None
        summary[f"{metric}_eligible_requests"] = int(per_request[metric].notna().sum())
    for k in sorted(set(ks)):
        for flag in ("has_text_embedding", "has_image_embedding"):
            for status in ("present", "missing", "unknown"):
                count_col = f"{flag}_{status}_clicks@{k}"
                hit_col = f"{flag}_{status}_hits@{k}"
                recall_col = f"{flag}_{status}_recall@{k}"
                if count_col in per_request:
                    denominator = int(per_request[count_col].sum())
                    hits_total = int(per_request[hit_col].sum())
                    summary[count_col] = denominator
                    summary[hit_col] = hits_total
                    summary[recall_col] = hits_total / denominator if denominator else None
    for col in per_request:
        if col.startswith(("median_article_age_h@", "unknown_age_share@", "age_le_24h_share@",
                           "age_24_72h_share@", "age_gt_72h_share@", "topic_article_coverage@",
                           "intra_list_diversity@", "shared_intra_list_diversity@",
                           "has_text_embedding_share@", "has_image_embedding_share@",
                           "has_text_embedding_missing_share@", "has_image_embedding_missing_share@",
                           "has_text_embedding_unknown_share@", "has_image_embedding_unknown_share@")):
            summary[col] = float(per_request[col].mean()) if per_request[col].notna().any() else None
        elif col.startswith("topic_coverage@"):
            summary[col] = float(per_request[col].mean()) if len(per_request) else None
        elif col.startswith("distinct_categories@"):
            summary[col] = float(per_request[col].mean()) if len(per_request) else None
    return summary, per_request, pd.DataFrame.from_records(top_records)


def evaluate_recorded_slates(scores_path: Path, labels: pd.DataFrame, *, k: int = 5) -> dict:
    """Macro AUC and nDCG@k on original shown candidates (diagnostic only)."""
    scores = read_table(scores_path, ["behaviour_id", "article_id", "score"])
    if scores[["behaviour_id", "article_id", "score"]].isna().any().any():
        raise ValueError("Recorded-slate scores contain null keys or scores")
    if scores.duplicated(["behaviour_id", "article_id"]).any():
        raise ValueError("Recorded-slate score keys are duplicated")
    label_lookup = labels.assign(behaviour_id=[
        _behavior_id(u, i) for u, i in zip(labels.user_id, labels.impression_id)
    ])[["behaviour_id", "candidate_article_id", "clicked"]].rename(
        columns={"candidate_article_id": "article_id"}
    )
    merged = scores.merge(label_lookup, on=["behaviour_id", "article_id"], how="outer", indicator=True)
    missing = int((merged._merge == "right_only").sum())
    extra = int((merged._merge == "left_only").sum())
    if missing or extra:
        raise ValueError(f"Recorded-slate score coverage mismatch: {missing} missing, {extra} unexpected")
    merged = merged.loc[merged._merge == "both"].copy()
    if merged.empty:
        raise ValueError("Recorded-slate evaluation has no rows; provide a non-empty matching score/label split")
    merged[["user_id", "impression_id"]] = merged.behaviour_id.str.split(":", expand=True).astype("int64")
    per = per_impression_metrics(merged, user_col="user_id", ks=(k,))
    aggregate = aggregate_per_impression(per).iloc[0].to_dict()
    return {"scope": "recorded_slate_diagnostic", "impressions": int(aggregate["impressions"]),
            "candidate_rows": int(aggregate["candidate_rows"]),
            "mean_impression_auc": aggregate["mean_impression_auc"],
            "auc_eligible_impressions": int(aggregate["impressions"] - aggregate["auc_impressions_excluded_single_class"]),
            "auc_excluded_single_class": int(aggregate["auc_impressions_excluded_single_class"]),
            f"mean_impression_ndcg@{k}": aggregate.get(f"ndcg@{k}"),
            f"ndcg@{k}_excluded_no_click": int(aggregate.get(f"ndcg@{k}_impressions_excluded_no_positive", 0))}


def add_continuation_diagnostics(
    per_request: pd.DataFrame,
    rankings: pd.DataFrame,
    labels_path: Path,
    clicked_map: dict[str, set[int]],
    ks: tuple[int, ...],
    *, allow_subset: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate optional observed continuation labels on clicked candidates only.

    The input must omit candidates whose continuation outcome is unobserved.
    Therefore, continuation rate is conditional on labeled clicked candidates;
    unclicked catalogue articles are never assigned a zero label.
    """
    cont = read_table(labels_path, ["behaviour_id", "article_id", "continued"])
    if cont.isna().any().any() or cont.duplicated(["behaviour_id", "article_id"]).any():
        raise ValueError("Continuation labels need unique keys and observed 0/1 outcomes only")
    cont["behaviour_id"] = _normalize_behaviour_ids(cont.behaviour_id)
    if cont.duplicated(["behaviour_id", "article_id"]).any():
        raise ValueError("Continuation labels contain duplicate keys after request-ID normalization")
    cont["article_id"] = checked_int64_series(cont.article_id, "continuation article_id")
    if not cont.continued.isin([0, 1, False, True]).all():
        raise ValueError("continued must be observed binary 0/1; omit censored/unobserved candidates")
    evaluated_requests, ranking_map = _validate_request_ranking_scope(per_request, rankings)
    out_of_scope = ~cont.behaviour_id.astype(str).isin(evaluated_requests)
    excluded_rows = int(out_of_scope.sum())
    excluded_requests = int(cont.loc[out_of_scope, "behaviour_id"].nunique())
    if excluded_rows and not allow_subset:
        raise ValueError(f"Continuation labels contain {excluded_requests} requests outside evaluated scope")
    cont = cont.loc[~out_of_scope].copy()
    for row in cont.itertuples(index=False):
        if int(row.article_id) not in clicked_map.get(str(row.behaviour_id), set()):
            raise ValueError("Continuation labels may only be attached to known clicked candidates")
    outcome = {(str(r.behaviour_id), int(r.article_id)): int(r.continued)
               for r in cont.itertuples(index=False)}
    outcomes_by_request: dict[str, dict[int, int]] = {}
    for (behaviour_id, article_id), label in outcome.items():
        outcomes_by_request.setdefault(behaviour_id, {})[article_id] = label
    record_map = per_request.copy()
    record_map["behaviour_id"] = _normalize_behaviour_ids(record_map.behaviour_id)
    record_map = record_map.set_index("behaviour_id")
    for behaviour_id in record_map.index:
        all_observed = outcomes_by_request.get(behaviour_id, {})
        record_map.loc[behaviour_id, "continuation_observed_candidates"] = len(all_observed)
        record_map.loc[behaviour_id, "continuation_positive_candidates"] = sum(all_observed.values())
        ranked = ranking_map[behaviour_id]
        for k in ks:
            top = ranked[:k]
            top_observed = [(a, outcome[(behaviour_id, a)]) for a in top if (behaviour_id, a) in outcome]
            pos_total = sum(all_observed.values())
            record_map.loc[behaviour_id, f"observed_continuation_recall@{k}"] = (
                sum(y for _, y in top_observed) / pos_total if pos_total else np.nan
            )
            record_map.loc[behaviour_id, f"continuation_observation_coverage@{k}"] = (
                len(top_observed) / len(top) if top else np.nan
            )
            record_map.loc[behaviour_id, f"continuation_rate_observed_topk@{k}"] = (
                sum(y for _, y in top_observed) / len(top_observed) if top_observed else np.nan
            )
    per_request = record_map.reset_index()
    labeled_requests = set(cont.behaviour_id.astype(str))
    cont_summary = {"evaluated_requests": int(len(evaluated_requests)),
                    "requests_with_rankings": int(len(evaluated_requests)),
                    "requests_with_observed_continuation_labels": int(len(labeled_requests)),
                    "requests_with_labels_and_rankings": int(len(labeled_requests & evaluated_requests)),
                    "labeled_clicked_candidates": int(len(cont)),
                    "positive_continuations": int(cont.continued.astype(bool).sum()),
                    "overall_observed_continuation_rate": float(cont.continued.mean()) if len(cont) else None,
                    "out_of_scope_rows_excluded": excluded_rows,
                    "out_of_scope_requests_excluded": excluded_requests,
                    "limitations": "Conditional on clicked candidates with observed follow-up; not a counterfactual outcome for unclicked catalogue items."}
    for k in ks:
        for name in (f"observed_continuation_recall@{k}",
                     f"continuation_observation_coverage@{k}",
                     f"continuation_rate_observed_topk@{k}"):
            cont_summary[name] = float(per_request[name].mean()) if per_request[name].notna().any() else None
    return per_request, cont_summary


def add_continuity_proxy(
    per_request: pd.DataFrame,
    rankings: pd.DataFrame,
    values_path: Path,
    articles: pd.DataFrame,
    ks: tuple[int, ...],
    *, allow_subset: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Summarize a fully pre-impression candidate continuity proxy in ranked lists.

    The file contains behaviour_id, article_id, continuity_score in [0,1]. This
    is a ranking-property summary, not an observed continuation outcome.
    """
    values = read_table(values_path, ["behaviour_id", "article_id", "continuity_score"])
    if values[["behaviour_id", "article_id", "continuity_score"]].isna().any().any():
        raise ValueError("Continuity proxy values must be non-null")
    if values.duplicated(["behaviour_id", "article_id"]).any():
        raise ValueError("Continuity proxy has duplicate request/article keys")
    values["behaviour_id"] = _normalize_behaviour_ids(values.behaviour_id)
    if values.duplicated(["behaviour_id", "article_id"]).any():
        raise ValueError("Continuity proxy has duplicate keys after request-ID normalization")
    valid_requests, ranking_map = _validate_request_ranking_scope(per_request, rankings)
    out_of_scope = ~values.behaviour_id.astype(str).isin(valid_requests)
    excluded_rows = int(out_of_scope.sum())
    excluded_requests = int(values.loc[out_of_scope, "behaviour_id"].nunique())
    if excluded_rows and not allow_subset:
        raise ValueError(f"Continuity proxy contains {excluded_requests} requests outside evaluated scope")
    values = values.loc[~out_of_scope].copy()
    values["article_id"] = checked_int64_series(values.article_id, "continuity proxy article_id")
    unknown_articles = set(values.article_id) - set(articles.index)
    if unknown_articles:
        raise ValueError(f"Continuity proxy contains {len(unknown_articles)} article IDs absent from catalogue")
    request_times = per_request.assign(
        behaviour_id=per_request.behaviour_id.astype(str)
    ).set_index("behaviour_id").impression_time
    proxy_times = values.behaviour_id.astype(str).map(request_times)
    publication_times = values.article_id.map(articles.published_time)
    if proxy_times.isna().any() or publication_times.isna().any():
        raise ValueError("Continuity proxy rows need known request and article publication times")
    if (publication_times > proxy_times).any():
        raise ValueError("Continuity proxy includes articles published after their request")
    values["continuity_score"] = pd.to_numeric(values.continuity_score, errors="raise")
    if not np.isfinite(values.continuity_score.to_numpy(float)).all() or not values.continuity_score.between(0, 1).all():
        raise ValueError("continuity_score must be finite and in [0,1]")
    lookup = {(str(row.behaviour_id), int(row.article_id)): float(row.continuity_score)
              for row in values.itertuples(index=False)}
    result = per_request.copy()
    result["behaviour_id"] = _normalize_behaviour_ids(result.behaviour_id)
    result = result.set_index("behaviour_id")
    for behaviour_id in result.index:
        ranked = ranking_map[behaviour_id]
        for k in ks:
            top = ranked[:k]
            observed = [lookup[(behaviour_id, article_id)] for article_id in top
                        if (behaviour_id, article_id) in lookup]
            result.loc[behaviour_id, f"session_continuity_proxy@{k}"] = (
                float(np.mean(observed)) if observed else np.nan
            )
            result.loc[behaviour_id, f"continuity_proxy_coverage@{k}"] = (
                len(observed) / len(top) if top else np.nan
            )
    per_request = result.reset_index()
    proxy_requests = set(values.behaviour_id.astype(str))
    report = {"definition": "Mean supplied pre-impression candidate continuity proxy among ranked items; not an observed continuation outcome.",
              "evaluated_requests": int(len(valid_requests)),
              "requests_with_rankings": int(len(valid_requests)),
              "requests_with_proxy_and_rankings": int(len(proxy_requests & valid_requests)),
              "scored_request_article_rows": int(len(values)),
              "requests_with_proxy_rows": int(values.behaviour_id.nunique()),
              "requests_without_proxy_rows": int(len(valid_requests - set(values.behaviour_id.astype(str)))),
              "out_of_scope_rows_excluded": excluded_rows,
              "out_of_scope_requests_excluded": excluded_requests,
              "note": "Upstream generation must guarantee proxy inputs use only information available before each impression."}
    for k in ks:
        for metric in (f"session_continuity_proxy@{k}", f"continuity_proxy_coverage@{k}"):
            report[metric] = float(per_request[metric].mean()) if per_request[metric].notna().any() else None
    return per_request, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path,
                        help="Candidate-level split features with user_id, impression_id, candidate_article_id, clicked, impression_time")
    parser.add_argument("--articles", required=True, type=Path,
                        help="The shared ordered article catalogue union with article_id and published_time; topics/vector flags optional")
    parser.add_argument("--rankings", required=True, type=Path,
                        help="Schema-v4 rankings file with behaviour_id and native article_ids list")
    parser.add_argument("--catalogue-scores", type=Path,
                        help="Optional schema-v4 top-K score rows; validated against ranking membership/order")
    parser.add_argument("--recorded-slate-scores", type=Path,
                        help="Optional complete diagnostics file: behaviour_id, article_id, score")
    parser.add_argument("--continuation-labels", type=Path,
                        help="Optional observed clicked-candidate outcomes: behaviour_id, article_id, continued (censored rows omitted)")
    parser.add_argument("--continuity-proxy", type=Path,
                        help="Optional pre-impression candidate proxy: behaviour_id, article_id, continuity_score in [0,1]")
    parser.add_argument("--similarity-constants", type=Path,
                        help="Optional training-derived s_same/s_diff/s_all JSON; adds fallback-adjusted diversity")
    parser.add_argument("--segments", type=Path,
                        help="Optional request-level table with behaviour_id and subgroup columns")
    parser.add_argument("--segment-columns", nargs="*", default=[],
                        help="Columns from --segments to summarize, e.g. history_length_group device_type")
    parser.add_argument("--export-k", type=int, default=200)
    parser.add_argument("--ks", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--split", choices=("train", "validation", "holdout", "test"), default="validation",
                        help="Name the supplied labelled partition; use holdout only after choices are frozen")
    parser.add_argument("--naive-timezone", default="UTC",
                        help="Timezone to assign to timezone-naive impression/article timestamps (default: UTC)")
    parser.add_argument("--category-fallback-for-topics", action="store_true",
                        help="Explicitly use category when an article has no topic labels; changes topic metric meaning")
    parser.add_argument("--run-config", type=Path,
                        help="Optional model run_config.json; validates schema-v4 and shared catalogue policy")
    parser.add_argument("--allow-subset", action="store_true",
                        help="Permit a declared development request sample; use the same IDs for every model")
    parser.add_argument("--slate-k", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path, help="JSON report path")
    parser.add_argument("--per-request-output", type=Path,
                        help="Optional CSV/Parquet with request-level metrics for paired/user bootstrap")
    parser.add_argument("--top-items-output", type=Path,
                        help="Optional CSV/Parquet with ranked items and known-click/age diagnostics")
    parser.add_argument("--hash-inputs", action="store_true", help="SHA-256 inputs (can take time on large files)")
    args = parser.parse_args()

    if args.export_k < 1 or args.slate_k < 1 or not args.ks or any(k < 1 for k in args.ks):
        parser.error("--export-k, --slate-k, and every --ks cutoff must be positive")
    if any(k > args.export_k for k in args.ks):
        parser.error("Every --ks cutoff must be <= --export-k")

    catalogue_manifest_path = args.articles.with_suffix(args.articles.suffix + ".manifest.json")
    catalogue_manifest = None
    if catalogue_manifest_path.exists():
        catalogue_manifest = json.loads(catalogue_manifest_path.read_text(encoding="utf-8"))
        manifest_timezone = catalogue_manifest.get("naive_timestamp_timezone")
        if manifest_timezone and manifest_timezone != args.naive_timezone:
            raise ValueError(
                f"Catalogue was built with naive timezone {manifest_timezone!r}; "
                f"use the same --naive-timezone (got {args.naive_timezone!r})"
            )
    else:
        warnings.warn("No catalogue manifest found beside --articles; source timezone policy cannot be verified")

    if "split" in table_columns(args.labels):
        split_values = set(read_table(args.labels, ["split"])["split"].dropna().astype(str).unique())
        accepted = {args.split}
        if split_values and not split_values.issubset(accepted):
            raise ValueError(f"Label table split values {sorted(split_values)} do not match --split {args.split}")
    request_frame, click_map = load_labels(args.labels, naive_timezone=args.naive_timezone)
    rankings = load_rankings(args.rankings)
    articles = load_articles(args.articles, naive_timezone=args.naive_timezone,
                             category_fallback_for_topics=args.category_fallback_for_topics)
    similarity_constants = (
        json.loads(args.similarity_constants.read_text(encoding="utf-8"))
        if args.similarity_constants else None
    )
    summary, per_request, top_items = evaluate_rankings(
        request_frame, click_map, rankings, articles, export_k=args.export_k, ks=tuple(args.ks),
        allow_subset=args.allow_subset, include_top_items=args.top_items_output is not None,
        similarity_constants=similarity_constants,
    )
    catalogue_fingerprint, catalogue_content_fingerprint = catalogue_fingerprints(articles)
    request_fingerprint = hashlib.sha256(
        "\n".join(sorted(per_request.behaviour_id.astype(str))).encode("utf-8")
    ).hexdigest()
    per_request["catalogue_id_fingerprint_sha256"] = catalogue_fingerprint
    per_request["catalogue_content_fingerprint_sha256"] = catalogue_content_fingerprint
    per_request["request_id_fingerprint_sha256"] = request_fingerprint
    per_request["split"] = args.split
    per_request["metric_definition_version"] = "shared_eval_v4"
    per_request["eligibility_policy"] = "published_at_or_before_request_v1"
    per_request["naive_timezone_policy"] = args.naive_timezone
    if similarity_constants is not None:
        constants_hash = hashlib.sha256(json.dumps(
            similarity_constants, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        per_request["similarity_constants_sha256"] = constants_hash
    per_request["topic_fallback_policy"] = (
        "category_for_missing_topics" if args.category_fallback_for_topics else "none"
    )
    per_request["export_k"] = int(args.export_k)
    per_request["metric_cutoffs_json"] = json.dumps(sorted(set(args.ks)))
    selection_cutoff = 100 if 100 in args.ks else (5 if 5 in args.ks else min(args.ks))
    report = {
        "protocol": "schema_v4_catalogue_and_stage2_evaluation",
        "split": args.split,
        "scope": {
            "train": "in_sample_training",
            "validation": "validation_development",
        }.get(args.split, "final_holdout_or_declared_test"),
        "export_k": args.export_k,
        "ks": sorted(set(args.ks)),
        "primary_metric": f"known_click_recall@{selection_cutoff}",
        "selection_tiebreaker": f"observed_click_ndcg@{selection_cutoff}",
        "metrics": summary,
        "catalogue_id_fingerprint_sha256": catalogue_fingerprint,
        "catalogue_content_fingerprint_sha256": catalogue_content_fingerprint,
        "request_id_fingerprint_sha256": request_fingerprint,
        "evaluated_request_count": int(len(per_request)),
        "eligibility_policy": "published_at_or_before_request_v1",
        "naive_timezone_policy": args.naive_timezone,
        "topic_fallback_policy": (
            "category_for_missing_topics" if args.category_fallback_for_topics else "none"
        ),
        "metric_definitions": {
            "known_click_recall@K": "Fraction of all known observed clicks recovered in top K; clicks excluded by eligibility remain in denominator.",
            "observed_click_ndcg@K": "Binary-gain nDCG using all known observed clicks, including ineligible clicks, in the ideal denominator.",
            "topic_coverage@K": "Mean number of distinct topic labels in top K (a count, not a percentage).",
            "topic_article_coverage@K": "Fraction of top-K articles with non-empty topic metadata.",
            "intra_list_diversity@K": "Mean pairwise topic Jaccard distance among articles with known non-empty topic sets.",
            "shared_intra_list_diversity@K": "Mean pairwise distance using topic Jaccard and training-estimated category/global fallback similarities from --similarity-constants.",
            "distinct_categories@K": "Number of distinct non-missing category values in top K.",
            "freshness_shares": "Age-bin shares use all ranked slots in top K as denominator; unknown-age share is reported separately.",
        },
        "limitations": [
            "Catalogue metrics recover original observed clicks; outcomes for unshown articles are unknown.",
            "Holdout/test outcomes must be loaded only after model and reranker choices are frozen.",
            "Session continuation is not inferred from missing telemetry; provide a validated outcome/proxy separately.",
        ],
        "inputs": {"labels": file_provenance(args.labels, hash_file=args.hash_inputs),
                   "articles": file_provenance(args.articles, hash_file=args.hash_inputs),
                   "rankings": file_provenance(args.rankings, hash_file=args.hash_inputs)},
    }
    if similarity_constants is not None:
        report["similarity_constants"] = similarity_constants
        report["similarity_constants_sha256"] = constants_hash
        report["inputs"]["similarity_constants"] = file_provenance(
            args.similarity_constants, hash_file=args.hash_inputs
        )
    if catalogue_manifest_path.exists():
        report["catalogue_manifest"] = catalogue_manifest
        report["inputs"]["catalogue_manifest"] = file_provenance(
            catalogue_manifest_path, hash_file=args.hash_inputs
        )
    if args.run_config:
        run_config = json.loads(args.run_config.read_text(encoding="utf-8"))
        if run_config.get("export_schema_version") != 4:
            raise ValueError("Run config must declare export_schema_version=4")
        if run_config.get("prediction_pool") != "time_eligible_catalogue":
            raise ValueError("Run config prediction_pool must be time_eligible_catalogue")
        pool_policy = run_config.get("pool_policy", {})
        policy_name = pool_policy.get("name", pool_policy.get("policy")) if isinstance(pool_policy, dict) else pool_policy
        if policy_name != "published_at_or_before_request_v1":
            raise ValueError(f"Unexpected catalogue eligibility policy: {policy_name}")
        run_catalogue_fingerprint = run_config.get("catalogue_id_fingerprint_sha256")
        if not run_catalogue_fingerprint:
            raise ValueError("Run config must include catalogue_id_fingerprint_sha256")
        if run_catalogue_fingerprint != catalogue_fingerprint:
            raise ValueError("Run config catalogue ID fingerprint does not match the evaluator catalogue")
        run_content_fingerprint = run_config.get("catalogue_content_fingerprint_sha256")
        if run_content_fingerprint and run_content_fingerprint != catalogue_content_fingerprint:
            raise ValueError("Run config catalogue content fingerprint does not match the evaluator catalogue")
        if not run_content_fingerprint:
            warnings.warn("Run config has no catalogue content fingerprint; metadata identity is verified only by article IDs")
        run_split = run_config.get("split")
        if run_split and run_split != args.split:
            raise ValueError(f"Run config split {run_split!r} does not match --split {args.split!r}")
        if not run_split:
            warnings.warn("Run config has no split field; split identity is taken from --split and the labels")
        run_timezone = run_config.get("naive_timezone_policy")
        if run_timezone and run_timezone != args.naive_timezone:
            raise ValueError("Run config naive timezone policy does not match --naive-timezone")
        request_scope = run_config.get("request_id_fingerprint_sha256")
        if request_scope and request_scope != report["request_id_fingerprint_sha256"]:
            raise ValueError("Run config request ID fingerprint does not match label request scope")
        declared_scope = run_config.get("export_request_scope")
        if declared_scope is None:
            warnings.warn("Run config has no export_request_scope; exact scope is recorded from evaluated rankings")
        elif isinstance(declared_scope, dict):
            declared_hash = declared_scope.get("request_id_fingerprint_sha256", declared_scope.get("fingerprint_sha256"))
            declared_count = declared_scope.get("request_count")
            if declared_hash and declared_hash != report["request_id_fingerprint_sha256"]:
                raise ValueError("Run config export_request_scope fingerprint does not match evaluated rankings")
            if declared_count is not None and int(declared_count) != len(per_request):
                raise ValueError("Run config export_request_scope request_count does not match evaluated rankings")
        elif isinstance(declared_scope, str) and len(declared_scope) == 64:
            if declared_scope != report["request_id_fingerprint_sha256"]:
                raise ValueError("Run config export_request_scope fingerprint does not match evaluated rankings")
        report["run_config"] = {"experiment_id": run_config.get("experiment_id"),
                                "run_id": run_config.get("run_id"),
                                "export_schema_version": run_config.get("export_schema_version"),
                                "prediction_pool": run_config.get("prediction_pool"),
                                "pool_policy": pool_policy,
                                "catalogue_id_fingerprint_sha256": run_config.get("catalogue_id_fingerprint_sha256"),
                                "catalogue_content_fingerprint_sha256": run_config.get("catalogue_content_fingerprint_sha256"),
                                "request_id_fingerprint_sha256": run_config.get("request_id_fingerprint_sha256"),
                                "naive_timezone_policy": run_config.get("naive_timezone_policy"),
                                "export_request_scope": run_config.get("export_request_scope")}
        per_request["run_id"] = run_config.get("run_id")
        report["inputs"]["run_config"] = file_provenance(args.run_config, hash_file=args.hash_inputs)
    if args.catalogue_scores:
        report["catalogue_score_export_check"] = validate_catalogue_scores(
            args.catalogue_scores, rankings,
        )
        report["inputs"]["catalogue_scores"] = file_provenance(
            args.catalogue_scores, hash_file=args.hash_inputs)
    if args.recorded_slate_scores:
        slate_labels = read_table(args.labels, ["user_id", "impression_id", "candidate_article_id", "clicked"])
        report["recorded_slate_diagnostics"] = evaluate_recorded_slates(
            args.recorded_slate_scores, slate_labels, k=args.slate_k,
        )
        report["inputs"]["recorded_slate_scores"] = file_provenance(
            args.recorded_slate_scores, hash_file=args.hash_inputs)
    if args.continuation_labels:
        per_request, report["session_continuation"] = add_continuation_diagnostics(
            per_request, rankings, args.continuation_labels, click_map, tuple(args.ks),
            allow_subset=args.allow_subset,
        )
        report["inputs"]["continuation_labels"] = file_provenance(
            args.continuation_labels, hash_file=args.hash_inputs)
        report["session_continuation"]["requests_with_observable_clicked_outcomes"] = int(
            report["session_continuation"]["requests_with_observed_continuation_labels"]
        )
        report["session_continuation"]["request_coverage"] = (
            report["session_continuation"]["requests_with_observable_clicked_outcomes"] / len(per_request)
            if len(per_request) else None
        )
    if args.continuity_proxy:
        per_request, report["session_continuity_proxy"] = add_continuity_proxy(
            per_request, rankings, args.continuity_proxy, articles, tuple(args.ks),
            allow_subset=args.allow_subset,
        )
        report["inputs"]["continuity_proxy"] = file_provenance(
            args.continuity_proxy, hash_file=args.hash_inputs)
    if args.segments:
        if not args.segment_columns:
            raise ValueError("Set --segment-columns when supplying --segments")
        segment_frame = read_table(args.segments)
        per_request = summarize_shared_segments(per_request, segment_frame, args.segment_columns)
        report["segment_summaries"] = per_request.attrs.get("segment_summaries", [])
        report["segment_matching"] = per_request.attrs.get("segment_matching", {})
        report["inputs"]["segments"] = file_provenance(args.segments, hash_file=args.hash_inputs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    if args.per_request_output:
        args.per_request_output.parent.mkdir(parents=True, exist_ok=True)
        if args.per_request_output.suffix.lower() == ".csv":
            per_request.to_csv(args.per_request_output, index=False)
        else:
            per_request.to_parquet(args.per_request_output, index=False)
    if args.top_items_output:
        args.top_items_output.parent.mkdir(parents=True, exist_ok=True)
        if args.top_items_output.suffix.lower() == ".csv":
            top_items.to_csv(args.top_items_output, index=False)
        else:
            top_items.to_parquet(args.top_items_output, index=False)
    print(json.dumps(report, indent=2, allow_nan=False, default=_json_default))


if __name__ == "__main__":
    main()
