"""Build impression-time candidate features from one EB-NeRD split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from array import array
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence


HOUR_US = 3_600_000_000
DAY_US = 24 * HOUR_US
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
LabelMode = Literal["auto", "required", "none"]
CATEGORICAL_COLUMNS = ("device_type", "category", "article_type", "premium_subscriber")
ONEHOT_STATE_VERSION = 1

ARTICLE_COLUMNS = (
    "article_id", "category", "article_type", "premium", "published_time", "topics"
)
HISTORY_COLUMNS = (
    "user_id", "article_id_fixed", "impression_time_fixed",
    "read_time_fixed", "scroll_percentage_fixed",
)
BEHAVIOR_REQUIRED = (
    "impression_id", "user_id", "impression_time", "article_ids_inview"
)
BEHAVIOR_OPTIONAL = (
    "article_id", "device_type", "is_sso_user", "is_subscriber",
    "article_ids_clicked",
)

# Current-feature predictors retained after EDA. IDs, labels, outcome fields,
# text, and the very sparse age/gender/postcode profile fields are excluded.
# Basic summaries and time representations make the source fields usable by a
# tabular model; the report's cross-field and window features are listed below.
ORIGINAL_FEATURE_COLUMNS = (
    "device_type", "is_sso_user", "is_subscriber", "is_front_page",
    "hour", "day_of_week", "slate_size",
    "has_history", "history_length", "median_read_time",
    "mean_scroll_percentage", "category", "article_type", "premium",
    "article_age_h", "article_age_missing",
)
ENGINEERED_FEATURE_COLUMNS = (
    "history_clicks_7d", "is_24h_eligible", "slate_relative_age_h",
    "category_affinity_72h", "category_affinity_missing",
    "last5_mean_topic_jaccard", "topic_affinity_missing",
    "has_any_topic_overlap",
    "same_category_mean_log1p_read_time",
    "same_category_read_depth_missing", "premium_subscriber",
)
LOG1P_FEATURE_VARIANTS = {
    "history_length": "history_length_log1p",
    "median_read_time": "median_read_time_log1p",
    "history_clicks_7d": "history_clicks_7d_log1p",
    "article_age_h": "article_age_log1p",
}
# Keep the existing name for code that imports the baseline definition.
BASELINE_FEATURE_COLUMNS = ORIGINAL_FEATURE_COLUMNS


@dataclass(slots=True)
class Article:
    category: str | None
    article_type: str | None
    premium: bool | None
    published_us: int | None
    topics: frozenset[str]


@dataclass(slots=True)
class HistoryProfile:
    latest_click_us: int | None
    history_length: int
    median_read_time: float | None
    mean_scroll_percentage: float | None
    category_affinity: dict[str, float] | None
    last5_topics: tuple[frozenset[str], ...]
    same_category_read_depth: dict[str, float]
    recent_click_us: array


def _timestamp_us(value: datetime | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise TypeError(f"Expected a datetime, got {type(value).__name__}")
    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    delta = aware.astimezone(timezone.utc) - EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, float) and math.isnan(value):
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError("Expected an array field, not text")
    return tuple(value)


def _number(value: Any, *, upper: float | None = None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < 0 or (upper is not None and result > upper):
        return None
    return result


def _nominal(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return bool(value)


def _topics(value: Any) -> frozenset[str]:
    return frozenset(str(item) for item in _sequence(value) if item is not None)


def load_articles(rows: Iterable[Mapping[str, Any]]) -> dict[int, Article]:
    """Keep only non-text catalogue fields needed by the feature definitions."""
    articles: dict[int, Article] = {}
    for row in rows:
        article_id = int(row["article_id"])
        if article_id in articles:
            raise ValueError(f"Duplicate article ID: {article_id}")
        articles[article_id] = Article(
            category=_nominal(row.get("category")),
            article_type=_nominal(row.get("article_type")),
            premium=_bool(row.get("premium")),
            published_us=_timestamp_us(row.get("published_time")),
            topics=_topics(row.get("topics")),
        )
    return articles


def prepare_history_profiles(
    rows: Iterable[Mapping[str, Any]],
    articles: Mapping[int, Article],
    *,
    split_start: datetime,
    user_modulus: int = 1,
    user_remainder: int = 0,
) -> dict[int, HistoryProfile]:
    """Summarise the fixed history available before the behaviour split starts."""
    _check_user_partition(user_modulus, user_remainder)
    start_us = _timestamp_us(split_start)
    if start_us is None:
        raise ValueError("split_start is required")
    profiles: dict[int, HistoryProfile] = {}

    for row in rows:
        user_id = int(row["user_id"])
        if user_id % user_modulus != user_remainder:
            continue
        if user_id in profiles:
            raise ValueError(f"Duplicate history row for user {user_id}")

        ids = _sequence(row.get("article_id_fixed"))
        times = _sequence(row.get("impression_time_fixed"))
        reads = _sequence(row.get("read_time_fixed"))
        scrolls = _sequence(row.get("scroll_percentage_fixed"))
        if len({len(ids), len(times), len(reads), len(scrolls)}) != 1:
            raise ValueError(f"Misaligned history arrays for user {user_id}")

        events: list[tuple[int, int, float | None, float | None]] = []
        for article_id, time, read, scroll in zip(ids, times, reads, scrolls):
            time_us = _timestamp_us(time)
            if time_us is None:
                continue
            if time_us >= start_us:
                raise ValueError(f"History for user {user_id} reaches the behaviour week")
            events.append((time_us, int(article_id), _number(read), _number(scroll, upper=100)))
        events.sort(key=lambda event: event[0])

        valid_reads = [read for _, _, read, _ in events if read is not None]
        valid_scrolls = [scroll for _, _, _, scroll in events if scroll is not None]
        median_read = float(statistics.median(valid_reads)) if valid_reads else None
        mean_scroll = float(statistics.fmean(valid_scrolls)) if valid_scrolls else None

        category_weight: dict[str, float] = defaultdict(float)
        reading_total: dict[str, float] = defaultdict(float)
        reading_count: dict[str, int] = defaultdict(int)
        denominator = 0.0
        latest_us = events[-1][0] if events else None
        for time_us, article_id, read, _ in events:
            article = articles.get(article_id)
            if article is None or article.category is None:
                continue
            if latest_us is None:
                raise RuntimeError("A history event has no latest timestamp")
            weight = math.exp(-((latest_us - time_us) / HOUR_US) / 72.0)
            category_weight[article.category] += weight
            denominator += weight
            if read is not None:
                reading_total[article.category] += math.log1p(read)
                reading_count[article.category] += 1

        affinity = (
            {category: weight / denominator for category, weight in category_weight.items()}
            if denominator > 0 else None
        )
        reading_depth = {
            category: total / reading_count[category]
            for category, total in reading_total.items()
        }
        last5_topics = tuple(
            articles[article_id].topics if article_id in articles else frozenset()
            for _, article_id, _, _ in events[-5:]
        )
        # Later seven-day windows cannot use clicks older than this cutoff.
        recent = array("q", (time_us for time_us, _, _, _ in events if time_us >= start_us - 7 * DAY_US))
        profiles[user_id] = HistoryProfile(
            latest_click_us=latest_us,
            history_length=len(events),
            median_read_time=median_read,
            mean_scroll_percentage=mean_scroll,
            category_affinity=affinity,
            last5_topics=last5_topics,
            same_category_read_depth=reading_depth,
            recent_click_us=recent,
        )
    return profiles


def _check_user_partition(modulus: int, remainder: int) -> None:
    if modulus < 1 or not 0 <= remainder < modulus:
        raise ValueError("Require user_modulus >= 1 and 0 <= user_remainder < user_modulus")


def _topic_affinity(
    candidate_topics: frozenset[str], profile: HistoryProfile | None
) -> float | None:
    if not candidate_topics or profile is None:
        return None
    scores = [
        len(candidate_topics & previous) / len(candidate_topics | previous)
        for previous in profile.last5_topics if previous
    ]
    return statistics.fmean(scores) if scores else None


def _premium_subscriber(premium: bool | None, subscriber: bool | None) -> str:
    if premium is None or subscriber is None:
        return "missing"
    access = "premium" if premium else "standard"
    status = "subscriber" if subscriber else "non_subscriber"
    return f"{access}_{status}"


def iter_feature_rows(
    behaviors: Iterable[Mapping[str, Any]],
    articles: Mapping[int, Article],
    profiles: Mapping[int, HistoryProfile],
    *,
    split: str,
    label_mode: LabelMode = "auto",
) -> Iterator[dict[str, Any]]:
    """Yield one row per shown candidate, including older and unmatched articles."""
    if label_mode not in ("auto", "required", "none"):
        raise ValueError("label_mode must be auto, required or none")
    for behavior in behaviors:
        user_id = int(behavior["user_id"])
        impression_id = int(behavior["impression_id"])
        impression_time = behavior["impression_time"]
        time_us = _timestamp_us(impression_time)
        if time_us is None:
            raise ValueError(f"Missing time for impression {impression_id}")
        candidate_ids = tuple(int(article_id) for article_id in _sequence(behavior["article_ids_inview"]))
        if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"Empty or duplicate candidate slate for impression {impression_id}")

        clicked_raw = behavior.get("article_ids_clicked")
        if label_mode == "required" and clicked_raw is None:
            raise ValueError(f"Missing click labels for impression {impression_id}")
        clicked_ids = (
            set(_sequence(clicked_raw))
            if label_mode != "none" and clicked_raw is not None else None
        )

        profile = profiles.get(user_id)
        if profile is not None and profile.latest_click_us is not None:
            if profile.latest_click_us >= time_us:
                raise ValueError(f"History reaches impression {impression_id}")
        if profile is None:
            recent_clicks = None
        else:
            recent_clicks = bisect_left(profile.recent_click_us, time_us) - bisect_left(
                profile.recent_click_us, time_us - 7 * DAY_US
            )

        ages: dict[int, float | None] = {}
        eligible_ages: list[float] = []
        for article_id in candidate_ids:
            article = articles.get(article_id)
            age = (
                (time_us - article.published_us) / HOUR_US
                if article is not None and article.published_us is not None else None
            )
            age = age if age is not None and age >= 0 else None
            ages[article_id] = age
            if age is not None and age <= 24.0:
                eligible_ages.append(age)
        eligible_median = statistics.median(eligible_ages) if eligible_ages else None

        subscriber = _bool(behavior.get("is_subscriber"))
        current_page_id = behavior.get("article_id")
        if isinstance(current_page_id, float) and math.isnan(current_page_id):
            current_page_id = None
        for article_id in candidate_ids:
            article = articles.get(article_id)
            category = article.category if article else None
            age = ages[article_id]
            eligible = age is not None and age <= 24.0
            affinity = (
                profile.category_affinity.get(category, 0.0)
                if profile is not None and profile.category_affinity is not None
                and category is not None else None
            )
            topic_affinity = _topic_affinity(article.topics, profile) if article else None
            read_depth = (
                profile.same_category_read_depth.get(category)
                if profile is not None and category is not None else None
            )
            median_read = profile.median_read_time if profile else None
            history_length = profile.history_length if profile else None
            premium = article.premium if article else None
            yield {
                "split": split,
                "impression_id": impression_id,
                "user_id": user_id,
                "candidate_article_id": article_id,
                "current_page_article_id": int(current_page_id) if current_page_id is not None else None,
                "impression_time": impression_time,
                "clicked": int(article_id in clicked_ids) if clicked_ids is not None else None,
                "device_type": _nominal(behavior.get("device_type")),
                "is_sso_user": _bool(behavior.get("is_sso_user")),
                "is_subscriber": subscriber,
                "is_front_page": current_page_id is None,
                "hour": impression_time.hour,
                "day_of_week": impression_time.weekday(),
                "slate_size": len(candidate_ids),
                "has_history": profile is not None,
                "history_length": history_length,
                "history_length_log1p": math.log1p(history_length) if history_length is not None else None,
                "median_read_time": median_read,
                "median_read_time_log1p": math.log1p(median_read) if median_read is not None else None,
                "mean_scroll_percentage": profile.mean_scroll_percentage if profile else None,
                "history_clicks_7d": recent_clicks,
                "history_clicks_7d_log1p": math.log1p(recent_clicks) if recent_clicks is not None else None,
                "category": category,
                "article_type": article.article_type if article else None,
                "premium": premium,
                "article_age_h": age,
                "article_age_log1p": math.log1p(age) if age is not None else None,
                "article_age_missing": age is None,
                "is_24h_eligible": eligible,
                "slate_relative_age_h": age - eligible_median if eligible and eligible_median is not None else None,
                "category_affinity_72h": affinity,
                "category_affinity_missing": affinity is None,
                "last5_mean_topic_jaccard": topic_affinity,
                "topic_affinity_missing": topic_affinity is None,
                "has_any_topic_overlap": topic_affinity > 0 if topic_affinity is not None else None,
                "same_category_mean_log1p_read_time": read_depth,
                "same_category_read_depth_missing": read_depth is None,
                "premium_subscriber": _premium_subscriber(premium, subscriber),
            }


def _selected_user_rows(batch: Any, modulus: int, remainder: int) -> list[dict[str, Any]]:
    """Convert only selected users' nested Arrow rows to Python objects."""
    if modulus == 1:
        return batch.to_pylist()
    import pyarrow as pa

    user_index = batch.schema.get_field_index("user_id")
    if user_index < 0:
        raise ValueError("User partitioning requires a user_id column")
    indices = [
        index for index, user_id in enumerate(batch.column(user_index).to_pylist())
        if int(user_id) % modulus == remainder
    ]
    if not indices:
        return []
    selected = pa.Table.from_batches([batch]).take(
        pa.array(indices, type=pa.int32())
    )
    return selected.to_pylist()


def _parquet_rows(
    path: Path,
    columns: tuple[str, ...],
    *,
    batch_size: int,
    user_modulus: int = 1,
    user_remainder: int = 0,
) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    missing = set(columns) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
        yield from _selected_user_rows(batch, user_modulus, user_remainder)


def _split_start(path: Path, *, batch_size: int) -> datetime:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if "impression_time" not in parquet.schema_arrow.names:
        raise ValueError(f"{path} lacks impression_time")
    earliest: datetime | None = None
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["impression_time"]):
        value = pc.min(batch.column(0)).as_py()
        if value is not None and (earliest is None or value < earliest):
            earliest = value
    if earliest is None:
        raise ValueError(f"No impression times in {path}")
    return earliest


def _output_schema():
    import pyarrow as pa

    return pa.schema([
        ("split", pa.string()), ("impression_id", pa.uint32()), ("user_id", pa.uint32()),
        ("candidate_article_id", pa.int32()), ("current_page_article_id", pa.int32()),
        ("impression_time", pa.timestamp("us")), ("clicked", pa.int8()),
        ("device_type", pa.string()), ("is_sso_user", pa.bool_()),
        ("is_subscriber", pa.bool_()), ("is_front_page", pa.bool_()),
        ("hour", pa.int8()), ("day_of_week", pa.int8()), ("slate_size", pa.int16()),
        ("has_history", pa.bool_()), ("history_length", pa.int32()),
        ("history_length_log1p", pa.float32()), ("median_read_time", pa.float32()),
        ("median_read_time_log1p", pa.float32()), ("mean_scroll_percentage", pa.float32()),
        ("history_clicks_7d", pa.int32()), ("history_clicks_7d_log1p", pa.float32()),
        ("category", pa.string()), ("article_type", pa.string()), ("premium", pa.bool_()),
        ("article_age_h", pa.float32()), ("article_age_log1p", pa.float32()),
        ("article_age_missing", pa.bool_()), ("is_24h_eligible", pa.bool_()),
        ("slate_relative_age_h", pa.float32()), ("category_affinity_72h", pa.float32()),
        ("category_affinity_missing", pa.bool_()),
        ("last5_mean_topic_jaccard", pa.float32()), ("topic_affinity_missing", pa.bool_()),
        ("has_any_topic_overlap", pa.bool_()),
        ("same_category_mean_log1p_read_time", pa.float32()),
        ("same_category_read_depth_missing", pa.bool_()),
        ("premium_subscriber", pa.string()),
    ])


def _validate_onehot_state(state: Mapping[str, Any]) -> dict[str, list[str]]:
    if state.get("version") != ONEHOT_STATE_VERSION:
        raise ValueError("Unsupported one-hot state version")
    if state.get("columns") != list(CATEGORICAL_COLUMNS):
        raise ValueError("One-hot state has different categorical columns")
    vocabulary = state.get("vocabulary")
    if not isinstance(vocabulary, dict):
        raise ValueError("One-hot state lacks a vocabulary")
    for column in CATEGORICAL_COLUMNS:
        values = vocabulary.get(column)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"Invalid one-hot vocabulary for {column}")
    return vocabulary


def onehot_column_names(
    state: Mapping[str, Any], *, columns: Sequence[str] | None = None
) -> tuple[str, ...]:
    """Stable numeric column names for all or selected categorical fields."""
    vocabulary = _validate_onehot_state(state)
    selected = tuple(CATEGORICAL_COLUMNS if columns is None else columns)
    if len(selected) != len(set(selected)) or any(
        column not in CATEGORICAL_COLUMNS for column in selected
    ):
        raise ValueError("One-hot columns must be distinct known categorical fields")
    names: list[str] = []
    for column in selected:
        names.extend(
            f"ohe_{column}_{index:03d}" for index in range(len(vocabulary[column]))
        )
        names.extend((f"ohe_{column}_missing", f"ohe_{column}_unknown"))
    return tuple(names)


def encoded_baseline_columns(
    state: Mapping[str, Any], *,
    include_engineered: bool = False,
    log_transform: bool = False,
) -> tuple[str, ...]:
    """Choose trainable columns; keep generated Parquet files as a reusable superset.

    The default uses all EDA-retained current features. ``include_engineered``
    adds the report's engineered features. ``log_transform`` substitutes the
    available log1p variant for each selected raw count, duration, or age.
    The same-category read-depth feature is inherently defined as a mean of
    log1p times, so that feature does not change with ``log_transform``.
    """
    selected = ORIGINAL_FEATURE_COLUMNS + (
        ENGINEERED_FEATURE_COLUMNS if include_engineered else ()
    )
    numeric_and_boolean = tuple(
        LOG1P_FEATURE_VARIANTS.get(column, column) if log_transform else column
        for column in selected if column not in CATEGORICAL_COLUMNS
    )
    categorical = tuple(
        column for column in selected if column in CATEGORICAL_COLUMNS
    )
    return numeric_and_boolean + onehot_column_names(state, columns=categorical)


def fit_onehot_state(
    train_feature_paths: str | Path | Sequence[str | Path],
    state_path: str | Path,
    *,
    min_frequency: int = 1,
    batch_size: int = 100_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit categorical vocabularies from training candidate exposures only."""
    import pyarrow.parquet as pq

    if min_frequency < 1 or batch_size < 1:
        raise ValueError("min_frequency and batch_size must be positive")
    paths = (
        [Path(train_feature_paths)]
        if isinstance(train_feature_paths, (str, Path))
        else [Path(path) for path in train_feature_paths]
    )
    if not paths:
        raise ValueError("At least one training feature file is required")
    state_path = Path(state_path)
    if state_path.exists() and not overwrite:
        raise FileExistsError(f"One-hot state exists: {state_path}")

    counts = {column: Counter() for column in CATEGORICAL_COLUMNS}
    row_count = 0
    selected_columns = ["split", *CATEGORICAL_COLUMNS]
    for path in paths:
        parquet = pq.ParquetFile(path)
        missing = set(selected_columns) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=selected_columns
        ):
            splits = batch.column(0).to_pylist()
            if any(value != "train" for value in splits):
                raise ValueError(f"One-hot vocabulary must be fit on train rows: {path}")
            row_count += len(splits)
            for index, column in enumerate(CATEGORICAL_COLUMNS, start=1):
                counts[column].update(
                    value for value in batch.column(index).to_pylist()
                    if value is not None
                )
    if row_count == 0:
        raise ValueError("No training rows were found for one-hot fitting")

    state: dict[str, Any] = {
        "version": ONEHOT_STATE_VERSION,
        "columns": list(CATEGORICAL_COLUMNS),
        "min_frequency": min_frequency,
        "training_rows": row_count,
        "vocabulary": {
            column: sorted(
                value for value, count in counts[column].items()
                if count >= min_frequency
            )
            for column in CATEGORICAL_COLUMNS
        },
    }
    onehot_column_names(state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{state_path.stem}.",
        suffix=".json", dir=state_path.parent, delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            json.dump(state, temporary, indent=2, ensure_ascii=False)
            temporary.write("\n")
        os.replace(temporary_path, state_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return state


def load_onehot_state(state_path: str | Path) -> dict[str, Any]:
    with Path(state_path).open(encoding="utf-8") as source:
        state = json.load(source)
    _validate_onehot_state(state)
    return state


def encode_feature_file(
    input_path: str | Path,
    state_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 10_000,
    overwrite: bool = False,
) -> dict[str, int]:
    """Replace the four nominal fields with train-fitted one-hot columns."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Encoded output must have a different path from raw features")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Encoded output exists: {output_path}")
    state = load_onehot_state(state_path)
    vocabulary = _validate_onehot_state(state)
    parquet = pq.ParquetFile(input_path)
    raw_schema = parquet.schema_arrow
    missing = set(CATEGORICAL_COLUMNS) - set(raw_schema.names)
    if missing:
        raise ValueError(f"{input_path} lacks categorical columns: {sorted(missing)}")
    preserved_fields = [
        field for field in raw_schema if field.name not in CATEGORICAL_COLUMNS
    ]
    encoded_names = onehot_column_names(state)
    output_schema = pa.schema([
        *preserved_fields,
        *(pa.field(name, pa.int8(), nullable=False) for name in encoded_names),
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.", suffix=".parquet", dir=output_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    row_count = 0
    try:
        with pq.ParquetWriter(
            temporary_path, output_schema, compression="snappy"
        ) as writer:
            for batch in parquet.iter_batches(batch_size=batch_size):
                arrays = [
                    batch.column(batch.schema.get_field_index(field.name))
                    for field in preserved_fields
                ]
                for column in CATEGORICAL_COLUMNS:
                    values = batch.column(batch.schema.get_field_index(column))
                    missing_mask = pc.is_null(values)
                    known_mask = (
                        pc.fill_null(
                            pc.is_in(
                                values,
                                value_set=pa.array(
                                    vocabulary[column], type=pa.string()
                                ),
                            ),
                            False,
                        )
                        if vocabulary[column]
                        else pa.array([False] * batch.num_rows, type=pa.bool_())
                    )
                    for category in vocabulary[column]:
                        match = pc.fill_null(
                            pc.equal(values, pa.scalar(category)), False
                        )
                        arrays.append(pc.cast(match, pa.int8()))
                    arrays.append(pc.cast(missing_mask, pa.int8()))
                    unknown_mask = pc.and_(
                        pc.invert(missing_mask), pc.invert(known_mask)
                    )
                    arrays.append(pc.cast(unknown_mask, pa.int8()))
                writer.write_table(pa.Table.from_arrays(
                    arrays, schema=output_schema
                ))
                row_count += batch.num_rows
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return {"rows": row_count, "onehot_columns": len(encoded_names)}


def build_feature_file(
    behaviors_path: str | Path,
    history_path: str | Path,
    articles_path: str | Path,
    output_path: str | Path,
    *,
    split: str,
    label_mode: LabelMode = "auto",
    batch_size: int = 1_000,
    user_modulus: int = 1,
    user_remainder: int = 0,
    max_impressions: int | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Stream one split to Parquet without exploding all candidates in memory."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _check_user_partition(user_modulus, user_remainder)
    if label_mode not in ("auto", "required", "none"):
        raise ValueError("label_mode must be auto, required or none")
    if batch_size < 1 or (max_impressions is not None and max_impressions < 1):
        raise ValueError("batch_size and max_impressions must be positive")
    behaviors_path, history_path, articles_path = map(
        Path, (behaviors_path, history_path, articles_path)
    )
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}; pass overwrite=True to replace it")

    articles = load_articles(_parquet_rows(articles_path, ARTICLE_COLUMNS, batch_size=batch_size))
    split_start = _split_start(behaviors_path, batch_size=batch_size)
    profiles = prepare_history_profiles(
        _parquet_rows(
            history_path, HISTORY_COLUMNS, batch_size=batch_size,
            user_modulus=user_modulus, user_remainder=user_remainder,
        ),
        articles,
        split_start=split_start,
        user_modulus=user_modulus,
        user_remainder=user_remainder,
    )

    behavior_file = pq.ParquetFile(behaviors_path)
    available = set(behavior_file.schema_arrow.names)
    if not set(BEHAVIOR_REQUIRED) <= available:
        raise ValueError(f"{behaviors_path} lacks required behavior columns")
    if label_mode == "required" and "article_ids_clicked" not in available:
        raise ValueError(f"{behaviors_path} has no click labels")
    columns = [column for column in (*BEHAVIOR_REQUIRED, *BEHAVIOR_OPTIONAL) if column in available]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.", suffix=".parquet", dir=output_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    counts = {"impressions": 0, "candidates": 0, "users_with_history": len(profiles)}
    try:
        with pq.ParquetWriter(temporary_path, _output_schema(), compression="snappy") as writer:
            for batch in behavior_file.iter_batches(batch_size=batch_size, columns=columns):
                selected = _selected_user_rows(batch, user_modulus, user_remainder)
                if max_impressions is not None:
                    remaining = max_impressions - counts["impressions"]
                    selected = selected[:remaining]
                if not selected:
                    if max_impressions is not None and counts["impressions"] >= max_impressions:
                        break
                    continue
                feature_rows = list(iter_feature_rows(
                    selected, articles, profiles, split=split, label_mode=label_mode
                ))
                writer.write_table(pa.Table.from_pylist(feature_rows, schema=_output_schema()))
                counts["impressions"] += len(selected)
                counts["candidates"] += len(feature_rows)
                if max_impressions is not None and counts["impressions"] >= max_impressions:
                    break
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return counts


def build_model_feature_file(
    behaviors_path: str | Path,
    history_path: str | Path,
    articles_path: str | Path,
    output_path: str | Path,
    *,
    split: str,
    onehot_state_path: str | Path,
    fit_onehot: bool = False,
    onehot_min_frequency: int = 1,
    label_mode: LabelMode = "auto",
    batch_size: int = 1_000,
    user_modulus: int = 1,
    user_remainder: int = 0,
    max_impressions: int | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Write one encoded Parquet file; discard the temporary raw feature file."""
    output_path = Path(output_path)
    onehot_state_path = Path(onehot_state_path)
    if output_path.resolve() == onehot_state_path.resolve():
        raise ValueError("Output and one-hot state must have different paths")
    if fit_onehot and split != "train":
        raise ValueError("Fit the one-hot state on the train split only")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}")
    if fit_onehot and onehot_state_path.exists() and not overwrite:
        raise FileExistsError(f"One-hot state exists: {onehot_state_path}")
    if not fit_onehot and not onehot_state_path.is_file():
        raise FileNotFoundError(
            f"Fit the train one-hot state before encoding {split}: {onehot_state_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".raw_features_", dir=output_path.parent) as temporary:
        raw_path = Path(temporary) / "features.parquet"
        counts = build_feature_file(
            behaviors_path, history_path, articles_path, raw_path,
            split=split, label_mode=label_mode, batch_size=batch_size,
            user_modulus=user_modulus, user_remainder=user_remainder,
            max_impressions=max_impressions,
        )
        if fit_onehot:
            state = fit_onehot_state(
                raw_path, onehot_state_path,
                min_frequency=onehot_min_frequency, overwrite=overwrite,
            )
            counts["onehot_training_rows"] = state["training_rows"]
        encoded = encode_feature_file(
            raw_path, onehot_state_path, output_path, overwrite=overwrite,
        )
    return {**counts, **encoded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behaviors", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--articles", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--label-mode", choices=("auto", "required", "none"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--user-modulus", type=int, default=1)
    parser.add_argument("--user-remainder", type=int, default=0)
    parser.add_argument("--max-impressions", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--onehot-state", type=Path)
    parser.add_argument("--fit-onehot", action="store_true")
    parser.add_argument("--onehot-min-frequency", type=int, default=1)
    parser.add_argument("--encoded-output", type=Path)
    parser.add_argument("--encoded-only", action="store_true",
                        help="Write only the final encoded Parquet at --output")
    args = parser.parse_args()
    if args.fit_onehot and args.split != "train":
        parser.error("--fit-onehot is only for the train split")
    if (args.fit_onehot or args.encoded_output or args.encoded_only) and args.onehot_state is None:
        parser.error("--onehot-state is required for fitting or encoding")
    if args.encoded_only and args.encoded_output is not None:
        parser.error("--encoded-only and --encoded-output cannot be combined")
    if args.encoded_only:
        counts = build_model_feature_file(
            args.behaviors, args.history, args.articles, args.output,
            split=args.split, onehot_state_path=args.onehot_state,
            fit_onehot=args.fit_onehot,
            onehot_min_frequency=args.onehot_min_frequency,
            label_mode=args.label_mode, batch_size=args.batch_size,
            user_modulus=args.user_modulus, user_remainder=args.user_remainder,
            max_impressions=args.max_impressions, overwrite=args.overwrite,
        )
        print(json.dumps({"output": str(args.output), **counts}, indent=2))
        return
    if args.encoded_output is not None:
        if args.encoded_output.resolve() == args.output.resolve():
            parser.error("--encoded-output must differ from --output")
        if args.encoded_output.exists() and not args.overwrite:
            parser.error("Encoded output exists; pass --overwrite to replace it")
    if args.fit_onehot and args.onehot_state.exists() and not args.overwrite:
        parser.error("One-hot state exists; pass --overwrite to replace it")
    if args.encoded_output and not args.fit_onehot and not args.onehot_state.exists():
        parser.error("Fit the train one-hot state before encoding validation or test")

    counts = build_feature_file(
        args.behaviors, args.history, args.articles, args.output,
        split=args.split, label_mode=args.label_mode, batch_size=args.batch_size,
        user_modulus=args.user_modulus, user_remainder=args.user_remainder,
        max_impressions=args.max_impressions, overwrite=args.overwrite,
    )
    result: dict[str, Any] = {"output": str(args.output), **counts}
    if args.fit_onehot:
        state = fit_onehot_state(
            args.output, args.onehot_state,
            min_frequency=args.onehot_min_frequency, overwrite=args.overwrite,
        )
        result["onehot_state"] = str(args.onehot_state)
        result["onehot_training_rows"] = state["training_rows"]
    if args.encoded_output is not None:
        encoded = encode_feature_file(
            args.output, args.onehot_state, args.encoded_output,
            overwrite=args.overwrite,
        )
        result["encoded_output"] = str(args.encoded_output)
        result.update(encoded)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
