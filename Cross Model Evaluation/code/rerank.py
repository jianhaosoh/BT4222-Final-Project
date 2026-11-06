"""Stage 2 freshness/diversity reranker for schema-v4 catalogue rankings.

The optional continuity term consumes externally estimated candidate-level
probabilities. This module does not infer continuation labels from unclicked
articles or train a continuation model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .evaluate import (
        _behavior_id, _normalize_behaviour_ids, _topic_set,
        catalogue_fingerprints, checked_int64_series, file_provenance,
        load_articles, load_rankings, read_table, table_columns, utc_times,
    )
    from .similarity import article_similarity
except ImportError:  # Allow `python rerank.py ...` from this directory.
    from evaluate import (
        _behavior_id, _normalize_behaviour_ids, _topic_set,
        catalogue_fingerprints, checked_int64_series, file_provenance,
        load_articles, load_rankings, read_table, table_columns, utc_times,
    )
    from similarity import article_similarity


def _category_key(value: Any) -> str | None:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _request_table(labels_path: Path, split: str, naive_timezone: str) -> pd.DataFrame:
    wanted = ["user_id", "impression_id", "impression_time"]
    if "split" in table_columns(labels_path):
        frame = read_table(labels_path, [*wanted, "split"])
        observed = set(frame["split"].dropna().astype(str).unique())
        if observed and observed != {split}:
            frame = frame.loc[frame["split"].astype(str).eq(split)].copy()
        if frame.empty:
            raise ValueError(f"No rows for split {split!r} in {labels_path}")
    else:
        frame = read_table(labels_path, wanted)
    missing = set(wanted) - set(frame)
    if missing:
        raise ValueError(f"Labels are missing columns: {sorted(missing)}")
    frame["user_id"] = checked_int64_series(frame.user_id, "labels user_id")
    frame["impression_id"] = checked_int64_series(frame.impression_id, "labels impression_id")
    frame["impression_time"] = utc_times(
        frame.impression_time, naive_timezone=naive_timezone, errors="raise"
    )
    if frame.impression_time.isna().any():
        raise ValueError("Request impression_time contains missing values")
    frame["behaviour_id"] = [
        _behavior_id(user, impression)
        for user, impression in zip(frame.user_id, frame.impression_id)
    ]
    request = frame.groupby("behaviour_id", sort=False, as_index=False).agg(
        user_id=("user_id", "first"), impression_id=("impression_id", "first"),
        impression_time=("impression_time", "first"),
        time_count=("impression_time", "nunique"),
    )
    if request.time_count.ne(1).any():
        raise ValueError("impression_time varies within a request")
    return request.drop(columns="time_count")


def estimate_similarity_constants(
    labels_path: Path,
    articles_path: Path,
    *,
    output_path: Path,
    split: str = "train",
    sample_pairs: int = 100_000,
    seed: int = 42,
    naive_timezone: str = "UTC",
) -> dict[str, Any]:
    """Estimate missing-topic fallback similarities from eligible training pools.

    Requests are sampled uniformly. For each sampled request, two distinct
    eligible articles are drawn uniformly without replacement. Repeated
    (request, unordered article pair) draws are discarded.
    """
    if sample_pairs < 1:
        raise ValueError("sample_pairs must be positive")
    requests = _request_table(labels_path, split, naive_timezone)
    articles = load_articles(articles_path, naive_timezone=naive_timezone)
    known = articles.loc[articles.published_time.notna()].copy()
    known = known.sort_values("published_time", kind="mergesort")
    pub_ns = known.published_time.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    article_ids = known.article_id.to_numpy(dtype=np.int64)
    topics = {int(a): _topic_set(t) for a, t in zip(known.article_id, known.get("topics", pd.Series([None] * len(known), index=known.index)))}
    categories = {int(a): _category_key(c) for a, c in zip(known.article_id, known.get("category", pd.Series([None] * len(known), index=known.index)))}
    request_ns = requests.impression_time.map(lambda t: pd.Timestamp(t).as_unit("ns").value).to_numpy(dtype=np.int64)
    eligible_counts = np.searchsorted(pub_ns, request_ns, side="right")
    eligible_request_indices = np.flatnonzero(eligible_counts >= 2)
    if not len(eligible_request_indices):
        raise ValueError("No training requests have at least two eligible articles")

    rng = np.random.default_rng(seed)
    seen_pairs: set[tuple[int, int, int]] = set()
    pair_counts = {"sampled_pairs": 0, "both_have_topics": 0,
                   "topic_pairs_same_category": 0, "topic_pairs_different_category": 0,
                   "topic_pairs_category_missing": 0}
    same_values: list[float] = []
    diff_values: list[float] = []
    all_values: list[float] = []
    attempts = 0
    max_attempts = sample_pairs * 20
    while pair_counts["sampled_pairs"] < sample_pairs and attempts < max_attempts:
        attempts += 1
        req_idx = int(rng.choice(eligible_request_indices))
        pool_n = int(eligible_counts[req_idx])
        i = int(rng.integers(pool_n))
        j = int(rng.integers(pool_n - 1))
        if j >= i:
            j += 1
        aid_a, aid_b = int(article_ids[i]), int(article_ids[j])
        lo, hi = sorted((aid_a, aid_b))
        pair_key = (req_idx, lo, hi)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        pair_counts["sampled_pairs"] += 1
        topics_a, topics_b = topics[aid_a], topics[aid_b]
        if not topics_a or not topics_b:
            continue
        similarity = len(topics_a & topics_b) / len(topics_a | topics_b)
        all_values.append(similarity)
        pair_counts["both_have_topics"] += 1
        cat_a, cat_b = categories[aid_a], categories[aid_b]
        if cat_a is None or cat_b is None:
            pair_counts["topic_pairs_category_missing"] += 1
        elif cat_a == cat_b:
            same_values.append(similarity)
            pair_counts["topic_pairs_same_category"] += 1
        else:
            diff_values.append(similarity)
            pair_counts["topic_pairs_different_category"] += 1
    if pair_counts["sampled_pairs"] < sample_pairs:
        raise RuntimeError(
            f"Could draw only {pair_counts['sampled_pairs']} unique request/article pairs "
            f"after {attempts} attempts (requested {sample_pairs})"
        )
    if not all_values:
        raise ValueError("Sample contained no pairs with topics on both articles; cannot estimate fallback")
    s_all = float(np.mean(all_values))
    constants = {
        "version": "shared_article_similarity_v1",
        "s_same": float(np.mean(same_values)) if same_values else s_all,
        "s_diff": float(np.mean(diff_values)) if diff_values else s_all,
        "s_all": s_all,
        "sampling": "uniform request; two distinct eligible articles sampled uniformly without replacement within each pair; repeated request/pair draws discarded",
        "split": split,
        "seed": int(seed),
        "sample_pairs_requested": int(sample_pairs),
        "sample_pairs_used": int(pair_counts["sampled_pairs"]),
        "pair_counts": pair_counts,
        "empty_estimate_fallback": {
            "s_same": "s_all" if not same_values else None,
            "s_diff": "s_all" if not diff_values else None,
        },
        "naive_timezone": naive_timezone,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(constants, indent=2) + "\n", encoding="utf-8")
    return constants


def load_continuity_scores(path: Path, ranked_ids: dict[str, list[int]]) -> dict[tuple[str, int], float]:
    frame = read_table(path, ["behaviour_id", "article_id", "continuation_score"])
    required = {"behaviour_id", "article_id", "continuation_score"}
    if required - set(frame):
        raise ValueError(f"Continuity score file needs columns {sorted(required)}")
    frame["behaviour_id"] = _normalize_behaviour_ids(frame.behaviour_id)
    frame["article_id"] = checked_int64_series(frame.article_id, "continuity article_id")
    frame["continuation_score"] = pd.to_numeric(frame.continuation_score, errors="raise")
    values = frame.continuation_score.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("continuation_score must be finite and in [0, 1]")
    if frame.duplicated(["behaviour_id", "article_id"]).any():
        raise ValueError("Continuity scores contain duplicate request/article pairs")
    wanted = {(bid, aid) for bid, ids in ranked_ids.items() for aid in ids}
    received = set(zip(frame.behaviour_id.astype(str), frame.article_id.astype(int)))
    missing = wanted - received
    if missing:
        raise ValueError(f"Continuity scores are missing {len(missing)} ranked request/article pairs")
    return dict(zip(zip(frame.behaviour_id.astype(str), frame.article_id.astype(int)), values))


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".parquet", ".pq"}:
        raise ValueError("Output must end in .csv, .parquet, or .pq")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        if suffix == ".csv":
            serial = frame.copy()
            serial["article_ids"] = serial.article_ids.map(lambda ids: json.dumps(ids, separators=(",", ":")))
            serial.to_csv(temp, index=False)
        else:
            frame.to_parquet(temp, index=False)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def rerank(
    rankings: pd.DataFrame,
    requests: pd.DataFrame,
    articles: pd.DataFrame,
    constants: dict[str, Any],
    *,
    freshness_grace_h: float = 0.0,
    freshness_half_life_h: float = 24.0,
    freshness_weight: float = 0.0,
    continuity_weight: float = 0.0,
    diversity_lambda: float = 0.8,
    output_k: int = 5,
    continuity_scores: dict[tuple[str, int], float] | None = None,
) -> pd.DataFrame:
    numeric_parameters = [freshness_grace_h, freshness_half_life_h,
                          freshness_weight, continuity_weight, diversity_lambda]
    if not np.isfinite(np.asarray(numeric_parameters, dtype=float)).all():
        raise ValueError("Reranking parameters must be finite")
    if not 0 <= diversity_lambda <= 1:
        raise ValueError("diversity_lambda must be in [0, 1]")
    if min(freshness_grace_h, freshness_weight, continuity_weight) < 0:
        raise ValueError("Freshness and continuity values/weights must be non-negative")
    if freshness_half_life_h <= 0 or output_k < 1:
        raise ValueError("freshness_half_life_h and output_k must be positive")
    if continuity_weight > 0 and continuity_scores is None:
        raise ValueError("continuity_weight > 0 requires validated candidate-level continuation scores")
    if not {"s_same", "s_diff", "s_all"}.issubset(constants):
        raise ValueError("Similarity constants need s_same, s_diff, and s_all")
    if any(not math.isfinite(float(constants[k])) or not 0 <= float(constants[k]) <= 1
           for k in ("s_same", "s_diff", "s_all")):
        raise ValueError("Similarity constants must be finite values in [0, 1]")

    request_map = requests.set_index("behaviour_id")
    if set(rankings.behaviour_id) != set(request_map.index):
        raise ValueError("Ranking request IDs must exactly match the supplied label requests")
    known_pub = articles.published_time.dropna().sort_values().astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    topic_sets = {int(a): _topic_set(t) for a, t in zip(articles.article_id, articles.get("topics", pd.Series([None] * len(articles), index=articles.index)))}
    categories = {int(a): _category_key(c) for a, c in zip(articles.article_id, articles.get("category", pd.Series([None] * len(articles), index=articles.index)))}
    ranking_map = dict(zip(rankings.behaviour_id.astype(str), rankings.article_ids))
    out_rows = []
    for behaviour_id, ids in ranking_map.items():
        row = request_map.loc[behaviour_id]
        imp_time = pd.Timestamp(row.impression_time)
        imp_ns = imp_time.as_unit("ns").value
        eligible_count = int(np.searchsorted(known_pub, imp_ns, side="right"))
        expected_input_length = min(200, eligible_count)
        if len(ids) != expected_input_length:
            raise ValueError(
                f"{behaviour_id}: got {len(ids)} Stage 1 articles; expected {expected_input_length} "
                "from the 200-item eligible-catalogue export"
            )
        if not ids:
            out_rows.append({"behaviour_id": behaviour_id, "article_ids": []})
            continue
        ranked_meta = articles.reindex(ids)
        if ranked_meta.article_id.isna().any():
            raise ValueError(f"{behaviour_id}: Stage 1 list contains IDs absent from the catalogue")
        pub = ranked_meta.published_time
        if pub.isna().any() or (pub > imp_time).any():
            raise ValueError(f"{behaviour_id}: Stage 1 list contains an ineligible article")

        n = len(ids)
        rank_relevance = np.ones(1, dtype=float) if n == 1 else 1.0 - np.arange(n, dtype=float) / (n - 1)
        age_h = ((imp_time - pub).dt.total_seconds().to_numpy(dtype=float) / 3600.0)
        fresh = np.exp2(-np.maximum(0.0, age_h - freshness_grace_h) / freshness_half_life_h)
        cont = np.zeros(n, dtype=float)
        if continuity_weight > 0:
            cont = np.asarray([continuity_scores[(behaviour_id, int(aid))] for aid in ids], dtype=float)
        base = rank_relevance * (1.0 + continuity_weight * cont) + freshness_weight * fresh

        remaining = list(range(n))
        chosen: list[int] = []
        while remaining and len(chosen) < output_k:
            best_index = None
            best_score = -float("inf")
            for candidate_index in remaining:
                penalty = 0.0
                if chosen:
                    penalty = max(
                        article_similarity(
                            topic_sets[int(ids[candidate_index])],
                            topic_sets[int(ids[selected])],
                            categories[int(ids[candidate_index])],
                            categories[int(ids[selected])], constants,
                        )
                        for selected in chosen
                    )
                score = diversity_lambda * base[candidate_index] - (1.0 - diversity_lambda) * penalty
                # remaining follows Stage 1 rank, so exact score ties keep the better base rank.
                if score > best_score:
                    best_score = score
                    best_index = candidate_index
            assert best_index is not None
            chosen.append(best_index)
            remaining.remove(best_index)
        out_rows.append({"behaviour_id": behaviour_id,
                         "article_ids": [int(ids[i]) for i in chosen]})
    return pd.DataFrame(out_rows, columns=["behaviour_id", "article_ids"])


def _fit_main() -> None:
    parser = argparse.ArgumentParser(description="Estimate diversity fallback constants from training pools")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--articles", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-pairs", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--naive-timezone", default="UTC")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = estimate_similarity_constants(
        args.labels, args.articles, output_path=args.output, split=args.split,
        sample_pairs=args.sample_pairs, seed=args.seed, naive_timezone=args.naive_timezone,
    )
    print(json.dumps(result, indent=2))


def _rerank_main() -> None:
    parser = argparse.ArgumentParser(description="Rerank Stage 1 top-200 lists to final top-K")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--articles", required=True, type=Path)
    parser.add_argument("--rankings", required=True, type=Path)
    parser.add_argument("--similarity-constants", required=True, type=Path)
    parser.add_argument("--continuity-scores", type=Path,
                        help="Optional validated probabilities: behaviour_id, article_id, continuation_score")
    parser.add_argument("--split", choices=("validation", "test", "train"), default="validation")
    parser.add_argument("--naive-timezone", default="UTC")
    parser.add_argument("--a0", type=float, default=0.0)
    parser.add_argument("--half-life-h", type=float, default=24.0)
    parser.add_argument("--freshness-weight", type=float, default=0.0)
    parser.add_argument("--continuity-weight", type=float, default=0.0)
    parser.add_argument("--lambda-mmr", type=float, default=1.0)
    parser.add_argument("--output-k", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    requests = _request_table(args.labels, args.split, args.naive_timezone)
    rankings = load_rankings(args.rankings)
    articles = load_articles(args.articles, naive_timezone=args.naive_timezone)
    constants = json.loads(args.similarity_constants.read_text(encoding="utf-8"))
    ranking_map = dict(zip(rankings.behaviour_id.astype(str), rankings.article_ids))
    continuity = None
    if args.continuity_scores is not None:
        continuity = load_continuity_scores(args.continuity_scores, ranking_map)
    output = rerank(
        rankings, requests, articles, constants,
        freshness_grace_h=args.a0, freshness_half_life_h=args.half_life_h,
        freshness_weight=args.freshness_weight, continuity_weight=args.continuity_weight,
        diversity_lambda=args.lambda_mmr, output_k=args.output_k,
        continuity_scores=continuity,
    )
    _write_table(output, args.output)
    catalogue_id_hash, catalogue_content_hash = catalogue_fingerprints(articles)
    request_hash = hashlib.sha256(
        "\n".join(sorted(output.behaviour_id.astype(str))).encode("utf-8")
    ).hexdigest()
    constants_hash = hashlib.sha256(json.dumps(
        constants, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest = {
        "protocol": "stage2_reranking_v1",
        "split": args.split,
        "ranking_schema": "behaviour_id,article_ids ordered list",
        "request_count": int(len(output)),
        "output_k": args.output_k,
        "request_id_fingerprint_sha256": request_hash,
        "catalogue_id_fingerprint_sha256": catalogue_id_hash,
        "catalogue_content_fingerprint_sha256": catalogue_content_hash,
        "similarity_constants_sha256": constants_hash,
        "similarity_constants": constants,
        "parameters": {
            "freshness_grace_h": args.a0,
            "freshness_half_life_h": args.half_life_h,
            "freshness_weight": args.freshness_weight,
            "continuity_weight": args.continuity_weight,
            "diversity_lambda": args.lambda_mmr,
        },
        "continuity_enabled": bool(args.continuity_weight > 0),
        "inputs": {
            "labels": file_provenance(args.labels),
            "articles": file_provenance(args.articles),
            "rankings": file_provenance(args.rankings),
            "similarity_constants": file_provenance(args.similarity_constants),
            "continuity_scores": file_provenance(args.continuity_scores) if args.continuity_scores else None,
        },
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({
        "split": args.split,
        "requests": int(len(output)),
        "output_k": args.output_k,
        "continuity_enabled": bool(args.continuity_weight > 0),
        "output": str(args.output),
        "manifest": str(manifest_path),
        "parameters": {"a0_h": args.a0, "half_life_h": args.half_life_h,
                       "freshness_weight": args.freshness_weight,
                       "continuity_weight": args.continuity_weight,
                       "lambda_mmr": args.lambda_mmr},
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit-similarity", "rerank"))
    args, remaining = parser.parse_known_args()
    import sys
    sys.argv = [sys.argv[0], *remaining]
    if args.mode == "fit-similarity":
        _fit_main()
    else:
        _rerank_main()


if __name__ == "__main__":
    main()
