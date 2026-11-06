"""Fast validation-sample search for freshness/diversity reranker weights.

The module tunes only freshness_weight and diversity_lambda. Freshness grace,
half-life, output K, and the candidate model rankings remain fixed. It searches
a deterministic user-level sample, applies a relevance guard, then confirms
the selected setting once on the full validation split.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from . import rerank as rerank_module
    from .evaluate import evaluate_rankings, load_articles, load_labels, load_rankings
except ImportError:  # Support importing from Evaluation/code on Colab.
    import rerank as rerank_module
    from evaluate import evaluate_rankings, load_articles, load_labels, load_rankings


RELEVANCE_METRICS = ("known_click_recall@5", "observed_click_ndcg@5")
FRESHNESS_METRIC = "age_le_24h_share@5"
DIVERSITY_METRIC = "shared_intra_list_diversity@5"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "model"


def _load_model_runs(model_runs: Sequence[dict[str, Any]]) -> list[tuple[str, Path, pd.DataFrame]]:
    if not model_runs:
        raise ValueError("Provide at least one model ranking to optimize against")
    loaded = []
    names = set()
    request_scope = None
    for run in model_runs:
        name = str(run["name"])
        path = Path(run.get("rankings", run.get("path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"{name} ranking file not found: {path}")
        if name.lower() in names:
            raise ValueError(f"Duplicate model name: {name}")
        names.add(name.lower())
        rankings = rerank_module.load_rankings(path)
        if rankings.behaviour_id.duplicated().any():
            raise ValueError(f"{name} ranking file has duplicate behaviour_id rows")
        ids = set(rankings.behaviour_id.astype(str))
        if request_scope is None:
            request_scope = ids
        elif ids != request_scope:
            raise ValueError("All optimization model rankings must contain identical request IDs")
        loaded.append((name, path, rankings))
    return loaded


def _top_k_prefix(rankings: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = []
    for behaviour_id, article_ids in zip(rankings.behaviour_id.astype(str), rankings.article_ids):
        if len(article_ids) < k:
            raise ValueError(f"{behaviour_id}: ranking has fewer than {k} articles")
        rows.append({"behaviour_id": behaviour_id, "article_ids": [int(a) for a in article_ids[:k]]})
    return pd.DataFrame(rows, columns=["behaviour_id", "article_ids"])


def _metric_summary(
    request_frame: pd.DataFrame,
    click_map: dict[str, set[int]],
    rankings: pd.DataFrame,
    articles: pd.DataFrame,
    constants: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    # Suppress the evaluator's per-1000-request progress lines during the grid.
    with contextlib.redirect_stdout(io.StringIO()):
        summary, per_request, _ = evaluate_rankings(
            request_frame,
            click_map,
            rankings,
            articles,
            export_k=5,
            ks=(5,),
            allow_subset=False,
            include_top_items=False,
            similarity_constants=constants,
        )
    return summary, per_request


def _install_similarity_cache(constants: dict[str, Any], maxsize: int = 500_000):
    """Cache article-pair similarities across parameter trials; return restore callback."""
    original = rerank_module.article_similarity

    @lru_cache(maxsize=maxsize)
    def cached_similarity(topics_a: tuple[str, ...], category_a: str | None,
                          topics_b: tuple[str, ...], category_b: str | None) -> float:
        return original(set(topics_a), set(topics_b), category_a, category_b, constants)

    def normalized_category(value: Any) -> str | None:
        return None if value is None else str(value)

    def wrapper(topics_a, topics_b, category_a, category_b, _constants):
        ta = tuple(sorted(topics_a))
        tb = tuple(sorted(topics_b))
        ca = normalized_category(category_a)
        cb = normalized_category(category_b)
        key_a = (ta, (ca is not None, ca or ""))
        key_b = (tb, (cb is not None, cb or ""))
        if key_b < key_a:
            ta, tb, ca, cb = tb, ta, cb, ca
        return cached_similarity(ta, ca, tb, cb)

    rerank_module.article_similarity = wrapper
    return lambda: setattr(rerank_module, "article_similarity", original)


def _sample_by_user(requests: pd.DataFrame, *, fraction: float, seed: int) -> tuple[pd.DataFrame, set[str]]:
    if not 0 < fraction <= 1:
        raise ValueError("sample_user_fraction must be in (0, 1]")
    users = np.sort(requests.user_id.unique())
    n_users = max(1, int(math.ceil(len(users) * fraction)))
    rng = np.random.default_rng(seed)
    selected_users = set(rng.choice(users, size=n_users, replace=False).tolist())
    sample = requests.loc[requests.user_id.isin(selected_users)].copy()
    return sample, set(sample.behaviour_id.astype(str))


def _allowed_drop(base: float, *, relative_limit: float, absolute_limit: float) -> float:
    """Use the stricter of the relative and absolute relevance-loss caps."""
    return min(max(0.0, abs(base) * relative_limit), absolute_limit)


def _normalized_gain(value: float, base: float) -> float:
    headroom = max(0.0, 1.0 - base)
    return (value - base) / headroom if headroom > 1e-12 else 0.0


def optimize_reranker_weights(
    *,
    validation_labels_path: Path,
    articles_path: Path,
    similarity_constants_path: Path,
    model_runs: Sequence[dict[str, Any]],
    output_dir: Path,
    freshness_weights: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75),
    diversity_lambdas: Sequence[float] = (0.6, 0.8, 0.9, 1.0),
    freshness_grace_h: float = 0.0,
    freshness_half_life_h: float = 24.0,
    output_k: int = 5,
    sample_user_fraction: float = 0.10,
    sample_seed: int = 42,
    max_relative_recall5_drop: float = 0.10,
    max_absolute_recall5_drop: float = 0.02,
    max_relative_ndcg5_drop: float = 0.10,
    max_absolute_ndcg5_drop: float = 0.02,
    freshness_objective_weight: float = 0.5,
    diversity_objective_weight: float = 0.5,
) -> dict[str, Any]:
    """Search a small grid on sampled users and confirm the selected weights.

    Selection: settings must stay within both Recall@5 and nDCG@5 relevance
    guards for every configured model. Among feasible settings, maximize the
    explicit weighted average of headroom-normalized freshness and diversity
    gains. If no non-no-op setting is feasible with positive gain, select the
    no-op and mark that no reranker setting passed the objective.
    """
    if output_k != 5:
        raise ValueError("This optimizer is defined for a final top-five list")
    if freshness_half_life_h <= 0 or freshness_grace_h < 0:
        raise ValueError("Freshness half-life must be positive and grace non-negative")
    if not np.isclose(freshness_objective_weight + diversity_objective_weight, 1.0):
        raise ValueError("Freshness and diversity objective weights must sum to 1")
    if min(freshness_objective_weight, diversity_objective_weight) < 0:
        raise ValueError("Objective weights must be non-negative")
    if min(max_relative_recall5_drop, max_absolute_recall5_drop,
           max_relative_ndcg5_drop, max_absolute_ndcg5_drop) < 0:
        raise ValueError("Relevance loss limits must be non-negative")
    if not freshness_weights or not diversity_lambdas:
        raise ValueError("Both parameter grids must be non-empty")
    if any(float(w) < 0 or not np.isfinite(float(w)) for w in freshness_weights):
        raise ValueError("Freshness weights must be finite and non-negative")
    if any(not 0 <= float(lam) <= 1 or not np.isfinite(float(lam)) for lam in diversity_lambdas):
        raise ValueError("Diversity lambdas must be finite and in [0, 1]")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_frame, click_map = load_labels(Path(validation_labels_path), naive_timezone="UTC")
    if request_frame.behaviour_id.astype(str).duplicated().any():
        raise ValueError("Validation labels must have one request row per behaviour_id")
    articles = load_articles(Path(articles_path), naive_timezone="UTC")
    constants = json.loads(Path(similarity_constants_path).read_text(encoding="utf-8"))
    loaded_runs = _load_model_runs(model_runs)
    sample_requests, sample_ids = _sample_by_user(
        request_frame, fraction=sample_user_fraction, seed=sample_seed
    )
    sample_clicks = {key: click_map[key] for key in sample_ids}
    sample_rankings_by_model = {
        name: frame.loc[frame.behaviour_id.astype(str).isin(sample_ids)].copy()
        for name, _, frame in loaded_runs
    }
    sample_n_users = int(sample_requests.user_id.nunique())

    # Ensure the exact no-op control is included even if omitted from the grid.
    configs = {(float(w), float(lam)) for w in freshness_weights for lam in diversity_lambdas}
    configs.add((0.0, 1.0))
    configs = sorted(configs)
    if (0.0, 1.0) not in configs:
        raise AssertionError("Internal error: no-op control missing")

    restore_similarity = _install_similarity_cache(constants)
    trial_records: list[dict[str, Any]] = []
    summaries_by_config: dict[tuple[float, float], dict[str, dict[str, Any]]] = {}
    try:
        for trial_number, (freshness_weight, diversity_lambda) in enumerate(configs, start=1):
            per_model = {}
            for name, _, full_rankings in loaded_runs:
                rankings = sample_rankings_by_model[name]
                reranked = rerank_module.rerank(
                    rankings,
                    sample_requests,
                    articles,
                    constants,
                    freshness_grace_h=freshness_grace_h,
                    freshness_half_life_h=freshness_half_life_h,
                    freshness_weight=freshness_weight,
                    continuity_weight=0.0,
                    diversity_lambda=diversity_lambda,
                    output_k=output_k,
                )
                summary, _ = _metric_summary(
                    sample_requests, sample_clicks, reranked, articles, constants
                )
                per_model[name] = summary
            summaries_by_config[(freshness_weight, diversity_lambda)] = per_model
            record = {
                "freshness_weight": freshness_weight,
                "diversity_lambda": diversity_lambda,
                "sample_requests": int(len(sample_requests)),
                "sample_users": sample_n_users,
            }
            for name, summary in per_model.items():
                prefix = _slug(name) + "_"
                for metric in (*RELEVANCE_METRICS, FRESHNESS_METRIC, DIVERSITY_METRIC):
                    record[prefix + metric] = summary.get(metric)
            trial_records.append(record)
            print(f"Tuned config {trial_number}/{len(configs)}: freshness_weight={freshness_weight}, lambda={diversity_lambda}")
    finally:
        restore_similarity()

    # Sampled no-op metrics are each model's comparison baseline.
    baseline = summaries_by_config[(0.0, 1.0)]
    feasible_records = []
    for record in trial_records:
        key = (record["freshness_weight"], record["diversity_lambda"])
        summaries = summaries_by_config[key]
        feasible = True
        for name, summary in summaries.items():
            base = baseline[name]
            for metric, rel_limit, abs_limit in (
                ("known_click_recall@5", max_relative_recall5_drop, max_absolute_recall5_drop),
                ("observed_click_ndcg@5", max_relative_ndcg5_drop, max_absolute_ndcg5_drop),
            ):
                base_value = base.get(metric)
                value = summary.get(metric)
                if base_value is not None and value is not None:
                    drop_limit = _allowed_drop(
                        float(base_value), relative_limit=rel_limit, absolute_limit=abs_limit
                    )
                    if float(value) < float(base_value) - drop_limit - 1e-12:
                        feasible = False
            freshness_gain = _normalized_gain(
                float(summary.get(FRESHNESS_METRIC) or 0.0),
                float(base.get(FRESHNESS_METRIC) or 0.0),
            )
            diversity_gain = _normalized_gain(
                float(summary.get(DIVERSITY_METRIC) or 0.0),
                float(base.get(DIVERSITY_METRIC) or 0.0),
            )
            record[_slug(name) + "_freshness_headroom_gain"] = freshness_gain
            record[_slug(name) + "_diversity_headroom_gain"] = diversity_gain
        record["relevance_guard_pass"] = bool(feasible)
        if feasible:
            # Average metrics across models so a model with more requests does not dominate.
            gains = []
            for name, summary in summaries.items():
                base = baseline[name]
                fresh_headroom = max(0.0, 1.0 - float(base.get(FRESHNESS_METRIC) or 0.0))
                div_headroom = max(0.0, 1.0 - float(base.get(DIVERSITY_METRIC) or 0.0))
                fresh_gain = _normalized_gain(
                    float(summary.get(FRESHNESS_METRIC) or 0.0),
                    float(base.get(FRESHNESS_METRIC) or 0.0),
                ) if fresh_headroom > 1e-12 else None
                div_gain = _normalized_gain(
                    float(summary.get(DIVERSITY_METRIC) or 0.0),
                    float(base.get(DIVERSITY_METRIC) or 0.0),
                ) if div_headroom > 1e-12 else None
                components = []
                weights = []
                if fresh_gain is not None:
                    components.append(freshness_objective_weight * fresh_gain)
                    weights.append(freshness_objective_weight)
                if div_gain is not None:
                    components.append(diversity_objective_weight * div_gain)
                    weights.append(diversity_objective_weight)
                gains.append(sum(components) / sum(weights) if weights and sum(weights) > 0 else 0.0)
            record["objective"] = float(np.mean(gains)) if gains else 0.0
            feasible_records.append(record)
        else:
            record["objective"] = np.nan

    results = pd.DataFrame(trial_records).sort_values(
        ["relevance_guard_pass", "objective", "freshness_weight", "diversity_lambda"],
        ascending=[False, False, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    results.to_csv(output_dir / "reranker_weight_search_sample.csv", index=False)

    valid_non_noop = [r for r in feasible_records
                      if (r["freshness_weight"], r["diversity_lambda"]) != (0.0, 1.0)
                      and float(r["objective"]) > 1e-12]
    if valid_non_noop:
        best_record = max(valid_non_noop, key=lambda r: r["objective"])
        no_improvement = False
    else:
        best_record = next(r for r in feasible_records
                           if (r["freshness_weight"], r["diversity_lambda"]) == (0.0, 1.0))
        no_improvement = True

    selected = {
        "freshness_weight": float(best_record["freshness_weight"]),
        "diversity_lambda": float(best_record["diversity_lambda"]),
        "freshness_grace_h": float(freshness_grace_h),
        "freshness_half_life_h": float(freshness_half_life_h),
        "continuity_weight": 0.0,
        "output_k": int(output_k),
        "tuning_sample_user_fraction": float(sample_user_fraction),
        "tuning_sample_seed": int(sample_seed),
        "no_non_noop_setting_met_objective": bool(no_improvement),
        "selection_objective": "headroom-normalized freshness and shared diversity gain under Recall@5 and nDCG@5 loss guards",
        "relevance_guards": {
            "relative_recall5_drop": float(max_relative_recall5_drop),
            "absolute_recall5_drop": float(max_absolute_recall5_drop),
            "relative_ndcg5_drop": float(max_relative_ndcg5_drop),
            "absolute_ndcg5_drop": float(max_absolute_ndcg5_drop),
        },
        "models": [name for name, _, _ in loaded_runs],
    }
    (output_dir / "selected_reranker_parameters.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )

    # One full-validation confirmation for the chosen configuration and its base top-five.
    full_records = []
    restore_similarity = _install_similarity_cache(constants)
    try:
        for name, source_path, rankings in loaded_runs:
            base_top5 = _top_k_prefix(rankings, output_k)
            base_summary, base_per = _metric_summary(
                request_frame, click_map, base_top5, articles, constants
            )
            selected_rankings = rerank_module.rerank(
                rankings,
                request_frame,
                articles,
                constants,
                freshness_grace_h=freshness_grace_h,
                freshness_half_life_h=freshness_half_life_h,
                freshness_weight=selected["freshness_weight"],
                continuity_weight=0.0,
                diversity_lambda=selected["diversity_lambda"],
                output_k=output_k,
            )
            selected_summary, selected_per = _metric_summary(
                request_frame, click_map, selected_rankings, articles, constants
            )
            for stage, summary in (("base_top5", base_summary), ("optimized_top5", selected_summary)):
                full_records.append({"model": name, "stage": stage, **summary})
    finally:
        restore_similarity()

    full_summary = pd.DataFrame(full_records)
    failures = []
    for model_name in full_summary.model.unique():
        rows = full_summary.loc[full_summary.model.eq(model_name)].set_index("stage")
        base = rows.loc["base_top5"]
        candidate = rows.loc["optimized_top5"]
        for metric, rel_limit, abs_limit in (
            ("known_click_recall@5", max_relative_recall5_drop, max_absolute_recall5_drop),
            ("observed_click_ndcg@5", max_relative_ndcg5_drop, max_absolute_ndcg5_drop),
        ):
            if pd.notna(base.get(metric)) and pd.notna(candidate.get(metric)):
                allowed = _allowed_drop(
                    float(base[metric]), relative_limit=rel_limit, absolute_limit=abs_limit
                )
                if float(candidate[metric]) < float(base[metric]) - allowed - 1e-12:
                    failures.append({
                        "model": model_name,
                        "metric": metric,
                        "base": float(base[metric]),
                        "candidate": float(candidate[metric]),
                        "allowed_drop": allowed,
                    })
    selected["full_validation_relevance_guard_pass"] = not failures
    selected["full_validation_guard_failures"] = failures
    (output_dir / "selected_reranker_parameters.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "selected_parameters": selected,
        "sample_trials": results,
        "full_validation_summary": full_summary,
        "output_dir": output_dir,
    }
