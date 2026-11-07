"""Shared article similarity definitions for reranking and evaluation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def normalize_category(value: Any) -> str | None:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def article_similarity(
    topics_a: set[str], topics_b: set[str],
    category_a: str | None, category_b: str | None,
    constants: dict[str, Any],
) -> float:
    """Jaccard for known topics; training-estimated category/global fallback otherwise."""
    if topics_a and topics_b:
        return len(topics_a & topics_b) / len(topics_a | topics_b)
    if category_a is not None and category_b is not None:
        return float(constants["s_same"] if category_a == category_b else constants["s_diff"])
    return float(constants["s_all"])


def validate_similarity_constants(constants: dict[str, Any]) -> None:
    if not {"s_same", "s_diff", "s_all"}.issubset(constants):
        raise ValueError("Similarity constants need s_same, s_diff, and s_all")
    for key in ("s_same", "s_diff", "s_all"):
        value = float(constants[key])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Similarity constant {key} must be finite and in [0, 1]")


def pairwise_shared_diversity_by_cutoff(
    topic_sets: Sequence[set[str]],
    categories: Sequence[str | None],
    cutoffs: Sequence[int],
    constants: dict[str, Any],
) -> dict[int, float]:
    """Mean pairwise distance under :func:`article_similarity` for each prefix.

    Incidence-matrix multiplication keeps the topic Jaccard calculation
    vectorized; fallback similarities fill pairs with missing topic metadata.
    """
    if len(topic_sets) != len(categories):
        raise ValueError("topic_sets and categories must have the same length")
    validate_similarity_constants(constants)
    cutoffs = sorted(set(int(k) for k in cutoffs))
    n = len(topic_sets)
    if n < 2:
        return {k: float("nan") for k in cutoffs}

    valid_topics = np.asarray([bool(value) for value in topic_sets], dtype=bool)
    similarity = np.empty((n, n), dtype=np.float64)
    category_known = np.asarray([value is not None for value in categories], dtype=bool)
    categories_array = np.asarray(["" if value is None else value for value in categories], dtype=object)
    both_categories = category_known[:, None] & category_known[None, :]
    same_category = categories_array[:, None] == categories_array[None, :]
    similarity[:] = float(constants["s_all"])
    similarity[both_categories & same_category] = float(constants["s_same"])
    similarity[both_categories & ~same_category] = float(constants["s_diff"])

    vocabulary = {topic: index for index, topic in enumerate(sorted(set().union(*[x for x in topic_sets if x])))}
    if vocabulary:
        incidence = np.zeros((n, len(vocabulary)), dtype=np.float64)
        sizes = np.zeros(n, dtype=np.float64)
        for row, topics in enumerate(topic_sets):
            if topics:
                cols = [vocabulary[topic] for topic in topics]
                incidence[row, cols] = 1.0
                sizes[row] = len(cols)
        intersections = incidence @ incidence.T
        unions = sizes[:, None] + sizes[None, :] - intersections
        known_pairs = valid_topics[:, None] & valid_topics[None, :]
        similarity[known_pairs] = intersections[known_pairs] / unions[known_pairs]

    upper_i, upper_j = np.triu_indices(n, k=1)
    distances = 1.0 - similarity[upper_i, upper_j]
    output = {}
    for cutoff in cutoffs:
        count = min(cutoff, n)
        mask = (upper_i < count) & (upper_j < count)
        output[cutoff] = float(distances[mask].mean()) if mask.any() else float("nan")
    return output
