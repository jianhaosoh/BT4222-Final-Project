"""Recorded-slate and catalogue-shortlist metrics (Experiments/README.md protocol)."""
from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_DEFINITIONS = {
    "known_click_recall@100": "PRIMARY (Stage 1 selection). Catalogue pool: for each request with known clicks, "
                              "|original clicked IDs in the exported top-100| / |original clicked IDs|; equal-weight "
                              "mean over those requests. Clicked articles rejected by eligibility stay in the "
                              "denominator; an empty eligible pool scores 0. Membership uses the export order "
                              "(descending score, ties by ascending article_id). Recovery of observed clicks only.",
    "observed_click_ndcg@100": "SECONDARY. Gain 1 for an article in the request's original known-click set, 0 observed "
                               "gain otherwise (bookkeeping, NOT an observed non-click label); DCG@100 = sum gain/log2(rank+1) "
                               "over the exported shortlist; ideal DCG places ALL original known clicks first, capped "
                               "at 100 (missed/excluded clicks count against it); same requests and macro-average as Recall@100.",
    "eligible_only_recall@100": "Recall@100 with the denominator restricted to eligible clicked articles; requests "
                                "without an eligible click excluded (reported separately from the primary metric).",
    "mean_impression_auc": "DIAGNOSTIC only. Recorded slates: Mann-Whitney AUC of clicked vs shown-unclicked candidates "
                           "per (user_id, impression_id), ties half credit, single-class impressions excluded, "
                           "macro-averaged. Not used for Stage 1 selection.",
    "mean_impression_ndcg@5": "DIAGNOSTIC only on recorded slates (expected credit over ties). nDCG@5 as an evaluation "
                              "metric belongs to Stage 2's final five; it is not a Stage 1 objective.",
}

for _name in ("known_click_recall", "observed_click_ndcg", "eligible_only_recall"):
    METRIC_DEFINITIONS[f"{_name}@200"] = METRIC_DEFINITIONS[f"{_name}@100"].replace("100", "200").replace(
        "PRIMARY (Stage 1 selection)", "EXPORT diagnostic (selection remains Recall@100)").replace(
        "SECONDARY.", "EXPORT diagnostic (selection tie-break remains nDCG@100).")


def request_retrieval(ranked_ids, clicked_ids, k: int = 100) -> tuple[int, int, float, float]:
    """(n_known_clicks, n_in_top_k, DCG@k, ideal DCG@k) for one request.

    `ranked_ids` must already be in the export order. The ideal places all
    original known clicks first (capped at k), so clicks that are ineligible,
    missing or discarded by the shortlist count against the metric."""
    clicked = set(int(c) for c in clicked_ids)
    n = len(clicked)
    hits, dcg = 0, 0.0
    for rank, article in enumerate(list(ranked_ids)[:k], start=1):
        if int(article) in clicked:
            hits += 1
            dcg += 1.0 / np.log2(rank + 1)
    ideal = float(sum(1.0 / np.log2(r + 1) for r in range(1, min(n, k) + 1)))
    return n, hits, dcg, ideal


def impression_metrics(group: np.ndarray, labels: np.ndarray, scores: np.ndarray, k: int = 5) -> dict:
    """group: contiguous or not, integer impression codes. Labels 0/1. Ties fair."""
    frame = pd.DataFrame({"g": np.asarray(group), "y": np.asarray(labels, dtype=np.float64),
                          "s": np.asarray(scores, dtype=np.float64)})
    if not np.isfinite(frame.s).all():
        raise ValueError("non-finite scores")
    codes, uniq = pd.factorize(frame.g, sort=True)
    frame["g"] = codes
    n_groups = len(uniq)
    sizes = np.bincount(codes, minlength=n_groups)
    pos = np.bincount(codes, weights=frame.y.to_numpy(), minlength=n_groups)
    neg = sizes - pos
    ranks = frame.groupby("g")["s"].rank(method="average").to_numpy()
    rank_sum = np.bincount(codes, weights=ranks * frame.y.to_numpy(), minlength=n_groups)
    auc_ok = (pos > 0) & (neg > 0)
    auc = (rank_sum - pos * (pos + 1) / 2)[auc_ok] / (pos[auc_ok] * neg[auc_ok])

    # Expected DCG under uniformly random order within tied-score blocks.
    frame["tie_mean"] = frame.groupby(["g", "s"])["y"].transform("mean")
    ranked = frame.sort_values(["g", "s"], ascending=[True, False], kind="stable")
    gcodes = ranked.g.to_numpy()
    position = ranked.groupby("g").cumcount().to_numpy()  # 0-based
    # For a tie block occupying positions [a, b), each member's expected discount
    # is the mean discount over those positions limited to the cutoff.
    block = ranked.groupby(["g", "s"], sort=False)
    start = block.cumcount().to_numpy()
    block_size = block["y"].transform("size").to_numpy()
    first = position - start
    cutoff = np.minimum(k, sizes[gcodes])
    disc = 1.0 / np.log2(np.arange(2, int(sizes.max()) + 3))
    cum = np.concatenate([[0.0], np.cumsum(disc)])
    lo = first
    hi = np.minimum(first + block_size, cutoff)
    expected_discount = np.where(hi > lo, (cum[np.maximum(hi, lo)] - cum[lo]) / block_size, 0.0)
    gains = ranked.y.to_numpy() * expected_discount  # per item: own label x expected discount of its block
    # Using own label x block-mean discount equals block-mean label x per-position discount summed.
    dcg = np.bincount(gcodes, weights=gains, minlength=n_groups)
    ideal_n = np.minimum(pos, np.minimum(k, sizes)).astype(int)
    idcg = cum[ideal_n]
    ndcg_ok = pos > 0
    ndcg = dcg[ndcg_ok] / idcg[ndcg_ok]
    return {
        "mean_impression_auc": float(auc.mean()) if auc.size else None,
        f"mean_impression_ndcg@{k}": float(ndcg.mean()) if ndcg.size else None,
        "candidate_rows": int(len(frame)), "impressions": int(n_groups),
        "auc_eligible_impressions": int(auc_ok.sum()),
        "auc_excluded_single_class": int((~auc_ok).sum()),
        "ndcg_eligible_impressions": int(ndcg_ok.sum()),
        "ndcg_excluded_no_positive": int((~ndcg_ok).sum()),
    }


def recall_summary(manifest: pd.DataFrame, k: int = 100) -> dict:
    """Macro catalogue metrics from a request manifest (one row per request)."""
    m = manifest[manifest.n_clicked > 0]
    out = {"requests_total": int(len(manifest)), "requests_with_known_clicks": int(len(m)),
           "requests_without_known_clicks": int(len(manifest) - len(m))}
    if not len(m):
        return out
    hits = m[f"n_clicked_in_top_{k}"] if f"n_clicked_in_top_{k}" in m else m.n_clicked_in_shortlist
    recall = hits / m.n_clicked
    ndcg = np.where(m[f"ideal_dcg_at_{k}"] > 0, m[f"observed_dcg_at_{k}"] / m[f"ideal_dcg_at_{k}"], 0.0)
    el = m[m.n_clicked_eligible > 0]
    out.update({
        f"known_click_recall@{k}": float(recall.mean()),
        f"observed_click_ndcg@{k}": float(np.mean(ndcg)),
        f"eligible_only_recall@{k}": float(((el[f"n_clicked_in_top_{k}"] if f"n_clicked_in_top_{k}" in el else el.n_clicked_in_shortlist) / el.n_clicked_eligible).mean()) if len(el) else None,
        "requests_with_no_eligible_click": int((m.n_clicked_eligible == 0).sum()),
        "empty_pool_requests_with_known_clicks": int((m.eligible_count == 0).sum()),
        "requests_with_ties_crossing_cutoff": int((m[f"tie_crosses_cutoff_{k}"] if f"tie_crosses_cutoff_{k}" in m else m.tie_crosses_cutoff).sum()),
        "clicked_articles_total": int(m.n_clicked.sum()),
        "clicked_articles_eligible": int(m.n_clicked_eligible.sum()),
        "clicked_excluded_not_in_catalogue": int(m.n_clicked_not_in_catalogue.sum()),
        "clicked_excluded_missing_published_time": int(m.n_clicked_missing_pub.sum()),
        "clicked_excluded_published_after_request": int(m.n_clicked_future.sum()),
        "eligibility_coverage_of_clicks": float(m.n_clicked_eligible.sum() / m.n_clicked.sum()),
        "clicked_articles_in_shortlist": int(hits.sum()),
    })
    return out


def better_selection(candidate: dict, best: dict | None) -> bool:
    """Selection order: higher Recall@100, then higher observed-click nDCG@100;
    exact ties keep the earlier checkpoint/trial (returns False)."""
    if best is None:
        return True
    c = (candidate["known_click_recall@100"], candidate["observed_click_ndcg@100"])
    b = (best["known_click_recall@100"], best["observed_click_ndcg@100"])
    return c > b
