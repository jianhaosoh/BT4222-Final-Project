"""Recorded-slate and catalogue-shortlist metrics (Experiments/README.md protocol)."""
from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_DEFINITIONS = {
    "mean_impression_auc": "Per (user_id, impression_id) recorded slate: Mann-Whitney AUC of clicked vs shown-unclicked "
                           "candidates with tied scores given half credit; equal-weight mean over impressions having "
                           "both classes. Primary model-selection metric (maximise).",
    "mean_impression_ndcg@5": "Binary click gains, cutoff min(5, slate_size), expected credit over tied scores "
                              "(including ties across the cutoff); impressions without a positive excluded. Secondary.",
    "known_click_recall@100": "Catalogue pool: per request with known clicks, |clicked ∩ top-100| / |original clicked|; "
                              "equal-weight mean. Ineligible clicked articles stay in the denominator. Measures recovery "
                              "of observed clicks, not unbiased catalogue relevance/CTR.",
    "eligible_only_recall@100": "Same, with the denominator restricted to clicked articles eligible in the catalogue; "
                                "requests with no eligible click excluded.",
}


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
    """From a request manifest with n_clicked, n_clicked_eligible, n_clicked_in_shortlist."""
    m = manifest[manifest.n_clicked > 0]
    if not len(m):
        return {"requests_with_known_clicks": 0}
    recall = m.n_clicked_in_shortlist / m.n_clicked
    el = m[m.n_clicked_eligible > 0]
    return {
        f"known_click_recall@{k}": float(recall.mean()),
        f"eligible_only_recall@{k}": float((el.n_clicked_in_shortlist / el.n_clicked_eligible).mean()) if len(el) else None,
        "requests_with_known_clicks": int(len(m)),
        "requests_with_no_eligible_click": int((m.n_clicked_eligible == 0).sum()),
        "clicked_articles_total": int(m.n_clicked.sum()),
        "clicked_articles_eligible": int(m.n_clicked_eligible.sum()),
        "clicked_excluded_not_in_catalogue": int(m.n_clicked_not_in_catalogue.sum()),
        "clicked_excluded_missing_published_time": int(m.n_clicked_missing_pub.sum()),
        "clicked_excluded_published_after_request": int(m.n_clicked_future.sum()),
        "eligibility_coverage_of_clicks": float(m.n_clicked_eligible.sum() / m.n_clicked.sum()),
        "clicked_articles_in_shortlist": int(m.n_clicked_in_shortlist.sum()),
    }
