"""Bounded validators for v4 top-200 Parquet and separate recorded slates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .catalogue import EXPORT_K, SCORE_SCHEMA, RANKING_SCHEMA
from .features import NO_PUB


def eligible_ids(ctx, art, pos: int) -> np.ndarray:
    cat = ctx.catalogue_idx
    pub = art.tensors["pub_s"].numpy()[cat]
    t = int(ctx.tensors["req_s"][pos])
    return art.ids[cat[(pub != NO_PUB) & (pub <= t)]]


def expected_top(ids: np.ndarray, scores: np.ndarray, k: int = EXPORT_K) -> list[int]:
    """Independent reference: descending scores, ascending original ID for ties."""
    return ids[np.lexsort((ids, -scores.astype(np.float64)))][:k].tolist()


def validate_export(ctx, art, out_root: Path, request_positions: np.ndarray) -> dict:
    """Read one request block at a time; check all retained pairs and empty pools.

    Running top-K is checked against each score-chunk merge by the exporter before
    discarding scores. This reader checks the persisted schemas, order, coverage,
    eligibility, pair/ranking agreement and probability semantics without another
    full-catalogue scoring pass.
    """
    root, split = Path(out_root), ctx.split
    scores = pq.ParquetFile(root / "predictions" / split / "submission_article_scores.parquet")
    ranks = pq.ParquetFile(root / "rankings" / split / "submission_top_200.parquet")
    manifests = pq.ParquetFile(root / "diagnostics" / "catalogue" / f"{split}_requests.parquet")
    errors = []
    def fail(msg):
        if len(errors) < 50:
            errors.append(msg)
    if not scores.schema_arrow.equals(SCORE_SCHEMA):
        fail("score Parquet schema differs from v4")
    if not ranks.schema_arrow.equals(RANKING_SCHEMA):
        fail("rankings must have string behaviour_id and native list<int64> article_ids")
    if len({scores.num_row_groups, ranks.num_row_groups, manifests.num_row_groups}) != 1:
        fail("score, ranking and request-manifest row groups disagree")
        return {"split": split, "passed": False, "errors": errors, "n_errors": len(errors), "checks": {}}
    positions = np.asarray(request_positions, dtype=np.int64)
    expected = ctx.requests.behaviour_id.to_numpy()[positions]
    if not pd.Index(expected).is_unique:
        fail("duplicate requested behaviour_id")
    cat = ctx.catalogue_idx
    cat_ids = art.ids[cat]
    pub = art.tensors["pub_s"].numpy()[cat]
    pub_sorted = np.sort(pub[pub != NO_PUB])
    pub_of = dict(zip(map(int, cat_ids), map(int, pub)))
    offset, checked_pairs, max_prob_error = 0, 0, 0.0
    for group in range(ranks.num_row_groups):
        sf = scores.read_row_group(group).to_pandas()
        rf = ranks.read_row_group(group).to_pandas()
        mf = manifests.read_row_group(group).to_pandas()
        n = len(rf)
        bids = expected[offset:offset + n]
        if not np.array_equal(rf.behaviour_id.to_numpy(), bids) or not np.array_equal(mf.behaviour_id.to_numpy(), bids):
            fail("ranking or manifest request coverage/order differs from declared scope")
        if sf.duplicated(["behaviour_id", "article_id"]).any():
            fail("duplicate retained article key")
        if not np.isfinite(sf.score).all():
            fail("non-finite relevance score")
        prob = sf.click_probability.to_numpy()
        if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
            fail("invalid click_probability")
        sigmoid = 0.5 * (1 + np.tanh(sf.score.to_numpy(np.float64) / 2))
        max_prob_error = max(max_prob_error, float(np.abs(sigmoid - prob).max()) if len(prob) else 0.0)
        score_groups = dict(tuple(sf.groupby("behaviour_id", sort=False)))
        for j, row in rf.iterrows():
            pos = positions[offset + j]
            req = ctx.requests.iloc[pos]
            if str(row.behaviour_id) != f"{int(req.user_id)}:{int(req.impression_id)}":
                fail("behaviour_id does not preserve original user/impression IDs")
            ids = list(map(int, row.article_ids))
            count = int(np.searchsorted(pub_sorted, int(ctx.tensors["req_s"][pos]), side="right"))
            if len(ids) != min(EXPORT_K, count) or int(mf.eligible_count.iloc[j]) != count:
                fail("shortlist length or eligible count differs from full catalogue")
            if len(ids) != len(set(ids)):
                fail("duplicate IDs in shortlist")
            if any(a not in pub_of or pub_of[a] == NO_PUB or pub_of[a] > int(ctx.tensors["req_s"][pos]) for a in ids):
                fail("shortlist contains ineligible article")
            g = score_groups.pop(str(row.behaviour_id), None)
            if not ids:
                if g is not None and len(g):
                    fail("empty-pool ranking has score rows")
            elif g is None or ids != g.article_id.tolist():
                fail("ranking and retained score article IDs disagree")
            else:
                if g["rank"].tolist() != list(range(1, len(ids) + 1)):
                    fail("nonconsecutive ranks")
                if ids != expected_top(g.article_id.to_numpy(), g.score.to_numpy()):
                    fail("score-descending / article-ID-ascending order differs")
        if score_groups:
            fail("score rows contain an unexpected request")
        checked_pairs += len(sf)
        offset += n
    if offset != len(positions):
        fail("missing or extra request rows")
    if max_prob_error > 1e-6:
        fail("click_probability is not sigmoid(logit)")
    return {"split": split, "passed": not errors, "errors": errors, "n_errors": len(errors),
            "checks": {"requests": offset, "retained_score_rows": checked_pairs,
                       "all_retained_pairs_checked": True, "max_abs_sigmoid_vs_probability": max_prob_error,
                       "topk_checked_during_scoring": True}}


def validate_recorded(path: Path, ctx) -> dict:
    """Check every original candidate in source order, without materialising the whole export."""
    errors, offset = [], 0
    rec = ctx.recorded
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000):
        frame = batch.to_pandas()
        hi = offset + len(frame)
        expected_bids = ctx.requests.behaviour_id.to_numpy()[rec["req_pos"][offset:hi]]
        if not np.array_equal(frame.behaviour_id, expected_bids) or not np.array_equal(frame.article_id, rec["article_id"][offset:hi]):
            errors.append("recorded slate source IDs/order differ")
        if "label" in frame or "clicked" in frame:
            errors.append("labels must be joined separately")
        if not np.isfinite(frame.score).all():
            errors.append("non-finite score")
        if not np.isfinite(frame.click_probability).all() or not frame.click_probability.between(0, 1).all():
            errors.append("invalid probability")
        offset = hi
    if offset != len(rec["req_pos"]):
        errors.append("missing or extra recorded candidate rows")
    return {"split": ctx.split, "passed": not errors, "errors": errors[:50], "rows": offset,
            "expected_rows": len(rec["req_pos"])}
