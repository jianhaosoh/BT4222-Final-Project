"""Validators for v3 exports: shortlist vs full catalogue scores/eligibility,
request coverage, IDs, empty pools, probability semantics and tie rules."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .catalogue import SHORTLIST_SIZE
from .features import NO_PUB


def _fail(errors, msg):
    errors.append(msg)


def eligible_ids(ctx, art, pos: int) -> np.ndarray:
    cat = ctx.catalogue_idx
    pub = art.tensors["pub_s"].numpy()[cat]
    t = int(ctx.tensors["req_s"][pos])
    ok = (pub != NO_PUB) & (pub <= t)
    return art.ids[cat[ok]]  # ascending article_id


def expected_top(ids: np.ndarray, scores: np.ndarray, k: int = SHORTLIST_SIZE) -> list[int]:
    """Reference ranking: descending score, ties by ascending article_id."""
    order = np.lexsort((ids, -scores.astype(np.float64)))
    return ids[order][:k].tolist()


def read_scores_bytes(data: bytes, header: bool) -> pd.DataFrame:
    """Parse score CSV bytes with pyarrow (correctly rounded floats on every pandas version)."""
    import pyarrow as pa
    import pyarrow.csv as pacsv

    names = ["behaviour_id", "article_id", "score", "click_probability"]
    table = pacsv.read_csv(io.BytesIO(data),
                           read_options=pacsv.ReadOptions(column_names=None if header else names,
                                                          autogenerate_column_names=False),
                           convert_options=pacsv.ConvertOptions(column_types={
                               "behaviour_id": pa.string(), "article_id": pa.int64(),
                               "score": pa.float64(), "click_probability": pa.float64()}))
    frame = table.to_pandas()
    # Shortest-repr float32 text round-trips exactly through float64 -> float32.
    frame["score"] = frame.score.astype(np.float32)
    return frame


def read_block(path: Path, start: int, end: int) -> pd.DataFrame:
    with open(path, "rb") as handle:
        handle.seek(start)
        data = handle.read(end - start)
    if not data:
        return pd.DataFrame({"behaviour_id": pd.Series(dtype=str), "article_id": pd.Series(dtype=np.int64),
                             "score": pd.Series(dtype=np.float32), "click_probability": pd.Series(dtype=np.float64)})
    return read_scores_bytes(data, header=False)


def validate_export(ctx, art, out_root: Path, request_positions: np.ndarray, *, full_scores: bool,
                    check_blocks: int | None = None, seed: int = 0, recorded_logits: np.ndarray | None = None,
                    score_tolerance: float = 2e-3) -> dict:
    """Validate one split's export under out_root.

    Rankings/manifest are checked for every exported request. If full scores were
    written, `check_blocks` sampled request blocks (None = all) are re-read via the
    byte index and compared exactly with the shortlist and eligibility set.
    `recorded_logits` (in ctx.recorded order) enables the recorded-vs-catalogue
    score consistency check for eligible recorded candidates in checked blocks."""
    split = ctx.split
    out_root = Path(out_root)
    errors, checks = [], {}
    request_positions = np.asarray(request_positions)
    rankings = pd.read_csv(out_root / "rankings" / split / "submission_top_100.csv", dtype={"behaviour_id": str})
    manifest = pd.read_csv(out_root / "diagnostics" / "catalogue" / f"{split}_requests.csv", dtype={"behaviour_id": str})
    expected_bids = ctx.requests.behaviour_id.to_numpy()[request_positions]
    checks["ranking_columns"] = list(rankings.columns)
    if list(rankings.columns) != ["behaviour_id", "article_ids"]:
        _fail(errors, f"ranking columns {list(rankings.columns)}")
    if rankings.behaviour_id.duplicated().any():
        _fail(errors, "duplicate behaviour_id in rankings")
    if not np.array_equal(rankings.behaviour_id.to_numpy(), expected_bids):
        _fail(errors, "rankings do not cover exactly the requested behaviours in order")
    # behaviour_id built from original integer IDs
    ref = (ctx.requests.user_id.astype(str) + ":" + ctx.requests.impression_id.astype(str)).to_numpy()[request_positions]
    if not np.array_equal(ref, expected_bids):
        _fail(errors, "behaviour_id is not f'{user_id}:{impression_id}'")
    lists = rankings.article_ids.map(json.loads)  # never eval
    lengths = lists.map(len).to_numpy()
    counts = manifest.eligible_count.to_numpy()
    if not np.array_equal(lengths, np.minimum(counts, SHORTLIST_SIZE)):
        _fail(errors, "shortlist length != min(100, eligible_count)")
    if lists.map(lambda v: len(set(v)) != len(v)).any():
        _fail(errors, "duplicate IDs within a shortlist")
    # Eligibility of every shortlisted ID (future-article exclusion) for every request
    cat_ids = art.ids[ctx.catalogue_idx]
    pub_of = dict(zip(cat_ids.tolist(), art.tensors["pub_s"].numpy()[ctx.catalogue_idx].tolist()))
    req_s = ctx.tensors["req_s"].numpy()
    future, not_in_cat, missing_pub = 0, 0, 0
    for pos, ids in zip(request_positions, lists):
        for a in ids:
            p = pub_of.get(a)
            if p is None:
                not_in_cat += 1
            elif p == NO_PUB:
                missing_pub += 1
            elif p > req_s[pos]:
                future += 1
    checks.update(shortlisted_future_articles=future, shortlisted_not_in_catalogue=not_in_cat,
                  shortlisted_missing_pub=missing_pub, empty_pools=int((counts == 0).sum()),
                  empty_pool_rows_present=int(((counts == 0) & (lengths == 0)).sum()),
                  requests=int(len(rankings)))
    if future or not_in_cat or missing_pub:
        _fail(errors, "shortlist contains ineligible articles")
    # Eligible counts recomputed independently for a sample
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(request_positions), size=min(200, len(request_positions)), replace=False) if len(request_positions) else []
    bad_counts = sum(len(eligible_ids(ctx, art, request_positions[j])) != counts[j] for j in sample)
    checks["eligible_count_recheck_sample"] = int(len(sample))
    if bad_counts:
        _fail(errors, f"{bad_counts} sampled eligible counts disagree with recomputation")

    if full_scores:
        scores_path = out_root / "predictions" / split / "submission_click_probabilities.csv"
        with open(scores_path, encoding="utf-8") as handle:
            header = handle.readline().strip()
        checks["score_header"] = header
        if header != "behaviour_id,article_id,score,click_probability":
            _fail(errors, f"score header {header}")
        blocks = pd.read_csv(out_root / "diagnostics" / "catalogue" / f"{split}_score_blocks.csv")
        if int(blocks.rows.sum()) != int(counts.sum()):
            _fail(errors, "score row count != sum of eligible counts")
        if blocks.byte_end.max() != scores_path.stat().st_size:
            _fail(errors, "score file size disagrees with block index (truncated/partial file?)")
        chosen = blocks.index if check_blocks is None else rng.choice(blocks.index, size=min(check_blocks, len(blocks)), replace=False)
        rec = ctx.recorded
        n_checked_req, n_rows, max_prob_err, rec_diffs = 0, 0, 0.0, []
        for b in sorted(chosen):
            row = blocks.loc[b]
            frame = read_block(scores_path, int(row.byte_start), int(row.byte_end))
            n_rows += len(frame)
            if len(frame) != row.rows:
                _fail(errors, f"block {b}: {len(frame)} rows, index says {row.rows}")
            if not np.isfinite(frame.score).all():
                _fail(errors, f"block {b}: non-finite score")
            p = frame.click_probability.to_numpy()
            if ((p < 0) | (p > 1)).any():
                _fail(errors, f"block {b}: probability outside [0,1]")
            sig = 1 / (1 + np.exp(-frame.score.to_numpy(np.float64)))
            max_prob_err = max(max_prob_err, float(np.abs(sig - p).max()) if len(p) else 0.0)
            if frame.duplicated(["behaviour_id", "article_id"]).any():
                _fail(errors, f"block {b}: duplicate (behaviour_id, article_id)")
            groups = dict(tuple(frame.groupby("behaviour_id", sort=False)))
            for j in range(int(row.first_request_index), int(row.last_request_index) + 1):
                pos = request_positions[j]
                bid = expected_bids[j]
                g = groups.get(bid)
                ids_expected = eligible_ids(ctx, art, pos)
                ids_got = g.article_id.to_numpy() if g is not None else np.zeros(0, np.int64)
                if not np.array_equal(ids_got, ids_expected):
                    _fail(errors, f"{bid}: scored IDs != eligible catalogue (order or membership)")
                    continue
                top = expected_top(ids_got, g.score.to_numpy() if g is not None else np.zeros(0))
                if top != lists.iloc[j]:
                    _fail(errors, f"{bid}: shortlist != top-{SHORTLIST_SIZE} of full scores with tie rule")
                n_checked_req += 1
                if recorded_logits is not None and rec and g is not None:
                    lo, hi = rec["offsets"][pos], rec["offsets"][pos + 1]
                    s = pd.Series(g.score.to_numpy(), index=g.article_id.to_numpy())
                    ids_r = rec["article_id"][lo:hi]
                    present = np.isin(ids_r, s.index)
                    if present.any():
                        rec_diffs.append(np.abs(s.loc[ids_r[present]].to_numpy() - recorded_logits[lo:hi][present]).max())
        checks.update(score_blocks_checked=int(len(chosen)), score_blocks_total=int(len(blocks)),
                      requests_checked_against_full_scores=n_checked_req, score_rows_checked=n_rows,
                      max_abs_sigmoid_vs_probability=max_prob_err,
                      recorded_vs_catalogue_max_abs_logit_diff=float(max(rec_diffs)) if rec_diffs else None)
        if max_prob_err > 1e-6:
            _fail(errors, "click_probability != sigmoid(score)")
        if rec_diffs and max(rec_diffs) > score_tolerance:
            _fail(errors, "recorded-slate scores disagree with catalogue scores for eligible recorded candidates")
    return {"split": split, "passed": not errors, "errors": errors[:50], "n_errors": len(errors), "checks": checks}


def validate_recorded(path: Path, ctx) -> dict:
    frame = read_scores_bytes(Path(path).read_bytes(), header=True)
    errors = []
    rec = ctx.recorded
    expected = pd.DataFrame({"behaviour_id": ctx.requests.behaviour_id.to_numpy()[rec["req_pos"]],
                             "article_id": rec["article_id"]})
    if list(frame.columns) != ["behaviour_id", "article_id", "score", "click_probability"]:
        errors.append(f"columns {list(frame.columns)}")
    if frame.duplicated(["behaviour_id", "article_id"]).any():
        errors.append("duplicate recorded candidates")
    merged = expected.merge(frame, on=["behaviour_id", "article_id"], how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        errors.append("recorded candidates missing or extra")
    if not np.isfinite(frame.score).all():
        errors.append("non-finite scores")
    if ((frame.click_probability < 0) | (frame.click_probability > 1)).any():
        errors.append("probability outside [0,1]")
    return {"split": ctx.split, "passed": not errors, "errors": errors, "rows": int(len(frame)),
            "expected_rows": int(len(expected))}
