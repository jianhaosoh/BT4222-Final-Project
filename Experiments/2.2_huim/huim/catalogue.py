"""Time-eligible catalogue scoring, top-200 shortlists and streamed v4 Parquet exports.

Eligibility (shared policy published_at_or_before_request_v1): for request
(user_id, impression_id) at time t, an article of the ordered train_val -> test
union catalogue (the same pool for every split) is eligible iff its
published_time is present and published_time <= t. Articles with a missing
publication time are excluded (cannot be verified as non-future) and counted.
No recency window, access rule or vector-coverage restriction is applied.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .features import NO_PUB
from .metrics import request_retrieval

SELECTION_K = 100
EXPORT_K = 200
SHORTLIST_SIZE = EXPORT_K
FINAL_DISPLAY_SIZE = 5
EXPORT_SCHEMA_VERSION = 4
PREDICTION_POOL = "time_eligible_catalogue"
ELIGIBILITY_RULES = [
    "policy published_at_or_before_request_v1: article belongs to the ordered union train_val/articles.parquet "
    "then test/articles.parquet (first record per article_id wins, incl. missing fields), the SAME pool for "
    "train, validation and test requests",
    "published_time is not missing (missing -> excluded, counted)",
    "published_time <= request impression_time (future publications excluded)",
    "no recency window, access, premium or embedding-coverage restriction",
]
SCORE_SEMANTICS = {
    "score": "Stage 1 logit (log-odds) from a BCE click model trained on recorded slates; higher = more relevant",
    "click_probability": "sigmoid(score); UNCALIBRATED click-probability estimate conditional on the recorded-slate "
                         "training population; not a catalogue-wide CTR; values need not sum to one per request",
}
SCORE_SCHEMA = pa.schema([
    ("behaviour_id", pa.string()), ("article_id", pa.int64()), ("rank", pa.int32()),
    ("score", pa.float32()), ("click_probability", pa.float64()),
])
RANKING_SCHEMA = pa.schema([("behaviour_id", pa.string()), ("article_ids", pa.list_(pa.int64()))])
MANIFEST_SCHEMA = pa.schema([
    ("behaviour_id", pa.string()), ("user_id", pa.int64()), ("impression_id", pa.int64()),
    ("impression_time", pa.string()), ("eligible_count", pa.int64()), ("shortlist_length", pa.int32()),
    ("n_recorded", pa.int64()), ("n_recorded_ineligible", pa.int64()),
    ("n_clicked", pa.int64()), ("n_clicked_eligible", pa.int64()), ("n_clicked_not_in_catalogue", pa.int64()),
    ("n_clicked_missing_pub", pa.int64()), ("n_clicked_future", pa.int64()),
    *[(name, typ) for k in (100, 200) for name, typ in (
        (f"n_clicked_in_top_{k}", pa.int64()), (f"observed_dcg_at_{k}", pa.float64()),
        (f"ideal_dcg_at_{k}", pa.float64()), (f"tie_crosses_cutoff_{k}", pa.bool_()))],
])


# ---------------------------------------------------------------- policy ----

def catalogue_eligibility(ctx, art) -> dict:
    """Exact per-request eligible counts via sorted publication times."""
    cat = ctx.catalogue_idx
    pub = art.tensors["pub_s"].numpy()[cat]
    ok = pub != NO_PUB
    pub_sorted = np.sort(pub[ok])
    req_s = ctx.tensors["req_s"].numpy()
    counts = np.searchsorted(pub_sorted, req_s, side="right").astype(np.int64)
    return {"eligible_count": counts, "catalogue_size": int(len(cat)),
            "catalogue_missing_published_time": int((~ok).sum()),
            "catalogue_published_after_last_request": int((pub_sorted > req_s.max()).sum()) if len(pub_sorted) else 0}


def click_eligibility(ctx, art, positions=None) -> list[dict | None]:
    """Per request: clicked IDs split into eligible / not in catalogue / missing
    pub / future. Original clicked IDs stay in the recall denominator.

    Returns a list indexed by request position; only `positions` (default: all)
    are computed, others are None. Membership uses a boolean mask over the global
    article index (O(clicks log N) per request), not a scan of the ~10^5-article pool."""
    n = len(ctx.requests)
    out: list[dict | None] = [None] * n
    in_pool = np.zeros(len(art.ids), dtype=bool)
    in_pool[ctx.catalogue_idx] = True
    in_pool[0] = False  # unknown article index
    pub = art.tensors["pub_s"].numpy()
    req_s = ctx.tensors["req_s"].numpy()
    clicked = ctx.requests["article_ids_clicked"]
    for i in (range(n) if positions is None else np.asarray(positions, dtype=np.int64)):
        ids = clicked.iloc[i]
        if ids is None:
            continue
        ids = np.unique(np.asarray(ids, dtype=np.int64))  # a clicked article counts once
        idx = art.lookup(ids)
        in_cat = in_pool[idx]
        p = pub[idx]
        missing = in_cat & (p == NO_PUB)
        future = in_cat & ~missing & (p > req_s[i])
        eligible = in_cat & ~missing & ~future
        out[i] = {"ids": ids, "eligible": eligible, "not_in_catalogue": ~in_cat, "missing": missing, "future": future}
    return out


def recorded_eligibility(ctx, art) -> pd.DataFrame:
    """Recorded candidates that are not eligible in the catalogue (reported, never dropped from slate metrics)."""
    rec = ctx.recorded
    catalogue_ids = art.ids[ctx.catalogue_idx]
    in_cat = np.isin(rec["article_id"], catalogue_ids)
    p = art.tensors["pub_s"].numpy()[rec["art_idx"]]
    t = ctx.tensors["req_s"].numpy()[rec["req_pos"]]
    reason = np.full(len(p), "", dtype=object)
    reason[~in_cat] = "not_in_catalogue"
    reason[in_cat & (p == NO_PUB)] = "missing_published_time"
    reason[in_cat & (p != NO_PUB) & (p > t)] = "published_after_request"
    bad = reason != ""
    return pd.DataFrame({"behaviour_id": ctx.requests.behaviour_id.to_numpy()[rec["req_pos"][bad]],
                         "article_id": rec["article_id"][bad], "reason": reason[bad]})


def feasibility_estimate(eligibility: dict, ctx, bytes_per_row: float | None = None,
                         pairs_per_second: float | None = None, csv_rows_per_second: float | None = None) -> dict:
    counts = eligibility["eligible_count"]
    pairs = int(counts.sum())
    if bytes_per_row is None:
        # behaviour_id + article_id(7) + float32 shortest repr (~11) + float64 prob (~19) + 3 commas + newline
        bid = float(ctx.requests.behaviour_id.str.len().mean()) if len(ctx.requests) else 15.0
        bytes_per_row = bid + 7 + 11 + 19 + 4
    est = {
        "requests": int(len(counts)), "eligible_pairs": pairs,
        "eligible_per_request": {"min": int(counts.min()) if len(counts) else 0,
                                 "median": float(np.median(counts)) if len(counts) else 0,
                                 "mean": float(counts.mean()) if len(counts) else 0,
                                 "max": int(counts.max()) if len(counts) else 0},
        "empty_pools": int((counts == 0).sum()),
        "pools_below_shortlist_size": int((counts < SHORTLIST_SIZE).sum()),
        "bytes_per_row_assumed": bytes_per_row,
        "retained_score_rows": int(np.minimum(counts, EXPORT_K).sum()),
        "retained_output_bytes_estimate": int(np.minimum(counts, EXPORT_K).sum() * 48 + len(counts) * (32 + EXPORT_K * 8)),

    }
    if pairs_per_second:
        est["scoring_hours_estimate"] = pairs / pairs_per_second / 3600
    if csv_rows_per_second:
        est["csv_write_hours_estimate"] = pairs / csv_rows_per_second / 3600
    return est


# ---------------------------------------------------------------- scoring ---

class CatalogueScorer:
    """Scores request blocks against the full split catalogue.

    Request/history states are computed once per request; article states once per
    catalogue; the (request x article) cross product is evaluated in pair chunks."""

    def __init__(self, model, ctx, art, device, pair_chunk: int = 32768, amp: bool = False):
        self.model, self.ctx, self.art, self.device = model.eval(), ctx, art, device
        self.pair_chunk, self.amp = int(pair_chunk), bool(amp) and device.type == "cuda"
        self.cat = torch.from_numpy(ctx.catalogue_idx).to(device)
        with torch.no_grad(), self._autocast():
            states = [model.article_state(art.tensors, self.cat[lo:lo + 65536]) for lo in range(0, len(self.cat), 65536)]
        self.ast = {k: torch.cat([s[k] for s in states]) for k in states[0]} if states else None
        self.pub = art.tensors["pub_s"][self.cat]
        self.pub_ok = ~art.tensors["pub_missing"][self.cat]
        self.has_text = art.tensors["has_text"][self.cat]
        self.category = art.tensors["category"][self.cat]

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp)

    @torch.no_grad()
    def logits(self, req_pos: np.ndarray) -> torch.Tensor:
        """[R, N] float32 logits for every catalogue article (ineligible included)."""
        R, N = len(req_pos), len(self.cat)
        out = torch.empty((R, N), dtype=torch.float32, device=self.device)
        if N == 0:
            return out
        rp = torch.from_numpy(np.asarray(req_pos, dtype=np.int64)).to(self.device)
        with self._autocast():
            rs = self.model.request_state(self.ctx.tensors, self.art.tensors, rp)
        nc = max(1, self.pair_chunk // max(R, 1))
        for lo in range(0, N, nc):
            hi = min(N, lo + nc)
            n = hi - lo
            rpos = torch.arange(R, device=self.device).repeat_interleave(n)
            apos = torch.arange(n, device=self.device).repeat(R)
            ast = {k: v[lo:hi] for k, v in self.ast.items()}
            with self._autocast():
                logit = self.model.score_from_states(self.ctx.tensors, self.art.tensors, len(self.art.ids),
                                                     rp[rpos], self.cat[lo:hi][apos], rs, rpos, ast, apos)
            out[:, lo:hi] = logit.float().reshape(R, n)
        if not torch.isfinite(out).all():
            raise FloatingPointError("non-finite catalogue logits")
        return out

    def eligible_mask(self, req_pos: np.ndarray) -> torch.Tensor:
        t = self.ctx.tensors["req_s"][torch.from_numpy(np.asarray(req_pos, dtype=np.int64)).to(self.device)]
        return self.pub_ok[None, :] & (self.pub[None, :] <= t[:, None])

    @torch.no_grad()
    def topk(self, req_pos: np.ndarray, k: int = EXPORT_K) -> dict:
        """Score the full pool in pair chunks and retain k+1 only.

        The extra item records a tie crossing k. Each merge sorts original IDs
        first, then scores stably, so the result is the genuine catalogue top-K
        even when a tie crosses an article-chunk boundary. No slate IDs enter it.
        """
        req_pos = np.asarray(req_pos, dtype=np.int64)
        R, N = len(req_pos), len(self.cat)
        rp = torch.from_numpy(req_pos).to(self.device)
        best_scores = torch.empty((R, 0), dtype=torch.float32, device=self.device)
        best_pos = torch.empty((R, 0), dtype=torch.int64, device=self.device)
        counts = torch.zeros(R, dtype=torch.int64, device=self.device)
        with self._autocast():
            rs = self.model.request_state(self.ctx.tensors, self.art.tensors, rp)
        nc = max(1, self.pair_chunk // max(R, 1))
        req_s = self.ctx.tensors["req_s"][rp]
        cat_ids = torch.as_tensor(self.art.ids[self.ctx.catalogue_idx], device=self.device)
        for lo in range(0, N, nc):
            hi = min(N, lo + nc)
            n = hi - lo
            rpos = torch.arange(R, device=self.device).repeat_interleave(n)
            apos = torch.arange(n, device=self.device).repeat(R)
            ast = {name: val[lo:hi] for name, val in self.ast.items()}
            with self._autocast():
                scores = self.model.score_from_states(
                    self.ctx.tensors, self.art.tensors, len(self.art.ids), rp[rpos], self.cat[lo:hi][apos],
                    rs, rpos, ast, apos).float().reshape(R, n)
            if not torch.isfinite(scores).all():
                raise FloatingPointError("non-finite catalogue logits")
            eligible = self.pub_ok[None, lo:hi] & (self.pub[None, lo:hi] <= req_s[:, None])
            counts += eligible.sum(1)
            scores = scores.masked_fill(~eligible, float("-inf"))
            positions = torch.arange(lo, hi, device=self.device)[None, :].expand(R, -1)
            merged_scores = torch.cat((best_scores, scores), 1)
            merged_pos = torch.cat((best_pos, positions), 1)
            by_id = torch.argsort(cat_ids[merged_pos], dim=1, stable=True)
            merged_scores = merged_scores.gather(1, by_id)
            merged_pos = merged_pos.gather(1, by_id)
            by_score = torch.argsort(merged_scores, dim=1, descending=True, stable=True)
            keep = by_score[:, :k + 1]
            best_scores = merged_scores.gather(1, keep)
            best_pos = merged_pos.gather(1, keep)
            # The discarded portion cannot outrank the retained prefix, checked
            # while its scores still exist (no second catalogue scoring pass).
            if by_score.shape[1] > k + 1:
                next_score = merged_scores.gather(1, by_score[:, k + 1:k + 2])[:, 0]
                if not (best_scores[:, -1] >= next_score).all():
                    raise AssertionError("running top-K merge discarded a higher score")
        def ties_at(cutoff):
            if best_scores.shape[1] <= cutoff:
                return torch.zeros(R, dtype=torch.bool, device=self.device)
            return (counts > cutoff) & (best_scores[:, cutoff - 1] == best_scores[:, cutoff])
        return {"positions": best_pos[:, :k], "scores": best_scores[:, :k],
                "lengths": counts.clamp_max(k), "eligible_count": counts,
                "tie_crosses_cutoff": ties_at(k), "selection_tie_crosses_cutoff": ties_at(SELECTION_K)}

    @staticmethod
    def rank(logits: torch.Tensor, eligible: torch.Tensor, k: int = SHORTLIST_SIZE):
        """Descending score; export ties broken by ascending article_id (= ascending
        catalogue position, via a stable sort). Returns positions [R,k], lengths and
        a flag for requests whose score at rank k equals the score at rank k+1
        (a tie crossing the cutoff, resolved by the article-ID rule)."""
        masked = torch.where(eligible, logits, torch.full_like(logits, float("-inf")))
        values, indices = torch.sort(masked, dim=1, descending=True, stable=True)
        order = indices[:, :k]
        n_elig = eligible.sum(1)
        lengths = torch.minimum(n_elig, torch.tensor(k, device=logits.device))
        if values.shape[1] > k:
            ties = (n_elig > k) & (values[:, k - 1] == values[:, k])
        else:
            ties = torch.zeros_like(n_elig, dtype=torch.bool)
        return order, lengths, ties


# ------------------------------------------------------------------ export --

def export_split(scorer: CatalogueScorer, out_root: Path, request_positions: np.ndarray, *,
                 click_info: list | None, recorded_counts: np.ndarray | None,
                 request_block: int = 16, log=print) -> dict:
    """Score full eligible pools; stream only genuine top-200 and request manifests.

    Parquet row groups bound write memory. Partial files replace final files only
    after all requests are written; this final workflow starts a fresh run folder.
    Labels never enter the recommendation or recorded-score schemas.
    """
    ctx, art = scorer.ctx, scorer.art
    split, root = ctx.split, Path(out_root)
    paths = {
        "scores": root / "predictions" / split / "submission_article_scores.parquet",
        "rankings": root / "rankings" / split / "submission_top_200.parquet",
        "manifest": root / "diagnostics" / "catalogue" / f"{split}_requests.parquet",
    }
    schemas = {"scores": SCORE_SCHEMA, "rankings": RANKING_SCHEMA, "manifest": MANIFEST_SCHEMA}
    temporary = {name: path.with_name("." + path.name + ".partial") for name, path in paths.items()}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    writers = {name: pq.ParquetWriter(temporary[name], schema, compression="zstd")
               for name, schema in schemas.items()}
    request_positions = np.asarray(request_positions, dtype=np.int64)
    cat_ids = art.ids[ctx.catalogue_idx]
    offsets = ctx.recorded.get("offsets") if ctx.recorded else None
    n_scores, n_pairs, start = 0, 0, time.time()
    try:
        for lo in range(0, len(request_positions), request_block):
            block = request_positions[lo:lo + request_block]
            top = scorer.topk(block, EXPORT_K)
            indices, values, lengths, counts = (top[name].cpu().numpy() for name in
                                               ("positions", "scores", "lengths", "eligible_count"))
            ties100 = top["selection_tie_crosses_cutoff"].cpu().numpy()
            ties200 = top["tie_crosses_cutoff"].cpu().numpy()
            score_rows, ranking_rows, manifest_rows = [], [], []
            for j, pos in enumerate(block):
                req = ctx.requests.iloc[pos]
                bid = str(req.behaviour_id)
                ids = cat_ids[indices[j, :lengths[j]]].tolist()
                scores = values[j, :lengths[j]]
                if not np.isfinite(scores).all() or len(ids) != len(set(ids)):
                    raise AssertionError("non-finite score or duplicate ID in retained top-K")
                # Verify score-descending / ID-ascending order before dropping
                # this block's retained scores; the scorer checked each merge.
                if ids != np.asarray(ids, dtype=np.int64)[np.lexsort((ids, -scores))].tolist():
                    raise AssertionError("top-K score/article-ID ordering disagrees")
                probabilities = torch.sigmoid(torch.from_numpy(scores.astype(np.float64))).numpy()
                for rank, (article, score, probability) in enumerate(zip(ids, scores, probabilities), 1):
                    score_rows.append({"behaviour_id": bid, "article_id": article, "rank": rank,
                                       "score": float(score), "click_probability": float(probability)})
                ranking_rows.append({"behaviour_id": bid, "article_ids": ids})
                row = {"behaviour_id": bid, "user_id": int(req.user_id), "impression_id": int(req.impression_id),
                       "impression_time": str(req.impression_time), "eligible_count": int(counts[j]),
                       "shortlist_length": int(lengths[j]),
                       "n_recorded": int(offsets[pos + 1] - offsets[pos]) if offsets is not None else None,
                       "n_recorded_ineligible": int(recorded_counts[pos]) if recorded_counts is not None else None,
                       "tie_crosses_cutoff_100": bool(ties100[j]), "tie_crosses_cutoff_200": bool(ties200[j])}
                info = click_info[pos] if click_info is not None else None
                if info is not None:
                    row.update(n_clicked=len(info["ids"]), n_clicked_eligible=int(info["eligible"].sum()),
                               n_clicked_not_in_catalogue=int(info["not_in_catalogue"].sum()),
                               n_clicked_missing_pub=int(info["missing"].sum()), n_clicked_future=int(info["future"].sum()))
                    for k in (SELECTION_K, EXPORT_K):
                        _, hits, dcg, ideal = request_retrieval(ids, info["ids"], k)
                        row.update({f"n_clicked_in_top_{k}": hits, f"observed_dcg_at_{k}": dcg,
                                    f"ideal_dcg_at_{k}": ideal})
                manifest_rows.append(row)
            for name, rows in (("scores", score_rows), ("rankings", ranking_rows), ("manifest", manifest_rows)):
                writers[name].write_table(pa.Table.from_pylist(rows, schema=schemas[name]))
            n_scores += len(score_rows)
            n_pairs += int(counts.sum())
            if lo == 0 or (lo // request_block + 1) % 100 == 0:
                log(f"{split}: {lo + len(block):,}/{len(request_positions):,} requests; {n_pairs:,} eligible pairs scored")
    finally:
        for writer in writers.values():
            writer.close()
    for name, path in paths.items():
        temporary[name].replace(path)
    return {"finished": True, "requests": len(request_positions), "rows_scores": n_scores,
            "scored_pairs": n_pairs, "seconds": time.time() - start,
            "topk_checked_during_scoring": True, "paths": {name: str(path) for name, path in paths.items()}}


def write_recorded_slate_scores(path: Path, ctx, logits: np.ndarray, batch_rows: int = 100_000) -> dict:
    """All original shown candidates, including nonclicks and outside the shortlist; no labels."""
    rec, path = ctx.recorded, Path(path)
    schema = pa.schema([field for field in SCORE_SCHEMA if field.name != "rank"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".partial")
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for lo in range(0, len(logits), batch_rows):
            hi = min(len(logits), lo + batch_rows)
            values = logits[lo:hi].astype(np.float32)
            table = pa.Table.from_pydict({
                "behaviour_id": ctx.requests.behaviour_id.to_numpy()[rec["req_pos"][lo:hi]].astype(str),
                "article_id": rec["article_id"][lo:hi], "score": values,
                "click_probability": torch.sigmoid(torch.from_numpy(values.astype(np.float64))).numpy(),
            }, schema=schema)
            writer.write_table(table)
    temporary.replace(path)
    return {"path": str(path), "rows": len(logits), "requests": int(len(np.unique(rec["req_pos"])))}


def sample_requests(n_requests: int, n: int, seed: int) -> np.ndarray:
    """Request-sampled development mode: whole requests, each with its full eligible pool."""
    if n >= n_requests:
        return np.arange(n_requests)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_requests, size=n, replace=False))
