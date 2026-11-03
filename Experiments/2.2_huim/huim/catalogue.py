"""Time-eligible catalogue scoring, top-100 shortlists and streamed v3 exports.

Eligibility (shared policy published_at_or_before_request_v1): for request
(user_id, impression_id) at time t, an article of the ordered train_val -> test
union catalogue (the same pool for every split) is eligible iff its
published_time is present and published_time <= t. Articles with a missing
publication time are excluded (cannot be verified as non-future) and counted.
No recency window, access rule or vector-coverage restriction is applied.
"""
from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import torch

from .features import NO_PUB
from .metrics import request_retrieval
from .io_utils import atomic_write_json, read_json

SHORTLIST_SIZE = 100
FINAL_DISPLAY_SIZE = 5
EXPORT_SCHEMA_VERSION = 3
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
MANIFEST_COLUMNS = [
    "behaviour_id", "user_id", "impression_id", "impression_time", "eligible_count", "shortlist_length",
    "n_recorded", "n_recorded_ineligible", "n_clicked", "n_clicked_eligible", "n_clicked_not_in_catalogue",
    "n_clicked_missing_pub", "n_clicked_future", "n_clicked_in_shortlist",
    "observed_dcg_at_100", "ideal_dcg_at_100", "tie_crosses_cutoff",
    "shortlist_share_with_text", "shortlist_median_age_h", "shortlist_distinct_categories",
]


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


def click_eligibility(ctx, art) -> list[dict]:
    """Per request: clicked IDs split into eligible / not in catalogue / missing
    pub / future. Original clicked IDs stay in the recall denominator."""
    out = []
    catalogue_ids = art.ids[ctx.catalogue_idx]
    pub = art.tensors["pub_s"].numpy()
    req_s = ctx.tensors["req_s"].numpy()
    clicked = ctx.requests["article_ids_clicked"]
    for i in range(len(ctx.requests)):
        ids = clicked.iloc[i]
        if ids is None:
            out.append(None)
            continue
        ids = np.unique(np.asarray(ids, dtype=np.int64))  # a clicked article counts once
        in_cat = np.isin(ids, catalogue_ids)
        p = pub[art.lookup(ids)]
        missing = in_cat & (p == NO_PUB)
        future = in_cat & ~missing & (p > req_s[i])
        eligible = in_cat & ~missing & ~future
        out.append({"ids": ids, "eligible": eligible, "not_in_catalogue": ~in_cat, "missing": missing, "future": future})
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
        "full_scores_csv_bytes_estimate": int(pairs * bytes_per_row),
        "top100_csv_bytes_estimate": int(len(counts) * (20 + 9 * SHORTLIST_SIZE)),
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

def _csv_bytes(table: pa.Table, header: bool) -> bytes:
    buf = io.BytesIO()
    try:
        opts = pacsv.WriteOptions(include_header=header, quoting_style="needed")
    except TypeError:  # older pyarrow
        opts = pacsv.WriteOptions(include_header=header)
    pacsv.write_csv(table, buf, write_options=opts)
    return buf.getvalue()


def _truncate_or_create(path: Path, size: int) -> None:
    """Cut a partial file back to the last checkpointed size. Never extend it:
    a file shorter than recorded means it was lost or partly copied."""
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.stat().st_size if path.exists() else 0
    if current < size:
        raise RuntimeError(f"{path} has {current} bytes but the export progress records {size}; "
                           "the partial export is damaged — remove its progress file to restart this split.")
    with open(path, "a+b") as handle:
        handle.truncate(size)


def export_split(scorer: CatalogueScorer, out_root: Path, request_positions: np.ndarray, *, write_full_scores: bool,
                 click_info: list | None, recorded_counts: tuple | None, request_block: int = 16,
                 checkpoint_every_blocks: int = 50, log=print, max_blocks: int | None = None,
                 model_digest: str | None = None) -> dict:
    """Stream rankings (+ optional full scores) for the given requests, resumably.

    Files under out_root:
      predictions/{split}/submission_click_probabilities.csv   (if write_full_scores)
      rankings/{split}/submission_top_100.csv
      diagnostics/catalogue/{split}_requests.csv               (eligibility/recall manifest)
      diagnostics/catalogue/{split}_score_blocks.csv           (byte ranges per request block)
      diagnostics/catalogue/{split}_export_progress.json       (resume state; written after each flush)
    `max_blocks` bounds work for an interruption test.
    """
    ctx, art = scorer.ctx, scorer.art
    split = ctx.split
    out_root = Path(out_root)
    paths = {
        "scores": out_root / "predictions" / split / "submission_click_probabilities.csv",
        "rankings": out_root / "rankings" / split / "submission_top_100.csv",
        "manifest": out_root / "diagnostics" / "catalogue" / f"{split}_requests.csv",
        "blocks": out_root / "diagnostics" / "catalogue" / f"{split}_score_blocks.csv",
    }
    progress_path = out_root / "diagnostics" / "catalogue" / f"{split}_export_progress.json"
    request_positions = np.asarray(request_positions, dtype=np.int64)
    signature = {"split": split, "n_requests": int(len(request_positions)),
                 "requests_sha": str(pd.util.hash_array(request_positions).sum()),
                 "write_full_scores": bool(write_full_scores), "request_block": int(request_block),
                 "model_digest": model_digest, "catalogue_sha": str(pd.util.hash_array(ctx.catalogue_idx).sum())}
    progress = None
    if progress_path.is_file():
        progress = read_json(progress_path)
        if progress.get("signature") != signature:
            raise RuntimeError(f"{progress_path} belongs to a different export configuration; "
                               "use a new output directory or remove it deliberately.")
        if progress.get("finished"):
            log(f"{split}: export already finished at {out_root}")
            return progress
        log(f"{split}: resuming export at request {progress['next_request']}")
    else:
        progress = {"signature": signature, "next_request": 0, "finished": False,
                    "bytes": {k: 0 for k in paths}, "rows_scores": 0, "seconds": 0.0, "blocks_done": 0}
    for key, path in paths.items():
        if key == "scores" and not write_full_scores:
            continue
        _truncate_or_create(path, progress["bytes"][key])

    handles = {k: open(p, "ab") for k, p in paths.items() if k != "scores" or write_full_scores}
    try:
        if progress["bytes"]["rankings"] == 0:
            handles["rankings"].write(b"behaviour_id,article_ids\n")
        if progress["bytes"]["manifest"] == 0:
            handles["manifest"].write((",".join(MANIFEST_COLUMNS) + "\n").encode())
        if progress["bytes"]["blocks"] == 0:
            handles["blocks"].write(b"block,first_request_index,last_request_index,byte_start,byte_end,rows\n")
        header_needed = write_full_scores and progress["bytes"]["scores"] == 0
        behaviour_ids = ctx.requests.behaviour_id.to_numpy()
        cat_ids = art.ids[ctx.catalogue_idx]
        cat_ids_t = pa.array(cat_ids, type=pa.int64())
        times = pd.to_datetime(ctx.requests.impression_time).astype(str).to_numpy()
        rec_offsets = ctx.recorded.get("offsets") if ctx.recorded else None
        t_start = time.time()
        blocks_this_call = 0
        i = progress["next_request"]
        while i < len(request_positions):
            if max_blocks is not None and blocks_this_call >= max_blocks:
                break
            block = request_positions[i:i + request_block]
            logits = scorer.logits(block)
            elig = scorer.eligible_mask(block)
            order, lengths, ties = scorer.rank(logits, elig)
            ties_np = ties.cpu().numpy()
            lengths_np = lengths.cpu().numpy()
            order_np = order.cpu().numpy()
            # Shortlist diagnostics (freshness / diversity / vector coverage)
            top_valid = torch.arange(order.shape[1], device=order.device)[None, :] < lengths[:, None]
            has_text = (scorer.has_text[order] & top_valid).sum(1).float() / lengths.clamp_min(1)
            t_req = ctx.tensors["req_s"][torch.from_numpy(block).to(order.device)]
            age_h = (t_req[:, None] - scorer.pub[order]).float() / 3600.0
            age_h = torch.where(top_valid, age_h, torch.full_like(age_h, float("nan")))
            med_age = torch.nanmedian(age_h, dim=1).values.cpu().numpy()
            cats = scorer.category[order].cpu().numpy()
            # --- full scores (eligible only), request-major then ascending article_id
            if write_full_scores:
                rows_idx, cols_idx = elig.nonzero(as_tuple=True)
                vals = logits[rows_idx, cols_idx].cpu().numpy().astype(np.float32)
                rows_np, cols_np = rows_idx.cpu().numpy(), cols_idx.cpu().numpy()
                table = pa.table({
                    "behaviour_id": pc.take(pa.array(behaviour_ids[block].astype(str)), pa.array(rows_np)),
                    "article_id": pc.take(cat_ids_t, pa.array(cols_np)),
                    "score": pa.array(vals, type=pa.float32()),
                    "click_probability": pa.array(1.0 / (1.0 + np.exp(-vals.astype(np.float64))), type=pa.float64()),
                })
                if header_needed:  # header written once, outside every block's byte range
                    handles["scores"].write(b"behaviour_id,article_id,score,click_probability\n")
                    header_needed = False
                start = handles["scores"].tell()
                handles["scores"].write(_csv_bytes(table, False))
                end = handles["scores"].tell()
                progress["rows_scores"] += len(vals)
            else:
                start = end = 0
                rows_np = None
            handles["blocks"].write(f"{progress['blocks_done']},{i},{i + len(block) - 1},{start},{end},"
                                    f"{int(elig.sum())}\n".encode())
            # --- rankings + manifest
            rank_buf, man_buf = io.StringIO(), io.StringIO()
            rank_writer, man_writer = csv.writer(rank_buf, lineterminator="\n"), csv.writer(man_buf, lineterminator="\n")
            elig_counts = elig.sum(1).cpu().numpy()
            for j, pos in enumerate(block):
                ids = cat_ids[order_np[j, :lengths_np[j]]].tolist()
                rank_writer.writerow([behaviour_ids[pos], json.dumps(ids)])
                info = click_info[pos] if click_info is not None else None
                if info is None:  # unlabelled (test): outcomes unknown, nothing computed
                    clicks = [""] * 8
                else:
                    n_clicked, hits, dcg, ideal = request_retrieval(ids, info["ids"], SHORTLIST_SIZE)
                    clicks = [n_clicked, int(info["eligible"].sum()), int(info["not_in_catalogue"].sum()),
                              int(info["missing"].sum()), int(info["future"].sum()), hits,
                              f"{dcg:.10f}", f"{ideal:.10f}"]
                clicks.append(int(bool(ties_np[j])))
                n_rec = int(rec_offsets[pos + 1] - rec_offsets[pos]) if rec_offsets is not None else ""
                n_rec_bad = recorded_counts[pos] if recorded_counts is not None else ""
                n_cat = len(set(cats[j, :lengths_np[j]].tolist()))
                req = ctx.requests.iloc[pos]
                man_writer.writerow([behaviour_ids[pos], int(req.user_id), int(req.impression_id), times[pos],
                                     int(elig_counts[j]), int(lengths_np[j]), n_rec, n_rec_bad, *clicks,
                                     f"{float(has_text[j]):.4f}", "" if np.isnan(med_age[j]) else f"{float(med_age[j]):.3f}",
                                     n_cat])
            handles["rankings"].write(rank_buf.getvalue().encode())
            handles["manifest"].write(man_buf.getvalue().encode())
            i += len(block)
            progress["blocks_done"] += 1
            blocks_this_call += 1
            if progress["blocks_done"] % checkpoint_every_blocks == 0 or i >= len(request_positions):
                for h in handles.values():
                    h.flush()
                progress["bytes"] = {k: (handles[k].tell() if k in handles else 0) for k in paths}
                progress["next_request"] = int(i)
                progress["seconds"] += time.time() - t_start
                t_start = time.time()
                atomic_write_json(progress_path, progress)
                log(f"{split}: {i}/{len(request_positions)} requests; {progress['rows_scores']:,} score rows; "
                    f"{progress['seconds'] / 60:.1f} min")
        for h in handles.values():
            h.flush()
        progress["bytes"] = {k: (handles[k].tell() if k in handles else 0) for k in paths}
        progress["next_request"] = int(i)
        progress["seconds"] += time.time() - t_start
        progress["finished"] = i >= len(request_positions)
        atomic_write_json(progress_path, progress)
    finally:
        for h in handles.values():
            h.close()
    progress["paths"] = {k: str(p) for k, p in paths.items() if k != "scores" or write_full_scores}
    return progress


def write_recorded_slate_scores(path: Path, ctx, logits: np.ndarray) -> dict:
    """Every original recorded candidate exactly once: behaviour_id,article_id,score,click_probability."""
    rec = ctx.recorded
    table = pa.table({
        "behaviour_id": pa.array(ctx.requests.behaviour_id.to_numpy()[rec["req_pos"]].astype(str)),
        "article_id": pa.array(rec["article_id"], type=pa.int64()),
        "score": pa.array(logits.astype(np.float32), type=pa.float32()),
        "click_probability": pa.array(1.0 / (1.0 + np.exp(-logits.astype(np.float64))), type=pa.float64()),
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial")
    with open(tmp, "wb") as handle:
        handle.write(_csv_bytes(table, True))
    tmp.replace(path)
    return {"path": str(path), "rows": int(len(table)), "requests": int(len(np.unique(rec["req_pos"])))}


def measure_throughput(scorer: CatalogueScorer, request_positions: np.ndarray, request_block: int = 16,
                       n_blocks: int = 4) -> dict:
    """Benchmark scoring pairs/s and CSV formatting rows/s on a few request blocks."""
    if not len(request_positions):
        return {}
    blocks = [request_positions[i:i + request_block] for i in range(0, min(len(request_positions), request_block * n_blocks), request_block)]
    scorer.logits(blocks[0])  # warm-up
    if scorer.device.type == "cuda":
        torch.cuda.synchronize()
    t0, pairs = time.time(), 0
    csv_rows, csv_bytes, csv_seconds = 0, 0, 0.0
    for b in blocks:
        logits = scorer.logits(b)
        elig = scorer.eligible_mask(b)
        if scorer.device.type == "cuda":
            torch.cuda.synchronize()
        pairs += logits.numel()
        r, c = elig.nonzero(as_tuple=True)
        vals = logits[r, c].cpu().numpy()
        t1 = time.time()
        table = pa.table({"behaviour_id": pc.take(pa.array(scorer.ctx.requests.behaviour_id.to_numpy()[b].astype(str)),
                                                  pa.array(r.cpu().numpy())),
                          "article_id": pa.array(scorer.art.ids[scorer.ctx.catalogue_idx][c.cpu().numpy()]),
                          "score": pa.array(vals.astype(np.float32)),
                          "click_probability": pa.array(1 / (1 + np.exp(-vals.astype(np.float64))))})
        data = _csv_bytes(table, False)
        csv_seconds += time.time() - t1
        csv_rows += len(vals)
        csv_bytes += len(data)
    seconds = time.time() - t0
    return {"pairs_scored": int(pairs), "seconds_total": seconds,
            "pairs_per_second_including_formatting": pairs / max(seconds, 1e-9),
            "pairs_per_second_scoring_only": pairs / max(seconds - csv_seconds, 1e-9),
            "csv_rows_per_second": csv_rows / max(csv_seconds, 1e-9),
            "csv_bytes_per_row": csv_bytes / max(csv_rows, 1),
            "note": "Excludes disk/Drive write speed; measured on a few blocks only."}


def sample_requests(n_requests: int, n: int, seed: int) -> np.ndarray:
    """Request-sampled development mode: whole requests, each with its full eligible pool."""
    if n >= n_requests:
        return np.arange(n_requests)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_requests, size=n, replace=False))
