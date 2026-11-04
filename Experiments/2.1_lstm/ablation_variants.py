"""Ablation switches, comparators and evaluation helpers for 2.1_lstm.

Used by ``ablation.ipynb``. This module only *adds* behaviour around the main
experiment code: ``lstm_model.py`` and ``lstm_data.py`` are imported, never
edited. Every ablation keeps the tuned main-run settings and changes one thing.

Model-side switches (``AblationConfig``), applied by subclassing:
  * ``use_text_vectors`` / ``use_image_vectors``: drop the frozen text or image
    vector block from the article encoder.
  * ``drop_article_numeric``: remove named article numeric inputs, for example the
    vector-availability flags ``in_catalogue``, ``has_text_vector`` and
    ``has_image_vector``.
  * ``use_recency``: remove the learned recency bias from attention (fixed at 0).
  * ``drop_event_features``: remove named per-event history inputs.

Data-side switches (numpy, applied to ``lstm_data.SplitArrays``): no history,
history truncation, and the shared ``permute_history`` order control.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch import nn

import lstm_data as D
import lstm_model as M

VECTOR_FLAG_COLUMNS = ("in_catalogue", "has_text_vector", "has_image_vector")
READ_EVENT_COLUMNS = ("log_read_time_z", "read_time_missing", "scroll_z", "scroll_missing")


# --------------------------------------------------------------------------
# Model-side switches
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AblationConfig:
    use_text_vectors: bool = True
    use_image_vectors: bool = True
    drop_article_numeric: tuple = ()
    use_recency: bool = True
    drop_event_features: tuple = ()

    def is_default(self) -> bool:
        return self == AblationConfig()

    def to_json(self) -> dict:
        data = asdict(self)
        data["drop_article_numeric"] = list(self.drop_article_numeric)
        data["drop_event_features"] = list(self.drop_event_features)
        return data


class AblationArticleEncoder(M.ArticleEncoder):
    """``ArticleEncoder`` with optional modality and numeric-column removal.
    Defaults reproduce the parent."""

    def __init__(self, cfg: M.ModelConfig, acfg: AblationConfig, numeric_names):
        keep = [i for i, n in enumerate(numeric_names) if n not in acfg.drop_article_numeric]
        reduced = replace(cfg, n_article_numeric=len(keep),
                          text_dim=cfg.text_dim if acfg.use_text_vectors else 0,
                          image_dim=cfg.image_dim if acfg.use_image_vectors else 0)
        with warnings.catch_warnings():   # zero-width Linear for a removed modality
            warnings.simplefilter("ignore")
            super().__init__(reduced)
        if not acfg.use_text_vectors:
            self.text = None
        if not acfg.use_image_vectors:
            self.image = None
        self.register_buffer("keep_numeric", torch.as_tensor(keep, dtype=torch.long))

    def encode_rows(self, rows: torch.Tensor, drop_text: bool = False) -> torch.Tensor:
        t, cfg = self._table, self.cfg
        topics = t["topics"][rows]
        topic_mask = (topics > 0).unsqueeze(-1).float()
        topic_vec = (self.topic(topics) * topic_mask).sum(-2) / topic_mask.sum(-2).clamp(min=1)
        has_text = t["has_text"][rows]
        numeric = t["numeric"][rows]
        if drop_text and self.training and cfg.text_dropout > 0:
            keep = (torch.rand(has_text.shape, device=has_text.device) >= cfg.text_dropout).float()
            has_text = has_text * keep
            numeric = numeric.clone()
            numeric[:, cfg.numeric_has_text_index] *= keep   # index in the full column list
        parts = [self.category(t["category"][rows]), self.article_type(t["article_type"][rows]),
                 topic_vec]
        if self.text is not None:
            parts.append(self.text(self.dropout(t["text"][t["text_row"][rows]])) * has_text.unsqueeze(-1))
        if self.image is not None:
            parts.append(self.image(self.dropout(t["image"][t["image_row"][rows]]))
                         * t["has_image"][rows].unsqueeze(-1))
        parts.append(numeric.index_select(-1, self.keep_numeric))
        if cfg.use_id_embedding:
            parts.append(self.ids(t["id_bucket"][rows]))
        return self.mlp(torch.cat(parts, dim=-1))


class AblationRanker(M.TargetAwareLSTMRanker):
    """``TargetAwareLSTMRanker`` with the ``AblationConfig`` switches.

    All scoring paths (recorded slates via ``forward`` and the catalogue via
    ``CatalogueScorer``) go through the overridden encoder/history methods.
    """

    def __init__(self, cfg: M.ModelConfig, acfg: AblationConfig, numeric_names, event_names):
        keep_event = [i for i, n in enumerate(event_names) if n not in acfg.drop_event_features]
        super().__init__(replace(cfg, n_event_features=len(keep_event)))
        self.articles = AblationArticleEncoder(cfg, acfg, numeric_names)
        self.register_buffer("keep_event", torch.as_tensor(keep_event, dtype=torch.long))
        if not acfg.use_recency:
            del self.recency_weight
            self.register_buffer("recency_weight", torch.zeros(()))

    def encode_history(self, hist_rows, hist_feats, lengths):
        return super().encode_history(hist_rows, hist_feats.index_select(-1, self.keep_event),
                                      lengths)


def build_arm_model(cfg: M.ModelConfig, acfg: AblationConfig | None, article_tensors: dict,
                    device, numeric_names, event_names):
    """Main-model class when every switch is at its default, else ``AblationRanker``."""
    if acfg is None or acfg.is_default():
        return M.build_model(cfg, article_tensors, device)
    model = AblationRanker(cfg, acfg, numeric_names, event_names).to(device)
    model.articles.set_table(article_tensors)
    return model


def with_id_buckets(article_tensors: dict, articles: D.ArticleTable, id_vocab: dict, device):
    """Copy of the article tensors with learned-ID buckets (0 = out of vocabulary)."""
    bucket = np.zeros(len(articles.article_ids), np.int64)
    bucket[1:] = [id_vocab.get(int(a), 0) for a in articles.article_ids[1:].tolist()]
    return {**article_tensors, "id_bucket": torch.as_tensor(bucket, device=device)}


# --------------------------------------------------------------------------
# Data-side switches
# --------------------------------------------------------------------------

def _copy(arrays: D.SplitArrays, **changes) -> D.SplitArrays:
    return D.SplitArrays(**{**arrays.__dict__, **changes})


def remove_history(arrays: D.SplitArrays, numeric_stats: dict) -> D.SplitArrays:
    """Every request takes the model's empty-history path.

    History events are zeroed and lengths set to 0, and the two history-derived
    request-context inputs (log history length, empty flag) are set to their
    empty-history values, so no history information remains.
    """
    names = list(arrays.context_names)
    ctx = arrays.imp_context.copy()
    stats = numeric_stats["log_history_length"]
    ctx[:, names.index("log_history_length_z")] = (0.0 - stats["mean"]) / stats["std"]
    ctx[:, names.index("history_empty")] = 1.0
    return _copy(arrays, imp_context=ctx,
                 hist_article_row=np.zeros_like(arrays.hist_article_row),
                 hist_times_s=np.zeros_like(arrays.hist_times_s),
                 hist_features=np.zeros_like(arrays.hist_features),
                 hist_lengths=np.zeros_like(arrays.hist_lengths))


def truncate_history(arrays: D.SplitArrays, max_len: int, event_names) -> D.SplitArrays:
    """Keep each user's most recent ``max_len`` retained events (oldest first).

    The first retained event's gap-to-previous is marked missing, as
    ``build_history`` does when it truncates at the main length. Request context
    (full history length) and the evaluated requests/candidates are unchanged.
    """
    names = list(event_names)
    gap, gap_missing = names.index("log_gap_prev_h_z"), names.index("gap_prev_missing")
    length = arrays.hist_lengths.astype(np.int64)
    new_len = np.minimum(length, max_len)
    pos = np.arange(max_len)[None, :]
    valid = pos < new_len[:, None]
    index = np.where(valid, (length - new_len)[:, None] + pos, 0)
    take = lambda a: np.take_along_axis(a, index if a.ndim == 2 else index[..., None], axis=1)
    rows, times, feats = take(arrays.hist_article_row), take(arrays.hist_times_s), \
        take(arrays.hist_features).copy()
    rows = np.where(valid, rows, 0)
    times = np.where(valid, times, 0)
    feats[~valid] = 0.0
    first = new_len > 0
    feats[first, 0, gap] = 0.0
    feats[first, 0, gap_missing] = 1.0
    return _copy(arrays, hist_article_row=rows, hist_times_s=times, hist_features=feats,
                 hist_lengths=new_len.astype(arrays.hist_lengths.dtype))


def hash_sample(user_id, impression_id, time_us, label: str, seed: int, limit: int,
                among=None):
    """Shared BLAKE2b request sampling with a distinct ``label`` in place of the
    split name, optionally restricted to the request indices ``among``.
    Returns indices into the full arrays, in chronological processing order."""
    among = np.arange(len(user_id)) if among is None else np.asarray(among, dtype=np.int64)
    local, bids = D.select_requests(np.asarray(user_id)[among], np.asarray(impression_id)[among],
                                    np.asarray(time_us)[among], label, seed, limit)
    return among[local], bids


# --------------------------------------------------------------------------
# Fixed-budget training (constant learning rate, no selection evaluation)
# --------------------------------------------------------------------------

def train_fixed_budget(model, train: M.SplitTensors, cfg: M.TrainConfig, epochs: int,
                       log=print) -> list:
    """Exactly ``epochs`` epochs at a constant learning rate.

    Same optimiser, AMP, gradient clipping, batch order and history-dropout
    generators as ``lstm_model.train_model``, without the per-epoch validation
    evaluation, LR scheduler or early stopping. With the main-run seed this
    replays the main run's first ``epochs`` epochs (its scheduler never reduced
    the rate before epoch 6), up to cuDNN/AMP nondeterminism.
    """
    device = train.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(cfg.seed)
    device_gen = torch.Generator(device=device).manual_seed(cfg.seed)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, count = time.time(), 0.0, 0
        batches = list(M.impression_batches(train.n_impressions, cfg.batch_impressions, True,
                                            generator))
        for step, imp in enumerate(batches, start=1):
            batch = train.batch(imp, cfg.history_dropout, device_gen)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(batch)
                loss = loss_fn(logits.float(), batch["labels"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(batch["labels"])
            count += len(batch["labels"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        history.append({"epoch": epoch, "train_bce": total / max(count, 1),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "steps": len(batches), "seconds": time.time() - t0})
        log(f"  epoch {epoch}: train BCE {history[-1]['train_bce']:.4f} "
            f"({history[-1]['seconds']:.0f}s)")
    model.eval()
    return history


# --------------------------------------------------------------------------
# Evaluation on the two reporting samples
# --------------------------------------------------------------------------

def reciprocal_rank(offsets: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Per-slate reciprocal rank of the first clicked candidate (NaN without a
    click). Tied scores get the expected value under a random order of the tie."""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    sizes = np.diff(offsets)
    group = np.repeat(np.arange(len(sizes)), sizes)
    clicked = labels == 1
    best = np.full(len(sizes), -np.inf)
    np.maximum.at(best, group[clicked], scores[clicked])
    has_click = np.isfinite(best)
    ref = best[group]
    above = np.bincount(group, weights=(scores > ref), minlength=len(sizes))
    tied = np.bincount(group, weights=(scores == ref), minlength=len(sizes)).astype(int)
    tied_clicks = np.bincount(group, weights=(scores == ref) & clicked,
                              minlength=len(sizes)).astype(int)
    out = np.full(len(sizes), np.nan)
    simple = has_click & (tied == 1)
    out[simple] = 1.0 / (above[simple] + 1)
    for g in np.flatnonzero(has_click & (tied > 1)):
        n, c, s = int(tied[g]), int(tied_clicks[g]), int(above[g])
        total = math.comb(n, c)
        out[g] = sum(math.comb(n - j, c - 1) / total / (s + j) for j in range(1, n - c + 2))
    return out


def slate_metrics(offsets, labels, scores) -> tuple[dict, dict]:
    """Recorded-slate diagnostics: mean impression AUC, nDCG@5 and MRR."""
    summary = D.impression_metrics(offsets, labels, scores, k=5, return_per_impression=True)
    per = summary.pop("_per_impression")
    mrr = reciprocal_rank(offsets, labels, scores)
    summary["mean_impression_mrr"] = float(np.nanmean(mrr)) if np.isfinite(mrr).any() else None
    summary["mrr_eligible_impressions"] = int(np.isfinite(mrr).sum())
    return summary, {"auc": per["auc"], "ndcg@5": per["ndcg@5"], "mrr": mrr}


def evaluate_catalogue(topk_fn, requests, click_eligibility: dict, k: int = 100,
                       tie_fn=None, sync=None) -> tuple[dict, dict]:
    """Full-pool top-k for each request: Recall@k / observed-click nDCG@k.

    ``topk_fn(i, k) -> (ids, scores)`` must return the exact top-k of request
    ``i``'s complete eligible pool (descending score, ascending ID for ties).
    """
    lookup = D.clicks_by_request(click_eligibility)
    hits, tops, top_scores, ties = {}, [], [], 0
    t0 = time.time()
    for i in requests:
        ids, scores = topk_fn(int(i), k)
        ids = ids.cpu().numpy() if torch.is_tensor(ids) else np.asarray(ids)
        scores = scores.cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
        tops.append(ids.astype(np.int64))
        top_scores.append(scores.astype(np.float32))
        h = D.click_hits(lookup, int(i), ids, (k,))
        if h is not None:
            hits[int(i)] = h
        if tie_fn is not None:
            ties += int(tie_fn())
    if sync is not None:
        sync()
    seconds = time.time() - t0
    summary = D.recall_from_hits(hits, click_eligibility, (k,), requests_scored=len(requests))
    summary.pop("split_click_eligibility", None)
    per = D.per_request_metrics(hits, requests, k)
    summary.update({"requests_scored": int(len(requests)), "ties_at_cutoff": ties,
                    "scoring_seconds": seconds})
    return summary, {"recall": per["recall"], "ndcg": per["ndcg"], "top": tops,
                     "top_scores": top_scores}


class EnsembleTopK:
    """Exact top-k of the mean logit of several ``CatalogueScorer``s (same index,
    chunk size and request split), averaged chunk by chunk."""

    def __init__(self, scorers):
        self.scorers = list(scorers)

    @torch.no_grad()
    def topk(self, i: int, k: int):
        device = self.scorers[0].device
        ids = torch.empty(0, dtype=torch.long, device=device)
        logits = torch.empty(0, dtype=torch.float32, device=device)
        for parts in zip(*(s.iter_score_chunks(i) for s in self.scorers)):
            chunk_ids = parts[0][0]
            mean = torch.stack([p[1].float() for p in parts]).mean(0)
            ids, logits = M.select_top_k(torch.cat([logits, mean]), torch.cat([ids, chunk_ids]), k)
        return ids, logits


class RerankTopK:
    """Recency as the candidate generator, the model as the ranker.

    For request ``i``, take the ``n`` newest eligible articles (the most-recent
    comparator's order and pool), score only those with the model and return the
    top k by logit (ties by ascending ID). Articles outside the newest ``n`` are
    never retrieved. Same arithmetic as ``CatalogueScorer.iter_score_chunks``.
    """

    def __init__(self, scorer, recency, index: D.CatalogueIndex, articles: D.ArticleTable, n: int):
        self.scorer, self.recency, self.articles, self.n = scorer, recency, articles, n
        self.position = np.zeros(len(articles.article_ids), np.int64)
        self.position[index.rows] = np.arange(len(index.rows))

    @torch.no_grad()
    def topk(self, i: int, k: int):
        s, d = self.scorer, self.scorer.data
        ids, _ = self.recency.topk(int(d.imp_time_us[i]), self.n)
        pos = torch.as_tensor(self.position[self.articles.rows_for(ids)], device=s.device)
        u = int(d.imp_user_row[i])
        length = int(d.hist_len[u])
        states_u, keys_u = s._user(u)
        t = d.imp_time[i]
        gap_h = (t - d.hist_time[u, :max(length, 1)]).double() / 3600.0
        gap_z = ((torch.log1p(gap_h.clamp(min=0)) - s.model.cfg.gap_mean) / s.model.cfg.gap_std).float()
        age_h = (t - s.pub[pos]).double() / 3600.0
        log_age = ((torch.log1p(age_h.clamp(min=0)) - s.age_mean) / s.age_std).float()
        feats = torch.stack([log_age, torch.zeros_like(log_age)], dim=1)
        logits = s.model.score_one_request(s.enc[pos], s.q[pos], states_u, keys_u, length, gap_z,
                                           feats, d.imp_ctx[i])
        return M.select_top_k(logits, s.ids[pos], k)


# --------------------------------------------------------------------------
# No-training comparators
# --------------------------------------------------------------------------

def load_click_events(paths, exclude_users=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """(click time in microseconds, clicked article ID) from behaviour files.

    One event per distinct (impression, clicked article); repeated entries in a
    click list count once. Reads only user/time/clicked columns. Pass only train and validation
    behaviour files; test-split behaviours are never read.
    """
    times, aids, info = [], [], {}
    for path in paths:
        table = pq.read_table(path, columns=["user_id", "impression_time", "article_ids_clicked"])
        clicked = table["article_ids_clicked"].combine_chunks()
        lengths = pc.list_value_length(clicked).fill_null(0).to_numpy(zero_copy_only=False)
        t_us = table["impression_time"].cast(pa.timestamp("us")).cast(pa.int64()).to_numpy(
            zero_copy_only=False)
        users = table["user_id"].to_numpy().astype(np.int64)
        keep = np.ones(len(users), bool) if exclude_users is None else \
            ~np.isin(users, np.asarray(exclude_users))
        flat = pc.list_flatten(clicked).to_numpy(zero_copy_only=False).astype(np.int64)
        events = pd.DataFrame({"behaviour": np.repeat(np.arange(len(users)), lengths),
                               "t": np.repeat(t_us, lengths), "aid": flat})
        events = events[np.repeat(keep, lengths)]
        unique = events.drop_duplicates(["behaviour", "aid"])   # one click per article per impression
        times.append(unique["t"].to_numpy())
        aids.append(unique["aid"].to_numpy())
        info[str(path)] = {"behaviours": int(keep.sum()), "click_entries": int(len(events)),
                           "distinct_impression_clicks": int(len(unique))}
    t, a = np.concatenate(times), np.concatenate(aids)
    order = np.argsort(t, kind="stable")
    return t[order], a[order], info


class RecencyPopularityRanker:
    """Most recent (``window_hours=None``) or time-aware popularity.

    Popularity of an article at request time t = clicks in [t - window, t),
    strictly before the request, so the request's own impression never counts.
    Order: click count (descending), then publication time (newest first), then
    article ID (ascending). Pool: the shared time-eligible catalogue.
    """

    def __init__(self, index: D.CatalogueIndex, articles: D.ArticleTable,
                 click_time_us=None, click_article_ids=None, window_hours=None):
        self.index, self.articles = index, articles
        self.window_us = None if window_hours is None else int(window_hours * 3600e6)
        self.in_pool = np.zeros(len(articles.article_ids), bool)
        self.in_pool[index.rows] = True          # shared catalogue with a known publication time
        if self.window_us is not None:
            rows = articles.rows_for(click_article_ids)
            ok = rows > 0
            self.click_t = np.asarray(click_time_us, dtype=np.int64)[ok]
            self.click_rows = rows[ok]
            order = np.lexsort((self.click_t, self.click_rows))
            self.t0 = int(self.click_t.min()) - self.window_us - 1
            self._keys = (self.click_rows[order].astype(np.int64) << 42) | \
                (self.click_t[order] - self.t0)

    def _counts_window(self, t_us: int):
        lo = np.searchsorted(self.click_t, t_us - self.window_us, side="left")
        hi = np.searchsorted(self.click_t, t_us, side="left")
        rows, counts = np.unique(self.click_rows[lo:hi], return_counts=True)
        ok = self.in_pool[rows] & (self.articles.published_us[rows] <= t_us)
        return rows[ok], counts[ok]

    def topk(self, t_us: int, k: int):
        idx = self.index
        hi = int(np.searchsorted(idx.published_us, t_us, side="right"))
        if hi == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        rows_pop, counts = (self._counts_window(t_us) if self.window_us is not None
                            else (np.zeros(0, np.int64), np.zeros(0, np.int64)))
        start = max(0, hi - (k + len(rows_pop)))
        while start > 0 and idx.published_us[start - 1] == idx.published_us[start]:
            start -= 1                           # keep every article tied at the boundary
        rows = np.unique(np.r_[idx.rows[start:hi], rows_pop])
        count = np.zeros(len(rows), np.int64)
        count[np.searchsorted(rows, rows_pop)] = counts
        ids, pub = self.articles.article_ids[rows], self.articles.published_us[rows]
        order = np.lexsort((ids, -pub, -count))[:k]
        return ids[order].astype(np.int64), count[order].astype(np.float32)

    def slate_scores(self, cand_rows: np.ndarray, cand_time_us: np.ndarray) -> np.ndarray:
        """Score recorded candidates with the same order: count + recency in [0, 1)."""
        pub = self.articles.published_s[cand_rows].astype(np.float64)
        recency = (pub - pub.min()) / (pub.max() - pub.min() + 1.0)
        if self.window_us is None:
            return recency
        t = np.asarray(cand_time_us, dtype=np.int64)
        offset_hi = np.clip(t - self.t0, 0, None)
        offset_lo = np.clip(t - self.window_us - self.t0, 0, None)
        rows = cand_rows.astype(np.int64) << 42
        upper = np.searchsorted(self._keys, rows | offset_hi, side="left")
        lower = np.searchsorted(self._keys, rows | offset_lo, side="left")
        return (upper - lower).astype(np.float64) + recency


# Feature list of ``Baseline Testing/train_baseline.py`` (NUMERIC, CATEGORICAL).
LR_NUMERIC = ["log_age_hours", "age_missing", "future_publication", "premium",
              "article_missing", "log_history_length", "category_affinity",
              "log_category_clicks", "previously_clicked", "hour_sin", "hour_cos"]
LR_CATEGORICAL = ["category", "article_type", "device_type", "weekday", "age_bucket"]


def history_profiles(history_path: Path, articles: D.ArticleTable) -> dict:
    """Baseline user profiles from a split's complete fixed history (all events,
    not the model's 50-event window): event count, per-category counts and the
    set of previously read articles."""
    table = pq.read_table(history_path, columns=["user_id", "article_id_fixed"])
    ids = table["article_id_fixed"].combine_chunks()
    lengths = pc.list_value_length(ids).fill_null(0).to_numpy(zero_copy_only=False)
    users = np.repeat(table["user_id"].to_numpy().astype(np.int64), lengths)
    flat = pc.list_flatten(ids)
    valid = pc.is_valid(flat).to_numpy(zero_copy_only=False)
    events = pd.DataFrame({"user_id": users[valid],
                           "article_id": flat.fill_null(0).to_numpy(zero_copy_only=False)
                           .astype(np.int64)[valid]})
    events["category"] = articles.category[articles.rows_for(events["article_id"].to_numpy())]
    return {"n": events.groupby("user_id").size().rename("n_hist"),
            "category": events.groupby(["user_id", "category"]).size().rename("cat_count"),
            "seen": events[["user_id", "article_id"]].drop_duplicates().assign(seen=1.0)}


def lr_features(frame: pd.DataFrame, articles: D.ArticleTable, profiles: dict,
                device_columns) -> pd.DataFrame:
    """Baseline logistic-regression features for recorded candidates.

    Same feature list and definitions as ``Baseline Testing`` with this
    project's shared inputs: category/type use the shared vocabulary codes,
    device type is decoded from the one-hot request columns, and article
    metadata comes from the union catalogue (so ``article_missing`` is 0).
    """
    rows = articles.rows_for(frame["candidate_article_id"].to_numpy())
    t = frame["impression_time"].astype("datetime64[us]")
    t_s = t.astype("int64").to_numpy() // 1_000_000
    pub_missing = articles.published_missing[rows]
    age = np.where(pub_missing, np.nan, (t_s - articles.published_s[rows]) / 3600.0)
    premium_col = articles.numeric_names.index("premium")
    out = pd.DataFrame({
        "user_id": frame["user_id"].to_numpy(),
        "article_id": frame["candidate_article_id"].to_numpy(),
        "category": articles.category[rows].astype(str),
        "article_type": articles.article_type[rows].astype(str),
        "premium": articles.numeric[rows, premium_col].astype(np.float32),
        "article_missing": (rows == 0).astype(np.float32),
        "age_missing": np.isnan(age).astype(np.float32),
        "future_publication": (age < 0).astype(np.float32),
        "log_age_hours": np.log1p(np.clip(np.nan_to_num(age, nan=0.0), 0, None)),
    })
    out["age_bucket"] = pd.cut(pd.Series(age), [-np.inf, 0, 6, 24, 72, 168, np.inf],
                               labels=["nonpositive", "0-6h", "6-24h", "1-3d", "3-7d", "7d+"]
                               ).astype("string").fillna("unknown").to_numpy()
    hour = (t.dt.hour + t.dt.minute / 60).to_numpy()
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    out["weekday"] = t.dt.weekday.astype(str).to_numpy()
    device = frame[list(device_columns)].to_numpy(dtype=float)
    labels = np.asarray([c.replace("ohe_device_type_", "") for c in device_columns], dtype=object)
    out["device_type"] = np.where(device.max(1) > 0, labels[device.argmax(1)], "none")
    out["_cat"] = articles.category[rows]
    out = out.merge(profiles["n"], left_on="user_id", right_index=True, how="left")
    out = out.merge(profiles["category"], left_on=["user_id", "_cat"], right_index=True, how="left")
    out = out.merge(profiles["seen"], on=["user_id", "article_id"], how="left")
    n = out["n_hist"].fillna(0).to_numpy()
    cat_count = out["cat_count"].fillna(0).to_numpy()
    out["log_history_length"] = np.log1p(n)
    out["category_affinity"] = cat_count / np.maximum(n, 1)
    out["log_category_clicks"] = np.log1p(cat_count)
    out["previously_clicked"] = out["seen"].fillna(0.0).to_numpy()
    for column in LR_NUMERIC:
        out[column] = out[column].astype(np.float32)
    return out[LR_NUMERIC + LR_CATEGORICAL]


def make_lr_pipeline():
    """Same estimator as ``Baseline Testing`` (scaler/encoder fitted on train only)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    features = ColumnTransformer([
        ("numeric", StandardScaler(), LR_NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), LR_CATEGORICAL),
    ], sparse_threshold=1.0)
    return Pipeline([("features", features),
                     ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=300))])


# --------------------------------------------------------------------------
# Segments, vector shares and paired inference
# --------------------------------------------------------------------------

HISTORY_BUCKETS = ((0, 0, "0 (empty)"), (1, 9, "1-9"), (10, 19, "10-19"), (20, 49, "20-49"),
                   (50, 99, "50-99"), (100, 199, "100-199"), (200, None, "200+"))


def history_bucket(full_lengths: np.ndarray) -> np.ndarray:
    out = np.full(len(full_lengths), "", dtype=object)
    for lo, hi, name in HISTORY_BUCKETS:
        mask = (full_lengths >= lo) & ((full_lengths <= hi) if hi is not None else True)
        out[mask] = name
    return out


def cold_warm_recall(tops, requests, click_lookup: dict, warm_ids: np.ndarray, k: int = 100):
    """Per-request Recall@k over cold clicked articles and over warm ones (NaN
    when the request has no click of that kind)."""
    cold = np.full(len(requests), np.nan)
    warm = np.full(len(requests), np.nan)
    for j, i in enumerate(requests):
        entry = click_lookup.get(int(i))
        if entry is None:
            continue
        aids = entry[0]
        hit = np.isin(aids, np.asarray(tops[j])[:k])
        is_warm = np.isin(aids, warm_ids)
        if (~is_warm).any():
            cold[j] = hit[~is_warm].mean()
        if is_warm.any():
            warm[j] = hit[is_warm].mean()
    return cold, warm


def pool_vectorless_share(index: D.CatalogueIndex, articles: D.ArticleTable,
                          request_time_us: np.ndarray, fresh_hours: float) -> dict:
    """Per-request share of vector-less articles (no text and no image vector)
    in the full eligible pool and in the fresh pool (published within
    ``fresh_hours`` before the request)."""
    vectorless = ~(articles.has_text[index.rows] | articles.has_image[index.rows])
    prefix = np.r_[0, np.cumsum(vectorless)]
    t = np.asarray(request_time_us, dtype=np.int64)
    hi = np.searchsorted(index.published_us, t, side="right")
    lo = np.searchsorted(index.published_us, t - int(fresh_hours * 3600e6), side="left")
    full = prefix[hi] / np.maximum(hi, 1)
    fresh = (prefix[hi] - prefix[lo]) / np.maximum(hi - lo, 1)
    return {"full_pool": full, "fresh_pool": np.where(hi > lo, fresh, np.nan),
            "fresh_pool_size": hi - lo}


def top_vectorless_share(tops, articles: D.ArticleTable, k: int = 100) -> np.ndarray:
    out = np.full(len(tops), np.nan)
    for j, ids in enumerate(tops):
        ids = np.asarray(ids)[:k]
        if len(ids):
            rows = articles.rows_for(ids)
            out[j] = float((~(articles.has_text[rows] | articles.has_image[rows])).mean())
    return out


def paired_delta(users, arm_runs: list, ref_runs: list, n_boot: int, seed: int = 0) -> dict:
    """Mean per-request difference (arm - reference), averaged over seed-matched
    runs, with a 95% user-cluster bootstrap interval (``user_cluster_bootstrap``)."""
    a = np.nanmean(np.vstack(arm_runs), axis=0) if len(arm_runs) > 1 else np.asarray(arm_runs[0])
    b = np.nanmean(np.vstack(ref_runs), axis=0) if len(ref_runs) > 1 else np.asarray(ref_runs[0])
    return D.user_cluster_bootstrap(np.asarray(users), a, b, n_boot=n_boot, seed=seed)


def differs(delta: dict, noise_floor: float | None) -> bool | None:
    """Pre-registered rule: the 95% CI excludes 0 and |delta| exceeds the noise floor."""
    if not delta or delta.get("ci95") is None or noise_floor is None:
        return None
    lo, hi = delta["ci95"]
    return bool((lo > 0 or hi < 0) and abs(delta["mean_difference"]) > noise_floor)


# --------------------------------------------------------------------------
# Per-request result frames (one folder per arm and seed)
# --------------------------------------------------------------------------

def catalogue_frame(requests, bids, users, per: dict) -> pd.DataFrame:
    return pd.DataFrame({"request_index": np.asarray(requests, dtype=np.int64),
                         "behaviour_id": list(bids), "user_id": np.asarray(users, dtype=np.int64),
                         "recall@100": per["recall"], "ndcg@100": per["ndcg"],
                         "top100": [np.asarray(t, dtype=np.int64) for t in per["top"]],
                         "top100_scores": [np.asarray(s, dtype=np.float32)
                                           for s in per["top_scores"]]})


def slate_frame(requests, bids, users, per: dict) -> pd.DataFrame:
    return pd.DataFrame({"request_index": np.asarray(requests, dtype=np.int64),
                         "behaviour_id": list(bids), "user_id": np.asarray(users, dtype=np.int64),
                         "auc": per["auc"], "ndcg@5": per["ndcg@5"], "mrr": per["mrr"]})
