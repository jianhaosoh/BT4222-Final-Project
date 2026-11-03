"""Request/article/history inputs for HUIM Stage 1.

One feature path serves training pairs, recorded-slate diagnostics and
catalogue-wide scoring. Every input below is constructible for an arbitrary
eligible article at request time (see FEATURE_AUDIT). Nothing depends on
recorded-slate membership, slate size, click counts, future totals or
post-impression outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

UNKNOWN_IDX = 0  # global article index 0 = unknown article / history padding
NO_PUB = np.iinfo(np.int64).max  # sentinel for a missing publication time
RECENCY_EDGES_H = [1, 3, 6, 12, 24, 48, 96, 168, 336, 672]  # fixed log-like bins

FEATURE_AUDIT = [
    # name, scope, status, reason
    ("category / subcategory / article_type / sentiment_label / topics / premium", "article", "used",
     "static catalogue metadata; available for any catalogue article"),
    ("body_word_count", "article", "used", "from the supplied snapshot (may post-date edits; documented caveat)"),
    ("text vector (gemini-embedding-2, 768-d, frozen)", "article", "used with mask",
     "precomputed lookup; missing vectors kept missing with has_text flag"),
    ("image vector (EB-NeRD supplied, 1024-d, frozen)", "article", "used with mask",
     "separate space from text; missing kept missing with has_image flag"),
    ("article age = request time - published_time", "pair", "used", "known at request; future articles are ineligible"),
    ("premium x subscriber access", "pair", "used", "request subscriber flag x article premium"),
    ("history category/type/topic preference crosses", "pair", "used",
     "computed from fixed history before the split start for any candidate"),
    ("candidate previously clicked in fixed history", "pair", "used", "fixed history only"),
    ("device / SSO / subscriber / hour / weekday", "request", "used", "request context"),
    ("history size, median read time, mean scroll, 7-day clicks, hours since last click", "request", "used",
     "fixed-history aggregates; past engagement only"),
    ("history items (article, read-time bin, scroll bin, recency bin)", "history", "used",
     "item-level attention input; events strictly before the split start"),
    ("slate_size / slate_relative_age_h", "pair", "excluded",
     "recorded-slate dependent; no catalogue equivalent (omitted from training and inference)"),
    ("is_front_page / current_page_article_id", "request", "excluded",
     "official test behaviours have no article_id column; shared test features mark every row front page"),
    ("total_inviews / total_pageviews / total_read_time / last_modified_time", "article", "excluded",
     "future aggregates / later than impression"),
    ("impression read_time / scroll_percentage / next_*", "request", "excluded", "post-impression outcomes"),
    ("clicked counts / in-view popularity / session co-occurrence", "pair", "excluded",
     "slate-, outcome- or future-dependent"),
    ("article/user ID embeddings", "model", "excluded", "cold-start catalogue; IDs are join keys only"),
]


def to_seconds(values) -> np.ndarray:
    """datetime-like -> int64 seconds since epoch (NaT -> NO_PUB)."""
    series = pd.to_datetime(pd.Series(values))
    out = np.full(len(series), NO_PUB, dtype=np.int64)
    ok = series.notna().to_numpy()
    out[ok] = series[ok].to_numpy().astype("datetime64[s]").astype(np.int64)
    return out


def _nominal(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    return str(value)


def _as_list(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return list(value)


# ============================================================ preprocessing ==

class Preprocessor:
    """Fitted, serialisable preprocessing. Fit on train-permitted data only."""

    def __init__(self, state: dict | None = None):
        self.state = state or {}

    # ---- fitting ------------------------------------------------------------
    @classmethod
    def fit(cls, vocabulary: dict, articles: pd.DataFrame, train_article_ids: np.ndarray,
            train_events: pd.DataFrame, history_len: int, max_topics: int = 12, max_subcats: int = 6) -> "Preprocessor":
        state: dict[str, Any] = {"version": 1, "history_len": int(history_len),
                                 "max_topics": int(max_topics), "max_subcats": int(max_subcats)}
        # Embedding vocabularies: the shared train-fitted Step 3 vocabulary.
        state["category_vocab"] = list(vocabulary["category"])
        state["article_type_vocab"] = list(vocabulary["article_type"])
        state["topic_vocab"] = list(vocabulary["topics"])
        state["device_vocab"] = list(vocabulary["device_type"])
        # Additional vocabularies fitted on train-referenced articles only.
        used = articles[articles.article_id.isin(set(map(int, train_article_ids)))]
        subcats = sorted({str(s) for v in used.subcategory for s in _as_list(v)})
        sentiments = sorted({s for s in used.sentiment_label.map(_nominal) if s is not None})
        state["subcategory_vocab"], state["sentiment_vocab"] = subcats, sentiments
        body = np.log1p(used.body_word_count.dropna().to_numpy(dtype="float64"))
        state["body_log_mean"] = float(body.mean()) if body.size else 0.0
        state["body_log_std"] = float(body.std()) if body.size and body.std() > 0 else 1.0
        topic_counts = np.log1p(used.topics.map(lambda v: len(_as_list(v))).to_numpy(dtype="float64"))
        state["topic_count_log_mean"] = float(topic_counts.mean()) if topic_counts.size else 0.0
        state["topic_count_log_std"] = float(topic_counts.std()) if topic_counts.size and topic_counts.std() > 0 else 1.0
        # Quantile bins for history engagement, fitted on train fixed-history events.
        def edges(values, q=8):
            values = values[np.isfinite(values)]
            if not values.size:
                return []
            return sorted(set(np.quantile(values, np.linspace(0, 1, q + 1)[1:-1]).round(6).tolist()))
        state["read_time_edges"] = edges(train_events.read_time.to_numpy(dtype="float64"))
        state["scroll_edges"] = edges(train_events.scroll.to_numpy(dtype="float64"))
        state["recency_edges_h"] = list(RECENCY_EDGES_H)
        # Cross-feature indexing over every category/type/topic value in the
        # loaded catalogue union: an index, not a fitted statistic (no labels).
        state["cross_categories"] = sorted({c for c in articles.category.map(_nominal) if c is not None})
        state["cross_types"] = sorted({c for c in articles.article_type.map(_nominal) if c is not None})
        state["cross_topics"] = sorted({str(t) for v in articles.topics for t in _as_list(v) if t is not None})
        return cls(state)

    def to_json(self) -> dict:
        return self.state

    @classmethod
    def from_json(cls, state: dict) -> "Preprocessor":
        return cls(state)

    # dense scalers are fitted later, from train recorded pairs/requests
    def set_dense_scaler(self, name: str, mean: np.ndarray, std: np.ndarray, columns: list[str]):
        std = np.where(np.asarray(std) > 1e-8, std, 1.0)
        self.state[f"{name}_scaler"] = {"columns": list(columns), "mean": np.asarray(mean, float).tolist(),
                                        "std": np.asarray(std, float).tolist()}

    def dense_scaler(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        s = self.state[f"{name}_scaler"]
        return np.asarray(s["mean"], np.float32), np.asarray(s["std"], np.float32)

    # sizes for the model
    def sizes(self) -> dict:
        s = self.state
        return {
            "n_category": len(s["category_vocab"]) + 2, "n_article_type": len(s["article_type_vocab"]) + 2,
            "n_topic": len(s["topic_vocab"]) + 2, "n_subcategory": len(s["subcategory_vocab"]) + 2,
            "n_sentiment": len(s["sentiment_vocab"]) + 2,
            "n_read_bin": len(s["read_time_edges"]) + 2, "n_scroll_bin": len(s["scroll_edges"]) + 2,
            "n_recency_bin": len(s["recency_edges_h"]) + 2,
            "n_request_dense": len(REQUEST_DENSE_COLUMNS(self)), "n_pair_dense": len(PAIR_DENSE_COLUMNS),
        }


def _index(values, vocab, missing=0, unknown=1):
    """nominal -> 0 missing, 1 unknown, 2.. vocabulary position."""
    lookup = {v: i + 2 for i, v in enumerate(vocab)}
    out = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        v = _nominal(v)
        out[i] = missing if v is None else lookup.get(v, unknown)
    return out


def _bin(values: np.ndarray, edges: list) -> np.ndarray:
    """0 = missing, 1.. = quantile bin."""
    values = np.asarray(values, dtype="float64")
    out = np.zeros(values.shape, dtype=np.int64)
    ok = np.isfinite(values)
    out[ok] = np.searchsorted(np.asarray(edges, dtype="float64"), values[ok], side="right") + 1
    return out


# ============================================================ article table ==

@dataclass
class ArticleTable:
    """Global article lookup. Index 0 is the unknown/padding article; 1..N are
    article IDs in ascending order, so ascending index == ascending article_id."""
    ids: np.ndarray
    tensors: dict
    report: dict

    def lookup(self, article_ids) -> np.ndarray:
        article_ids = np.asarray(article_ids, dtype=np.int64)
        pos = np.searchsorted(self.ids[1:], article_ids) + 1
        pos = np.clip(pos, 1, len(self.ids) - 1)
        found = self.ids[pos] == article_ids
        return np.where(found, pos, UNKNOWN_IDX).astype(np.int64)

    def to(self, device) -> "ArticleTable":
        return ArticleTable(self.ids, {k: v.to(device) for k, v in self.tensors.items()}, self.report)


def build_article_table(articles: pd.DataFrame, pre: Preprocessor, text_ids: np.ndarray, text_vecs: np.ndarray,
                        image_ids: np.ndarray, image_vecs: np.ndarray, vector_dtype=torch.float16) -> ArticleTable:
    s = pre.state
    articles = articles.sort_values("article_id").reset_index(drop=True)
    ids = np.concatenate([[-1], articles.article_id.to_numpy(np.int64)])
    n = len(ids)

    def pad(values):
        return np.concatenate([[0], values])

    pub = pad(to_seconds(articles.published_time))
    pub[0] = NO_PUB
    t = {}
    t["pub_s"] = torch.from_numpy(pub)
    t["pub_missing"] = torch.from_numpy(pub == NO_PUB)
    t["known"] = torch.from_numpy(np.arange(n) > 0)
    t["category"] = torch.from_numpy(pad(_index(articles.category.tolist(), s["category_vocab"])))
    t["article_type"] = torch.from_numpy(pad(_index(articles.article_type.tolist(), s["article_type_vocab"])))
    t["sentiment"] = torch.from_numpy(pad(_index(articles.sentiment_label.tolist(), s["sentiment_vocab"])))
    premium = articles.premium.map(lambda v: np.nan if v is None or pd.isna(v) else float(bool(v))).to_numpy("float64")
    t["premium"] = torch.from_numpy(pad(np.nan_to_num(premium, nan=0.0)).astype(np.float32))
    t["premium_missing"] = torch.from_numpy(pad(np.isnan(premium)).astype(np.float32))
    t["premium_missing"][0] = 1.0
    body = np.log1p(articles.body_word_count.to_numpy("float64"))
    body_z = (body - s["body_log_mean"]) / s["body_log_std"]
    t["body_z"] = torch.from_numpy(pad(np.nan_to_num(body_z, nan=0.0)).astype(np.float32))
    t["body_missing"] = torch.from_numpy(pad(np.isnan(body)).astype(np.float32))
    t["body_missing"][0] = 1.0
    topic_lists = [[str(x) for x in _as_list(v) if x is not None] for v in articles.topics]
    topic_lists = [list(dict.fromkeys(v)) for v in topic_lists]  # unique, order kept
    tc = np.log1p(np.array([len(v) for v in topic_lists], dtype="float64"))
    t["topic_count_z"] = torch.from_numpy(pad((tc - s["topic_count_log_mean"]) / s["topic_count_log_std"]).astype(np.float32))
    # Embedding-bag inputs (0 = pad, 1 = unknown topic, 2.. vocab)
    topic_lookup = {v: i + 2 for i, v in enumerate(s["topic_vocab"])}
    tp = np.zeros((n, s["max_topics"]), dtype=np.int64)
    for i, lst in enumerate(topic_lists, start=1):
        for j, topic in enumerate(lst[: s["max_topics"]]):
            tp[i, j] = topic_lookup.get(topic, 1)
    t["topics_pad"] = torch.from_numpy(tp)
    sub_lookup = {v: i + 2 for i, v in enumerate(s["subcategory_vocab"])}
    sp = np.zeros((n, s["max_subcats"]), dtype=np.int64)
    for i, v in enumerate(articles.subcategory, start=1):
        for j, sub in enumerate(_as_list(v)[: s["max_subcats"]]):
            sp[i, j] = sub_lookup.get(str(sub), 1)
    t["subcats_pad"] = torch.from_numpy(sp)
    # Cross-feature columns over the full value sets (exact Step 3 semantics).
    cat_cross = {v: i for i, v in enumerate(s["cross_categories"])}
    type_cross = {v: i for i, v in enumerate(s["cross_types"])}
    topic_cross = {v: i for i, v in enumerate(s["cross_topics"])}
    cc = np.full(n, -1, dtype=np.int64)
    cc[1:] = [cat_cross.get(_nominal(v), -1) for v in articles.category]
    tcx = np.full(n, -1, dtype=np.int64)
    tcx[1:] = [type_cross.get(_nominal(v), -1) for v in articles.article_type]
    mh = np.zeros((n, max(len(topic_cross), 1)), dtype=np.float16)
    ntop = np.zeros(n, dtype=np.float32)
    for i, lst in enumerate(topic_lists, start=1):
        cols = [topic_cross[x] for x in lst if x in topic_cross]
        mh[i, cols] = 1.0
        ntop[i] = len(lst)
    t["cross_category"] = torch.from_numpy(cc)
    t["cross_type"] = torch.from_numpy(tcx)
    t["topic_multi_hot"] = torch.from_numpy(mh)
    t["n_topics"] = torch.from_numpy(ntop)

    # Frozen vectors, stored compactly with pointers (-1 = no usable vector).
    def attach(name, vid, vec):
        ptr = np.full(n, -1, dtype=np.int64)
        if len(vid):
            pos = np.searchsorted(ids[1:], vid) + 1
            pos = np.clip(pos, 1, n - 1)
            ok = ids[pos] == vid
            ptr[pos[ok]] = np.arange(len(vid))[ok]
        norms = np.linalg.norm(vec, axis=1, keepdims=True) if len(vec) else np.zeros((0, 1))
        unit = (vec / np.where(norms > 0, norms, 1.0)).astype(np.float32) if len(vec) else vec
        t[f"{name}_ptr"] = torch.from_numpy(ptr)
        t[f"{name}_vectors"] = torch.from_numpy(unit).to(vector_dtype) if len(vec) else torch.zeros((1, 1), dtype=vector_dtype)
        t[f"has_{name}"] = torch.from_numpy(ptr >= 0)
        return int((ptr >= 0).sum())

    covered_text = attach("text", text_ids, text_vecs)
    covered_image = attach("image", image_ids, image_vecs)
    for key, value in t.items():  # guard against dtype promotion by padding
        if value.dtype == torch.float64:
            t[key] = value.float()
    report = {"articles": n - 1, "text_vectors_attached": covered_text, "image_vectors_attached": covered_image,
              "published_time_missing": int((pub[1:] == NO_PUB).sum()),
              "text_dim": int(text_vecs.shape[1]) if len(text_vecs) else 0,
              "image_dim": int(image_vecs.shape[1]) if len(image_vecs) else 0}
    return ArticleTable(ids, t, report)


# ============================================================ split context ==

def REQUEST_DENSE_COLUMNS(pre: Preprocessor) -> list[str]:
    devices = [f"device_{d}" for d in pre.state["device_vocab"]] + ["device_missing", "device_unknown"]
    return devices + ["is_sso_user", "sso_missing", "is_subscriber", "subscriber_missing",
                      "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
                      "log_history_size", "history_empty", "log_median_read_time", "read_time_missing",
                      "mean_scroll_frac", "scroll_missing", "log_history_clicks_7d",
                      "log_hours_since_last_click", "last_click_missing"]


REQUEST_SCALED = {"log_history_size", "log_median_read_time", "mean_scroll_frac", "log_history_clicks_7d",
                  "log_hours_since_last_click"}

PAIR_DENSE_COLUMNS = ["log_article_age_h", "article_age_missing",
                      "access_premium_subscriber", "access_premium_non_subscriber",
                      "access_standard_subscriber", "access_standard_non_subscriber", "access_missing",
                      "category_preference", "category_preference_missing",
                      "article_type_preference", "article_type_preference_missing",
                      "topic_preference", "topic_preference_missing", "previously_clicked"]
PAIR_SCALED = {"log_article_age_h", "category_preference", "article_type_preference", "topic_preference"}


def _optional_bool(series) -> tuple[np.ndarray, np.ndarray]:
    values = pd.Series(series)
    missing = values.isna().to_numpy()
    return np.where(missing, 0.0, values.fillna(False).astype(bool)).astype(np.float32), missing.astype(np.float32)


@dataclass
class SplitContext:
    """Per-split request/history tensors plus the split's catalogue."""
    split: str
    requests: pd.DataFrame          # behaviour_id, user_id, impression_id, impression_time, ...
    tensors: dict                   # torch tensors (moved with .to())
    catalogue_idx: np.ndarray       # global indices of the split catalogue, ascending article_id
    recorded: dict                  # offsets/article_idx/labels/article_id for recorded slates
    report: dict

    def to(self, device) -> "SplitContext":
        return SplitContext(self.split, self.requests, {k: v.to(device) for k, v in self.tensors.items()},
                            self.catalogue_idx, self.recorded, self.report)


def build_split_context(split: str, behaviours: pd.DataFrame, events: pd.DataFrame, history_users: np.ndarray,
                        art: ArticleTable, pre: Preprocessor, catalogue_ids: np.ndarray,
                        pairs: pd.DataFrame | None) -> SplitContext:
    s = pre.state
    H = s["history_len"]
    req = behaviours.reset_index(drop=True).copy()
    R = len(req)
    users = np.asarray(history_users, dtype=np.int64)
    missing_users = np.setdiff1d(req.user_id.to_numpy(np.int64), users)
    if missing_users.size:
        # Not dropped: these requests get an empty history (with an explicit flag).
        users = np.union1d(users, missing_users)
    U = len(users)
    user_row = np.searchsorted(users, req.user_id.to_numpy(np.int64))
    req_s = to_seconds(req.impression_time)

    # ---- history events -> global indices, per-user profile, last-H items ----
    ev = events.copy()
    ev["row"] = np.searchsorted(users, ev.user_id.to_numpy(np.int64))
    ev["art"] = art.lookup(ev.article_id.to_numpy(np.int64))
    ev["t"] = to_seconds(ev.time)
    ev = ev.sort_values(["row", "t", "event_index"], kind="stable").reset_index(drop=True)
    counts = np.bincount(ev.row.to_numpy(), minlength=U)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    size = counts.astype(np.float32)

    cat_cols = art.tensors["cross_category"].numpy()[ev.art.to_numpy()]
    type_cols = art.tensors["cross_type"].numpy()[ev.art.to_numpy()]
    n_cat, n_type = len(s["cross_categories"]), len(s["cross_types"])
    mh = art.tensors["topic_multi_hot"].numpy()
    denom = np.maximum(size, 1.0)[:, None]
    cat_frac = np.zeros((U, max(n_cat, 1)), np.float32)
    ok = cat_cols >= 0
    np.add.at(cat_frac, (ev.row.to_numpy()[ok], cat_cols[ok]), 1.0)
    type_frac = np.zeros((U, max(n_type, 1)), np.float32)
    ok = type_cols >= 0
    np.add.at(type_frac, (ev.row.to_numpy()[ok], type_cols[ok]), 1.0)
    topic_frac = np.zeros((U, mh.shape[1]), np.float32)
    np.add.at(topic_frac, ev.row.to_numpy(), mh[ev.art.to_numpy()].astype(np.float32))
    cat_frac, type_frac, topic_frac = cat_frac / denom, type_frac / denom, topic_frac / denom

    grouped = ev.groupby("row")
    median_read = grouped.read_time.median().reindex(range(U)).to_numpy("float64")
    mean_scroll = grouped.scroll.mean().reindex(range(U)).to_numpy("float64")
    last_t = grouped.t.max().reindex(range(U)).to_numpy("float64")

    # Last H events per user (most recent by time; deterministic tie order).
    hist_idx = np.zeros((U, H), np.int64)
    hist_t = np.zeros((U, H), np.int64)
    hist_read = np.zeros((U, H), np.int64)
    hist_scroll = np.zeros((U, H), np.int64)
    hist_mask = np.zeros((U, H), bool)
    rb = _bin(ev.read_time.to_numpy("float64"), s["read_time_edges"])
    sb = _bin(ev.scroll.to_numpy("float64"), s["scroll_edges"])
    ev_art, ev_t = ev.art.to_numpy(), ev.t.to_numpy()
    for u in range(U):
        lo, hi = offsets[u], offsets[u + 1]
        lo = max(lo, hi - H)
        k = hi - lo
        if k:
            hist_idx[u, :k] = ev_art[lo:hi]
            hist_t[u, :k] = ev_t[lo:hi]
            hist_read[u, :k] = rb[lo:hi]
            hist_scroll[u, :k] = sb[lo:hi]
            hist_mask[u, :k] = True
    # An unknown history article keeps its slot (mask True) with the unknown embedding.

    # Membership keys over the *full* fixed history (for previously_clicked).
    keys = np.unique(ev.row.to_numpy(np.int64) * (len(art.ids) + 1) + ev_art.astype(np.int64))

    # ---- request dense features -------------------------------------------
    dev_vocab = s["device_vocab"]
    dev = req["device_type"].map(_nominal) if "device_type" in req else pd.Series([None] * R)
    dev_cols = np.zeros((R, len(dev_vocab) + 2), np.float32)
    for i, v in enumerate(dev):
        if v is None:
            dev_cols[i, len(dev_vocab)] = 1
        elif v in dev_vocab:
            dev_cols[i, dev_vocab.index(v)] = 1
        else:
            dev_cols[i, len(dev_vocab) + 1] = 1
    sso, sso_m = _optional_bool(req["is_sso_user"] if "is_sso_user" in req else [None] * R)
    sub, sub_m = _optional_bool(req["is_subscriber"] if "is_subscriber" in req else [None] * R)
    ts = pd.to_datetime(req.impression_time)
    hour = (ts.dt.hour + ts.dt.minute / 60.0).to_numpy("float64")
    dow = ts.dt.day_of_week.to_numpy("float64")
    # 7-day clicks: fixed-history events in [t - 7d, t).
    big = np.int64(1) << np.int64(34)
    ev_keys = ev.row.to_numpy(np.int64) * big + ev_t.astype(np.int64)
    upper = np.searchsorted(ev_keys, user_row.astype(np.int64) * big + req_s, side="left")
    lower = np.searchsorted(ev_keys, user_row.astype(np.int64) * big + (req_s - 7 * 86400), side="left")
    clicks7 = (upper - lower).astype(np.float64)
    usize = size[user_row].astype("float64")
    mread = median_read[user_row]
    mscroll = mean_scroll[user_row]
    since = (req_s - last_t[user_row]) / 3600.0
    dense = np.column_stack([
        dev_cols, sso, sso_m, sub, sub_m,
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), (dow >= 5).astype(np.float32),
        np.log1p(usize), (usize == 0).astype(np.float32),
        np.nan_to_num(np.log1p(mread), nan=0.0), np.isnan(mread).astype(np.float32),
        np.nan_to_num(mscroll / 100.0, nan=0.0), np.isnan(mscroll).astype(np.float32),
        np.log1p(clicks7),
        np.nan_to_num(np.log1p(np.clip(since, 0, None)), nan=0.0), np.isnan(since).astype(np.float32),
    ]).astype(np.float32)
    columns = REQUEST_DENSE_COLUMNS(pre)
    assert dense.shape[1] == len(columns), (dense.shape, len(columns))
    if not np.isfinite(dense).all():
        raise ValueError(f"{split}: non-finite request features")

    tensors = {
        "req_s": torch.from_numpy(req_s), "req_user": torch.from_numpy(user_row.astype(np.int64)),
        "req_dense_raw": torch.from_numpy(dense), "req_subscriber": torch.from_numpy(sub),
        "req_subscriber_missing": torch.from_numpy(sub_m),
        "hist_idx": torch.from_numpy(hist_idx), "hist_s": torch.from_numpy(hist_t),
        "hist_read_bin": torch.from_numpy(hist_read), "hist_scroll_bin": torch.from_numpy(hist_scroll),
        "hist_mask": torch.from_numpy(hist_mask),
        "user_cat_frac": torch.from_numpy(cat_frac), "user_type_frac": torch.from_numpy(type_frac),
        "user_topic_frac": torch.from_numpy(topic_frac),
        "user_hist_size": torch.from_numpy(size), "hist_keys": torch.from_numpy(keys),
    }

    # ---- recorded slates (training pairs / diagnostics) ---------------------
    recorded = {}
    if pairs is not None:
        key = pd.MultiIndex.from_frame(req[["user_id", "impression_id"]])
        pos = key.get_indexer(pd.MultiIndex.from_frame(pairs[["user_id", "impression_id"]]))
        if (pos < 0).any():
            raise ValueError(f"{split}: recorded pairs reference unknown requests")
        order = np.lexsort((pairs.candidate_article_id.to_numpy(), pos))
        pos = pos[order]
        cand = pairs.candidate_article_id.to_numpy(np.int64)[order]
        labels = pairs.clicked.to_numpy("float64")[order]
        recorded = {"req_pos": pos.astype(np.int64), "article_id": cand, "art_idx": art.lookup(cand),
                    "label": labels, "offsets": np.concatenate([[0], np.cumsum(np.bincount(pos, minlength=R))])}

    cat_idx = art.lookup(catalogue_ids)
    if (cat_idx == UNKNOWN_IDX).any():
        raise ValueError(f"{split}: catalogue IDs missing from the article table")
    report = {"requests": R, "users": U, "users_without_history_row": int(missing_users.size),
              "catalogue_articles": int(len(cat_idx)),
              "history_events": int(len(ev)), "history_unknown_articles": int((ev.art == 0).sum()),
              "recorded_pairs": int(len(recorded.get("req_pos", []))),
              "recorded_unknown_articles": int((recorded.get("art_idx", np.ones(1)) == 0).sum()) if recorded else 0}
    return SplitContext(split, req, tensors, np.sort(cat_idx), recorded, report)


# ====================================================== pair dense (torch) ==

def pair_dense(ctx_t: dict, art_t: dict, req_pos: torch.Tensor, art_idx: torch.Tensor, n_articles: int) -> torch.Tensor:
    """Raw (unscaled) pair features for aligned (request, article) index pairs."""
    t_req = ctx_t["req_s"][req_pos]
    pub = art_t["pub_s"][art_idx]
    missing = art_t["pub_missing"][art_idx]
    age_h = (t_req - torch.where(missing, t_req, pub)).to(torch.float32) / 3600.0
    log_age = torch.log1p(age_h.clamp_min(0.0)) * (~missing)
    sub = ctx_t["req_subscriber"][req_pos]
    sub_missing = ctx_t["req_subscriber_missing"][req_pos] > 0
    prem = art_t["premium"][art_idx]
    prem_missing = art_t["premium_missing"][art_idx] > 0
    acc_missing = (sub_missing | prem_missing).float()
    known = 1.0 - acc_missing
    acc = torch.stack([prem * sub, prem * (1 - sub), (1 - prem) * sub, (1 - prem) * (1 - sub)], dim=1) * known[:, None]
    user = ctx_t["req_user"][req_pos]
    cat = art_t["cross_category"][art_idx]
    cat_ok = cat >= 0
    cat_pref = ctx_t["user_cat_frac"][user, cat.clamp_min(0)] * cat_ok
    typ = art_t["cross_type"][art_idx]
    typ_ok = typ >= 0
    typ_pref = ctx_t["user_type_frac"][user, typ.clamp_min(0)] * typ_ok
    ntop = art_t["n_topics"][art_idx]
    top_ok = ntop > 0
    topic_sum = (ctx_t["user_topic_frac"][user].float() * art_t["topic_multi_hot"][art_idx].float()).sum(dim=1)
    topic_pref = torch.where(top_ok, topic_sum / ntop.clamp_min(1.0), torch.zeros_like(topic_sum))
    keys = ctx_t["hist_keys"]
    query = user.to(torch.int64) * (n_articles + 1) + art_idx.to(torch.int64)
    if keys.numel():
        loc = torch.searchsorted(keys, query).clamp_max(keys.numel() - 1)
        previously = (keys[loc] == query) & (art_idx > 0)
    else:
        previously = torch.zeros_like(query, dtype=torch.bool)
    return torch.cat([
        log_age[:, None], missing.float()[:, None], acc, acc_missing[:, None],
        cat_pref[:, None], (~cat_ok).float()[:, None], typ_pref[:, None], (~typ_ok).float()[:, None],
        topic_pref[:, None], (~top_ok).float()[:, None], previously.float()[:, None],
    ], dim=1)


def scaled(raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, scaled_mask: torch.Tensor) -> torch.Tensor:
    return torch.where(scaled_mask, (raw - mean) / std, raw)


def fit_dense_scalers(pre: Preprocessor, ctx: SplitContext, art: ArticleTable, chunk: int = 1 << 18) -> None:
    """Fit request and pair standardisation on TRAIN only (recorded pairs for pair
    features; train requests for request features). Flags remain 0/1."""
    cols = REQUEST_DENSE_COLUMNS(pre)
    raw = ctx.tensors["req_dense_raw"].numpy().astype("float64")
    mask = np.array([c in REQUEST_SCALED for c in cols])
    mean = np.where(mask, raw.mean(0), 0.0)
    std = np.where(mask, raw.std(0), 1.0)
    pre.set_dense_scaler("request", mean, std, cols)
    total = np.zeros(len(PAIR_DENSE_COLUMNS))
    total_sq = np.zeros(len(PAIR_DENSE_COLUMNS))
    n = 0
    rp = torch.from_numpy(ctx.recorded["req_pos"])
    ai = torch.from_numpy(ctx.recorded["art_idx"])
    for lo in range(0, len(rp), chunk):
        block = pair_dense(ctx.tensors, art.tensors, rp[lo:lo + chunk], ai[lo:lo + chunk], len(art.ids)).double().numpy()
        total += block.sum(0)
        total_sq += (block ** 2).sum(0)
        n += len(block)
    pmask = np.array([c in PAIR_SCALED for c in PAIR_DENSE_COLUMNS])
    pmean = total / max(n, 1)
    pstd = np.sqrt(np.maximum(total_sq / max(n, 1) - pmean ** 2, 0))
    pre.set_dense_scaler("pair", np.where(pmask, pmean, 0.0), np.where(pmask, pstd, 1.0), PAIR_DENSE_COLUMNS)
    pre.state["request_scaled_columns"] = [c for c in cols if c in REQUEST_SCALED]
    pre.state["pair_scaled_columns"] = [c for c in PAIR_DENSE_COLUMNS if c in PAIR_SCALED]


def consistency_with_shared_features(feature_path, ctx: SplitContext, art: ArticleTable, max_rows: int = 200_000) -> dict:
    """Compare our request-time features with the shared Step 3 feature table on
    the same recorded pairs (a check of alignment and cutoffs, not an input)."""
    import pyarrow.parquet as pq

    cols = ["user_id", "impression_id", "candidate_article_id", "article_age_h", "history_size",
            "history_clicks_7d", "candidate_category_preference", "candidate_article_type_preference",
            "candidate_topic_preference"]
    pf = pq.ParquetFile(feature_path)
    got = []
    n = 0
    for batch in pf.iter_batches(batch_size=65536, columns=cols):
        got.append(batch.to_pandas())
        n += batch.num_rows
        if n >= max_rows:
            break
    shared = pd.concat(got, ignore_index=True).head(max_rows)
    key = pd.MultiIndex.from_frame(ctx.requests[["user_id", "impression_id"]])
    pos = key.get_indexer(pd.MultiIndex.from_frame(shared[["user_id", "impression_id"]].astype("int64")))
    ai = art.lookup(shared.candidate_article_id.to_numpy(np.int64))
    raw = pair_dense(ctx.tensors, art.tensors, torch.from_numpy(pos), torch.from_numpy(ai), len(art.ids)).numpy()
    dense = ctx.tensors["req_dense_raw"].numpy()[pos]

    def diff(ours, theirs):
        theirs = pd.to_numeric(theirs, errors="coerce").to_numpy("float64")
        ok = np.isfinite(theirs) & np.isfinite(ours)
        if not ok.any():
            return {"compared": 0}
        d = np.abs(ours[ok] - theirs[ok])
        return {"compared": int(ok.sum()), "max_abs_diff": float(d.max()), "share_gt_1e-3": float((d > 1e-3).mean())}

    age_ours = np.expm1(raw[:, 0])
    age_ours[raw[:, 1] > 0] = np.nan
    return {
        "rows_compared": int(len(shared)),
        "article_age_h": diff(age_ours, shared.article_age_h.where(shared.article_age_h > 0)),
        "history_size": diff(ctx.tensors["user_hist_size"].numpy()[ctx.tensors["req_user"].numpy()[pos]].astype("float64"),
                             shared.history_size),
        "candidate_category_preference": diff(raw[:, 7].astype("float64"), shared.candidate_category_preference),
        "candidate_article_type_preference": diff(raw[:, 9].astype("float64"), shared.candidate_article_type_preference),
        "candidate_topic_preference": diff(raw[:, 11].astype("float64"), shared.candidate_topic_preference),
        "history_clicks_7d": diff(np.expm1(dense[:, -3].astype("float64")), shared.history_clicks_7d),
        "note": "article_age_h compared where both are positive; our age is clipped at 0 for future-published "
                "recorded candidates (never eligible in the catalogue).",
    }
