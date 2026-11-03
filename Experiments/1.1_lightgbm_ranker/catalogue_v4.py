"""Catalogue-scoring helpers for experiment 1.1 (export schema v4).

Rebuilds the shared engineered features (Preprocessing/step3_feature_engineering.ipynb, Jianhao) for ANY
article in the eligible catalogue, so a model fitted on recorded slates can score articles that were never
shown. Definitions follow the shared notebook exactly; `compare_to_shared` proves this on recorded rows.

Not rebuilt (pool-dependent, undefined for an article outside a slate, so they are removed from the model):
  slate_size, slate_relative_age_h
"""
import hashlib, math, statistics
from bisect import bisect_left
from collections import Counter, defaultdict
import numpy as np, pandas as pd, pyarrow.dataset as ds, scipy.sparse as sp

POOL_DEPENDENT = ["slate_size", "slate_relative_age_h"]
POLICY = "published_at_or_before_request_v1"


def nominal(v):
    if v is None or v is pd.NA or (isinstance(v, (float, np.floating)) and math.isnan(v)):
        return None
    return str(v)


def valid_number(v, upper=None):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or x < 0 or (upper is not None and x > upper):
        return None
    return x


def as_seq(v):
    if v is None or v is pd.NA or (isinstance(v, (float, np.floating)) and math.isnan(v)):
        return ()
    return tuple(v)


# ------------------------------------------------------------------ catalogue
class Catalogue:
    """Union catalogue (first source record wins) with the article fields the shared features use."""

    def __init__(self, article_paths, state):
        frames = []
        for p in article_paths:
            d = pd.read_parquet(p, columns=["article_id", "category", "article_type", "premium",
                                            "published_time", "topics", "body"])
            d["category"] = d["category"].map(nominal)
            d["article_type"] = d["article_type"].map(nominal)
            d["premium"] = d["premium"].map(lambda v: None if v is None or pd.isna(v) else bool(v))
            d["published_time"] = pd.to_datetime(d["published_time"])
            d["topics"] = d["topics"].map(lambda v: tuple(dict.fromkeys(
                str(t) for t in as_seq(v) if t is not None and not pd.isna(t))))
            d["body_word_count"] = d["body"].map(
                lambda s: len(s.split()) if isinstance(s, str) and s.strip() else None)
            frames.append(d.drop(columns="body"))
        a = pd.concat(frames, ignore_index=True).drop_duplicates("article_id", keep="first").reset_index(drop=True)
        self.n = len(a)
        self.ids = a.article_id.to_numpy("int64")
        self.pos = pd.Series(np.arange(self.n), index=self.ids)
        self.published = a.published_time.to_numpy("datetime64[us]")
        self.premium = a.premium.map(lambda v: np.nan if v is None else float(v)).to_numpy("float64")
        self.body_wc = pd.to_numeric(a.body_word_count, errors="coerce").to_numpy("float64")
        self.topic_count = a.topics.map(len).to_numpy("int64")
        self.cat_names = sorted({c for c in a.category if c is not None})
        self.type_names = sorted({c for c in a.article_type if c is not None})
        self.cat_g = a.category.map({c: i for i, c in enumerate(self.cat_names)}).fillna(-1).astype(int).to_numpy()
        self.type_g = a.article_type.map({c: i for i, c in enumerate(self.type_names)}).fillna(-1).astype(int).to_numpy()
        topics_all = sorted({t for ts in a.topics for t in ts})
        self.topic_names = topics_all
        self.topic_g = {t: i for i, t in enumerate(topics_all)}
        rows, cols = [], []
        for i, ts in enumerate(a.topics):
            for t in ts:
                rows.append(i); cols.append(self.topic_g[t])
        self.T = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(self.n, len(topics_all)))
        self.topics = list(a.topics)
        self.cat_str = list(a.category); self.type_str = list(a.article_type)
        # shared one-hot vocabularies (fitted on train by the shared pipeline)
        self.state = state
        for field, strs in (("category", self.cat_str), ("article_type", self.type_str)):
            known = {t: i for i, t in enumerate(state[field])}
            idx = np.array([-2 if s is None else known.get(s, -1) for s in strs])  # -2 missing, -1 unknown
            setattr(self, f"{field}_ohe", idx)
        self.fingerprint = hashlib.sha256(np.sort(self.ids).tobytes()).hexdigest()[:16]

    def eligible(self, t):
        """published_at_or_before_request_v1: known publication time and published_time <= request time."""
        pub = self.published
        return np.flatnonzero(~np.isnat(pub) & (pub <= np.datetime64(t, "us")))


# ------------------------------------------------------------------ user profiles
class Profile:
    __slots__ = ("size", "med_body", "med_read", "mean_scroll", "cat_frac", "type_frac", "topic_frac", "aff",
                 "same_read", "last5", "times", "hist")


def build_profiles(history_path, users, cat, state):
    """Per-user history profiles for `users`, following aggregate_history_features in the shared notebook."""
    users = [int(u) for u in users]
    t = ds.dataset(history_path).to_table(
        columns=["user_id", "article_id_fixed", "impression_time_fixed", "read_time_fixed", "scroll_percentage_fixed"],
        filter=ds.field("user_id").isin(users)).to_pandas()
    voc = {k: set(state[k]) for k in ("category", "topics", "article_type")}
    out = {}
    for row in t.itertuples(index=False):
        ids, times = as_seq(row.article_id_fixed), as_seq(row.impression_time_fixed)
        reads, scrolls = as_seq(row.read_time_fixed), as_seq(row.scroll_percentage_fixed)
        events = []
        for aid, rt, rd, sc in zip(ids, times, reads, scrolls):
            if rt is None or pd.isna(rt):
                continue
            events.append((pd.Timestamp(rt), int(aid), valid_number(rd), valid_number(sc, 100)))
        events.sort(key=lambda e: e[0])
        size = len(events)
        cats, tops, types = Counter(), Counter(), Counter()
        cw, creads = defaultdict(float), defaultdict(list)
        bodies, rts, scs = [], [], []
        latest = events[-1][0] if events else None
        for tm, aid, rd, sc in events:
            p = cat.pos.get(aid)
            if rd is not None: rts.append(rd)
            if sc is not None: scs.append(sc)
            if p is None:
                continue
            c, ty = cat.cat_str[p], cat.type_str[p]
            if c is not None:
                cats[c] += 1
                cw[c] += math.exp(-((latest - tm).total_seconds() / 3600.0) / 72.0)
                if rd is not None: creads[c].append(rd)
            if ty is not None: types[ty] += 1
            for tp in set(cat.topics[p]): tops[tp] += 1
            b = cat.body_wc[p]
            if not np.isnan(b) and b > 0: bodies.append(b)
        last5 = []
        for _, aid, _, _ in events[-5:]:
            p = cat.pos.get(aid)
            last5.append(frozenset(cat.topics[p]) if p is not None else frozenset())
        den = size or 1
        P = Profile()
        P.size = size
        P.med_body = float(statistics.median(bodies)) if bodies else np.nan
        P.med_read = float(statistics.median(rts)) if rts else np.nan
        P.mean_scroll = float(statistics.fmean(scs)) if scs else np.nan
        P.cat_frac = np.zeros(len(cat.cat_names)); P.type_frac = np.zeros(len(cat.type_names))
        P.topic_frac = np.zeros(len(cat.topic_names)); P.aff = np.full(len(cat.cat_names), np.nan)
        P.same_read = np.full(len(cat.cat_names), np.nan)
        for tok, c in cats.items(): P.cat_frac[cat.cat_names.index(tok)] = c / den
        for tok, c in types.items(): P.type_frac[cat.type_names.index(tok)] = c / den
        for tok, c in tops.items(): P.topic_frac[cat.topic_g[tok]] = c / den
        wt = sum(cw.values())
        if wt:
            P.aff[:] = 0.0
            for tok, w in cw.items(): P.aff[cat.cat_names.index(tok)] = w / wt
        for tok, v in creads.items(): P.same_read[cat.cat_names.index(tok)] = statistics.fmean(v)
        prev = [s for s in last5 if s]
        P.last5 = [np.array([cat.topic_g[x] for x in s if x in cat.topic_g], dtype=int) for s in prev]
        P.times = np.array([e[0].to_datetime64() for e in events], dtype="datetime64[us]")
        hist = {}
        for field, counts, prefix in (("category", cats, "hist_category_frac"), ("topics", tops, "hist_topic_frac"),
                                      ("article_type", types, "hist_article_type_frac")):
            for i, tok in enumerate(state[field]): hist[f"{prefix}_{i:03d}"] = counts[tok] / den
            hist[f"{prefix}_other"] = sum(c for tok, c in counts.items() if tok not in voc[field]) / den
        P.hist = hist
        out[int(row.user_id)] = P
    return out


# ------------------------------------------------------------------ request features
def request_features(cat, prof, ctx, ix, features):
    """Feature matrix (len(ix) x len(features), float32) for catalogue articles `ix` at one request.

    ctx: impression_time (np.datetime64), is_sso_user, is_subscriber, is_front_page, hour, day_of_week,
         plus the request's device one-hot columns (ohe_device_type_*).
    """
    n = len(ix)
    t = np.datetime64(ctx["impression_time"], "us")
    f = {}
    for k in ("is_sso_user", "is_subscriber", "is_front_page", "hour", "day_of_week"):
        f[k] = ctx[k]
    f["is_weekend"] = float(ctx["day_of_week"] >= 5)
    for k, v in ctx.items():
        if k.startswith("ohe_device_type_"): f[k] = v
    f["premium"] = cat.premium[ix]
    f["body_word_count"] = cat.body_wc[ix]
    tc = cat.topic_count[ix]
    f["topic_count"] = tc.astype(float); f["has_topics"] = (tc > 0).astype(float)
    age = (t - cat.published[ix]) / np.timedelta64(1, "s") / 3600.0
    age = np.where(np.isnan(age) | (age < 0), np.nan, age)
    f["article_age_h"] = age; f["article_age_missing"] = np.isnan(age).astype(float)
    f["is_24h_eligible"] = ((age >= 0) & (age <= 24)).astype(float)
    # history scalars
    f["history_size"] = float(prof.size)
    f["median_history_body_word_count"] = prof.med_body
    f["median_history_read_time"] = prof.med_read
    f["mean_history_scroll_percentage"] = prof.mean_scroll
    f["history_body_word_count_missing"] = float(np.isnan(prof.med_body))
    f["history_read_time_missing"] = float(np.isnan(prof.med_read))
    f["history_scroll_missing"] = float(np.isnan(prof.mean_scroll))
    f["history_clicks_7d"] = float(np.searchsorted(prof.times, t, "left") - np.searchsorted(prof.times, t - np.timedelta64(7, "D"), "left"))
    for k, v in prof.hist.items(): f[k] = v
    # candidate-history features
    cg, tg = cat.cat_g[ix], cat.type_g[ix]
    cmiss, tmiss = cg < 0, tg < 0
    cp = np.where(cmiss, np.nan, prof.cat_frac[np.maximum(cg, 0)])
    tp = np.where(tmiss, np.nan, prof.type_frac[np.maximum(tg, 0)])
    f["candidate_category_preference"] = cp; f["candidate_category_preference_missing"] = np.isnan(cp).astype(float)
    f["candidate_article_type_preference"] = tp; f["candidate_article_type_preference_missing"] = np.isnan(tp).astype(float)
    Tm = cat.T[ix]
    with np.errstate(invalid="ignore", divide="ignore"):
        topp = np.where(tc > 0, (Tm @ prof.topic_frac) / np.maximum(tc, 1), np.nan)
    f["candidate_topic_preference"] = topp; f["candidate_topic_preference_missing"] = np.isnan(topp).astype(float)
    has_aff = not np.all(np.isnan(prof.aff))
    aff = np.where(cmiss, np.nan, prof.aff[np.maximum(cg, 0)]) if has_aff else np.full(n, np.nan)
    f["category_affinity_72h"] = aff; f["category_affinity_missing"] = np.isnan(aff).astype(float)
    if prof.last5:
        P = sp.csr_matrix((np.ones(sum(len(p) for p in prof.last5), np.float32),
                           (np.repeat(np.arange(len(prof.last5)), [len(p) for p in prof.last5]),
                            np.concatenate(prof.last5) if any(len(p) for p in prof.last5) else np.array([], int))),
                          shape=(len(prof.last5), len(cat.topic_names)))
        inter = np.asarray((Tm @ P.T).todense())
        plen = np.asarray(P.sum(axis=1)).ravel()
        union = tc[:, None] + plen[None, :] - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            jac = np.where(tc > 0, (inter / union).mean(axis=1), np.nan)
    else:
        jac = np.full(n, np.nan)
    f["last5_mean_topic_jaccard"] = jac; f["topic_affinity_missing"] = np.isnan(jac).astype(float)
    f["has_any_topic_overlap"] = np.where(np.isnan(jac), np.nan, (jac > 0).astype(float))
    sr = np.where(cmiss, np.nan, prof.same_read[np.maximum(cg, 0)])
    f["same_category_mean_read_time"] = sr; f["same_category_read_depth_missing"] = np.isnan(sr).astype(float)
    # one-hots (shared vocabulary)
    for field, idx_arr in (("category", cat.category_ohe[ix]), ("article_type", cat.article_type_ohe[ix])):
        for i in range(len(cat.state[field])): f[f"ohe_{field}_{i:03d}"] = (idx_arr == i).astype(float)
        f[f"ohe_{field}_missing"] = (idx_arr == -2).astype(float)
        f[f"ohe_{field}_unknown"] = (idx_arr == -1).astype(float)
    sub = ctx["is_subscriber"]
    prem = cat.premium[ix]
    vocab = cat.state["premium_subscriber"]
    token_idx = np.full(n, -2)
    if not (isinstance(sub, float) and np.isnan(sub)):
        for j, tok in enumerate(vocab):
            is_prem, is_sub = tok.startswith("premium"), tok.endswith("_subscriber") and not tok.endswith("non_subscriber")
            token_idx = np.where(~np.isnan(prem) & (prem == float(is_prem)) & (bool(sub) == is_sub), j, token_idx)
    for i in range(len(vocab)): f[f"ohe_premium_subscriber_{i:03d}"] = (token_idx == i).astype(float)
    f["ohe_premium_subscriber_missing"] = (token_idx == -2).astype(float)
    f["ohe_premium_subscriber_unknown"] = np.zeros(n)
    M = np.empty((n, len(features)), np.float32)
    for j, name in enumerate(features):
        M[:, j] = f[name]
    return M


# ------------------------------------------------------------------ requests, sampling, metrics
def behaviour_id(user, impression):
    return f"{int(user)}:{int(impression)}"


def hash_order(requests, split, seed=42):
    """Shared deterministic request sampling (3.1 policy): blake2b hash priority, then behaviour_id."""
    r = requests.copy()
    r["behaviour_id"] = [behaviour_id(u, i) for u, i in zip(r.user_id, r.impression_id)]
    r["_h"] = [hashlib.blake2b(f"{seed}:{split}:{b}".encode(), digest_size=8).hexdigest() for b in r.behaviour_id]
    return r.sort_values(["_h", "behaviour_id"]).drop(columns="_h").reset_index(drop=True)


def top_k(scores, ids, k):
    """Genuine top-k by descending score then ascending article id; returns (order, ties_cross_cutoff)."""
    n = len(scores)
    if n == 0:
        return np.array([], int), False
    if n > k:
        thr = np.partition(-scores, k - 1)[k - 1]
        cand = np.flatnonzero(-scores <= thr)
    else:
        cand = np.arange(n)
    order = cand[np.lexsort((ids[cand], -scores[cand]))]
    cross = len(order) > k and scores[order[k - 1]] == scores[order[k]]
    return order[:k], bool(cross)


def retrieval_metrics(ranked_ids, known, ks=(100, 200)):
    out = {}
    for k in ks:
        top = ranked_ids[:k]
        hit = np.array([a in known for a in top], dtype=float)
        out[f"recall@{k}"] = hit.sum() / len(known)
        dcg = (hit / np.log2(np.arange(2, len(hit) + 2))).sum()
        ideal = (1 / np.log2(np.arange(2, min(len(known), k) + 2))).sum()
        out[f"ndcg@{k}"] = dcg / ideal
    return out


def compare_to_shared(shared_rows, ours, features, tol=1e-4):
    """Per-column comparison of rebuilt features vs the shared file on identical candidate rows."""
    rep = []
    for j, name in enumerate(features):
        a, b = shared_rows[name].to_numpy("float64"), ours[:, j].astype("float64")
        both_nan = np.isnan(a) & np.isnan(b)
        bad = ~both_nan & ~(np.abs(a - b) <= tol * np.maximum(1.0, np.abs(a)))
        bad = bad | (np.isnan(a) ^ np.isnan(b))
        if bad.any():
            rep.append((name, int(bad.sum()), float(np.nanmax(np.abs(a - b)[bad])) if (~np.isnan(a - b))[bad].any() else np.nan))
    return rep
