"""Traditional-track rerun on the 20,738-article SMALL catalogue (shared rule since 5 Nov).

Refits LightGBM, XGBoost and logistic regression on recorded train slates, selects each by known-click Recall@100 on the
same 200 validation requests, exports all validation requests for all three, then exports the test holdout for the model
with the highest full-validation Recall@100. Test labels are never loaded. Finished export blocks are kept, so a failed run
resumes without repeating them.

  python rerun_small_catalogue.py                 # complete run
  python rerun_small_catalogue.py --dev           # tiny smoke test written to a scratch run id
"""
import argparse, json, os, pickle, platform, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent; EXP = HERE.parent; sys.path.insert(0, str(EXP / "1.1_lightgbm_ranker"))
import catalogue_v4 as cv, export_full_split as ef
sys.path.insert(0, str(HERE))
import project_paths

PATHS = project_paths.resolve(); SHARED, FEATURES_DIR, ARTICLES, HISTORY = PATHS.shared, PATHS.features, PATHS.articles, PATHS.history
SMALL_FP = "eff069a2586ad6cd"; EXPORT_K, BLOCK = 200, 1000; ROUNDS = 400
CHECK = sorted({*range(25, 201, 25), 300, 400})
FOLDERS = {"lightgbm": "1.1_lightgbm_ranker", "logistic": "1.2_logistic_regression", "xgboost": "1.3_xgboost"}
LOG_FEATURES = ["body_word_count", "history_size", "median_history_body_word_count", "median_history_read_time", "history_clicks_7d", "same_category_mean_read_time"]
TRIALS = dict(lightgbm={"leaves31": 31, "leaves63": 63, "leaves127": 127}, xgboost={"leaves31": 31, "leaves63": 63, "leaves127": 127}, logistic={"C0.01": 0.01, "C0.1": 0.1, "C1": 1.0})


def parse():
    p = argparse.ArgumentParser(); p.add_argument("--run-id", default="20261105_small_seed42"); p.add_argument("--workers", type=int, default=5); p.add_argument("--threads", type=int, default=3)
    p.add_argument("--dev", action="store_true"); p.add_argument("--train-only", action="store_true"); p.add_argument("--stage", default="all"); p.add_argument("--worker", type=int, default=-1); p.add_argument("--jobs", default="")
    a = p.parse_args()
    if a.dev: a.run_id = "_dev_small"
    a.sel_n, a.max_req, a.frac = (6, 40, 0.03) if a.dev else (200, 0, 0.0)
    return a


def out_dir(model, a): return EXP / FOLDERS[model] / "outputs" / a.run_id
def persist_dir(model, a): return (SHARED / "Experiments" / "_dev_scratch" / FOLDERS[model] / a.run_id) if a.dev else (SHARED / "Experiments" / FOLDERS[model] / a.run_id)
def log(a, m):
    line = time.strftime("%H:%M:%S ") + m; print(line, flush=True)
    with open(EXP / "1.3_xgboost" / "outputs" / f"{a.run_id}_progress.log", "a", encoding="utf-8") as f: f.write(line + "\n")


def catalogue():
    state = json.load(open(FEATURES_DIR / "feature_vocabulary.json", encoding="utf-8")); cat = cv.Catalogue([ARTICLES], state)
    assert cat.n == 20738 and cat.fingerprint == SMALL_FP, "not the shared SMALL catalogue"
    return state, cat


def features_list():
    skip = {"split", "user_id", "impression_id", "candidate_article_id", "current_page_article_id", "impression_time", "clicked", *cv.POOL_DEPENDENT}
    feat = [c for c in pq.ParquetFile(FEATURES_DIR / "train_features.parquet").schema_arrow.names if c not in skip and not c.endswith("_log1p")]
    assert len(feat) == 200
    return feat


def request_table(split, a):
    """Requests in shared hash order. For test, the click column is never read."""
    path = FEATURES_DIR / f"{split}_features.parquet"; names = pq.ParquetFile(path).schema_arrow.names
    use = ["user_id", "impression_id", "candidate_article_id", "impression_time", "is_sso_user", "is_subscriber", "is_front_page", "hour", "day_of_week"] + [c for c in names if c.startswith("ohe_device_type_")]
    if split != "test": use.append("clicked")
    d = pd.read_parquet(path, columns=use); assert not d.duplicated(use[:3]).any()
    d["behaviour_id"] = d.user_id.astype("int64").astype(str) + ":" + d.impression_id.astype("int64").astype(str)
    first = d.drop_duplicates("behaviour_id").set_index("behaviour_id"); reqs = cv.hash_order(first.reset_index()[["user_id", "impression_id"]], split, 42)
    if split == "test": reqs["known"] = [set() for _ in range(len(reqs))]
    else:
        k = d[d.clicked == 1].groupby("behaviour_id").candidate_article_id.apply(lambda s: set(s.astype("int64"))); reqs["known"] = reqs.behaviour_id.map(k).map(lambda s: s if isinstance(s, set) else set())
    return (reqs.head(a.max_req) if a.max_req else reqs), first


def lr_base(X, feat):
    ai, lx = feat.index("article_age_h"), [feat.index(c) for c in LOG_FEATURES]
    X = np.array(X, "float32"); g = X[:, ai].copy(); X[:, lx] = np.log1p(np.maximum(X[:, lx], 0)); X[:, ai] = 0
    return np.hstack([X, np.stack([g <= 6, (g > 6) & (g <= 24), (g > 24) & (g <= 72), (g > 72) & (g <= 168), g > 168], axis=1).astype("float32")])


def lr_transform(X, pr, feat):
    X = lr_base(X, feat); q = np.isnan(X); X[q] = np.take(pr["median"], np.where(q)[1])
    return np.clip((X - pr["mean"]) / pr["std"], -pr["clip"], pr["clip"])


def scorer(model, folder, feat, threads):
    if model == "lightgbm":
        import lightgbm as lgb; b = lgb.Booster(model_file=str(folder / "lightgbm_model.txt")); return lambda M: b.predict(M, raw_score=True, num_threads=threads)
    if model == "xgboost":
        import xgboost as xgb; b = xgb.Booster(); b.load_model(str(folder / "model.json")); b.set_param({"nthread": threads}); return lambda M: b.inplace_predict(M, predict_type="margin", missing=np.nan)
    pr = pickle.load(open(folder / "preprocessing.pkl", "rb")); m = pickle.load(open(folder / "model.pkl", "rb")); w, c = m.coef_[0].astype("float64"), float(m.intercept_[0])
    return lambda M: lr_transform(M, pr, feat).astype("float64") @ w + c


def rebuild_check(split, reqs, first, state, cat, feat, n=50):
    d = pd.read_parquet(FEATURES_DIR / f"{split}_features.parquet", columns=["user_id", "impression_id", "candidate_article_id"] + feat)
    d["b"] = d.user_id.astype("int64").astype(str) + ":" + d.impression_id.astype("int64").astype(str); hs = reqs.head(n)
    keep = d[d.b.isin(set(hs.behaviour_id))]; prof = cv.build_profiles(HISTORY[split], hs.user_id.unique(), cat, state); bad, rows = 0, 0
    for r in hs.itertuples():
        rec = keep[keep.b == r.behaviour_id]; ix = cat.pos.reindex(rec.candidate_article_id.to_numpy()).to_numpy().astype(int)
        bad += len(cv.compare_to_shared(rec[feat], cv.request_features(cat, prof[int(r.user_id)], ef.ctx_of(first.loc[r.behaviour_id]), ix, feat), feat)); rows += len(rec)
    assert bad == 0, f"{split}: rebuilt features differ"; return rows


def fit_stage(a):
    import lightgbm as lgb, xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    state, cat = catalogue(); feat = features_list(); reqs, first = request_table("validation", a); sel = reqs.head(a.sel_n)
    for sp in ("validation", "test"):
        r2, f2 = (reqs, first) if sp == "validation" else request_table("test", a); log(a, f"{sp}: feature rebuild matches shared file on {rebuild_check(sp, r2, f2, state, cat, feat)} rows")
    prof = cv.build_profiles(HISTORY["validation"], sel.user_id.unique(), cat, state)
    tr = pd.read_parquet(FEATURES_DIR / "train_features.parquet", columns=["user_id", "impression_id"] + feat + ["clicked"])
    if a.frac: bid = tr.user_id.astype(str) + ":" + tr.impression_id.astype(str); ids = bid.unique(); tr = tr[bid.isin(set(np.random.default_rng(42).choice(ids, int(len(ids) * a.frac), replace=False)))]
    X, y = tr[feat].to_numpy("float32"), tr.clicked.to_numpy(); del tr; rounds = 50 if a.dev else ROUNDS; check = [c for c in CHECK if c <= rounds] + ([rounds] if a.dev else [])
    base_l = dict(objective="binary", learning_rate=0.08, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=127, seed=42, verbosity=-1, deterministic=True, force_row_wise=True, metric="None", num_threads=14)
    base_x = dict(objective="binary:logistic", tree_method="hist", grow_policy="lossguide", max_depth=0, eta=0.08, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, max_bin=127, min_child_weight=5, seed=42, nthread=14, eval_metric="logloss")
    fits = {}; dx = lgb.Dataset(X, y, free_raw_data=False)
    for n, l in TRIALS["lightgbm"].items():
        t = time.time(); b = lgb.train({**base_l, "num_leaves": l}, dx, rounds); fits[("lightgbm", n)] = b; log(a, f"lightgbm {n} trained ({time.time()-t:.0f}s)")
    del dx; dm = xgb.DMatrix(X, label=y, missing=np.nan, nthread=14)
    for n, l in TRIALS["xgboost"].items():
        t = time.time(); b = xgb.train({**base_x, "max_leaves": l}, dm, rounds); fits[("xgboost", n)] = b; log(a, f"xgboost {n} trained ({time.time()-t:.0f}s)")
    del dm
    Xb = lr_base(X, feat); del X; med = np.nanmedian(Xb, axis=0); med = np.where(np.isnan(med), 0, med); q = np.isnan(Xb); Xb[q] = np.take(med, np.where(q)[1]); del q
    mu, sd = Xb.mean(axis=0), Xb.std(axis=0); sd = np.where(sd < 1e-8, 1.0, sd); bn = ((Xb == 0) | (Xb == 1)).all(axis=0); mu[bn], sd[bn] = 0.0, 1.0
    pr = dict(features=feat, extra_columns=["age_le_6h", "age_6_24h", "age_24_72h", "age_72_168h", "age_gt_168h"], log1p_features=LOG_FEATURES, median=med, mean=mu, std=sd, clip=5.0, fitted_on="train only")
    Xs = np.clip((Xb - mu) / sd, -5, 5).astype("float32"); del Xb
    for n, c in TRIALS["logistic"].items():
        t = time.time(); m = LogisticRegression(C=c, solver="lbfgs", max_iter=300, tol=1e-4, random_state=42).fit(Xs, y); assert m.n_iter_[0] < 300; fits[("logistic", n)] = m; log(a, f"logistic {n} fitted ({time.time()-t:.0f}s)")
    del Xs
    w = {n: (fits[("logistic", n)].coef_[0].astype("float64"), float(fits[("logistic", n)].intercept_[0])) for n in TRIALS["logistic"]}
    res = {}; t0 = time.time()
    for i, r in enumerate(sel.itertuples(), 1):
        row = first.loc[r.behaviour_id]; ix = cat.eligible(row.impression_time.to_datetime64()); ids = cat.ids[ix]; M = cv.request_features(cat, prof[int(r.user_id)], ef.ctx_of(row), ix, feat); Z = lr_transform(M, pr, feat).astype("float64")
        def add(k, s): o, c_ = cv.top_k(s, ids, EXPORT_K); res.setdefault(k, []).append({**cv.retrieval_metrics(ids[o], r.known, (100, 200)), "tie": int(c_)})
        for n in TRIALS["lightgbm"]:
            for k in check: add(("lightgbm", n, k), fits[("lightgbm", n)].predict(M, num_iteration=k, raw_score=True, num_threads=14))
        for n in TRIALS["xgboost"]:
            for k in check: add(("xgboost", n, k), fits[("xgboost", n)].inplace_predict(M, iteration_range=(0, k), predict_type="margin", missing=np.nan))
        for n, (wv, c) in w.items(): add(("logistic", n, 0), Z @ wv + c)
        if i % 50 == 0: log(a, f"selection {i}/{len(sel)} ({time.time()-t0:.0f}s)")
    rows = pd.DataFrame([dict(model=k[0], trial=k[1], checkpoint=k[2], **pd.DataFrame(v).mean().to_dict()) for k, v in res.items()])
    for model in TRIALS:
        o = out_dir(model, a); (o / "tuning").mkdir(parents=True, exist_ok=True); (o / "model").mkdir(parents=True, exist_ok=True); t = rows[rows.model == model].copy(); t.to_csv(o / "tuning" / "trials.csv", index=False)
        t["_o"] = t.trial.map({n: i for i, n in enumerate(TRIALS[model])}); win = t.sort_values(["recall@100", "ndcg@100", "_o", "checkpoint"], ascending=[False, False, True, True]).iloc[0]; n, k = win.trial, int(win.checkpoint)
        if model == "lightgbm": fits[(model, n)].save_model(str(o / "model/lightgbm_model.txt"), num_iteration=k)
        elif model == "xgboost": fits[(model, n)][:k].save_model(str(o / "model/model.json"))
        else: pickle.dump(fits[(model, n)], open(o / "model/model.pkl", "wb")); pickle.dump(pr, open(o / "model/preprocessing.pkl", "wb"))
        (o / "model" / "selection.json").write_text(json.dumps(dict(trial=n, trees=k, recall100=float(win["recall@100"]), ndcg100=float(win["ndcg@100"]), recall200=float(win["recall@200"]), selection_requests=len(sel)), indent=2))
        shutil.copy(EXP / "1.1_lightgbm_ranker/catalogue_v4.py", o / "model/catalogue_v4.py"); shutil.copy(Path(__file__), o / "model/rerun_small_catalogue.py"); shutil.copy(HERE / "project_paths.py", o / "model/project_paths.py"); shutil.copy(EXP / "1.1_lightgbm_ranker/export_full_split.py", o / "model/export_full_split.py")
        log(a, f"SELECTED {model}: {n}, trees {k}, Recall@100 {win['recall@100']:.4f}, nDCG@100 {win['ndcg@100']:.4f}")


def worker(a):
    state, cat = catalogue(); feat = features_list(); cache = {}
    for job in a.jobs.split(","):
        model, split = job.split(":"); o = out_dir(model, a); blocks = o / "_blocks" / split; blocks.mkdir(parents=True, exist_ok=True)
        cache.setdefault(split, request_table(split, a)); reqs, first = cache[split]; score = scorer(model, o / "model", feat, a.threads); n = int(np.ceil(len(reqs) / BLOCK))
        for b in range(n):
            done = blocks / f"manifest_{b:05d}.parquet"
            if b % a.workers != a.worker or done.exists(): continue
            part = reqs.iloc[b * BLOCK:(b + 1) * BLOCK]; prof = cv.build_profiles(HISTORY[split], part.user_id.unique(), cat, state); bids, ia, sa, man = [], [], [], []
            for r in part.itertuples():
                row = first.loc[r.behaviour_id]; ix = cat.eligible(row.impression_time.to_datetime64()); ids = cat.ids[ix]; sc = score(cv.request_features(cat, prof[int(r.user_id)], ef.ctx_of(row), ix, feat)); o2, cross = cv.top_k(sc, ids, EXPORT_K); top = ids[o2].astype("int64")
                rec = dict(behaviour_id=r.behaviour_id, impression_time=row.impression_time.to_datetime64(), eligible_pool=len(ix), shortlist_len=len(top), known_clicks=len(r.known), known_clicks_eligible=len(r.known & set(ids)), tie_at_cutoff=bool(cross))
                if r.known: rec.update(cv.retrieval_metrics(top, r.known, (100, 200)))
                bids.append(r.behaviour_id); ia.append(top); sa.append(np.asarray(sc[o2], "float64")); man.append(rec)
            ln = [len(x) for x in ia]
            pd.DataFrame({"behaviour_id": np.repeat(np.array(bids, dtype=object), ln), "article_id": np.concatenate(ia), "rank": np.concatenate([np.arange(1, x + 1) for x in ln]), "score": np.concatenate(sa)}).to_parquet(blocks / f"scores_{b:05d}.parquet", index=False)
            tmp = blocks / f"manifest_{b:05d}.tmp"; pd.DataFrame(man).to_parquet(tmp, index=False); os.replace(tmp, done)
            log(a, f"{model} {split} block {b + 1}/{n}")


def auc_slates(g, y, s):
    d = pd.DataFrame({"g": g, "y": y.astype(float), "s": s}); d["r"] = d.groupby("g", sort=False)["s"].rank(method="average"); t = d.assign(ry=d.r * d.y).groupby("g", sort=False).agg(n=("y", "size"), p=("y", "sum"), ry=("ry", "sum")); q = t.n - t.p; ok = (t.p > 0) & (q > 0)
    return float(((t.ry - t.p * (t.p + 1) / 2) / (t.p * q))[ok].mean())


def merge(a, model, split):
    state, cat = catalogue(); feat = features_list(); o = out_dir(model, a); reqs, first = request_table(split, a); n = int(np.ceil(len(reqs) / BLOCK)); bl = o / "_blocks" / split
    S = pd.concat([pd.read_parquet(bl / f"scores_{b:05d}.parquet") for b in range(n)], ignore_index=True); m = pd.concat([pd.read_parquet(bl / f"manifest_{b:05d}.parquet") for b in range(n)], ignore_index=True); S["click_probability"] = ef.sig(S.score.to_numpy()); assert np.isfinite(S.score).all()
    for d_ in (f"predictions/{split}", f"rankings/{split}", "diagnostics/recorded_slates"): (o / d_).mkdir(parents=True, exist_ok=True)
    S.to_parquet(o / f"predictions/{split}/submission_article_scores.parquet", index=False, compression="zstd"); m.to_parquet(o / f"predictions/{split}/request_manifest.parquet", index=False)
    lists = S.groupby("behaviour_id", sort=False).article_id.apply(lambda x: x.astype("int64").tolist()); pd.DataFrame({"behaviour_id": lists.index.to_numpy(), "article_ids": lists.to_list()}).to_parquet(o / f"rankings/{split}/submission_top_200.parquet", index=False, compression="zstd")
    R = pd.read_parquet(o / f"rankings/{split}/submission_top_200.parquet"); S2 = pd.read_parquet(o / f"predictions/{split}/submission_article_scores.parquet")
    assert len(R) == len(reqs) and set(R.behaviour_id) == set(reqs.behaviour_id) and (R.article_ids.map(len) == np.minimum(200, m.set_index("behaviour_id").loc[R.behaviour_id].eligible_pool.to_numpy())).all()
    assert np.array_equal(np.concatenate([np.asarray(x, "int64") for x in R.article_ids]), S2.article_id.to_numpy()) and not S2.duplicated(["behaviour_id", "article_id"]).any() and S2.groupby("behaviour_id", sort=False).score.apply(lambda x: bool((np.diff(x.to_numpy()) <= 0).all())).all()
    cols = ["user_id", "impression_id", "candidate_article_id"] + feat + ([] if split == "test" else ["clicked"]); d = pd.read_parquet(FEATURES_DIR / f"{split}_features.parquet", columns=cols); d["behaviour_id"] = d.user_id.astype("int64").astype(str) + ":" + d.impression_id.astype("int64").astype(str)
    s = scorer(model, o / "model", feat, 14)(d[feat].to_numpy("float32")); pd.DataFrame({"behaviour_id": d.behaviour_id, "article_id": d.candidate_article_id.astype("int64"), "score": s, "click_probability": ef.sig(np.asarray(s, "float64"))}).sort_values(["behaviour_id", "article_id"]).to_parquet(o / f"diagnostics/recorded_slates/{split}_scores.parquet", index=False, compression="zstd")
    res = dict(requests=len(R), score_rows=len(S2), ties_at_cutoff=int(m.tie_at_cutoff.sum()), median_eligible_pool=float(m.eligible_pool.median()), empty_pools=int((m.eligible_pool == 0).sum()))
    if split != "test":
        k = m[m.known_clicks > 0]; res.update({c: float(k[c].mean()) for c in ("recall@100", "ndcg@100", "recall@200", "ndcg@200")}); res.update(known_clicks=int(k.known_clicks.sum()), eligible_known_clicks=int(k.known_clicks_eligible.sum()), recorded_auc=auc_slates(pd.factorize(d.behaviour_id)[0], d.clicked.to_numpy(), np.asarray(s, "float64")))
    mf = o / "metrics.json"; all_m = json.loads(mf.read_text()) if mf.exists() else dict(note="development results; test labels were not read"); all_m[split] = res; mf.write_text(json.dumps(all_m, indent=2))
    log(a, f"MERGED {model} {split}: {res}"); return res


def finish(a, model, splits):
    o = out_dir(model, a); feat = features_list(); sel = json.load(open(o / "model/selection.json", encoding="utf-8")); git = lambda *x: subprocess.run(["git", *x], capture_output=True, text=True, cwd=HERE).stdout.strip()
    import lightgbm, xgboost, sklearn
    cfg = dict(experiment_id=FOLDERS[model], run_id=a.run_id, owner="Nigel Yap", seed=42, request_sampling_seed=42, export_schema_version=4, prediction_pool="time_eligible_catalogue", selection_k=100, export_k=EXPORT_K, shortlist_size=EXPORT_K, final_display_size=5, model=model, selected=sel,
        scope=dict(splits_exported=splits, in_sample_splits=[s for s in splits if s == "train"], all_requests_in_split=a.max_req == 0, sampled_run=bool(a.max_req), train_catalogue_export="train" in splits, training="all recorded train rows" if not a.frac else f"DEV {a.frac}"),
        eligibility=dict(policy=cv.POLICY, catalogue="EB-NeRD SMALL, shared rule from commit 57b3475", catalogue_articles=20738, catalogue_fingerprint=SMALL_FP, sources=["train_val/articles.parquet", "test/articles.parquet (SHA-256 b1cea52e..., identical file)"]),
        selection_rule=dict(primary="mean known-click Recall@100", secondary="mean observed-click nDCG@100", tie_rule="then earlier trial/checkpoint", partition="same 200 validation requests as earlier runs (development results)"), score_semantics=dict(score_type="logit", click_probability="sigmoid(score), uncalibrated"),
        features=feat, histories=dict(validation="small validation history", test="small validation history (project holdout)"), test_labels="not loaded", packages=dict(python=platform.python_version(), lightgbm=lightgbm.__version__, xgboost=xgboost.__version__, sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__), code_commit=git("rev-parse", "HEAD"), uncommitted_changes=bool(git("status", "--porcelain")), persistent_artifact_location=str(persist_dir(model, a)))
    (o / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str)); (o / "requirements.txt").write_text(f"lightgbm=={lightgbm.__version__}\nxgboost=={xgboost.__version__}\nscikit-learn=={sklearn.__version__}\nnumpy=={np.__version__}\npandas=={pd.__version__}\npyarrow\nscipy\n")
    p = persist_dir(model, a); shutil.copytree(o, p, dirs_exist_ok=True, ignore=shutil.ignore_patterns("_blocks"))
    state, cat = catalogue(); split = splits[0]; reqs, first = request_table(split, a); probe = reqs.tail(5); prof = cv.build_profiles(HISTORY[split], probe.user_id.unique(), cat, state); sc = scorer(model, p / "model", feat, 14); ex = pd.read_parquet(p / f"predictions/{split}/submission_article_scores.parquet"); worst = 0.0
    for r in probe.itertuples():
        row = first.loc[r.behaviour_id]; ix = cat.eligible(row.impression_time.to_datetime64()); s = sc(cv.request_features(cat, prof[int(r.user_id)], ef.ctx_of(row), ix, feat)); f = ex[ex.behaviour_id == r.behaviour_id].set_index("article_id").score; worst = max(worst, float(np.abs(pd.Series(s, index=cat.ids[ix]).reindex(f.index).to_numpy() - f.to_numpy()).max()))
    assert worst < 1e-5, worst; log(a, f"{model} reload check passed, max gap {worst}")


def run_workers(a, jobs, env):
    base = [sys.executable, str(Path(__file__).resolve()), "--run-id", a.run_id, "--workers", str(a.workers), "--threads", str(a.threads), "--jobs", jobs] + (["--dev"] if a.dev else [])
    ps = [subprocess.Popen(base + ["--worker", str(i)], env=env) for i in range(a.workers)]; codes = [p.wait() for p in ps]
    if any(codes): log(a, f"WORKER FAILED {codes}: rerun to resume"); sys.exit(1)


def main():
    a = parse(); (EXP / "1.3_xgboost" / "outputs").mkdir(exist_ok=True)
    if a.worker >= 0: return worker(a)
    if os.name == "nt":
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    env = {**os.environ, "OMP_NUM_THREADS": str(a.threads), "OPENBLAS_NUM_THREADS": str(a.threads), "MKL_NUM_THREADS": str(a.threads)}; t0 = time.time(); log(a, "start")
    if a.train_only:     # load-only: score train requests with the saved final XGBoost model (in-sample)
        run_workers(a, "xgboost:train", env); merge(a, "xgboost", "train"); finish(a, "xgboost", ["validation", "test", "train"]); log(a, f"train export finished minutes={(time.time() - t0) / 60:.1f}"); return
    if not all((out_dir(m, a) / "model" / "selection.json").exists() for m in FOLDERS): fit_stage(a)
    run_workers(a, ",".join(f"{m}:validation" for m in FOLDERS), env); res = {m: merge(a, m, "validation") for m in FOLDERS}
    best = max(res, key=lambda m: (res[m]["recall@100"], res[m]["ndcg@100"])); log(a, f"FINAL MODEL by full validation Recall@100: {best}")
    run_workers(a, f"{best}:test", env); merge(a, best, "test")
    for m in FOLDERS: finish(a, m, ["validation", "test"] if m == best else ["validation"])
    (EXP / "1.3_xgboost" / "outputs" / f"{a.run_id}_final.json").write_text(json.dumps(dict(final_model=best, validation=res), indent=2)); log(a, f"finished minutes={(time.time() - t0) / 60:.1f}")


if __name__ == "__main__":
    main()
