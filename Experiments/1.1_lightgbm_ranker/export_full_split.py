"""Load-only full catalogue export for experiment 1.1 (no training, no refitting).

Reloads the saved LightGBM model from an existing run and scores every request of one split against the
full time-eligible union catalogue. Work is split into fixed blocks of requests written atomically, so an
interrupted run resumes without repeating finished blocks. The merged result is checked against the earlier
sampled export and reloaded from its persistent copy.

Example:
  python export_full_split.py --split validation --workers 5 --threads 3
"""
import argparse, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import catalogue_v4 as cv

SHARED = Path(r"C:\Users\nigel\BT4222 Shared Final Folder")
SMALL = Path(r"C:\Users\nigel\bt4222-data\ebnerd_small")
TEST_ARTICLES = Path(r"C:\Users\nigel\OneDrive\Desktop\BT4222 Final\ebnerd_large\articles.parquet")
EXPORT_K, BLOCK = 200, 1000
sig = lambda x: 1 / (1 + np.exp(-x))


def args_():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="validation", choices=["validation"])
    p.add_argument("--source-run", default="20261104_v4_union_seed42")
    p.add_argument("--run-id", default="20261104_v4_union_fullval_seed42")
    p.add_argument("--features", default=str(SHARED / "feature_engineering_nov4"))
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--max-requests", type=int, default=0, help="smoke test only; 0 means every request")
    p.add_argument("--persist", default=str(SHARED / "Experiments" / "1.1_lightgbm_ranker"))
    p.add_argument("--worker", type=int, default=-1)
    p.add_argument("--merge", action="store_true")
    return p.parse_args()


def paths(a):
    return dict(src=HERE / "outputs" / a.source_run, out=HERE / "outputs" / a.run_id,
                blocks=HERE / "outputs" / a.run_id / "_blocks", persist=Path(a.persist) / a.run_id)


def log(path, msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_requests(a):
    names = pd.read_parquet(Path(a.features) / f"{a.split}_features.parquet", columns=["user_id"]).shape  # existence check
    import pyarrow.parquet as pq
    cols = pq.ParquetFile(Path(a.features) / f"{a.split}_features.parquet").schema_arrow.names
    use = ["user_id", "impression_id", "candidate_article_id", "impression_time", "is_sso_user", "is_subscriber", "is_front_page",
           "hour", "day_of_week", "clicked"] + [c for c in cols if c.startswith("ohe_device_type_")]
    d = pd.read_parquet(Path(a.features) / f"{a.split}_features.parquet", columns=use)
    assert not d.duplicated(["user_id", "impression_id", "candidate_article_id"]).any()
    d["behaviour_id"] = d.user_id.astype("int64").astype(str) + ":" + d.impression_id.astype("int64").astype(str)
    first = d.drop_duplicates("behaviour_id").set_index("behaviour_id")
    reqs = cv.hash_order(first.reset_index()[["user_id", "impression_id"]], a.split, 42)
    known = d[d.clicked == 1].groupby("behaviour_id").candidate_article_id.apply(lambda s: set(s.astype("int64")))
    reqs["known"] = reqs.behaviour_id.map(known).map(lambda s: s if isinstance(s, set) else set())
    return (reqs.head(a.max_requests) if a.max_requests else reqs), first


def ctx_of(r):
    sub = r.is_subscriber
    c = {"impression_time": r.impression_time.to_datetime64(), "is_sso_user": float(r.is_sso_user) if pd.notna(r.is_sso_user) else np.nan,
         "is_subscriber": bool(sub) if pd.notna(sub) else float("nan"), "is_front_page": float(r.is_front_page),
         "hour": float(r.hour), "day_of_week": float(r.day_of_week)}
    c.update({k: float(r[k]) for k in r.index if k.startswith("ohe_device_type_")})
    return c


def load_inputs(a, p):
    state = json.load(open(Path(a.features) / "feature_vocabulary.json", encoding="utf-8"))
    cat = cv.Catalogue([SMALL / "articles.parquet", TEST_ARTICLES], state)
    cfg = json.load(open(p["src"] / "model" / "model_config.json", encoding="utf-8"))
    assert cat.fingerprint == json.load(open(p["src"] / "run_config.json", encoding="utf-8"))["eligibility"]["catalogue_fingerprint"], "pool differs from source run"
    return state, cat, cfg


def worker(a, p):
    import lightgbm as lgb
    reqs, first = load_requests(a)
    state, cat, cfg = load_inputs(a, p)
    booster = lgb.Booster(model_file=str(p["src"] / "model" / "lightgbm_model.txt"))
    features = cfg["feature_names"]
    n_blocks = int(np.ceil(len(reqs) / BLOCK))
    for b in range(n_blocks):
        if b % a.workers != a.worker:
            continue
        done = p["blocks"] / f"manifest_{b:05d}.parquet"
        if done.exists():
            continue
        part = reqs.iloc[b * BLOCK:(b + 1) * BLOCK]
        profiles = cv.build_profiles(SMALL / "validation" / "history.parquet", part.user_id.unique(), cat, state)
        bids, ids_all, sc_all, man = [], [], [], []
        for r in part.itertuples():
            row = first.loc[r.behaviour_id]
            t = row.impression_time.to_datetime64()
            ix = cat.eligible(t); ids = cat.ids[ix]
            sc = booster.predict(cv.request_features(cat, profiles[int(r.user_id)], ctx_of(row), ix, features), raw_score=True, num_threads=a.threads)
            order, cross = cv.top_k(sc, ids, EXPORT_K)
            top = ids[order].astype("int64")
            bids.append(r.behaviour_id); ids_all.append(top); sc_all.append(sc[order].astype("float64"))
            rec = dict(behaviour_id=r.behaviour_id, impression_time=t, eligible_pool=int(len(ix)), shortlist_len=int(len(top)), known_clicks=len(r.known),
                       known_clicks_eligible=len(r.known & set(ids)), tie_at_cutoff=bool(cross))
            if r.known:
                rec.update(cv.retrieval_metrics(top, r.known, (100, 200)))
            man.append(rec)
        lens = np.array([len(x) for x in ids_all])
        s = pd.DataFrame({"behaviour_id": np.repeat(np.array(bids, dtype=object), lens), "article_id": np.concatenate(ids_all),
                          "rank": np.concatenate([np.arange(1, n + 1) for n in lens]), "score": np.concatenate(sc_all)})
        s.to_parquet(p["blocks"] / f"scores_{b:05d}.parquet", index=False)
        tmp = p["blocks"] / f"manifest_{b:05d}.tmp"
        pd.DataFrame(man).to_parquet(tmp, index=False)
        os.replace(tmp, done)     # a block counts as finished only after this atomic step
        log(p["out"] / "progress.log", f"block {b + 1}/{n_blocks} done by worker {a.worker}")


def merge(a, p):
    import lightgbm as lgb
    out, sp = p["out"], a.split
    reqs, first = load_requests(a)
    n_blocks = int(np.ceil(len(reqs) / BLOCK))
    S = pd.concat([pd.read_parquet(p["blocks"] / f"scores_{b:05d}.parquet") for b in range(n_blocks)], ignore_index=True)
    m = pd.concat([pd.read_parquet(p["blocks"] / f"manifest_{b:05d}.parquet") for b in range(n_blocks)], ignore_index=True)
    S["click_probability"] = sig(S.score.to_numpy())
    assert np.isfinite(S.score).all()
    for d in (f"predictions/{sp}", f"rankings/{sp}", "diagnostics/recorded_slates"):
        (out / d).mkdir(parents=True, exist_ok=True)
    S.to_parquet(out / "predictions" / sp / "submission_article_scores.parquet", index=False, compression="zstd")
    lists = S.groupby("behaviour_id", sort=False).article_id.apply(lambda x: x.astype("int64").tolist())
    pd.DataFrame({"behaviour_id": lists.index.to_numpy(), "article_ids": lists.to_list()}).to_parquet(
        out / "rankings" / sp / "submission_top_200.parquet", index=False, compression="zstd")
    m.to_parquet(out / "predictions" / sp / "request_manifest.parquet", index=False)

    # integrity, read back from disk
    S2 = pd.read_parquet(out / "predictions" / sp / "submission_article_scores.parquet")
    R2 = pd.read_parquet(out / "rankings" / sp / "submission_top_200.parquet")
    assert len(R2) == len(reqs) and R2.behaviour_id.is_unique and set(R2.behaviour_id) == set(reqs.behaviour_id), "ranking file does not cover every request once"
    assert (R2.article_ids.map(len).to_numpy() == np.minimum(EXPORT_K, m.set_index("behaviour_id").loc[R2.behaviour_id].eligible_pool.to_numpy())).all(), "not min(200, pool)"
    assert np.array_equal(np.concatenate([np.asarray(x, dtype="int64") for x in R2.article_ids]), S2.article_id.to_numpy()), "scores and rankings disagree"
    assert not S2.duplicated(["behaviour_id", "article_id"]).any(), "duplicate keys"
    assert S2.groupby("behaviour_id", sort=False).score.apply(lambda x: bool((np.diff(x.to_numpy()) <= 0).all())).all(), "scores not descending"

    # regression: every request in the earlier sampled export must be reproduced exactly
    old = pd.read_parquet(p["src"] / "predictions" / sp / "submission_article_scores.parquet")
    both = old[old.behaviour_id.isin(set(S2.behaviour_id))].merge(S2, on=["behaviour_id", "rank"], suffixes=("_old", "_new"))
    assert len(both) == int(old.behaviour_id.isin(set(S2.behaviour_id)).sum()), "rank rows missing"
    assert (both.article_id_old == both.article_id_new).all(), "article order differs from the earlier export"
    gap = float((both.score_old - both.score_new).abs().max()) if len(both) else float("nan")
    assert not gap > 1e-9, f"scores differ from the earlier export: {gap}"

    # reuse the complete recorded-slate diagnostic and the saved model; those did not change
    shutil.copy(p["src"] / "diagnostics" / "recorded_slates" / f"{sp}_scores.parquet", out / "diagnostics" / "recorded_slates" / f"{sp}_scores.parquet")
    shutil.copytree(p["src"] / "model", out / "model", dirs_exist_ok=True)
    shutil.copytree(p["src"] / "tuning", out / "tuning", dirs_exist_ok=True)
    shutil.copy(p["src"] / "requirements.txt", out / "requirements.txt")
    shutil.copy(HERE / "export_full_split.py", out / "model" / "export_full_split.py")

    k = m[m.known_clicks > 0]
    metric = {c: float(k[c].mean()) for c in ("recall@100", "ndcg@100", "recall@200", "ndcg@200")}
    metric.update(requests_with_known_clicks=int(len(k)), known_clicks=int(k.known_clicks.sum()), eligible_known_clicks=int(k.known_clicks_eligible.sum()),
                  ties_at_cutoff=int(m.tie_at_cutoff.sum()), empty_pools=int((m.eligible_pool == 0).sum()), median_eligible_pool=float(m.eligible_pool.median()))
    src_metrics = json.load(open(p["src"] / "metrics.json", encoding="utf-8"))
    (out / "metrics.json").write_text(json.dumps(dict(
        note="development results: validation was used to select the model on 200 requests; this run re-scores the same saved model on every validation request",
        primary="known-click Recall@100", secondary="observed-click nDCG@100", full_validation=metric,
        recorded_slate_diagnostic=src_metrics.get("recorded_slate_diagnostic", {}).get("validation"), selection_validation=src_metrics.get("selection_validation"),
        ablation=src_metrics.get("ablation")), indent=2, default=str))
    cfg = json.load(open(p["src"] / "run_config.json", encoding="utf-8"))
    git = lambda *x: subprocess.run(["git", *x], capture_output=True, text=True, cwd=HERE).stdout.strip()
    cfg.update(run_id=a.run_id, parent_run=a.source_run, created_by="export_full_split.py (load-only; model not refitted)", persistent_artifact_location=str(p["persist"]),
               code_commit=git("rev-parse", "HEAD"), uncommitted_changes=bool(git("status", "--porcelain")),
               request_id_list="predictions/validation/request_manifest.parquet")
    cfg["scope"] = dict(split=sp, export_requests=len(R2), all_requests_in_split=(a.max_requests == 0), sampled_run=False, train_and_test_in_this_bundle=False,
                        training="unchanged: model reused from parent run")
    cfg["exports"] = {sp: dict(requests=len(R2), score_rows=len(S2))}
    cfg["eligibility"]["test_catalogue_provenance"] = ("test catalogue is the file with SHA-256 07ee3568d8f003c0... ; the local file hash and the train_val hash were both "
                                                       "verified against catalogue/article_union.parquet.manifest.json; published_time also matches article_union.parquet")
    cfg["eligibility"]["shared_catalogue_id_fingerprint_sha256"] = "46bc36568a9bc1c6fdcfd101079b2cfef6f4192f4505032766fc858490c5f877"
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str))

    # persistent copy, then reload from it without refitting
    p["persist"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(out, p["persist"], dirs_exist_ok=True, ignore=shutil.ignore_patterns("_blocks"))
    state, cat, c = load_inputs(a, p)
    rl = lgb.Booster(model_file=str(p["persist"] / "model" / "lightgbm_model.txt"))
    ex = pd.read_parquet(p["persist"] / "predictions" / sp / "submission_article_scores.parquet")
    probe = reqs.tail(5); worst = 0.0
    prof = cv.build_profiles(SMALL / "validation" / "history.parquet", probe.user_id.unique(), cat, state)
    for r in probe.itertuples():
        row = first.loc[r.behaviour_id]; ix = cat.eligible(row.impression_time.to_datetime64())
        sc = rl.predict(cv.request_features(cat, prof[int(r.user_id)], ctx_of(row), ix, c["feature_names"]), raw_score=True, num_threads=a.threads)
        ref = ex[ex.behaviour_id == r.behaviour_id].set_index("article_id").score
        worst = max(worst, float(np.abs(pd.Series(sc, index=cat.ids[ix]).reindex(ref.index).to_numpy() - ref.to_numpy()).max()))
    assert worst < 1e-6, f"reload check failed: {worst}"
    log(out / "progress.log", f"MERGED requests={len(R2)} rows={len(S2)} | regression rows={len(both)} max_gap={gap} | reload max_gap={worst} | {metric}")


def main():
    a = args_(); p = paths(a)
    p["blocks"].mkdir(parents=True, exist_ok=True)
    if a.worker >= 0:
        return worker(a, p)
    if a.merge:
        return merge(a, p)
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)     # keep awake while running; nothing persistent changes
    base = [sys.executable, str(Path(__file__).resolve()), "--split", a.split, "--source-run", a.source_run, "--run-id", a.run_id, "--features", a.features,
            "--workers", str(a.workers), "--threads", str(a.threads), "--max-requests", str(a.max_requests), "--persist", a.persist]
    env = {**os.environ, "OMP_NUM_THREADS": str(a.threads), "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    t0 = time.time(); log(p["out"] / "progress.log", f"start {a.workers} workers, {a.threads} threads each")
    procs = [subprocess.Popen(base + ["--worker", str(i)], env=env) for i in range(a.workers)]
    codes = [q.wait() for q in procs]
    if any(codes):
        log(p["out"] / "progress.log", f"WORKER FAILED {codes}; rerun the same command to resume finished blocks"); sys.exit(1)
    r = subprocess.run(base + ["--merge"], env=env)
    log(p["out"] / "progress.log", f"finished exit={r.returncode} minutes={(time.time() - t0) / 60:.1f}")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
