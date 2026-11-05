"""Load-only full validation export for experiment 1.3 (no training). Same block design and checks as 1.1's export."""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; L = HERE.parent / "1.1_lightgbm_ranker"
sys.path.insert(0, str(L))
import catalogue_v4 as cv, export_full_split as ef

SHARED = ef.SHARED; EXPORT_K, BLOCK = 200, 1000
FEATURES_DIR = str(SHARED / "feature_engineering_nov4")
SOURCE = HERE / "outputs" / "20261104_val3000_seed42"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="20261104_fullval_seed42"); p.add_argument("--workers", type=int, default=5); p.add_argument("--threads", type=int, default=3)
    p.add_argument("--max-requests", type=int, default=0); p.add_argument("--persist", default=str(SHARED / "Experiments" / "1.3_xgboost"))
    p.add_argument("--worker", type=int, default=-1); p.add_argument("--merge", action="store_true")
    return p.parse_args()


def setup(a):
    return SimpleNamespace(split="validation", features=FEATURES_DIR, max_requests=a.max_requests), HERE / "outputs" / a.run_id


def inputs():
    state = json.load(open(Path(FEATURES_DIR) / "feature_vocabulary.json", encoding="utf-8"))
    cat = cv.Catalogue([ef.SMALL / "articles.parquet", ef.TEST_ARTICLES], state)
    assert cat.fingerprint == "cb89570c1bc70654", "pool differs from 1.1"
    feat = json.load(open(SOURCE / "run_config.json", encoding="utf-8"))["features"]
    return state, cat, feat


def worker(a):
    import xgboost as xgb
    ns, out = setup(a); blocks = out / "_blocks"; reqs, first = ef.load_requests(ns); state, cat, feat = inputs()
    b_ = xgb.Booster(); b_.load_model(str(SOURCE / "model" / "model.json")); b_.set_param({"nthread": a.threads})
    n = int(np.ceil(len(reqs) / BLOCK))
    for b in range(n):
        done = blocks / f"manifest_{b:05d}.parquet"
        if b % a.workers != a.worker or done.exists():
            continue
        part = reqs.iloc[b * BLOCK:(b + 1) * BLOCK]; prof = cv.build_profiles(ef.SMALL / "validation" / "history.parquet", part.user_id.unique(), cat, state)
        bids, ia, sa, man = [], [], [], []
        for r in part.itertuples():
            row = first.loc[r.behaviour_id]; ix = cat.eligible(row.impression_time.to_datetime64()); ids = cat.ids[ix]
            sc = b_.inplace_predict(cv.request_features(cat, prof[int(r.user_id)], ef.ctx_of(row), ix, feat), predict_type="margin", missing=np.nan)
            order, cross = cv.top_k(sc, ids, EXPORT_K); top = ids[order].astype("int64")
            bids.append(r.behaviour_id); ia.append(top); sa.append(np.asarray(sc[order], "float64"))
            man.append(dict(behaviour_id=r.behaviour_id, impression_time=row.impression_time.to_datetime64(), eligible_pool=len(ix), shortlist_len=len(top), known_clicks=len(r.known),
                            known_clicks_eligible=len(r.known & set(ids)), tie_at_cutoff=bool(cross), **cv.retrieval_metrics(top, r.known, (100, 200))))
        lens = [len(x) for x in ia]
        pd.DataFrame({"behaviour_id": np.repeat(np.array(bids, dtype=object), lens), "article_id": np.concatenate(ia), "rank": np.concatenate([np.arange(1, x + 1) for x in lens]),
                      "score": np.concatenate(sa)}).to_parquet(blocks / f"scores_{b:05d}.parquet", index=False)
        tmp = blocks / f"manifest_{b:05d}.tmp"; pd.DataFrame(man).to_parquet(tmp, index=False); os.replace(tmp, done)
        ef.log(out / "progress.log", f"block {b + 1}/{n} done by worker {a.worker}")


def merge(a):
    import xgboost as xgb
    ns, out = setup(a); blocks = out / "_blocks"; reqs, first = ef.load_requests(ns); persist = Path(a.persist) / a.run_id
    n = int(np.ceil(len(reqs) / BLOCK))
    S = pd.concat([pd.read_parquet(blocks / f"scores_{b:05d}.parquet") for b in range(n)], ignore_index=True); m = pd.concat([pd.read_parquet(blocks / f"manifest_{b:05d}.parquet") for b in range(n)], ignore_index=True)
    S["click_probability"] = ef.sig(S.score.to_numpy()); assert np.isfinite(S.score).all()
    for d in ("predictions/validation", "rankings/validation", "diagnostics/recorded_slates"): (out / d).mkdir(parents=True, exist_ok=True)
    S.to_parquet(out / "predictions/validation/submission_article_scores.parquet", index=False, compression="zstd")
    lists = S.groupby("behaviour_id", sort=False).article_id.apply(lambda x: x.astype("int64").tolist())
    pd.DataFrame({"behaviour_id": lists.index.to_numpy(), "article_ids": lists.to_list()}).to_parquet(out / "rankings/validation/submission_top_200.parquet", index=False, compression="zstd")
    m.to_parquet(out / "predictions/validation/request_manifest.parquet", index=False)
    S2 = pd.read_parquet(out / "predictions/validation/submission_article_scores.parquet"); R2 = pd.read_parquet(out / "rankings/validation/submission_top_200.parquet")
    assert len(R2) == len(reqs) and R2.behaviour_id.is_unique and set(R2.behaviour_id) == set(reqs.behaviour_id)
    assert (R2.article_ids.map(len) == 200).all() and np.array_equal(np.concatenate([np.asarray(x, "int64") for x in R2.article_ids]), S2.article_id.to_numpy()) and not S2.duplicated(["behaviour_id", "article_id"]).any()
    assert S2.groupby("behaviour_id", sort=False).score.apply(lambda x: bool((np.diff(x.to_numpy()) <= 0).all())).all()
    old = pd.read_parquet(SOURCE / "predictions/validation/submission_article_scores.parquet"); old = old[old.behaviour_id.isin(set(S2.behaviour_id))]
    both = old.merge(S2, on=["behaviour_id", "rank"], suffixes=("_old", "_new")); assert len(both) == len(old) and (both.article_id_old == both.article_id_new).all()
    gap = float((both.score_old - both.score_new).abs().max()); assert gap < 1e-5, gap
    for sub in ("model", "tuning"): shutil.copytree(SOURCE / sub, out / sub, dirs_exist_ok=True)
    shutil.copy(SOURCE / "diagnostics/recorded_slates/validation_scores.parquet", out / "diagnostics/recorded_slates/validation_scores.parquet"); shutil.copy(SOURCE / "requirements.txt", out / "requirements.txt")
    shutil.copy(Path(__file__), out / "model" / "export_full_validation.py"); shutil.copy(L / "export_full_split.py", out / "model" / "export_full_split.py")
    metric = {c: float(m[c].mean()) for c in ("recall@100", "ndcg@100", "recall@200", "ndcg@200")}
    metric.update(requests=len(m), known_clicks=int(m.known_clicks.sum()), eligible_known_clicks=int(m.known_clicks_eligible.sum()), ties_at_cutoff=int(m.tie_at_cutoff.sum()), median_eligible_pool=float(m.eligible_pool.median()))
    src = json.load(open(SOURCE / "metrics.json", encoding="utf-8"))
    (out / "metrics.json").write_text(json.dumps(dict(note="development results; selection used 200 validation requests", full_validation=metric, recorded_slate_diagnostic=src["recorded_slate_diagnostic"], selected_selection=src["selected_selection"]), indent=2))
    cfg = json.load(open(SOURCE / "run_config.json", encoding="utf-8")); git = lambda *x: subprocess.run(["git", *x], capture_output=True, text=True, cwd=HERE).stdout.strip()
    cfg.update(run_id=a.run_id, parent_run=SOURCE.name, created_by="export_full_validation.py (load-only)", persistent_artifact_location=str(persist), code_commit=git("rev-parse", "HEAD"), uncommitted_changes=bool(git("status", "--porcelain")))
    cfg["scope"] = dict(split="validation", export_requests=len(R2), all_requests_in_split=a.max_requests == 0, sampled_run=False, train_and_test_in_this_bundle=False, training="unchanged: model reused from parent run")
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    persist.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(out, persist, dirs_exist_ok=True, ignore=shutil.ignore_patterns("_blocks"))
    state, cat, feat = inputs(); rb = xgb.Booster(); rb.load_model(str(persist / "model" / "model.json")); ex = pd.read_parquet(persist / "predictions/validation/submission_article_scores.parquet")
    probe = reqs.tail(5); prof = cv.build_profiles(ef.SMALL / "validation/history.parquet", probe.user_id.unique(), cat, state); worst = 0.0
    for r in probe.itertuples():
        row = first.loc[r.behaviour_id]; ix = cat.eligible(row.impression_time.to_datetime64()); sc = rb.inplace_predict(cv.request_features(cat, prof[int(r.user_id)], ef.ctx_of(row), ix, feat), predict_type="margin", missing=np.nan)
        rf = ex[ex.behaviour_id == r.behaviour_id].set_index("article_id").score; worst = max(worst, float(np.abs(pd.Series(sc, index=cat.ids[ix]).reindex(rf.index).to_numpy() - rf.to_numpy()).max()))
    assert worst < 1e-5, worst
    ef.log(out / "progress.log", f"MERGED requests={len(R2)} rows={len(S2)} | regression rows={len(both)} max_gap={gap} | reload max_gap={worst} | {metric}")


def main():
    a = parse(); ns, out = setup(a); (out / "_blocks").mkdir(parents=True, exist_ok=True)
    if a.worker >= 0: return worker(a)
    if a.merge: return merge(a)
    import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    base = [sys.executable, str(Path(__file__).resolve()), "--run-id", a.run_id, "--workers", str(a.workers), "--threads", str(a.threads), "--max-requests", str(a.max_requests), "--persist", a.persist]
    env = {**os.environ, "OMP_NUM_THREADS": str(a.threads), "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}; t0 = time.time(); ef.log(out / "progress.log", "start")
    ps = [subprocess.Popen(base + ["--worker", str(i)], env=env) for i in range(a.workers)]; codes = [p.wait() for p in ps]
    if any(codes): ef.log(out / "progress.log", f"WORKER FAILED {codes}"); sys.exit(1)
    r = subprocess.run(base + ["--merge"], env=env); ef.log(out / "progress.log", f"finished exit={r.returncode} minutes={(time.time() - t0) / 60:.1f}"); sys.exit(r.returncode)


if __name__ == "__main__":
    main()
