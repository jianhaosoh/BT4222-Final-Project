"""Orchestration used by experiment.ipynb: configuration presets, input
preparation, preflight, training/loading, exports, persistence and reload."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import catalogue as C
from . import data as D
from . import features as F
from .io_utils import (disk_free_bytes, environment_info, git_info, mirror_tree, read_json,
                       write_completion_marker, marker_is_valid)
from .metrics import METRIC_DEFINITIONS, impression_metrics, recall_summary
from .model import ModelConfig, build_model, parameter_counts
from .persist import MODEL_FILES, check_article_ids, load_model, save_model_files
from .io_utils import atomic_write_json, copy_verified
from . import sampling
from .io_utils import file_record
from .sampling import POOL_POLICY
from .selection import CatalogueSelector
from .train import Trainer, evaluate_recorded, score_recorded, seed_everything
from .validate import validate_export, validate_recorded

EXPERIMENT_ID = "2.2_huim"

BASE_CONFIG = {
    "experiment_id": EXPERIMENT_ID,
    "seed": 42,                            # model-training seed (independent of request sampling)
    "feature_case": "not_applicable",      # inputs are built from source tables; see run_config['inputs']
    # Shared pool policy published_at_or_before_request_v1 (3.1): ONE ordered union, same for every split.
    "catalogue_policy": {"train": ["train_val", "test"], "validation": ["train_val", "test"], "test": ["train_val", "test"]},
    "vector_source_order": ["train_val", "test"],
    "text_dim": 768, "image_dim": 1024,
    # model
    "history_len": 50, "d_model": 64, "attention_mode": "softmax", "attention_hidden": [32, 16],
    "fusion": "concat", "feature_hidden": 64, "head_hidden": [128, 64], "dropout": 0.2, "modality_dropout": 0.15,
    # optimisation
    "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 4096, "max_epochs": 4,
    "grad_clip": 1.0, "amp": True, "checkpoint_every_steps": 200, "log_every_steps": 200,
    # Shared request sampling (3.1): BLAKE2b-64 priority with seed 42; three SEPARATE limits.
    # None = every request of the agreed population. Every selected inference request uses its FULL pool.
    "request_sampling_seed": 42,
    "train_requests": None,               # complete recorded slates of the selected training requests
    "selection_requests": 100,            # validation-selection scope (fixed before training)
    "export_requests": {"train": None, "validation": None, "test": None},   # exported scope per split
    "reference_31_run": None,             # optional 3.1 run folder: reuse its saved request IDs when present
    # selection (Experiments/README.md): catalogue validation Recall@100 on the fixed selection requests
    "selection_interval_steps": 300,      # plus an evaluation at every epoch end
    "early_stopping_patience": 3,         # selection evaluations without improvement
    "min_improvement": 0.0,               # 0: any better (Recall, then nDCG) checkpoint resets patience
    "scheduler_patience": 1,              # ReduceLROnPlateau on Recall@100, in evaluations
    # catalogue scoring/export and resource guards
    "pair_chunk": 32768, "request_block": 16, "inference_amp": False,
    "full_score_splits": ["train", "validation", "test"],  # write full score CSVs for exported requests (storage gate)
    "max_export_pairs": 20_000_000,       # stop before exporting more pairs; raise deliberately for full scope
    "max_selection_pairs_per_evaluation": 50_000_000,
    "full_score_storage_margin": 1.25,    # require free space >= margin x estimate
    "validation_check_blocks": 50,        # sampled score blocks re-read for large exports (None = all)
}

PRESETS = {
    # Labelled smoke run: tiny, explicitly bounded request scopes (prefixes of the shared hash order).
    "smoke": {"train_requests": 1000, "selection_requests": 25,
              "export_requests": {"train": 5, "validation": 5, "test": 5},
              "max_epochs": 1, "batch_size": 1024, "selection_interval_steps": 4, "checkpoint_every_steps": 0},
    # Aligned with the running 3.1 development scope: 5,000 training requests, 100 validation-selection
    # requests, 25 export requests per split (sampled development, not a submission).
    "development": {"train_requests": 5000, "selection_requests": 100,
                    "export_requests": {"train": 25, "validation": 25, "test": 25},
                    "batch_size": 1024, "max_epochs": 6, "selection_interval_steps": 20},
    # Explicit full-request scope. Requires raising max_export_pairs after reading the budgets.
    "full": {"train_requests": None, "selection_requests": 100,
             "export_requests": {"train": None, "validation": None, "test": None}},
}


def make_config(mode: str, **overrides) -> dict:
    if mode not in PRESETS:
        raise ValueError(f"RUN_MODE must be one of {list(PRESETS)}")
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.update(copy.deepcopy(PRESETS[mode]))
    cfg.update(overrides)
    cfg["run_mode"] = mode
    return cfg


@dataclass
class Workspace:
    paths: D.DataPaths
    vocabulary: dict
    behaviours: dict
    articles: pd.DataFrame
    catalogues: dict
    vector_reports: dict
    events: dict
    history: dict
    pairs: dict
    pre: F.Preprocessor
    art: F.ArticleTable
    ctx: dict
    reports: dict = field(default_factory=dict)
    device_cache: dict = field(default_factory=dict)


def prepare_inputs(cfg: dict, data_root: Path, pre: F.Preprocessor | None = None, path_overrides: dict | None = None,
                   log=print) -> Workspace:
    """Load the shared inputs and build request/article/history tensors.

    If `pre` is None, preprocessing is fitted on train-permitted data; otherwise the
    saved preprocessing is reused unchanged (load-only path)."""
    t0 = time.time()
    paths = D.resolve_paths(Path(data_root), path_overrides)
    vocabulary = read_json(paths.vocabulary)
    behaviours = {s: D.load_behaviours(paths.behaviours[s], s) for s in D.SPLITS}
    starts = {s: pd.Timestamp(behaviours[s].impression_time.min()) for s in D.SPLITS}
    if not (behaviours["train"].impression_time.max() < starts["validation"]):
        raise ValueError("train and validation periods overlap")
    union = POOL_POLICY["source_order"]
    if any(list(v) != union for v in cfg["catalogue_policy"].values()):
        raise ValueError(f"catalogue_policy must use the shared ordered union {union} for every split "
                         f"({POOL_POLICY['name']}); alternative policies must be named and compared separately")
    if list(cfg["vector_source_order"]) != union:
        raise ValueError(f"vector_source_order must be {union} under {POOL_POLICY['name']}")
    articles, catalogues = D.load_articles(paths, union)
    log(f"articles: {len(articles):,} unique IDs from {union}")
    text_ids, text_vecs, text_rep = D.load_vectors(paths, "text", cfg["vector_source_order"], cfg["text_dim"])
    image_ids, image_vecs, image_rep = D.load_vectors(paths, "image", cfg["vector_source_order"], cfg["image_dim"])
    log(f"vectors: text {len(text_ids):,}, image {len(image_ids):,}")
    events, history = {}, {}
    for s in D.SPLITS:
        events[s], history[s] = D.load_history(paths.history[s], starts[s])
    pairs = {s: D.load_recorded_pairs(paths.features[s], s, behaviours[s]) for s in D.SPLITS}
    log("recorded pairs: " + ", ".join(f"{s} {len(p):,}" for s, p in pairs.items()))
    # Union lookup must cover the declared catalogues plus every referenced history/slate article.
    referenced = np.unique(np.concatenate([events[s].article_id.to_numpy(np.int64) for s in D.SPLITS] +
                                          [pairs[s].candidate_article_id.to_numpy(np.int64) for s in D.SPLITS]))
    not_found = np.setdiff1d(referenced, articles.article_id.to_numpy())
    if pre is None:
        train_ids = np.union1d(pairs["train"].candidate_article_id.to_numpy(), events["train"].article_id.to_numpy())
        pre = F.Preprocessor.fit(vocabulary, articles, train_ids, events["train"], cfg["history_len"])
    art = F.build_article_table(articles, pre, text_ids, text_vecs, image_ids, image_vecs)
    ctx = {}
    members = np.unique(np.concatenate([catalogues["members"][n] for n in union]))  # the same union for every split
    for s in D.SPLITS:
        ctx[s] = F.build_split_context(s, behaviours[s], events[s], history[s]["users"], art, pre, members, pairs[s])
    pool = sampling.pool_fingerprint(art.ids[ctx["train"].catalogue_idx],
                                     art.tensors["pub_s"].numpy()[ctx["train"].catalogue_idx],
                                     {n: {k: v for k, v in file_record(paths.articles[n], None).items()
                                          if k != "sha256"} for n in union})
    for s in D.SPLITS:
        assert np.array_equal(ctx[s].catalogue_idx, ctx["train"].catalogue_idx), "pool differs across splits"
    lookup = {s: {"history_articles_not_in_union": int(len(np.setdiff1d(events[s].article_id.to_numpy(np.int64),
                                                                       articles.article_id.to_numpy()))),
                  "recorded_candidates_not_in_union": int(len(np.setdiff1d(pairs[s].candidate_article_id.to_numpy(np.int64),
                                                                          articles.article_id.to_numpy())))}
              for s in D.SPLITS}
    reports = {"catalogues": catalogues["report"], "referenced_articles_not_in_any_catalogue": int(len(not_found)),
               "pool_policy": POOL_POLICY, "pool_fingerprint": pool, "article_lookup_coverage": lookup,
               "history": {s: history[s]["report"] for s in D.SPLITS},
               "contexts": {s: ctx[s].report for s in D.SPLITS}, "article_table": art.report,
               "split_starts": {s: str(v) for s, v in starts.items()},
               "prepare_seconds": time.time() - t0}
    log(f"inputs ready in {time.time() - t0:.0f}s")
    return Workspace(paths, vocabulary, behaviours, articles, catalogues, {"text": text_rep, "image": image_rep},
                     events, history, pairs, pre, art, ctx, reports)


def coverage_report(ws: Workspace) -> dict:
    """Vector coverage of each split's catalogue and of its recorded candidates."""
    out = {}
    has_t = ws.art.tensors["has_text"].numpy()
    has_i = ws.art.tensors["has_image"].numpy()
    for s, ctx in ws.ctx.items():
        cat = ctx.catalogue_idx
        rec = ctx.recorded["art_idx"]
        out[s] = {"catalogue_articles": int(len(cat)),
                  "catalogue_with_text": int(has_t[cat].sum()), "catalogue_with_image": int(has_i[cat].sum()),
                  "catalogue_text_share": float(has_t[cat].mean()) if len(cat) else None,
                  "catalogue_image_share": float(has_i[cat].mean()) if len(cat) else None,
                  "recorded_candidates_text_share": float(has_t[rec].mean()) if len(rec) else None,
                  "recorded_candidates_image_share": float(has_i[rec].mean()) if len(rec) else None,
                  "recorded_candidates_unknown_article": int((rec == 0).sum())}
    return out


def preflight(ws: Workspace, cfg: dict, output_roots: list[Path], scopes: dict | None = None) -> dict:
    """Independent checks of the full Drive artifacts before training/export."""
    scopes = scopes or resolve_request_scopes(ws, cfg)
    report = {"vector_decoding": ws.vector_reports, "coverage": coverage_report(ws), "eligibility": {},
              "recorded_ineligible": {}, "history": ws.reports["history"],
              "pool_policy": ws.reports["pool_policy"], "pool_fingerprint": ws.reports["pool_fingerprint"],
              "article_lookup_coverage": ws.reports["article_lookup_coverage"],
              "request_scopes": {k: v["info"] for k, v in scopes.items()},
              "budget": export_budget(ws, cfg, scopes)}
    for s, ctx in ws.ctx.items():
        el = C.catalogue_eligibility(ctx, ws.art)
        report["eligibility"][s] = {k: v for k, v in el.items() if k != "eligible_count"}
        report["eligibility"][s]["feasibility"] = C.feasibility_estimate(el, ctx)
        bad = C.recorded_eligibility(ctx, ws.art)
        report["recorded_ineligible"][s] = bad.reason.value_counts().to_dict()
    total_pairs = sum(v["feasibility"]["eligible_pairs"] for v in report["eligibility"].values())
    total_bytes = sum(v["feasibility"]["full_scores_csv_bytes_estimate"] for v in report["eligibility"].values())
    report["all_splits"] = {"eligible_pairs": total_pairs, "full_scores_csv_bytes_estimate": total_bytes,
                            "full_scores_csv_gib_estimate": total_bytes / 1024 ** 3}
    report["disk_free_bytes"] = {str(p): disk_free_bytes(p) for p in output_roots}
    try:
        report["consistency_with_shared_features"] = {
            s: F.consistency_with_shared_features(ws.paths.features[s], ws.ctx[s], ws.art) for s in D.SPLITS}
    except Exception as error:  # diagnostic only; reported, never silently ignored
        report["consistency_with_shared_features"] = {"error": repr(error)}
    # Hard checks
    for s, ctx in ws.ctx.items():
        assert (ctx.tensors["hist_s"][ctx.tensors["hist_mask"]] < ctx.tensors["req_s"].min()).all(), \
            f"{s}: history event at/after the split's first request"
        assert ctx.requests.behaviour_id.is_unique
    return report


# ------------------------------------------------------------------ model --

def model_config(cfg: dict, pre: F.Preprocessor, art: F.ArticleTable) -> ModelConfig:
    sizes = pre.sizes()
    return ModelConfig(**sizes, text_dim=art.report["text_dim"], image_dim=art.report["image_dim"],
                       d_model=cfg["d_model"], attention_mode=cfg["attention_mode"],
                       attention_hidden=tuple(cfg["attention_hidden"]), fusion=cfg["fusion"],
                       feature_hidden=cfg["feature_hidden"], head_hidden=tuple(cfg["head_hidden"]),
                       dropout=cfg["dropout"], modality_dropout=cfg["modality_dropout"],
                       recency_edges_h=list(pre.state["recency_edges_h"]))


def resolve_request_scopes(ws: Workspace, cfg: dict) -> dict:
    """Training, selection-validation and export request positions (shared hash
    order; saved 3.1 IDs preferred when a reference run is configured)."""
    seed = cfg["request_sampling_seed"]
    ref = sampling.read_reference_ids(Path(cfg["reference_31_run"])) if cfg.get("reference_31_run") else {}
    scopes = {}
    pos, info = sampling.resolve_scope(ws.ctx["train"].requests, "train", "training", cfg["train_requests"], seed)
    scopes["training"] = {"positions": pos, "info": info}
    pos, info = sampling.resolve_scope(ws.ctx["validation"].requests, "validation", "selection_validation",
                                       cfg["selection_requests"], seed, ref.get("selection_validation"))
    scopes["selection_validation"] = {"positions": pos, "info": info}
    for s in D.SPLITS:
        pos, info = sampling.resolve_scope(ws.ctx[s].requests, s, f"export_{s}", cfg["export_requests"][s], seed,
                                           ref.get(f"export_{s}"))
        scopes[f"export_{s}"] = {"positions": pos, "info": info}
    if scopes["selection_validation"]["info"]["requests"] < 1:
        raise ValueError("selection scope is empty")
    return scopes


def scope_ids(ws: Workspace, scopes: dict) -> dict:
    """Saved request IDs for every scope (written to selection/request_scopes.json)."""
    split_of = {"training": "train", "selection_validation": "validation",
                **{f"export_{s}": s for s in D.SPLITS}}
    return {name: {**sc["info"], "behaviour_ids": ws.ctx[split_of[name]].requests.behaviour_id.to_numpy()[
        sc["positions"]].tolist()} for name, sc in scopes.items()}


def training_subset(ws: Workspace, cfg: dict, scopes: dict | None = None) -> tuple[np.ndarray | None, dict]:
    """Recorded pairs of the selected training requests (complete slates)."""
    scopes = scopes or resolve_request_scopes(ws, cfg)
    sc = scopes["training"]
    rec = ws.ctx["train"].recorded
    if cfg["train_requests"] is None:
        return None, {**sc["info"], "pairs": int(len(rec["req_pos"])), "scope": "all training requests"}
    idx = np.flatnonzero(np.isin(rec["req_pos"], sc["positions"]))
    return idx, {**sc["info"], "pairs": int(len(idx)), "scope": "hash-sampled training requests, complete slates"}


def device_and_tensors(ws: Workspace, device):
    """Device copies of the article/split tensors, cached per device (one GPU copy)."""
    key = str(device)
    if key not in ws.device_cache:
        ws.device_cache.clear()
        ws.device_cache[key] = (ws.art.to(device), {s: c.to(device) for s, c in ws.ctx.items()})
    return ws.device_cache[key]


def make_selector(ws: Workspace, cfg: dict, device, scopes: dict | None = None) -> CatalogueSelector:
    """The fixed validation-selection requests (same for every checkpoint, control and trial)."""
    art, ctx = device_and_tensors(ws, device)
    scopes = scopes or resolve_request_scopes(ws, cfg)
    sc = scopes["selection_validation"]
    el = C.catalogue_eligibility(ws.ctx["validation"], ws.art)["eligible_count"][sc["positions"]]
    if int(el.sum()) > cfg["max_selection_pairs_per_evaluation"]:
        raise RuntimeError(f"selection needs {int(el.sum()):,} pairs per evaluation (> max_selection_pairs_per_"
                           "evaluation); reduce selection_requests or raise the declared budget deliberately")
    return CatalogueSelector(ws.ctx["validation"], ws.art, ctx["validation"], art, sc["positions"], device, cfg,
                             sample_info=sc["info"], pool=ws.reports["pool_fingerprint"])


def train_model(ws: Workspace, cfg: dict, run_dir: Path, persist_dir: Path | None, device, resume: bool = True,
                log=print, selector: CatalogueSelector | None = None, on_evaluation=None) -> tuple:
    seed_everything(cfg["seed"])
    if "request_scaler" not in ws.pre.state:
        F.fit_dense_scalers(ws.pre, ws.ctx["train"], ws.art)
    mcfg = model_config(cfg, ws.pre, ws.art)
    model = build_model(mcfg, ws.pre).to(device)
    art, ctx = device_and_tensors(ws, device)
    scopes = resolve_request_scopes(ws, cfg)
    subset, subset_info = training_subset(ws, cfg, scopes)
    selector = selector or make_selector(ws, cfg, device, scopes)
    sample_path = Path(run_dir) / "selection" / "selection_requests.json"
    atomic_write_json(sample_path, selector.sample_record())
    if persist_dir is not None:
        copy_verified(sample_path, Path(persist_dir) / "selection" / "selection_requests.json")
    tcfg = dict(cfg, train_pair_indices=subset)
    trainer = Trainer(model, art, ctx["train"], ctx["validation"], device, tcfg, run_dir, persist_dir, log,
                      selector=selector, on_evaluation=on_evaluation)
    rec = Path(run_dir) / "recovery" / "last.pt"
    persisted = Path(persist_dir) / "recovery" / "last.pt" if persist_dir else None
    if resume and not rec.is_file() and persisted is not None and persisted.is_file():
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_bytes(persisted.read_bytes())
        sel = Path(persist_dir) / "model" / "weights.pt"
        if sel.is_file():
            (Path(run_dir) / "model").mkdir(parents=True, exist_ok=True)
            (Path(run_dir) / "model" / "weights.pt").write_bytes(sel.read_bytes())
    if resume and rec.is_file():
        trainer.load_recovery(rec)
    state = trainer.fit()
    if state["best"] is None:
        raise RuntimeError("no selection evaluation ran; shortlist selection is incomplete (no AUC fallback)")
    # Restore the selected (best Recall@100) checkpoint, not the last one, and re-evaluate it.
    model.load_state_dict(torch.load(Path(run_dir) / "model" / "weights.pt", map_location=device, weights_only=True))
    model.eval()
    restored = selector.evaluate(model)
    best = state["best"]
    selection = {**selector.protocol,
                 "selected_checkpoint": {k: best[k] for k in ("eval_index", "epoch", "step_in_epoch", "global_step")},
                 "selected_known_click_recall@100_selection_sample": best["known_click_recall@100"],
                 "selected_observed_click_ndcg@100_selection_sample": best["observed_click_ndcg@100"],
                 "selected_diagnostic_recorded_slate_auc": best["diagnostic_recorded_slate_auc"],
                 "restored_checkpoint_reevaluation": {k: restored[k] for k in (
                     "known_click_recall@100", "observed_click_ndcg@100", "requests_evaluated")},
                 "restored_matches_recorded": bool(
                     abs(restored["known_click_recall@100"] - best["known_click_recall@100"]) <= 1e-9
                     and abs(restored["observed_click_ndcg@100"] - best["observed_click_ndcg@100"]) <= 1e-6),
                 "evaluations": len(state["history"]), "stop_reason": state["stop_reason"],
                 "selection_sample_file": "selection/selection_requests.json"}
    save_model_files(run_dir, model, mcfg, ws.pre, ws.art, selection)
    return model, mcfg, state, subset_info, selection


# ---------------------------------------------------------------- exports --

def export_recorded(model, ws: Workspace, run_dir: Path, device) -> dict:
    art, ctx = device_and_tensors(ws, device)
    out, logits_by_split = {}, {}
    for s in D.SPLITS:
        logits = score_recorded(model, ctx[s], art, device)
        logits_by_split[s] = logits
        path = Path(run_dir) / "diagnostics" / "recorded_slates" / f"{s}_scores.csv"
        info = C.write_recorded_slate_scores(path, ws.ctx[s], logits)
        info["validation"] = validate_recorded(path, ws.ctx[s])
        bad = C.recorded_eligibility(ws.ctx[s], ws.art)
        bad.to_csv(Path(run_dir) / "diagnostics" / "recorded_slates" / f"{s}_catalogue_ineligible.csv", index=False)
        info["catalogue_ineligible_recorded_candidates"] = bad.reason.value_counts().to_dict()
        if s in D.LABELLED_SPLITS:
            info["metrics"] = impression_metrics(ws.ctx[s].recorded["req_pos"],
                                                 ws.ctx[s].recorded["label"].astype(int), logits, k=5)
        out[s] = info
    return out, logits_by_split


def _recorded_bad_counts(ws, s):
    bad = C.recorded_eligibility(ws.ctx[s], ws.art)
    counts = np.zeros(len(ws.ctx[s].requests), np.int64)
    if len(bad):
        pos = pd.Index(ws.ctx[s].requests.behaviour_id).get_indexer(bad.behaviour_id)
        np.add.at(counts, pos, 1)
    return counts


def recompute_shortlist_check(scorer, ws: Workspace, split: str, out_root: Path, positions: np.ndarray, seed: int,
                              n: int = 32, block: int = 16) -> dict:
    """Without a full score file, re-score sampled requests over the full eligible
    catalogue and compare the exported shortlist with the reference ranking."""
    import json as _json
    from .validate import eligible_ids, expected_top

    ranks = pd.read_csv(Path(out_root) / "rankings" / split / "submission_top_100.csv", dtype={"behaviour_id": str})
    rng = np.random.default_rng(seed + 7)
    pick = np.sort(rng.choice(len(positions), size=min(n, len(positions)), replace=False)) if len(positions) else []
    mismatches = 0
    rb = block
    for j in pick:
        pos = positions[j]
        # Re-score the exact export block so matrix shapes (and GPU kernels) match the export.
        start = (j // rb) * rb
        blk = positions[start:start + rb]
        logits = scorer.logits(blk)[j - start]
        elig = scorer.eligible_mask(blk)[j - start]
        ids = ws.art.ids[ws.ctx[split].catalogue_idx][elig.cpu().numpy()]
        if not np.array_equal(ids, eligible_ids(ws.ctx[split], ws.art, pos)):
            mismatches += 1
            continue
        if expected_top(ids, logits[elig].cpu().numpy()) != _json.loads(ranks.article_ids.iloc[j]):
            mismatches += 1
    return {"requests_checked": int(len(pick)), "mismatches": int(mismatches), "passed": mismatches == 0}


def model_digest(model) -> str:
    """SHA-256 of the state dict: ties a resumable export to the exact weights."""
    from .train import params_digest
    return params_digest(model)


def export_catalogue(model, ws: Workspace, cfg: dict, out_root: Path, device, request_sets: dict,
                     full_scores: dict, recorded_logits: dict | None = None, log=print, check_blocks=None) -> dict:
    """Run + validate catalogue exports for {split: request positions}."""
    art, ctx = device_and_tensors(ws, device)
    results = {}
    for s, positions in request_sets.items():
        scorer = C.CatalogueScorer(model, ctx[s], art, device, cfg["pair_chunk"], cfg["inference_amp"])
        clicks = C.click_eligibility(ws.ctx[s], ws.art) if s in D.LABELLED_SPLITS else None
        progress = C.export_split(scorer, out_root, positions, write_full_scores=full_scores[s], click_info=clicks,
                                  recorded_counts=_recorded_bad_counts(ws, s), request_block=cfg["request_block"],
                                  log=log, model_digest=model_digest(model))
        validation = validate_export(ws.ctx[s], ws.art, out_root, positions, full_scores=full_scores[s],
                                     check_blocks=check_blocks, seed=cfg["seed"],
                                     recorded_logits=None if recorded_logits is None else recorded_logits[s])
        if not full_scores[s]:
            validation["checks"]["recomputed_shortlist_check"] = recompute_shortlist_check(
                scorer, ws, s, Path(out_root), positions, cfg["seed"], block=cfg["request_block"])
            if not validation["checks"]["recomputed_shortlist_check"]["passed"]:
                validation["passed"] = False
                validation["errors"].append("shortlist != top-100 of recomputed full catalogue scores")
        manifest = pd.read_csv(Path(out_root) / "diagnostics" / "catalogue" / f"{s}_requests.csv",
                               dtype={"behaviour_id": str})
        summary = {"requests": int(len(manifest)), "scored_pairs": int(manifest.eligible_count.sum()),
                   "empty_pools": int((manifest.eligible_count == 0).sum()),
                   "shortlist_length": manifest.shortlist_length.describe().to_dict(),
                   "shortlist_share_with_text_mean": float(manifest.shortlist_share_with_text.mean()),
                   "shortlist_median_age_h_median": float(manifest.shortlist_median_age_h.median()),
                   "shortlist_distinct_categories_mean": float(manifest.shortlist_distinct_categories.mean()),
                   "seconds": progress["seconds"], "full_scores_written": bool(full_scores[s])}
        if s in D.LABELLED_SPLITS:
            summary["recall"] = recall_summary(manifest)
        results[s] = {"progress": {k: v for k, v in progress.items() if k != "signature"}, "validation": validation,
                      "summary": summary}
        status = "PASSED" if validation["passed"] else f"FAILED ({validation['n_errors']} errors)"
        log(f"{s}: catalogue export validation {status}")
    return results


def export_budget(ws: Workspace, cfg: dict, scopes: dict, bytes_per_row: float | None = None,
                  pairs_per_second: float | None = None) -> dict:
    """Exact eligible-pair counts for the PLANNED exports (union pool) and for
    every agreed request, plus selection cost per evaluation and training pairs."""
    out = {}
    for s in D.SPLITS:
        el = C.catalogue_eligibility(ws.ctx[s], ws.art)
        counts = el["eligible_count"]
        planned = counts[scopes[f"export_{s}"]["positions"]]
        bpr = bytes_per_row or C.feasibility_estimate(el, ws.ctx[s])["bytes_per_row_assumed"]
        def block(c):
            pairs = int(c.sum())
            b = {"requests": int(len(c)), "eligible_pairs": pairs, "empty_pools": int((c == 0).sum()),
                 "pools_below_shortlist_size": int((c < C.SHORTLIST_SIZE).sum()),
                 "eligible_per_request_median": float(np.median(c)) if len(c) else 0.0,
                 "full_scores_csv_bytes_estimate": int(pairs * bpr)}
            if pairs_per_second:
                b["scoring_hours_estimate"] = pairs / pairs_per_second / 3600
            return b
        out[s] = {"planned_export": block(planned), "all_requests": block(counts), "bytes_per_row": bpr,
                  "export_scope": scopes[f"export_{s}"]["info"]["limit"]}
    sel = C.catalogue_eligibility(ws.ctx["validation"], ws.art)["eligible_count"][
        scopes["selection_validation"]["positions"]]
    out["selection_per_evaluation_pairs"] = int(sel.sum())
    out["training_requests"] = int(scopes["training"]["info"]["requests"])
    out["planned_export_pairs_total"] = int(sum(out[s]["planned_export"]["eligible_pairs"] for s in D.SPLITS))
    out["all_requests_pairs_total"] = int(sum(out[s]["all_requests"]["eligible_pairs"] for s in D.SPLITS))
    out["max_export_pairs"] = cfg["max_export_pairs"]
    out["pool"] = ws.reports["pool_fingerprint"]["policy"]
    return out


def check_export_budget(budget: dict) -> None:
    """Stop an infeasible export before it starts (never shrink the pool instead)."""
    if budget["planned_export_pairs_total"] > budget["max_export_pairs"]:
        raise RuntimeError(f"planned export is {budget['planned_export_pairs_total']:,} pairs > max_export_pairs="
                           f"{budget['max_export_pairs']:,}. Inspect the budget; use a sampled export scope or raise "
                           "max_export_pairs deliberately after checking runtime/storage.")


def full_score_gate(budget: dict, cfg: dict, out_roots: list) -> dict:
    """Per split: may the full score CSV for the planned export requests be written?

    Never silently changes the pool: a split that fails the gate still gets
    complete top-100 shortlists for every planned request (re-validated against
    recomputed full scores), and the manifest records why scores are absent."""
    frees = [disk_free_bytes(r) for r in out_roots]
    free = min(f for f in frees if f is not None) if any(f is not None for f in frees) else 0
    decisions, remaining = {}, free
    for s in D.SPLITS:
        need = budget[s]["planned_export"]["full_scores_csv_bytes_estimate"]
        wanted = s in cfg["full_score_splits"]
        ok = wanted and need * cfg["full_score_storage_margin"] <= remaining
        decisions[s] = {"requested": wanted, "write": bool(ok), "estimated_bytes": int(need),
                        "free_bytes_at_decision": int(remaining),
                        "reason": None if ok else ("not requested" if not wanted else
                                                   "insufficient free space at the export location")}
        if ok:
            remaining -= need
    return decisions


# ------------------------------------------------------------ diagnostics --

def vector_masked_metrics(model, ws: Workspace, device, selector: CatalogueSelector | None = None) -> dict:
    """Selection-sample Recall@100 / nDCG@100 (and diagnostic recorded-slate AUC)
    with every text/image vector hidden.

    The test catalogue is mostly without vectors; this shows how much shortlist
    quality depends on them (a robustness diagnostic, not a model choice)."""
    art, ctx = device_and_tensors(ws, device)
    masked = F.ArticleTable(art.ids, dict(art.tensors), art.report)
    masked.tensors["has_text"] = torch.zeros_like(art.tensors["has_text"])
    masked.tensors["has_image"] = torch.zeros_like(art.tensors["has_image"])
    diag, _ = evaluate_recorded(model, ctx["validation"], masked, device)
    out = {"diagnostic_recorded_slate_auc": diag["mean_impression_auc"]}
    if selector is not None:
        sel = selector.evaluate(model, art_dev=masked)
        out.update({k: sel[k] for k in ("known_click_recall@100", "observed_click_ndcg@100",
                                        "eligible_only_recall@100", "requests_evaluated")})
    return out


@torch.no_grad()
def attention_examples(model, ws: Workspace, device, n_requests: int = 3, seed: int = 0) -> dict:
    """Attention weights over history items for clicked vs unclicked recorded
    candidates in a few validation impressions, plus mean attention entropy
    relative to uniform pooling. Diagnostic only: not evidence of reader motives."""
    art, ctx = device_and_tensors(ws, device)
    v, rec = ctx["validation"], ws.ctx["validation"].recorded
    rp = torch.from_numpy(rec["req_pos"]).to(device)
    ai = torch.from_numpy(rec["art_idx"]).to(device)
    take = torch.arange(min(len(rp), 50_000), device=device)
    _, w = model(v.tensors, art.tensors, len(art.ids), rp[take], ai[take], return_weights=True)
    mask = v.tensors["hist_mask"][v.tensors["req_user"][rp[take]]]
    n_valid = mask.sum(1).clamp_min(1).float()
    entropy_ratio = None
    if model.cfg.attention_mode == "softmax":
        ent = -(w.clamp_min(1e-12).log() * w * mask).sum(1)
        ok = n_valid > 1
        entropy_ratio = float((ent[ok] / n_valid[ok].log()).mean()) if ok.any() else None
    rng = np.random.default_rng(seed)
    labels = rec["label"]
    cand = [p for p in np.unique(rec["req_pos"]) if labels[rec["offsets"][p]:rec["offsets"][p + 1]].sum() > 0]
    examples = []
    for pos in rng.choice(cand, size=min(n_requests, len(cand)), replace=False) if cand else []:
        lo, hi = rec["offsets"][pos], rec["offsets"][pos + 1]
        user = int(ws.ctx["validation"].tensors["req_user"][pos])
        hist_idx = ws.ctx["validation"].tensors["hist_idx"][user].numpy()
        hist_mask = ws.ctx["validation"].tensors["hist_mask"][user].numpy()
        r = torch.full((hi - lo,), int(pos), device=device)
        a = torch.from_numpy(rec["art_idx"][lo:hi]).to(device)
        logit, ww = model(v.tensors, art.tensors, len(art.ids), r, a, return_weights=True)
        ww = ww.cpu().numpy()
        cats = ws.art.tensors["category"].numpy()
        for j in range(hi - lo):
            top = np.argsort(-ww[j] * hist_mask)[:5]
            examples.append({"behaviour_id": ws.ctx["validation"].requests.behaviour_id.iloc[pos],
                             "candidate_article_id": int(rec["article_id"][lo + j]), "clicked": int(labels[lo + j]),
                             "logit": float(logit[j]),
                             "candidate_category_index": int(cats[rec["art_idx"][lo + j]]),
                             "top_history_article_ids": ws.art.ids[hist_idx[top]].tolist(),
                             "top_history_category_index": cats[hist_idx[top]].tolist(),
                             "top_weights": ww[j][top].round(4).tolist()})
    return {"attention_mode": model.cfg.attention_mode,
            "mean_entropy_ratio_vs_uniform": entropy_ratio,
            "pairs_used_for_entropy": int(len(take)), "examples": examples,
            "caveat": "Attention weights are model internals; they are not causal explanations of reader behaviour."}


# ------------------------------------------------------------ persistence --

REQUIRED_RUN_FILES = ["run_config.json", "metrics.json", "requirements.txt", "export_manifest.json"] + MODEL_FILES


def persist_run(run_dir: Path, persist_dir: Path, log=print) -> dict:
    """Copy + verify every file. No completion marker yet: see mark_run_complete."""
    records = mirror_tree(run_dir, persist_dir)
    required = [r for r in REQUIRED_RUN_FILES if (Path(run_dir) / r).is_file()]
    missing = sorted(set(REQUIRED_RUN_FILES) - set(required))
    if missing:
        raise FileNotFoundError(f"run bundle incomplete; missing {missing}")
    log(f"persisted {len(records)} files to {persist_dir} (not yet marked complete)")
    return {"files": len(records)}


def mark_run_complete(run_dir: Path, persist_dir: Path, reload_result: dict, log=print) -> dict:
    """Write _COMPLETE_RUN.json LAST, only after the reload check from the
    persistent copy passed; re-mirrors any files written since persist_run."""
    if not reload_result.get("passed"):
        raise RuntimeError("reload check did not pass; the run is NOT marked complete")
    mirror_tree(run_dir, persist_dir)
    required = [r for r in REQUIRED_RUN_FILES if (Path(run_dir) / r).is_file()] + ["reload_check.json"]
    marker = write_completion_marker(persist_dir, "_COMPLETE_RUN.json", required, {"reload_check_passed": True})
    log(f"completion marker written last: {persist_dir / '_COMPLETE_RUN.json'}")
    return marker


def run_is_complete(persist_dir: Path) -> bool:
    return marker_is_valid(persist_dir, "_COMPLETE_RUN.json")


def reload_check(persist_dir: Path, ws: Workspace, cfg: dict, device, n_requests: int = 4,
                 tolerance: float = 1e-4) -> dict:
    """Load model + preprocessing from the persistent copy (no fitting) and
    compare predictions with the persisted exports."""
    model, mcfg, pre, meta = load_model(persist_dir, device)
    out = {"loaded_from": str(persist_dir), "article_ids_match": check_article_ids(persist_dir, ws.art),
           "preprocessing_matches_workspace": pre.to_json() == ws.pre.to_json()}
    art, ctx = device_and_tensors(ws, device)
    # (1) recorded-slate validation scores
    rec_path = Path(persist_dir) / "diagnostics" / "recorded_slates" / "validation_scores.csv"
    from .validate import read_scores_bytes
    saved = read_scores_bytes(rec_path.read_bytes(), header=True)
    take = np.arange(min(len(saved), 5000))
    v = ctx["validation"]
    r = torch.from_numpy(ws.ctx["validation"].recorded["req_pos"][take]).to(device)
    a = torch.from_numpy(ws.ctx["validation"].recorded["art_idx"][take]).to(device)
    with torch.no_grad():
        new = model(v.tensors, art.tensors, len(art.ids), r, a).float().cpu().numpy()
    out["recorded_max_abs_logit_diff"] = float(np.abs(new - saved.score.to_numpy()[take]).max())
    # (2) catalogue: re-score the first exported validation requests from the persistent copy
    root = Path(persist_dir)
    blocks_path = root / "diagnostics" / "catalogue" / "validation_score_blocks.csv"
    scores_path = root / "predictions" / "validation" / "submission_click_probabilities.csv"
    scopes_path = root / "selection" / "request_scopes.json"
    if scopes_path.is_file():
        saved_pool = read_json(scopes_path).get("pool_fingerprint", {}).get("ids_and_published_sha256")
        out["pool_fingerprint_matches"] = saved_pool == ws.reports["pool_fingerprint"]["ids_and_published_sha256"]
    if blocks_path.is_file() and scores_path.is_file() and scopes_path.is_file():
        from .validate import read_block
        blocks = pd.read_csv(blocks_path)
        first = blocks.iloc[0]
        frame = read_block(scores_path, int(first.byte_start), int(first.byte_end))
        ids = read_json(scopes_path)["export_validation"]["behaviour_ids"]
        all_pos = pd.Index(ws.ctx["validation"].requests.behaviour_id).get_indexer(ids)
        positions = all_pos[int(first.first_request_index):int(first.last_request_index) + 1][:n_requests]
        scorer = C.CatalogueScorer(model, v, art, device, cfg["pair_chunk"], False)
        logits = scorer.logits(positions)
        elig = scorer.eligible_mask(positions)
        rows, cols = elig.nonzero(as_tuple=True)
        recomputed = logits[rows, cols].cpu().numpy()
        k = len(recomputed)
        out["catalogue_pairs_compared"] = int(k)
        out["catalogue_max_abs_logit_diff"] = float(np.abs(recomputed - frame.score.to_numpy()[:k]).max()) if k else None
        out["catalogue_ids_match"] = bool(np.array_equal(ws.art.ids[ws.ctx["validation"].catalogue_idx][cols.cpu().numpy()],
                                                         frame.article_id.to_numpy()[:k]))
    out["tolerance"] = tolerance
    out["passed"] = bool(out["article_ids_match"] and out["preprocessing_matches_workspace"]
                         and out["recorded_max_abs_logit_diff"] <= tolerance
                         and (out.get("catalogue_max_abs_logit_diff") is None or out["catalogue_max_abs_logit_diff"] <= tolerance)
                         and out.get("catalogue_ids_match", True) and out.get("pool_fingerprint_matches", True))
    return out


def requirements_text() -> str:
    env = environment_info()
    lines = [f"{k}=={v}" for k, v in env["packages"].items() if v]
    return "# Versions used by this run (load-time dependencies)\n" + "\n".join(lines) + "\n"


__all__ = ["make_config", "prepare_inputs", "preflight", "train_model", "export_recorded", "export_catalogue",
           "full_score_gate", "export_budget", "check_export_budget", "resolve_request_scopes", "scope_ids",
           "persist_run", "mark_run_complete", "run_is_complete", "reload_check", "requirements_text",
           "METRIC_DEFINITIONS", "parameter_counts", "git_info", "environment_info", "evaluate_recorded"]
