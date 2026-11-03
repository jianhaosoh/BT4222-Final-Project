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
import pyarrow.parquet as pq

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
    "max_export_pairs": None,       # stop before exporting more pairs; raise deliberately for full scope
    "max_selection_pairs_per_evaluation": 50_000_000,
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


def make_config(mode: str = "full", **overrides) -> dict:
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
    # Eligibility compares whole seconds; 3.1 compares full timestamps. Refuse data where that could differ.
    for name, values in [("articles.published_time", articles.published_time)] + \
            [(f"{s}.impression_time", behaviours[s].impression_time) for s in D.SPLITS]:
        ts = pd.to_datetime(values).dropna()
        if len(ts) and (ts.dt.microsecond.ne(0).any() or ts.dt.nanosecond.ne(0).any()):
            raise ValueError(f"{name} has sub-second timestamps; second-resolution eligibility would differ from 3.1")
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
    # Pool identity = policy + sorted IDs + publication times (content), never file paths/mtimes.
    pool = sampling.pool_fingerprint(art.ids[ctx["train"].catalogue_idx],
                                     art.tensors["pub_s"].numpy()[ctx["train"].catalogue_idx], {})
    pool_sources = {n: file_record(paths.articles[n], 0) for n in union}  # size/mtime only; not a content hash
    for s in D.SPLITS:
        if not np.array_equal(ctx[s].catalogue_idx, ctx["train"].catalogue_idx):
            raise RuntimeError(f"{s}: pool differs from train under {POOL_POLICY['name']}")
    lookup = {s: {"history_articles_not_in_union": int(len(np.setdiff1d(events[s].article_id.to_numpy(np.int64),
                                                                       articles.article_id.to_numpy()))),
                  "recorded_candidates_not_in_union": int(len(np.setdiff1d(pairs[s].candidate_article_id.to_numpy(np.int64),
                                                                          articles.article_id.to_numpy())))}
              for s in D.SPLITS}
    reports = {"catalogues": catalogues["report"], "referenced_articles_not_in_any_catalogue": int(len(not_found)),
               "pool_policy": POOL_POLICY, "pool_fingerprint": pool, "pool_source_files": pool_sources,
               "article_lookup_coverage": lookup,
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
    total_bytes = sum(v["feasibility"]["retained_output_bytes_estimate"] for v in report["eligibility"].values())
    report["all_splits"] = {"eligible_pairs": total_pairs, "retained_output_bytes_estimate": total_bytes,
                            "retained_output_gib_estimate": total_bytes / 1024 ** 3}
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
        path = Path(run_dir) / "diagnostics" / "recorded_slates" / f"{s}_scores.parquet"
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


def export_catalogue(model, ws: Workspace, cfg: dict, out_root: Path, device, request_sets: dict,
                     log=print) -> dict:
    """Full eligible-catalogue inference, retained Parquet outputs and validation."""
    art, ctx = device_and_tensors(ws, device)
    results = {}
    for s, positions in request_sets.items():
        scorer = C.CatalogueScorer(model, ctx[s], art, device, cfg["pair_chunk"], cfg["inference_amp"])
        clicks = C.click_eligibility(ws.ctx[s], ws.art, positions) if s in D.LABELLED_SPLITS else None
        progress = C.export_split(scorer, out_root, positions, click_info=clicks,
                                  recorded_counts=_recorded_bad_counts(ws, s), request_block=cfg["request_block"], log=log)
        validation = validate_export(ws.ctx[s], ws.art, out_root, positions)
        manifests = pq.ParquetFile(Path(out_root) / "diagnostics" / "catalogue" / f"{s}_requests.parquet")
        summaries = {k: [] for k in (C.SELECTION_K, C.EXPORT_K)}
        lengths, requests, empty = 0, 0, 0
        min_len, max_len = C.EXPORT_K, 0
        for batch in manifests.iter_batches(batch_size=10_000):
            frame = batch.to_pandas()
            requests += len(frame)
            lengths += int(frame.shortlist_length.sum())
            empty += int((frame.eligible_count == 0).sum())
            if len(frame):
                min_len, max_len = min(min_len, int(frame.shortlist_length.min())), max(max_len, int(frame.shortlist_length.max()))
            if s in D.LABELLED_SPLITS:
                for k in summaries:
                    summaries[k].append(recall_summary(frame, k))
        summary = {"requests": requests, "scored_pairs": progress["scored_pairs"], "empty_pools": empty,
                   "retained_score_rows": progress["rows_scores"], "seconds": progress["seconds"],
                   "shortlist_length": {"min": min_len if requests else 0, "max": max_len,
                                        "mean": lengths / requests if requests else 0.0}}
        if s in D.LABELLED_SPLITS:
            metrics = {}
            for k, chunks in summaries.items():
                known = sum(c["requests_with_known_clicks"] for c in chunks)
                with_eligible = sum(c["requests_with_known_clicks"] - c.get("requests_with_no_eligible_click", 0) for c in chunks)
                for name in (f"known_click_recall@{k}", f"observed_click_ndcg@{k}"):
                    metrics[name] = sum(c.get(name, 0) * c["requests_with_known_clicks"] for c in chunks) / known if known else None
                name = f"eligible_only_recall@{k}"
                metrics[name] = sum((c.get(name) or 0) * (c["requests_with_known_clicks"] - c.get("requests_with_no_eligible_click", 0))
                                    for c in chunks) / with_eligible if with_eligible else None
                metrics[f"requests_with_ties_crossing_cutoff_{k}"] = sum(c.get("requests_with_ties_crossing_cutoff", 0) for c in chunks)
                metrics[f"clicked_articles_in_top_{k}"] = sum(c.get("clicked_articles_in_shortlist", 0) for c in chunks)
            for name in ("requests_total", "requests_with_known_clicks", "requests_without_known_clicks",
                         "requests_with_no_eligible_click", "empty_pool_requests_with_known_clicks", "clicked_articles_total",
                         "clicked_articles_eligible", "clicked_excluded_not_in_catalogue", "clicked_excluded_missing_published_time",
                         "clicked_excluded_published_after_request"):
                metrics[name] = sum(c.get(name, 0) for c in summaries[C.SELECTION_K])
            metrics["eligibility_coverage_of_clicks"] = (metrics["clicked_articles_eligible"] / metrics["clicked_articles_total"]
                                                        if metrics["clicked_articles_total"] else None)
            summary["recall"] = metrics
        results[s] = {"progress": progress, "validation": validation, "summary": summary}
        log(f"{s}: catalogue export validation {'PASSED' if validation['passed'] else 'FAILED'}")
        if not validation["passed"]:
            raise RuntimeError(f"{s}: export validation failed: {validation['errors']}")
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
                 "retained_score_rows": int(np.minimum(c, C.EXPORT_K).sum()),
                 "retained_output_bytes_estimate": int(np.minimum(c, C.EXPORT_K).sum() * 48 + len(c) * (32 + C.EXPORT_K * 8))}
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
    if budget["max_export_pairs"] is not None and budget["planned_export_pairs_total"] > budget["max_export_pairs"]:
        raise RuntimeError(f"planned export is {budget['planned_export_pairs_total']:,} pairs > max_export_pairs="
                           f"{budget['max_export_pairs']:,}. Inspect the budget; use a sampled export scope or raise "
                           "max_export_pairs deliberately after checking runtime/storage.")


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
    """Copy and verify the selected model, exports and run records."""
    records = mirror_tree(run_dir, persist_dir)
    required = [r for r in REQUIRED_RUN_FILES if (Path(run_dir) / r).is_file()]
    missing = sorted(set(REQUIRED_RUN_FILES) - set(required))
    if missing:
        raise FileNotFoundError(f"run bundle incomplete; missing {missing}")
    log(f"persisted {len(records)} files to {persist_dir}")
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
    """Reload the persistent model/preprocessing; compare small saved score batches."""
    root = Path(persist_dir)
    model, _, pre, _ = load_model(root, device)
    names = read_json(root / "model" / "feature_dictionary.json")
    out = {"loaded_from": str(root), "article_ids_match": check_article_ids(root, ws.art),
           "preprocessing_matches_workspace": pre.to_json() == ws.pre.to_json(),
           "feature_names_match": names["ordered_features"]["request_context"] == F.REQUEST_DENSE_COLUMNS(pre)
                                  and names["ordered_features"]["candidate"] == F.PAIR_DENSE_COLUMNS}
    art, contexts = device_and_tensors(ws, device)
    context = contexts["validation"]
    lookup = pd.Index(ws.ctx["validation"].requests.behaviour_id)
    for name, path in {
        "recorded": root / "diagnostics" / "recorded_slates" / "validation_scores.parquet",
        "catalogue": root / "predictions" / "validation" / "submission_article_scores.parquet",
    }.items():
        parquet = pq.ParquetFile(path)
        batch = next(parquet.iter_batches(batch_size=max(1, n_requests * C.EXPORT_K)), None)
        if batch is None:
            out[f"{name}_pairs_compared"] = 0
            out[f"{name}_max_abs_logit_diff"] = 0.0
            continue
        frame = batch.to_pandas()
        frame = frame[frame.behaviour_id.isin(frame.behaviour_id.drop_duplicates().iloc[:n_requests])]
        req = lookup.get_indexer(frame.behaviour_id)
        if (req < 0).any():
            raise ValueError("saved reload probe has an unknown request")
        ai = ws.art.lookup(frame.article_id.to_numpy(np.int64))
        with torch.no_grad():
            logits = model(context.tensors, art.tensors, len(art.ids),
                           torch.from_numpy(req).to(device), torch.from_numpy(ai).to(device)).float().cpu().numpy()
        out[f"{name}_pairs_compared"] = len(frame)
        out[f"{name}_max_abs_logit_diff"] = float(np.abs(logits - frame.score.to_numpy()).max()) if len(frame) else 0.0
    scopes = root / "selection" / "request_scopes.json"
    if scopes.is_file():
        saved_pool = read_json(scopes).get("pool_fingerprint", {}).get("ids_and_published_sha256")
        out["pool_fingerprint_matches"] = saved_pool == ws.reports["pool_fingerprint"]["ids_and_published_sha256"]
    out["tolerance"] = tolerance
    out["passed"] = bool(out["article_ids_match"] and out["preprocessing_matches_workspace"] and out["feature_names_match"]
                         and out["recorded_pairs_compared"] > 0
                         and out["recorded_max_abs_logit_diff"] <= tolerance
                         and out["catalogue_max_abs_logit_diff"] <= tolerance
                         and out.get("pool_fingerprint_matches", True))
    return out


def requirements_text() -> str:
    env = environment_info()
    lines = [f"{k}=={v}" for k, v in env["packages"].items() if v]
    return "# Versions used by this run (load-time dependencies)\n" + "\n".join(lines) + "\n"


__all__ = ["make_config", "prepare_inputs", "preflight", "train_model", "export_recorded", "export_catalogue",
           "export_budget", "check_export_budget", "resolve_request_scopes", "scope_ids",
           "persist_run", "mark_run_complete", "run_is_complete", "reload_check", "requirements_text",
           "METRIC_DEFINITIONS", "parameter_counts", "git_info", "environment_info", "evaluate_recorded"]
