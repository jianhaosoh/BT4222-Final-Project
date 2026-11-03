"""Catalogue validation used for checkpoint/trial selection (Experiments/README.md).

Objective: maximise macro known-click Recall@100 over a FIXED sample of
validation requests, each scored against its FULL time-eligible catalogue; ties
by observed-click nDCG@100, then the earlier checkpoint/trial. Rankings are
computed in batches with the export's deterministic order (descending score,
ascending article_id); no score CSV is written during selection.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch

from . import catalogue as C
from .metrics import request_retrieval

SELECTION_OBJECTIVE = "validation catalogue known-click Recall@100 (macro over requests with known clicks)"
# v2: shared union pool (published_at_or_before_request_v1) + shared BLAKE2b request sampling.
# v1 (separate split catalogues, numpy-sampled requests) checkpoints are refused, not reinterpreted.
SELECTION_PROTOCOL_VERSION = "catalogue_recall@100_v2_union_pool_hash_requests"


class CatalogueSelector:
    """Evaluates a model on the fixed validation selection sample."""

    def __init__(self, ctx_cpu, art_cpu, ctx_dev, art_dev, positions: np.ndarray, device, cfg: dict,
                 sample_info: dict | None = None, pool: dict | None = None):
        self.ctx_cpu, self.art_cpu, self.ctx_dev, self.art_dev = ctx_cpu, art_cpu, ctx_dev, art_dev
        self.positions = np.asarray(positions, dtype=np.int64)
        self.device, self.cfg = device, cfg
        clicks = C.click_eligibility(ctx_cpu, art_cpu, self.positions)
        self.clicks = [clicks[p] for p in self.positions]
        if any(c is None for c in self.clicks):
            raise ValueError("selection requires a labelled split (validation)")
        ids = ctx_cpu.requests.behaviour_id.to_numpy()[self.positions]
        self.behaviour_ids = ids.tolist()
        digest = hashlib.sha256("\n".join(sorted(self.behaviour_ids)).encode()).hexdigest()
        cat_digest = hashlib.sha256(art_cpu.ids[ctx_cpu.catalogue_idx].tobytes()).hexdigest()
        if sample_info is not None and sample_info.get("behaviour_ids_sha256") not in (None, digest):
            raise ValueError("selection positions disagree with their sampling record")
        self.protocol = {
            "version": SELECTION_PROTOCOL_VERSION, "objective": SELECTION_OBJECTIVE, "direction": "maximize",
            "secondary": "observed-click nDCG@100 (ideal over all original known clicks, capped at 100)",
            "tie_breaker": "higher Recall@100, then higher nDCG@100, then the earlier checkpoint/trial",
            "selection_k": C.SELECTION_K, "export_k": C.EXPORT_K, "shortlist_size": C.SHORTLIST_SIZE, "ranking_order": "descending score, ties by ascending article_id",
            "partition": "official validation split (development data; used repeatedly for selection)",
            "pool": "full time-eligible catalogue for every sampled request (no sampled negatives)",
            "eligibility_rules": C.ELIGIBILITY_RULES, "pool_policy": "published_at_or_before_request_v1",
            "catalogue_policy": cfg["catalogue_policy"]["validation"], "catalogue_ids_sha256": cat_digest,
            "pool_fingerprint": pool,
            "sample": {"scope": "selection_validation", "requests": int(len(self.positions)),
                       "of_validation_requests": int(len(ctx_cpu.requests)),
                       "seed": cfg.get("request_sampling_seed"),
                       "rule": (sample_info or {}).get("rule", "explicit positions"),
                       "used": (sample_info or {}).get("used"), "reference": (sample_info or {}).get("reference"),
                       "fixed": "before training; shared by every checkpoint, control and trial",
                       "behaviour_ids_sha256": digest},
            "interval_steps": cfg["selection_interval_steps"], "evaluate_at_epoch_end": True,
            "early_stopping_patience_evaluations": cfg["early_stopping_patience"],
            "min_improvement": cfg["min_improvement"],
            "reserved_final_partition": None,  # none is declared by the shared protocol; test has no labels
        }

    def sample_record(self) -> dict:
        return {"protocol": self.protocol, "positions": self.positions.tolist(), "behaviour_ids": self.behaviour_ids}

    @torch.no_grad()
    def evaluate(self, model, art_dev=None) -> dict:
        """Recall@100 / nDCG@100 (+ eligible-only recall) of the current weights."""
        t0 = time.time()
        was_training = model.training
        scorer = C.CatalogueScorer(model, self.ctx_dev, art_dev or self.art_dev, self.device,
                                   self.cfg["pair_chunk"], self.cfg["inference_amp"])
        cat_ids = self.art_cpu.ids[self.ctx_cpu.catalogue_idx]
        recall, ndcg, eligible_recall = [], [], []
        stats = {"requests_without_known_clicks": 0, "empty_pool_requests_with_known_clicks": 0,
                 "requests_with_ties_crossing_cutoff": 0, "scored_pairs": 0,
                 "clicked_articles_total": 0, "clicked_articles_eligible": 0, "clicked_articles_in_shortlist": 0}
        block = self.cfg["request_block"]
        for lo in range(0, len(self.positions), block):
            pos = self.positions[lo:lo + block]
            top = scorer.topk(pos, k=C.SELECTION_K)
            order, lengths, ties = (top[key].cpu().numpy() for key in
                                    ("positions", "lengths", "tie_crosses_cutoff"))
            stats["scored_pairs"] += int(top["eligible_count"].sum())
            for j in range(len(pos)):
                info = self.clicks[lo + j]
                ranked = cat_ids[order[j, :lengths[j]]]
                n, hits, dcg, ideal = request_retrieval(ranked, info["ids"], C.SELECTION_K)
                stats["requests_with_ties_crossing_cutoff"] += int(ties[j])
                if n == 0:
                    stats["requests_without_known_clicks"] += 1
                    continue
                stats["empty_pool_requests_with_known_clicks"] += int(lengths[j] == 0)
                recall.append(hits / n)
                ndcg.append(dcg / ideal if ideal > 0 else 0.0)
                n_el = int(info["eligible"].sum())
                if n_el:
                    eligible_recall.append(hits / n_el)
                stats["clicked_articles_total"] += n
                stats["clicked_articles_eligible"] += n_el
                stats["clicked_articles_in_shortlist"] += hits
        if was_training:
            model.train()
        return {"known_click_recall@100": float(np.mean(recall)) if recall else 0.0,
                "observed_click_ndcg@100": float(np.mean(ndcg)) if ndcg else 0.0,
                "eligible_only_recall@100": float(np.mean(eligible_recall)) if eligible_recall else None,
                "requests_evaluated": int(len(recall)), **stats, "seconds": time.time() - t0,
                "pool": "selection sample of validation requests x full eligible catalogue"}
