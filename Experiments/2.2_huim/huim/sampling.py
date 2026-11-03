"""Shared request sampling and catalogue-pool policy (aligned with 3.1).

Request sampling (Experiments/README.md, "Shared request sampling"):
  priority = blake2b(f"{seed}:{split}:{behaviour_id}", digest_size=8).hexdigest()
  select the smallest N by (priority, behaviour_id); process the selected set by
  (impression_time, user_id, impression_id). Seed 42 is the shared sampling seed,
  independent of model-training seeds. A smaller N is a prefix of the same order.
Requests are sampled, never articles: every selected request keeps its complete
recorded slate (training/diagnostics) and its FULL eligible article pool (inference).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUEST_SAMPLING_SEED = 42
SAMPLING_RULE = ("smallest N by (BLAKE2b-64 hexdigest of f'{seed}:{split}:{behaviour_id}', behaviour_id); "
                 "processed by (impression_time, user_id, impression_id); requests sampled, never articles")

POOL_POLICY = {
    "name": "published_at_or_before_request_v1",
    "sources": ["train_val/articles.parquet", "test/articles.parquet"],
    "source_order": ["train_val", "test"],
    "duplicate_precedence": "first source record per article_id wins, including its missing fields",
    "same_pool_for_splits": ["train", "validation", "test"],
    "publication_field": "published_time",
    "eligible_if": "published_time is known and published_time <= impression_time",
    "missing_publication": "exclude (known clicks on such articles stay in metric denominators/ideals)",
    "age_limit_hours": None, "access_filter": None, "embedding_coverage_filter": None,
    "vector_precedence": "train_val then test; first source record wins, including legitimately missing vectors",
    "reference": "Experiments/3.1_pretrained_similarity/catalogue_pipeline.py POLICY",
}


def priority(seed: int, split: str, behaviour_id: str) -> str:
    return hashlib.blake2b(f"{seed}:{split}:{behaviour_id}".encode(), digest_size=8).hexdigest()


def ids_digest(behaviour_ids) -> str:
    """SHA-256 over the sorted behaviour IDs (set identity, order-independent)."""
    return hashlib.sha256("\n".join(sorted(map(str, behaviour_ids))).encode()).hexdigest()


def chronological(requests: pd.DataFrame, positions) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    sub = requests.iloc[positions]
    order = np.lexsort((sub.impression_id.to_numpy(), sub.user_id.to_numpy(),
                        pd.to_datetime(sub.impression_time).to_numpy()))
    return positions[order]


def hash_prefix(requests: pd.DataFrame, split: str, n: int | None, seed: int = REQUEST_SAMPLING_SEED) -> np.ndarray:
    """Positions (into `requests`) of the first N requests in hash-priority order,
    returned in processing order. n=None selects every request."""
    bids = requests.behaviour_id.to_numpy().astype(str)
    if n is None:
        return chronological(requests, np.arange(len(bids)))
    if int(n) < 1:
        raise ValueError("request limits must be positive, or None for all requests")
    pr = np.array([priority(seed, split, b) for b in bids])
    order = np.lexsort((bids, pr))  # primary: priority; tie: behaviour_id
    return chronological(requests, order[: int(n)])


def read_reference_ids(reference_run: Path | None) -> dict:
    """Saved request IDs from a 3.1 run folder, when available (never required):
    input_audit.json['selection_requests'] and eligibility/{split}_requests.csv."""
    out = {}
    if reference_run is None:
        return out
    root = Path(reference_run)
    audit = root / "input_audit.json"
    if audit.is_file():
        data = json.loads(audit.read_text())
        if data.get("selection_requests"):
            out["selection_validation"] = {"ids": list(map(str, data["selection_requests"])), "source": str(audit),
                                           "seed": data.get("seed"),
                                           "policy": (data.get("eligibility_policy") or {}).get("name")}
    for split in ("train", "validation", "test"):
        path = root / "eligibility" / f"{split}_requests.csv"
        if path.is_file():
            with open(path, newline="") as handle:
                ids = [row["behaviour_id"] for row in csv.DictReader(handle)]
            out[f"export_{split}"] = {"ids": ids, "source": str(path)}
    return out


def resolve_scope(requests: pd.DataFrame, split: str, purpose: str, n: int | None,
                  seed: int = REQUEST_SAMPLING_SEED, reference: dict | None = None) -> tuple[np.ndarray, dict]:
    """Request positions for one scope (training / selection_validation / export),
    preferring saved 3.1 IDs when supplied. Always records our own hash selection
    and whether it agrees, so identical requests can be proven, not assumed."""
    own = hash_prefix(requests, split, n, seed)
    bids = requests.behaviour_id.to_numpy().astype(str)
    info = {"purpose": purpose, "split": split, "limit": n, "seed": seed, "rule": SAMPLING_RULE,
            "population_requests": int(len(requests))}
    used = own
    if reference is not None:
        index = pd.Index(bids)
        pos = index.get_indexer(reference["ids"])
        if (pos < 0).any():
            raise ValueError(f"{purpose}/{split}: {int((pos < 0).sum())} reference IDs from {reference['source']} are "
                             "not in this request population; the inputs differ — do not compare these scopes")
        own_same_n = hash_prefix(requests, split, len(reference["ids"]), seed)
        info["reference"] = {"source": reference["source"], "requests": len(reference["ids"]),
                             "reference_seed": reference.get("seed"), "reference_policy": reference.get("policy"),
                             "matches_own_hash_selection": bool(set(bids[own_same_n]) == set(reference["ids"]))}
        if n is not None and len(reference["ids"]) != n:
            info["reference"]["note"] = f"reference has {len(reference['ids'])} requests; configured limit {n}"
        used = chronological(requests, pos)
        info["used"] = "reference_ids"
    else:
        info["used"] = "own_hash_selection"
    ids = bids[used]
    info.update(requests=int(len(used)), behaviour_ids_sha256=ids_digest(ids),
                first_ids=ids[:5].tolist())
    return used, info


def pool_fingerprint(article_ids: np.ndarray, published_s: np.ndarray, source_records: dict) -> dict:
    """Identity of the catalogue pool: policy, ordered sources, sorted IDs and
    their publication times (a content fingerprint of the pool itself)."""
    order = np.argsort(article_ids)
    h = hashlib.sha256()
    h.update(json.dumps(POOL_POLICY, sort_keys=True).encode())
    h.update(np.asarray(article_ids, dtype=np.int64)[order].tobytes())
    h.update(np.asarray(published_s, dtype=np.int64)[order].tobytes())
    return {"policy": POOL_POLICY["name"], "articles": int(len(article_ids)),
            "ids_and_published_sha256": h.hexdigest(), "sources": source_records}
