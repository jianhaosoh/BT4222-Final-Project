"""Run-bundle persistence and reload-from-persistent-copy checks."""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import torch

from .features import Preprocessor, REQUEST_DENSE_COLUMNS, PAIR_DENSE_COLUMNS
from .io_utils import atomic_write_json, read_json, sha256_file
from .model import ModelConfig, build_model

MODEL_FILES = ["model/weights.pt", "model/model_config.json", "model/preprocessing.json",
               "model/article_ids.npy", "model/feature_dictionary.json"]


def save_model_files(run_dir: Path, model, model_cfg: ModelConfig, pre: Preprocessor, art, selection: dict) -> dict:
    run_dir = Path(run_dir)
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    weights = run_dir / "model" / "weights.pt"
    if not weights.is_file():
        raise FileNotFoundError("selected weights were not saved by training")
    atomic_write_json(run_dir / "model" / "model_config.json", {
        "class": "huim.model.HUIMStage1", "model_config": model_cfg.to_json(), "selection": selection,
        "frozen_components": ["text vectors (gemini-embedding-2, 768-d) — lookup, not fine-tuned",
                              "image vectors (EB-NeRD supplied, 1024-d) — lookup, not fine-tuned"],
        "trained_components": ["text/image linear projections", "metadata embeddings", "event encoder",
                               "candidate attention", "feature branch", "fusion/gate", "MLP head"],
        "weights_sha256": sha256_file(weights),
        "load": "huim.persist.load_model(bundle_dir, device); state_dict loaded with torch.load(weights_only=True)",
    })
    atomic_write_json(run_dir / "model" / "preprocessing.json", pre.to_json())
    state = pre.to_json()
    dictionary = {
        "model_family": "HUIM-inspired candidate-aware attention",
        "ordered_features": {"request_context": REQUEST_DENSE_COLUMNS(pre),
                             "candidate": PAIR_DENSE_COLUMNS,
                             "history_event": ["article representation", "read_time_bin", "scroll_bin", "recency_bin"]},
        "categorical_index_to_value": {
            key: {**{"0": "padding/missing", "1": "unknown"},
                  **{str(i + 2): value for i, value in enumerate(values)}}
            for key, values in state.items()
            if key in ("category_vocab", "article_type_vocab", "sentiment_vocab", "topic_vocab", "subcategory_vocab")},
        "device_one_hot_values": state["device_vocab"],
        "fitted_transformations": "preprocessing.json (saved bins, cross-feature columns, scaling and vocabulary order)",
        "article_row_to_original_id": "article_ids.npy",
        "score": "click logit; sigmoid probability is uncalibrated",
        "embedding_interpretation": "Vector coordinates are not word features; preserve source/dimensions and interpret meaningful groups.",
    }
    atomic_write_json(run_dir / "model" / "feature_dictionary.json", dictionary)
    code_dir = run_dir / "model" / "code" / "huim"
    code_dir.mkdir(parents=True, exist_ok=True)
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, code_dir / source.name)
    tmp = run_dir / "model" / ".article_ids.npy.tmp"
    with open(tmp, "wb") as handle:
        np.save(handle, art.ids)
    tmp.replace(run_dir / "model" / "article_ids.npy")
    return {f: sha256_file(run_dir / f) for f in MODEL_FILES}


def load_model(bundle_dir: Path, device) -> tuple:
    bundle_dir = Path(bundle_dir)
    meta = read_json(bundle_dir / "model" / "model_config.json")
    cfg = ModelConfig.from_json(meta["model_config"])
    pre = Preprocessor.from_json(read_json(bundle_dir / "model" / "preprocessing.json"))
    model = build_model(cfg, pre)
    state = torch.load(bundle_dir / "model" / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if sha256_file(bundle_dir / "model" / "weights.pt") != meta["weights_sha256"]:
        raise ValueError("weights file hash differs from model_config.json")
    return model.to(device).eval(), cfg, pre, meta


def check_article_ids(bundle_dir: Path, art) -> bool:
    saved = np.load(Path(bundle_dir) / "model" / "article_ids.npy")
    return bool(np.array_equal(saved, art.ids))
