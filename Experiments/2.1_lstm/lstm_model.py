"""LSTM with target-aware attention for impression-slate click scoring (2.1_lstm).

Adapted from the course Week 8 recipe (chronological item history -> LSTM ->
candidate projected into an attention query -> masked attention over LSTM
states -> prediction head), with the objective changed from rating regression
to binary click prediction (BCE on logits).

Variants (``ModelConfig.variant``):
  attention           main model: candidate-specific attention over LSTM states
  last_state          simpler history summary: final LSTM state, same for all
                      candidates in the slate (no target-aware weighting)
  attention_shuffled  order control: same as ``attention`` but trained/evaluated
                      on histories in a fixed random order (prepared in data)
  attention_no_lstm   pooled control: attention directly over per-event inputs,
                      no recurrence
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

VARIANTS = ("attention", "last_state", "attention_shuffled", "attention_no_lstm")
# Positions in lstm_data.build_split_arrays event features (asserted in the notebook).
GAP_PREV_FEATURE, GAP_PREV_MISSING = 4, 5


@dataclass
class ModelConfig:
    variant: str = "attention"
    # Article encoder
    n_categories: int = 26          # 0 pad, 1 missing, 2 unknown, 3.. vocab
    n_types: int = 15
    n_topics: int = 80              # 0 pad, 1 unknown, 2.. vocab
    n_article_numeric: int = 9
    text_dim_in: int = 768
    image_dim_in: int = 1024
    category_dim: int = 16
    type_dim: int = 8
    topic_dim: int = 16
    text_dim: int = 64
    image_dim: int = 32
    use_id_embedding: bool = False
    n_id_buckets: int = 1
    id_dim: int = 16
    article_dim: int = 128
    # History encoder
    n_event_features: int = 8
    lstm_hidden: int = 128
    lstm_layers: int = 1
    # Attention and head
    attention_dim: int = 64
    interaction_dim: int = 64
    n_candidate_features: int = 3
    n_context_features: int = 22
    head_hidden: tuple = (256, 128)
    dropout: float = 0.2
    embedding_dropout: float = 0.1
    # Recency normalisation for the attention bias (train-fitted constants)
    gap_mean: float = 0.0
    gap_std: float = 1.0

    def to_json(self) -> dict:
        data = asdict(self)
        data["head_hidden"] = list(self.head_hidden)
        return data

    @classmethod
    def from_json(cls, data: dict) -> "ModelConfig":
        data = dict(data)
        data["head_hidden"] = tuple(data.get("head_hidden", (256, 128)))
        return cls(**data)


class ArticleEncoder(nn.Module):
    """Metadata + frozen text/image vectors (+ optional learned ID) -> vector.

    Supplied vectors are fixed lookup inputs (frozen); only the projections
    below are trained. Text and image spaces are projected separately.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.category = nn.Embedding(cfg.n_categories, cfg.category_dim, padding_idx=0)
        self.article_type = nn.Embedding(cfg.n_types, cfg.type_dim, padding_idx=0)
        self.topic = nn.Embedding(cfg.n_topics, cfg.topic_dim, padding_idx=0)
        self.text = nn.Linear(cfg.text_dim_in, cfg.text_dim)
        self.image = nn.Linear(cfg.image_dim_in, cfg.image_dim)
        width = (cfg.category_dim + cfg.type_dim + cfg.topic_dim + cfg.text_dim
                 + cfg.image_dim + cfg.n_article_numeric)
        if cfg.use_id_embedding:
            self.ids = nn.Embedding(cfg.n_id_buckets, cfg.id_dim, padding_idx=0)
            width += cfg.id_dim
        self.dropout = nn.Dropout(cfg.embedding_dropout)
        self.mlp = nn.Sequential(nn.Linear(width, cfg.article_dim), nn.ReLU(),
                                 nn.LayerNorm(cfg.article_dim))
        self._table: dict[str, torch.Tensor] = {}

    def set_table(self, table: dict[str, torch.Tensor]) -> None:
        """Attach article lookup tensors (not part of the state_dict)."""
        self._table = table

    def encode_rows(self, rows: torch.Tensor) -> torch.Tensor:
        t = self._table
        topics = t["topics"][rows]
        topic_mask = (topics > 0).unsqueeze(-1).float()
        topic_vec = (self.topic(topics) * topic_mask).sum(-2) / topic_mask.sum(-2).clamp(min=1)
        text = self.text(self.dropout(t["text"][rows])) * t["has_text"][rows].unsqueeze(-1)
        image = self.image(self.dropout(t["image"][rows])) * t["has_image"][rows].unsqueeze(-1)
        parts = [self.category(t["category"][rows]), self.article_type(t["article_type"][rows]),
                 topic_vec, text, image, t["numeric"][rows]]
        if self.cfg.use_id_embedding:
            parts.append(self.ids(t["id_bucket"][rows]))
        return self.mlp(torch.cat(parts, dim=-1))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """Encode each distinct article once, then scatter back."""
        unique, inverse = torch.unique(rows, return_inverse=True)
        encoded = self.encode_rows(unique)
        return encoded[inverse]


class TargetAwareLSTMRanker(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.variant not in VARIANTS:
            raise ValueError(f"unknown variant {cfg.variant}")
        self.cfg = cfg
        self.articles = ArticleEncoder(cfg)
        event_width = cfg.article_dim + cfg.n_event_features
        self.event_in = nn.Sequential(nn.Linear(event_width, cfg.lstm_hidden), nn.ReLU(),
                                      nn.Dropout(cfg.dropout))
        if cfg.variant != "attention_no_lstm":
            self.lstm = nn.LSTM(cfg.lstm_hidden, cfg.lstm_hidden, num_layers=cfg.lstm_layers,
                                batch_first=True,
                                dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0)
        H, A, I = cfg.lstm_hidden, cfg.attention_dim, cfg.interaction_dim
        self.query = nn.Linear(cfg.article_dim, A)
        self.key = nn.Linear(H, A)
        self.recency_weight = nn.Parameter(torch.tensor(-0.5))
        self.empty_history = nn.Parameter(torch.zeros(H))
        self.cand_proj = nn.Linear(cfg.article_dim, I)
        self.ctx_proj = nn.Linear(H, I)
        width = (cfg.article_dim + H + I + 1 + cfg.n_candidate_features
                 + cfg.n_context_features)
        layers = []
        for hidden in cfg.head_hidden:
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(cfg.dropout)]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*layers)

    # ---- history -----------------------------------------------------------
    def encode_history(self, hist_rows, hist_feats, lengths):
        """[U, L] article rows + [U, L, F] event features -> [U, L, H] states."""
        U, L = hist_rows.shape
        H = self.cfg.lstm_hidden
        if L == 0:
            return hist_feats.new_zeros(U, 0, H)
        art = self.articles(hist_rows.reshape(-1)).reshape(U, L, -1)
        x = self.event_in(torch.cat([art, hist_feats], dim=-1))
        if self.cfg.variant == "attention_no_lstm":
            mask = (torch.arange(L, device=x.device)[None, :] < lengths[:, None])
            return x * mask.unsqueeze(-1)
        safe = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(x, safe, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=L)
        mask = (torch.arange(L, device=x.device)[None, :] < lengths[:, None])
        return out * mask.unsqueeze(-1)

    # ---- full forward --------------------------------------------------------
    def forward(self, batch: dict, return_attention: bool = False):
        cfg = self.cfg
        states = self.encode_history(batch["hist_rows"], batch["hist_feats"],
                                     batch["hist_len"])            # [U, L, H]
        U, L, H = states.shape
        cand_user = batch["imp_user"][batch["cand_imp"]]          # [N]
        cand = self.articles(batch["cand_rows"])                  # [N, D]
        lengths = batch["hist_len"][cand_user]                    # [N]
        has_hist = (lengths > 0)
        attention = None
        if cfg.variant == "last_state":
            last = (batch["hist_len"] - 1).clamp(min=0)
            user_last = states[torch.arange(U, device=states.device), last] if L else \
                states.new_zeros(U, H)
            context = user_last[cand_user]
        else:
            if L:
                keys = self.key(states)                               # [U, L, A]
                q = self.query(cand)                                  # [N, A]
                with torch.autocast(device_type=states.device.type, enabled=False):
                    context, weights = self._attend(q.float(), keys.float(), states.float(),
                                                    cand_user, lengths, has_hist, batch)
                attention = weights
            else:
                context = states.new_zeros(len(cand_user), H)
        context = torch.where(has_hist[:, None], context.to(cand.dtype),
                              self.empty_history.to(cand.dtype).expand_as(context))
        p_c, p_h = self.cand_proj(cand), self.ctx_proj(context)
        product = p_c * p_h
        dot = product.sum(-1, keepdim=True) / math.sqrt(cfg.interaction_dim)
        features = torch.cat([cand, context, product, dot, batch["cand_feats"],
                              batch["imp_ctx"][batch["cand_imp"]]], dim=-1)
        logits = self.head(features).squeeze(-1)
        if return_attention:
            return logits, attention
        return logits

    def _attend(self, q, keys, states, cand_user, lengths, has_hist, batch):
        """fp32 masked attention; empty histories get zero weights."""
        cfg = self.cfg
        L = keys.shape[1]
        scores = torch.einsum("na,nla->nl", q, keys[cand_user])
        scores = scores / math.sqrt(cfg.attention_dim)
        gap_h = (batch["imp_time"][:, None] - batch["hist_time"][batch["imp_user"]]
                 ).float() / 3600.0                          # [I, L]
        gap = (torch.log1p(gap_h.clamp(min=0)) - cfg.gap_mean) / cfg.gap_std
        scores = scores + self.recency_weight * gap[batch["cand_imp"]]
        mask = torch.arange(L, device=scores.device)[None, :] < lengths[:, None]
        scores = scores.masked_fill(~mask, float("-inf"))
        scores = torch.where(has_hist[:, None], scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1) * mask
        context = torch.einsum("nl,nlh->nh", weights, states[cand_user])
        return context, weights


# --------------------------------------------------------------------------
# Device-resident split data and batching
# --------------------------------------------------------------------------

def article_table_tensors(table, device) -> dict:
    as_t = lambda a, dtype: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)
    return {
        "category": as_t(table.category, torch.long),
        "article_type": as_t(table.article_type, torch.long),
        "topics": as_t(table.topics, torch.long),
        "numeric": as_t(table.numeric, torch.float32),
        "text": as_t(table.text, torch.float32),
        "has_text": as_t(table.has_text, torch.float32),
        "image": as_t(table.image, torch.float32),
        "has_image": as_t(table.has_image, torch.float32),
        "id_bucket": as_t(table.id_bucket, torch.long),
    }


class SplitTensors:
    """All arrays of one split on the target device; batches by impression."""

    def __init__(self, arrays, device):
        t = lambda a, dtype: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)
        self.name = arrays.split
        self.n_impressions = arrays.n_impressions
        self.n_candidates = arrays.n_candidates
        self.offsets = t(arrays.offsets, torch.long)
        self.imp_time = t(arrays.imp_time_s, torch.long)
        self.imp_user_row = t(arrays.imp_user_row, torch.long)
        self.imp_ctx = t(arrays.imp_context, torch.float32)
        self.cand_rows = t(arrays.cand_article_row, torch.long)
        self.cand_feats = t(arrays.cand_features, torch.float32)
        labels = np.nan_to_num(arrays.labels, nan=-1.0)
        self.labels = t(labels, torch.float32)
        self.hist_rows = t(arrays.hist_article_row, torch.long)
        self.hist_time = t(arrays.hist_times_s, torch.long)
        self.hist_feats = t(arrays.hist_features, torch.float32)
        self.hist_len = t(arrays.hist_lengths, torch.long)
        self.device = device

    def batch(self, imp_index: torch.Tensor, history_dropout: float = 0.0,
              generator: torch.Generator | None = None) -> dict:
        imp_index = imp_index.to(self.device)
        starts, ends = self.offsets[imp_index], self.offsets[imp_index + 1]
        sizes = ends - starts
        cand_imp = torch.repeat_interleave(torch.arange(len(imp_index), device=self.device), sizes)
        first = torch.cumsum(sizes, 0) - sizes
        cand_index = starts[cand_imp] + (torch.arange(int(sizes.sum()), device=self.device)
                                         - first[cand_imp])
        users, imp_user = torch.unique(self.imp_user_row[imp_index], return_inverse=True)
        hist_len = self.hist_len[users]
        max_len = int(hist_len.max()) if len(users) else 0
        hist_rows = self.hist_rows[users, :max_len]
        hist_feats = self.hist_feats[users, :max_len]
        hist_time = self.hist_time[users, :max_len]
        if history_dropout > 0 and max_len > 0:
            hist_rows, hist_feats, hist_time, hist_len = drop_history_events(
                hist_rows, hist_feats, hist_time, hist_len, history_dropout, generator)
        return {
            "imp_index": imp_index, "imp_user": imp_user,
            "imp_time": self.imp_time[imp_index], "imp_ctx": self.imp_ctx[imp_index],
            "cand_imp": cand_imp, "cand_index": cand_index,
            "cand_rows": self.cand_rows[cand_index], "cand_feats": self.cand_feats[cand_index],
            "labels": self.labels[cand_index],
            "hist_rows": hist_rows, "hist_feats": hist_feats,
            "hist_time": hist_time, "hist_len": hist_len,
        }


def drop_history_events(rows, feats, times, lengths, p, generator=None):
    """Training-only augmentation: drop each non-latest event with prob p and
    left-compact the survivors (chronological order preserved)."""
    U, L = rows.shape
    device = rows.device
    pos = torch.arange(L, device=device)[None, :]
    valid = pos < lengths[:, None]
    latest = pos == (lengths[:, None] - 1)
    rand = torch.rand((U, L), device=device, generator=generator)
    keep = valid & ((rand >= p) | latest)
    # A survivor whose predecessor was dropped no longer has a valid "gap to
    # previous event": mark it missing (as for a first event) instead of stale.
    prev_kept = torch.cat([torch.ones_like(keep[:, :1]), keep[:, :-1]], dim=1)
    stale = keep & ~prev_kept
    feats = feats.clone()
    feats[..., GAP_PREV_FEATURE] = feats[..., GAP_PREV_FEATURE].masked_fill(stale, 0.0)
    feats[..., GAP_PREV_MISSING] = feats[..., GAP_PREV_MISSING].masked_fill(stale, 1.0)
    order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
    gather = lambda a: torch.gather(a, 1, order if a.dim() == 2 else
                                    order.unsqueeze(-1).expand_as(a))
    new_len = keep.sum(1)
    rows, feats, times = gather(rows), gather(feats), gather(times)
    new_valid = pos < new_len[:, None]
    rows = rows.masked_fill(~new_valid, 0)
    feats = feats.masked_fill(~new_valid.unsqueeze(-1), 0.0)
    times = times.masked_fill(~new_valid, 0)
    return rows, feats, times, new_len


def impression_batches(n: int, batch_size: int, shuffle: bool, generator=None):
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


@torch.no_grad()
def predict_logits(model, data: SplitTensors, batch_size: int = 256,
                   impression_subset: np.ndarray | None = None) -> np.ndarray:
    """fp32 inference in the split's canonical candidate order."""
    model.eval()
    out = torch.empty(data.n_candidates, dtype=torch.float32, device=data.device)
    filled = torch.zeros(data.n_candidates, dtype=torch.bool, device=data.device)
    indices = (torch.arange(data.n_impressions) if impression_subset is None
               else torch.as_tensor(impression_subset, dtype=torch.long))
    for start in range(0, len(indices), batch_size):
        batch = data.batch(indices[start:start + batch_size])
        out[batch["cand_index"]] = model(batch).float()
        filled[batch["cand_index"]] = True
    result = out.cpu().numpy().astype(np.float64)
    if impression_subset is None and not bool(filled.all()):
        raise AssertionError("some candidates were not scored")
    result[~filled.cpu().numpy()] = np.nan
    return result


@torch.no_grad()
def collect_attention(model, data: SplitTensors, impression_subset, batch_size=256):
    """Attention weights for the given impressions (diagnostics only)."""
    model.eval()
    records = []
    idx = torch.as_tensor(impression_subset, dtype=torch.long)
    for start in range(0, len(idx), batch_size):
        batch = data.batch(idx[start:start + batch_size])
        logits, weights = model(batch, return_attention=True)
        if weights is None:
            return None
        users = batch["imp_user"][batch["cand_imp"]]
        records.append({
            "cand_index": batch["cand_index"].cpu().numpy(),
            "logits": logits.float().cpu().numpy(),
            "weights": weights.float().cpu().numpy(),
            "hist_len": batch["hist_len"][users].cpu().numpy(),
            "hist_rows": batch["hist_rows"][users].cpu().numpy(),
        })
    return records


# --------------------------------------------------------------------------
# Training with early stopping, checkpoints and resumption
# --------------------------------------------------------------------------

@dataclass
class TrainConfig:
    max_epochs: int = 8
    batch_impressions: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    early_stopping_patience: int = 2
    lr_factor: float = 0.5
    lr_patience: int = 0
    min_lr: float = 1e-5
    history_dropout: float = 0.0
    use_amp: bool = True
    eval_batch_impressions: int = 256
    max_train_impressions_per_epoch: int | None = None
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch_save(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train_model(model, train: SplitTensors, validation: SplitTensors, evaluate,
                cfg: TrainConfig, out_dir: Path, resume: bool = True,
                on_epoch_end=None, trial=None, log=print):
    """Train with BCE on logits; select the epoch with the best validation
    mean impression AUC (ties keep the earlier epoch).

    ``evaluate(scores) -> dict`` computes validation metrics from fp32 logits.
    Saves ``best_model_state.pt`` (inference weights) and
    ``recovery/last_checkpoint.pt`` (resume state) under ``out_dir``.
    """
    out_dir = Path(out_dir)
    device = train.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience,
        min_lr=cfg.min_lr)
    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.BCEWithLogitsLoss()
    history, best = [], {"epoch": None, "auc": -math.inf, "metrics": None}
    start_epoch, stale = 1, 0
    recovery = out_dir / "recovery" / "last_checkpoint.pt"
    best_path = out_dir / "best_model_state.pt"
    generator = torch.Generator().manual_seed(cfg.seed)
    device_gen = torch.Generator(device=device).manual_seed(cfg.seed)

    if resume and recovery.is_file():
        state = torch.load(recovery, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state["scaler"]:   # empty when saved from a CPU / non-AMP run
            scaler.load_state_dict(state["scaler"])
        history, best, stale = state["history"], state["best"], state["stale"]
        start_epoch = state["epoch"] + 1
        restore_rng_state(state["rng"])
        generator.set_state(state["generator"])
        log(f"Resumed from {recovery} after epoch {state['epoch']} "
            f"(best epoch {best['epoch']}, AUC {best['auc']:.4f}). Exact bitwise "
            "resumption is not guaranteed (cuDNN/AMP nondeterminism).")
        if state.get("finished"):
            log("Recovery checkpoint marks training as finished; skipping the loop.")
            start_epoch = cfg.max_epochs + 1

    for epoch in range(start_epoch, cfg.max_epochs + 1):
        model.train()
        t0, total, count = time.time(), 0.0, 0
        n = train.n_impressions
        limit = cfg.max_train_impressions_per_epoch
        batches = list(impression_batches(n, cfg.batch_impressions, True, generator))
        if limit:
            batches = batches[: max(1, math.ceil(limit / cfg.batch_impressions))]
        for step, imp in enumerate(batches, start=1):
            batch = train.batch(imp, cfg.history_dropout, device_gen)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(batch)
                loss = loss_fn(logits.float(), batch["labels"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss) * len(batch["labels"])
            count += len(batch["labels"])
            if step % 200 == 0:
                log(f"  epoch {epoch} step {step}/{len(batches)} "
                    f"loss {total / count:.4f} ({time.time() - t0:.0f}s)")
        scores = predict_logits(model, validation, cfg.eval_batch_impressions)
        metrics = evaluate(scores)
        auc = metrics["mean_impression_auc"]
        if auc is None:
            raise ValueError("No validation impression has both a click and a non-click, so "
                             "AUC-based selection is impossible; enlarge the (smoke) sample.")
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(auc)
        row = {"epoch": epoch, "train_bce": total / max(count, 1),
               "validation_auc": auc,
               "validation_ndcg@5": metrics["mean_impression_ndcg@5"],
               "learning_rate": lr, "seconds": time.time() - t0}
        history.append(row)
        improved = auc > best["auc"]
        if improved:
            best = {"epoch": epoch, "auc": auc, "metrics": metrics}
            atomic_torch_save(model.state_dict(), best_path)
            stale = 0
        else:
            stale += 1
        log(f"epoch {epoch}: train BCE {row['train_bce']:.4f} | val AUC {auc:.4f} "
            f"nDCG@5 {row['validation_ndcg@5']:.4f} | lr {lr:.2e} | "
            f"{row['seconds']:.0f}s{' *best*' if improved else ''}")
        # Stop after `early_stopping_patience` consecutive epochs without improvement.
        finished = stale >= cfg.early_stopping_patience or epoch == cfg.max_epochs
        atomic_torch_save({
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "history": history, "best": best, "stale": stale, "rng": rng_state(),
            "generator": generator.get_state(), "finished": finished,
            "train_config": asdict(cfg),
        }, recovery)
        (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
        if on_epoch_end is not None:
            on_epoch_end(epoch, row)
        if trial is not None:
            trial.report(auc, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        if finished:
            if stale >= cfg.early_stopping_patience:
                log(f"Early stopping: no validation AUC gain for {stale} epochs.")
            break

    if best["epoch"] is None:
        raise RuntimeError("no epoch completed")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return history, best


def build_model(cfg: ModelConfig, article_tensors: dict, device) -> TargetAwareLSTMRanker:
    model = TargetAwareLSTMRanker(cfg).to(device)
    model.articles.set_table(article_tensors)
    return model


def count_parameters(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable_parameters": int(trainable),
            "frozen_inputs": "text (768-d) and image (1024-d) article vectors are fixed "
                             "lookup tensors, not parameters; their encoders are not fine-tuned"}
