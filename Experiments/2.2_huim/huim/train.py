"""Training with BCE on recorded pairs, AUC-based checkpoint selection and
interruption-safe recovery checkpoints."""
from __future__ import annotations

import copy
import hashlib
import math
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from .io_utils import atomic_torch_save, atomic_write_json, copy_verified
from .metrics import better_selection, impression_metrics


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # RNG states must be CPU ByteTensors even if the checkpoint was mapped elsewhere.
    torch.set_rng_state(state["torch_cpu"].cpu() if hasattr(state["torch_cpu"], "cpu") else state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW; no weight decay on biases, norms and embedding tables."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name.lower() or isinstance_embedding(model, name):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def isinstance_embedding(model: nn.Module, param_name: str) -> bool:
    module_name = param_name.rsplit(".", 1)[0]
    module = dict(model.named_modules()).get(module_name)
    return isinstance(module, (nn.Embedding, nn.EmbeddingBag))


@torch.no_grad()
def score_recorded(model, ctx, art, device, batch_pairs: int = 65536, amp: bool = False) -> np.ndarray:
    """Logits for every recorded candidate of a split (in ctx.recorded order)."""
    model.eval()
    rp = torch.from_numpy(ctx.recorded["req_pos"])
    ai = torch.from_numpy(ctx.recorded["art_idx"])
    out = np.empty(len(rp), dtype=np.float32)
    for lo in range(0, len(rp), batch_pairs):
        r = rp[lo:lo + batch_pairs].to(device)
        a = ai[lo:lo + batch_pairs].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logit = model(ctx.tensors, art.tensors, len(art.ids), r, a)
        out[lo:lo + batch_pairs] = logit.float().cpu().numpy()
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite recorded-slate logits")
    return out


def evaluate_recorded(model, ctx, art, device, amp=False) -> tuple[dict, np.ndarray]:
    scores = score_recorded(model, ctx, art, device, amp=amp)
    labels = ctx.recorded["label"]
    if np.isnan(labels).any():
        raise ValueError(f"{ctx.split} has no labels; recorded-slate metrics unavailable")
    return impression_metrics(ctx.recorded["req_pos"], labels.astype(int), scores, k=5), scores


def params_digest(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ProtocolMismatch(RuntimeError):
    """A recovery checkpoint was produced under a different selection protocol."""


class Trainer:
    """Step-based trainer with catalogue-validation selection.

    * Objective: BCE on logits of the supplied recorded pairs (clicked=1, shown
      but unclicked=0). No catalogue negatives; unshown outcomes stay unknown.
    * Selection: every `selection_interval_steps` optimiser steps and at each
      epoch end, `selector.evaluate` scores a FIXED validation-request sample
      against the full eligible catalogue. Maximise Recall@100; ties -> higher
      observed-click nDCG@100 -> earlier checkpoint. Early stopping counts
      selection evaluations without improvement (`early_stopping_patience`,
      `min_improvement`). ReduceLROnPlateau follows the same Recall@100.
    * Diagnostic: recorded-slate validation AUC at every evaluation (not used to select).
    * Recovery: `recovery/last.pt` (model, optimiser, scheduler, GradScaler, RNG,
      progress, selected weights, selection protocol), written atomically and
      mirrored to persistent storage. A checkpoint from another protocol (e.g. an
      earlier AUC-selected run) is refused rather than reinterpreted.
    """

    def __init__(self, model, art, train_ctx, val_ctx, device, cfg: dict, run_dir: Path,
                 persist_dir: Path | None = None, log: Callable = print, selector=None,
                 on_evaluation: Callable | None = None):
        self.model, self.art, self.train_ctx, self.val_ctx = model, art, train_ctx, val_ctx
        self.device, self.cfg, self.run_dir, self.persist_dir, self.log = device, cfg, Path(run_dir), persist_dir, log
        self.selector, self.on_evaluation = selector, on_evaluation
        self.amp = bool(cfg.get("amp", True)) and device.type == "cuda"
        self.optimizer = make_optimizer(model, cfg["lr"], cfg["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=cfg.get("scheduler_patience", 1))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.state = {"protocol": selector.protocol if selector is not None else None,
                      "epoch": 0, "step_in_epoch": 0, "global_step": 0, "best": None,
                      "evals_without_improvement": 0, "last_eval_global_step": -1,
                      "loss_sum": 0.0, "loss_count": 0, "history": [], "finished": False, "stop_reason": None}
        rec = train_ctx.recorded
        self.train_idx = np.arange(len(rec["req_pos"]))
        if cfg.get("train_pair_indices") is not None:
            self.train_idx = np.asarray(cfg["train_pair_indices"])
        self.rp = torch.from_numpy(rec["req_pos"])
        self.ai = torch.from_numpy(rec["art_idx"])
        self.y = torch.from_numpy(rec["label"].astype(np.float32))

    # ------------------------------------------------------------ checkpoints
    def recovery_path(self) -> Path:
        return self.run_dir / "recovery" / "last.pt"

    def save_recovery(self) -> None:
        selected = self.run_dir / "model" / "weights.pt"
        payload = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                   # the selected inference weights travel with the recovery state, so a
                   # crash between saving them and saving recovery cannot mislabel them
                   "selected_weights": torch.load(selected, map_location="cpu", weights_only=True)
                   if selected.is_file() else None,
                   "scheduler": self.scheduler.state_dict(), "grad_scaler": self.scaler.state_dict(),
                   "amp": self.amp, "state": copy.deepcopy(self.state), "rng": rng_state(),
                   "train_cfg": {k: v for k, v in self.cfg.items() if k != "train_pair_indices"}}
        atomic_torch_save(payload, self.recovery_path())
        if self.persist_dir is not None:
            copy_verified(self.recovery_path(), Path(self.persist_dir) / "recovery" / "last.pt")

    def load_recovery(self, path: Path) -> None:
        # Load on CPU first: RNG states must stay CPU tensors; the optimiser
        # state is cast to each parameter's device by load_state_dict.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        saved_protocol = payload.get("state", {}).get("protocol")
        if saved_protocol != self.state["protocol"]:
            raise ProtocolMismatch(
                f"{path} was written under a different selection protocol "
                f"({(saved_protocol or {}).get('version', 'pre-v1 / AUC-selected')}); it cannot be resumed as a "
                f"{(self.state['protocol'] or {}).get('version')} run. Start a new RUN_ID; keep the old run as a "
                "diagnostic with its original selection provenance.")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["grad_scaler"])
        self.state = payload["state"]
        if payload.get("selected_weights") is not None:
            atomic_torch_save(payload["selected_weights"], self.run_dir / "model" / "weights.pt")
        restore_rng_state(payload["rng"])
        best = self.state.get("best") or {}
        self.log(f"Resumed from {path}: epoch {self.state['epoch']}, step {self.state['step_in_epoch']}, "
                 f"best Recall@100 {best.get('known_click_recall@100')}")

    def save_selected(self) -> None:
        sel = self.run_dir / "model" / "weights.pt"
        atomic_torch_save({k: v.detach().cpu() for k, v in self.model.state_dict().items()}, sel)

    # --------------------------------------------------------------- training
    def steps_per_epoch(self) -> int:
        return math.ceil(len(self.train_idx) / self.cfg["batch_size"])

    def epoch_order(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng(self.cfg["seed"] * 1009 + epoch)
        return self.train_idx[rng.permutation(len(self.train_idx))]

    def train_steps(self, epoch: int, start_step: int, stop_step: int | None = None) -> tuple[float, int]:
        """Run steps [start_step, stop_step) of an epoch; checkpoint every N steps."""
        bs = self.cfg["batch_size"]
        order = self.epoch_order(epoch)
        n_steps = math.ceil(len(order) / bs)
        stop = n_steps if stop_step is None else min(stop_step, n_steps)
        self.model.train()
        total, count = 0.0, 0
        every = self.cfg.get("checkpoint_every_steps") or 0
        t0 = time.time()
        for step in range(start_step, stop):
            batch = torch.from_numpy(order[step * bs:(step + 1) * bs])
            r, a, y = self.rp[batch].to(self.device), self.ai[batch].to(self.device), self.y[batch].to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                logit = self.model(self.train_ctx.tensors, self.art.tensors, len(self.art.ids), r, a)
            loss = self.loss_fn(logit.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.get("grad_clip", 1.0))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += float(loss) * len(batch)
            count += len(batch)
            self.state["loss_sum"] += float(loss) * len(batch)
            self.state["loss_count"] += len(batch)
            self.state["step_in_epoch"] = step + 1
            self.state["global_step"] += 1
            if every and (step + 1) % every == 0 and step + 1 < n_steps:
                self.save_recovery()
            if (step + 1) % self.cfg.get("log_every_steps", 500) == 0:
                self.log(f"epoch {epoch} step {step + 1}/{n_steps} loss {total / max(count, 1):.5f} "
                         f"({(time.time() - t0) / (step + 1 - start_step):.3f}s/step)")
        return total / max(count, 1), n_steps

    def evaluate(self) -> dict:
        """One selection evaluation; updates best checkpoint, patience and scheduler."""
        cfg = self.cfg
        t0 = time.time()
        sel = self.selector.evaluate(self.model)
        diag, _ = evaluate_recorded(self.model, self.val_ctx, self.art, self.device, amp=False)
        self.model.train()
        record = {"eval_index": len(self.state["history"]), "epoch": self.state["epoch"],
                  "step_in_epoch": self.state["step_in_epoch"], "global_step": self.state["global_step"],
                  "train_loss_since_last_eval": self.state["loss_sum"] / max(self.state["loss_count"], 1),
                  "train_pairs_since_last_eval": self.state["loss_count"],
                  "lr": self.optimizer.param_groups[0]["lr"],
                  "known_click_recall@100": sel["known_click_recall@100"],
                  "observed_click_ndcg@100": sel["observed_click_ndcg@100"],
                  "eligible_only_recall@100": sel["eligible_only_recall@100"],
                  "requests_with_ties_crossing_cutoff": sel["requests_with_ties_crossing_cutoff"],
                  "diagnostic_recorded_slate_auc": diag["mean_impression_auc"],
                  "selection_seconds": sel["seconds"], "evaluation_seconds": time.time() - t0}
        previous = self.state["best"]
        is_better = better_selection(record, previous)
        min_delta = cfg.get("min_improvement", 0.0) or 0.0
        if previous is None:
            improved = True
        elif min_delta > 0:
            improved = record["known_click_recall@100"] >= previous["known_click_recall@100"] + min_delta
        else:
            improved = is_better
        record["selected_so_far"] = bool(is_better)
        if is_better:
            self.state["best"] = {k: record[k] for k in ("eval_index", "epoch", "step_in_epoch", "global_step",
                                                         "known_click_recall@100", "observed_click_ndcg@100",
                                                         "eligible_only_recall@100", "diagnostic_recorded_slate_auc")}
            self.state["best"]["selection_detail"] = sel
            self.save_selected()
        self.state["evals_without_improvement"] = 0 if improved else self.state["evals_without_improvement"] + 1
        self.scheduler.step(record["known_click_recall@100"])
        self.state["history"].append(record)
        self.state["last_eval_global_step"] = self.state["global_step"]
        self.state["loss_sum"], self.state["loss_count"] = 0.0, 0
        if self.state["evals_without_improvement"] > cfg["early_stopping_patience"]:
            self.state["finished"] = True
            self.state["stop_reason"] = (f"early stopping: {self.state['evals_without_improvement']} selection "
                                         f"evaluations without improvement")
        atomic_write_json(self.run_dir / "training_history.json", self.state["history"])
        self.save_recovery()  # selected weights + state persisted together
        self.log(f"eval {record['eval_index']} (epoch {record['epoch']} step {record['step_in_epoch']}): "
                 f"Recall@100 {record['known_click_recall@100']:.5f}  nDCG@100 {record['observed_click_ndcg@100']:.5f}  "
                 f"[diag AUC {record['diagnostic_recorded_slate_auc']:.5f}]{'  *selected*' if is_better else ''}")
        if self.on_evaluation is not None:
            self.on_evaluation(record)  # e.g. Optuna report/prune; may raise to stop the trial
        return record

    def fit(self) -> dict:
        if self.selector is None:
            raise ValueError("fit() needs a catalogue selector; selection must not silently fall back to AUC")
        cfg = self.cfg
        interval = cfg.get("selection_interval_steps") or None
        n_steps = self.steps_per_epoch()
        def pending() -> bool:  # an evaluation boundary reached but not evaluated (e.g. crash before eval)
            g = self.state["global_step"]
            at_boundary = (bool(interval) and g > 0 and g % interval == 0) or self.state["step_in_epoch"] >= n_steps
            return at_boundary and self.state["last_eval_global_step"] != g

        while self.state["epoch"] < cfg["max_epochs"] and not self.state["finished"]:
            epoch = self.state["epoch"]
            if pending():
                self.evaluate()
            while self.state["step_in_epoch"] < n_steps and not self.state["finished"]:
                start = self.state["step_in_epoch"]
                stop = n_steps
                if interval:
                    until_eval = interval - (self.state["global_step"] % interval)
                    stop = min(n_steps, start + until_eval)
                self.train_steps(epoch, start, stop)
                at_interval = bool(interval) and self.state["global_step"] % interval == 0
                at_epoch_end = self.state["step_in_epoch"] >= n_steps
                if (at_interval or at_epoch_end) and self.state["last_eval_global_step"] != self.state["global_step"]:
                    self.evaluate()
            if self.state["step_in_epoch"] >= n_steps:
                self.state["epoch"] = epoch + 1
                self.state["step_in_epoch"] = 0
                self.save_recovery()
        if not self.state["finished"]:
            self.state["stop_reason"] = f"max_epochs={cfg['max_epochs']} reached"
        self.state["finished"] = True
        self.save_recovery()
        return self.state


def interruption_check(build_trainer: Callable[[Path], "Trainer"], scratch: Path, steps: int = 40) -> dict:
    """Optional, bounded resume check (prepared; must be executed in Colab).

    A: train 2*steps continuously. B: train `steps`, save recovery, rebuild a
    fresh trainer, load it, train `steps` more. Compare parameters. The epoch
    must contain more than 2*steps steps so the resume is genuinely mid-epoch."""
    probe = build_trainer(scratch / "probe")
    n_steps = math.ceil(len(probe.train_idx) / probe.cfg["batch_size"])
    del probe
    if n_steps <= 2 * steps:
        raise ValueError(f"epoch has only {n_steps} steps; use a larger training subset or fewer steps")
    a = build_trainer(scratch / "A")
    a.train_steps(0, 0, 2 * steps)
    digest_a = params_digest(a.model)
    params_a = {k: v.detach().cpu().clone() for k, v in a.model.state_dict().items()}
    b = build_trainer(scratch / "B")
    b.train_steps(0, 0, steps)
    b.save_recovery()
    c = build_trainer(scratch / "C")
    c.load_recovery(scratch / "B" / "recovery" / "last.pt")
    assert c.state["step_in_epoch"] == steps, "resume did not restore the mid-epoch position"
    c.train_steps(0, c.state["step_in_epoch"], 2 * steps)
    params_c = {k: v.detach().cpu() for k, v in c.model.state_dict().items()}
    max_diff = max(float((params_a[k].float() - params_c[k].float()).abs().max()) for k in params_a)
    return {"steps_each_half": steps, "steps_per_epoch": n_steps, "digest_continuous": digest_a, "digest_resumed": params_digest(c.model),
            "bitwise_identical": digest_a == params_digest(c.model), "max_abs_param_diff": max_diff,
            "note": "GPU kernels (e.g. embedding backward/index_add) may be nondeterministic; "
                    "a tiny nonzero diff is possible without a resume bug."}
