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
from .metrics import impression_metrics


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


class Trainer:
    """Epoch-based trainer.

    * Objective: BCE on logits of the supplied recorded pairs (clicked=1, shown
      but unclicked=0). No catalogue negatives.
    * Selection: maximise validation mean impression AUC; ties -> earlier epoch.
    * Recovery: `recovery/last.pt` (model+optimizer+scheduler+GradScaler+RNG+
      progress), written atomically and mirrored to persistent storage.
    """

    def __init__(self, model, art, train_ctx, val_ctx, device, cfg: dict, run_dir: Path,
                 persist_dir: Path | None = None, log: Callable = print):
        self.model, self.art, self.train_ctx, self.val_ctx = model, art, train_ctx, val_ctx
        self.device, self.cfg, self.run_dir, self.persist_dir, self.log = device, cfg, Path(run_dir), persist_dir, log
        self.amp = bool(cfg.get("amp", True)) and device.type == "cuda"
        self.optimizer = make_optimizer(model, cfg["lr"], cfg["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=0)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.state = {"epoch": 0, "step_in_epoch": 0, "global_step": 0, "best_auc": -math.inf, "best_epoch": None,
                      "epochs_without_improvement": 0, "history": [], "finished": False}
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
                   # the selected (best-AUC) inference weights travel with the recovery state, so a
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
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["grad_scaler"])
        self.state = payload["state"]
        if payload.get("selected_weights") is not None:
            atomic_torch_save(payload["selected_weights"], self.run_dir / "model" / "weights.pt")
        restore_rng_state(payload["rng"])
        self.log(f"Resumed from {path}: epoch {self.state['epoch']}, step {self.state['step_in_epoch']}, "
                 f"best AUC {self.state['best_auc']}")

    def save_selected(self) -> None:
        sel = self.run_dir / "model" / "weights.pt"
        atomic_torch_save({k: v.detach().cpu() for k, v in self.model.state_dict().items()}, sel)

    # --------------------------------------------------------------- training
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
            self.state["step_in_epoch"] = step + 1
            self.state["global_step"] += 1
            if every and (step + 1) % every == 0 and step + 1 < n_steps:
                self.save_recovery()
            if (step + 1) % self.cfg.get("log_every_steps", 500) == 0:
                self.log(f"epoch {epoch} step {step + 1}/{n_steps} loss {total / max(count, 1):.5f} "
                         f"({(time.time() - t0) / (step + 1 - start_step):.3f}s/step)")
        return total / max(count, 1), n_steps

    def fit(self) -> dict:
        cfg = self.cfg
        while self.state["epoch"] < cfg["max_epochs"] and not self.state["finished"]:
            epoch = self.state["epoch"]
            t0 = time.time()
            loss, n_steps = self.train_steps(epoch, self.state["step_in_epoch"])
            metrics, _ = evaluate_recorded(self.model, self.val_ctx, self.art, self.device, amp=False)
            auc = metrics["mean_impression_auc"]
            record = {"epoch": epoch, "train_loss_partial_epoch_if_resumed": loss, "steps": n_steps,
                      "lr": self.optimizer.param_groups[0]["lr"], "seconds": time.time() - t0,
                      "validation_auc": auc, "validation_ndcg@5": metrics["mean_impression_ndcg@5"]}
            self.state["history"].append(record)
            self.log(f"epoch {epoch}: loss {loss:.5f}  val AUC {auc:.5f}  nDCG@5 {metrics['mean_impression_ndcg@5']:.5f}")
            if auc > self.state["best_auc"]:  # strict: ties keep the earlier epoch
                self.state.update(best_auc=auc, best_epoch=epoch, best_metrics=metrics, epochs_without_improvement=0)
                self.save_selected()
                if self.persist_dir is not None:
                    copy_verified(self.run_dir / "model" / "weights.pt", Path(self.persist_dir) / "model" / "weights.pt")
            else:
                self.state["epochs_without_improvement"] += 1
            self.scheduler.step(auc)
            self.state["epoch"] = epoch + 1
            self.state["step_in_epoch"] = 0
            if self.state["epochs_without_improvement"] > cfg["early_stopping_patience"]:
                self.log("early stopping")
                self.state["finished"] = True
            atomic_write_json(self.run_dir / "training_history.json", self.state["history"])
            self.save_recovery()
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
