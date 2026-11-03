"""Optional, bounded Optuna search (off by default).

Objective: validation mean impression AUC on recorded slates (maximise), the
common selection metric. Validation is reused for selection, so tuned scores are
development results. The study lives in a local SQLite file; after every trial a
consistent snapshot (sqlite3 backup API) and trials.csv are copied to the
persistent run folder, so an interrupted Colab session can resume the study."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .io_utils import copy_verified

SEARCH_SPACE = {
    "lr": ("log_float", 3e-4, 3e-3),
    "dropout": ("float", 0.1, 0.4),
    "weight_decay": ("log_float", 1e-6, 1e-3),
    "d_model": ("categorical", [48, 64, 96]),
    "modality_dropout": ("float", 0.0, 0.3),
}


def snapshot(db_path: Path, dst_dir: Path) -> None:
    """Consistent SQLite backup made on LOCAL disk, then a verified copy to Drive
    (SQLite on the Drive mount is unreliable)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp = db_path.parent / f".{db_path.name}.snapshot"
    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(tmp)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    copy_verified(tmp, dst_dir / db_path.name)
    tmp.unlink(missing_ok=True)


def run_study(objective_builder, study_name: str, local_dir: Path, persist_dir: Path | None, n_trials: int,
              seed: int, timeout_hours: float | None = None):
    import optuna  # installed only when tuning is enabled

    local_dir.mkdir(parents=True, exist_ok=True)
    db = local_dir / "optuna_study.sqlite3"
    if persist_dir is not None and (persist_dir / "tuning" / db.name).is_file() and not db.is_file():
        copy_verified(persist_dir / "tuning" / db.name, db)  # resume from persistent snapshot
    study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{db}", direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed), load_if_exists=True)

    def suggest(trial):
        params = {}
        for name, spec in SEARCH_SPACE.items():
            if spec[0] == "log_float":
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif spec[0] == "float":
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            else:
                params[name] = trial.suggest_categorical(name, spec[1])
        return params

    def callback(study, trial):
        study.trials_dataframe().to_csv(local_dir / "trials.csv", index=False)
        if persist_dir is not None:
            snapshot(db, persist_dir / "tuning")
            copy_verified(local_dir / "trials.csv", persist_dir / "tuning" / "trials.csv")

    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, n_trials - done)
    study.optimize(lambda t: objective_builder(t, suggest(t)), n_trials=remaining,
                   timeout=None if timeout_hours is None else timeout_hours * 3600, callbacks=[callback])
    return study
