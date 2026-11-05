"""One place for input paths, so the traditional-track rerun works in Colab and on the original laptop.

Colab uses the shared Drive layout. Elsewhere, set BT4222_SHARED to a local copy of that layout, or the
original laptop folders are used. The notebooks from earlier runs keep their original paths as run records.
"""
import os
from pathlib import Path
from types import SimpleNamespace

COLAB = Path("/content/drive/MyDrive/Colab Notebooks/BT4222 Shared Final Folder")
LAPTOP = Path(r"C:\Users\nigel\BT4222 Shared Final Folder")
LAPTOP_SMALL = Path(r"C:\Users\nigel\bt4222-data\ebnerd_small")


def resolve():
    custom = os.environ.get("BT4222_SHARED")
    root = Path(custom) if custom else COLAB if COLAB.exists() else None
    if root is not None:      # shared Drive layout
        return SimpleNamespace(shared=root, features=root / "feature_engineering", articles=root / "train_val" / "articles.parquet",
                               history={"train": root / "train_val" / "train" / "history.parquet", "validation": root / "train_val" / "validation" / "history.parquet",
                                        "test": root / "test" / "history.parquet"})
    history = {"train": LAPTOP_SMALL / "train" / "history.parquet", "validation": LAPTOP_SMALL / "validation" / "history.parquet", "test": LAPTOP_SMALL / "validation" / "history.parquet"}
    return SimpleNamespace(shared=LAPTOP, features=LAPTOP / "feature_engineering_nov4", articles=LAPTOP / "updated_nov5" / "articles.parquet", history=history)
