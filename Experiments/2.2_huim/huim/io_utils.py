"""Small, dependency-light I/O helpers: hashing, atomic writes, verified copies,
completion markers, environment and code-revision provenance."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, hash_limit_bytes: int | None = 2 * 1024 ** 3) -> dict:
    """Size, mtime and (for files up to hash_limit_bytes) SHA-256."""
    path = Path(path)
    stat = path.stat()
    record = {"path": str(path), "bytes": stat.st_size,
              "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime))}
    if hash_limit_bytes is None or stat.st_size <= hash_limit_bytes:
        record["sha256"] = sha256_file(path)
    else:
        record["sha256"] = None
        record["sha256_skipped_reason"] = f"larger than {hash_limit_bytes} bytes"
    return record


class _Encoder(json.JSONEncoder):
    def default(self, value):  # noqa: D401 - json hook
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        return str(value)


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, cls=_Encoder, allow_nan=False) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, dumps(obj))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_torch_save(obj: Any, path: Path) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def copy_verified(src: Path, dst: Path) -> dict:
    """Copy via a temporary name, verify size + SHA-256, then rename into place."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.partial")
    shutil.copyfile(src, tmp)
    src_hash, tmp_hash = sha256_file(src), sha256_file(tmp)
    if src.stat().st_size != tmp.stat().st_size or src_hash != tmp_hash:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Verified copy failed for {src} -> {dst}")
    os.replace(tmp, dst)
    return {"src": str(src), "dst": str(dst), "bytes": dst.stat().st_size, "sha256": src_hash}


def mirror_tree(src_root: Path, dst_root: Path, skip_names: Iterable[str] = ()) -> list[dict]:
    """Copy every file under src_root to dst_root with verification.

    Completion markers (names starting with "_COMPLETE") are skipped here; the
    caller writes them last, after every required file has been verified.
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    skip = set(skip_names)
    records = []
    for path in sorted(p for p in src_root.rglob("*") if p.is_file()):
        rel = path.relative_to(src_root)
        if path.name.startswith("_COMPLETE") or path.name in skip or path.name.endswith(".partial"):
            continue
        target = dst_root / rel
        if target.is_file() and target.stat().st_size == path.stat().st_size \
                and sha256_file(target) == sha256_file(path):
            records.append({"src": str(path), "dst": str(target), "bytes": path.stat().st_size,
                            "status": "already_identical"})
            continue
        records.append(copy_verified(path, target))
    return records


def write_completion_marker(root: Path, name: str, required: Iterable[str], extra: dict | None = None) -> dict:
    """Write `name` only after every required relative path exists and is non-empty."""
    root = Path(root)
    missing = [rel for rel in required if not (root / rel).is_file() or (root / rel).stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Cannot mark {root} complete; missing/empty: {missing}")
    record = {
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {rel: {"bytes": (root / rel).stat().st_size,
                        "sha256": sha256_file(root / rel) if (root / rel).stat().st_size < 2 * 1024 ** 3 else None}
                  for rel in required},
    }
    if extra:
        record.update(extra)
    atomic_write_json(root / name, record)
    return record


def marker_is_valid(root: Path, name: str, check_hashes: bool = True) -> bool:
    """A marker counts only if every file it lists still exists with the same size/hash."""
    path = Path(root) / name
    if not path.is_file():
        return False
    try:
        record = read_json(path)
        for rel, info in record["files"].items():
            target = Path(root) / rel
            if not target.is_file() or target.stat().st_size != info["bytes"]:
                return False
            if check_hashes and info.get("sha256") and sha256_file(target) != info["sha256"]:
                return False
    except (KeyError, ValueError, OSError):
        return False
    return True


def package_versions(names: Iterable[str]) -> dict:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_info(extra_packages: Iterable[str] = ()) -> dict:
    names = ["numpy", "pandas", "pyarrow", "torch", "scikit-learn", "matplotlib", "optuna", *extra_packages]
    info = {"python": sys.version, "platform": platform.platform(), "packages": package_versions(names)}
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as error:  # pragma: no cover - informational only
        info["torch_error"] = repr(error)
    return info


def git_info(repo: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)

    head = run("rev-parse", "HEAD")
    if head.returncode != 0:
        return {"available": False, "error": head.stderr.strip()}
    status = run("status", "--porcelain")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    return {
        "available": True,
        "commit": head.stdout.strip(),
        "branch": branch.stdout.strip(),
        "uncommitted_changes": bool(status.stdout.strip()),
        "uncommitted_paths": status.stdout.strip().splitlines()[:50],
    }


def disk_free_bytes(path: Path) -> int | None:
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


class RunLogger:
    """Append-only run log plus a separate error log with tracebacks."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "logs"
        self.dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with open(self.dir / "run.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def stage(self, name: str):
        logger = self

        class _Stage:
            def __enter__(self):
                self.t0 = time.time()
                logger(f"[start] {name}")
                return self

            def __exit__(self, exc_type, exc, tb):
                if exc_type is None:
                    logger(f"[done] {name} ({time.time() - self.t0:.1f}s)")
                    return False
                import traceback

                with open(logger.dir / "errors.log", "a", encoding="utf-8") as handle:
                    handle.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} stage '{name}' failed\n")
                    handle.write("".join(traceback.format_exception(exc_type, exc, tb)) + "\n")
                logger(f"[FAILED] {name}: {exc!r} (traceback in logs/errors.log)")
                return False

        return _Stage()
