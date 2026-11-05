"""Study 2's run registry: the same append-only format as study 1, kept under .runs/kuairec/."""
from __future__ import annotations

import hashlib
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from receval.registry import code_hash, config_hash, git_state, read, record

from . import config


def data_hash(cache: Path = config.CACHE) -> str:
    manifest = cache / "manifest.json"
    if not manifest.exists():
        return "unavailable: no KuaiRec cache manifest yet"
    return hashlib.sha256(manifest.read_bytes()).hexdigest()[:16]


@contextmanager
def logged(kind: str, cfg: dict | None = None, runs: Path = config.RUNS, **extra):
    """Start entry, then ok or failed; the caller adds keys (output paths, hashes) to the yielded dict."""
    cfg = cfg or {}
    base = {"run_id": f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}-{uuid.uuid4().hex[:6]}", "kind": kind, "config": cfg,
            "config_hash": config_hash(cfg), **git_state(), "code_hash": code_hash(), "data_hash": data_hash(), **extra}
    record({**base, "status": "started", "time": time.strftime("%Y-%m-%d %H:%M:%S")}, runs)
    state = {"run_id": base["run_id"]}
    try:
        yield state
    except BaseException as exc:
        record({**base, **state, "status": "failed", "error": repr(exc), "time": time.strftime("%Y-%m-%d %H:%M:%S")}, runs)
        raise
    record({**base, **state, "status": "ok", "time": time.strftime("%Y-%m-%d %H:%M:%S")}, runs)


def entries(runs: Path = config.RUNS) -> list[dict]:
    return read(runs)


def new_run_dir(kind: str, runs: Path = config.RUNS) -> tuple[str, Path]:
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}-{uuid.uuid4().hex[:6]}"
    path = runs / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return run_id, path
