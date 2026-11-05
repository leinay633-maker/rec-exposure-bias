"""Append-only run registry: every run records code version, config hash, data hash and outcome."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import config


def git_state(repo: Path = config.ROOT) -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout
    return {"commit": head, "dirty": bool(dirty.strip())}


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def data_hash(cache: Path = config.CACHE) -> str:
    return hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest()[:16]


def code_hash(repo: Path = config.ROOT) -> str:
    """Fingerprint of the code a run used: every .py under src/, pyproject.toml and uv.lock (LF-normalised)."""
    h = hashlib.sha256()
    for path in sorted((repo / "src").rglob("*.py")) + [repo / "pyproject.toml", repo / "uv.lock"]:
        if path.exists():
            h.update(path.relative_to(repo).as_posix().encode() + b"\0")
            h.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:16]


def _fingerprint(fn) -> str:
    """A fingerprint, or an explicit marker when its input cannot be read, so the run is still registered."""
    try:
        return fn()
    except OSError as exc:
        return f"unavailable: {exc!r}"


@contextmanager
def logged(kind: str, cfg: dict | None = None, runs: Path = config.RUNS, **extra):
    """Record a start entry, then ok or failed, so an interrupted or crashed run still leaves a trace.

    Yields a dict holding the run_id; keys the caller adds to it go into the final entry.
    """
    cfg = cfg or {}
    base = {"run_id": f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}-{uuid.uuid4().hex[:6]}", "kind": kind,
            "config": cfg, "config_hash": config_hash(cfg), **git_state(), "code_hash": code_hash(),
            "data_hash": _fingerprint(lambda: data_hash(config.CACHE)), **extra}
    record({**base, "status": "started", "time": time.strftime("%Y-%m-%d %H:%M:%S")}, runs)
    state = {"run_id": base["run_id"]}
    try:
        yield state
    except BaseException as exc:
        record({**base, **state, "status": "failed", "error": repr(exc), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
               runs)
        raise
    record({**base, **state, "status": "ok", "time": time.strftime("%Y-%m-%d %H:%M:%S")}, runs)


def succeeded(kind: str, runs: Path = config.RUNS, **match) -> bool:
    """Whether some run of this kind finished ok with every given field equal."""
    return any(e.get("kind") == kind and e.get("status") == "ok" and all(e.get(k) == v for k, v in match.items())
               for e in read(runs))


def new_run(kind: str, runs: Path = config.RUNS) -> tuple[str, Path]:
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}-{uuid.uuid4().hex[:6]}"
    path = runs / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return run_id, path


def record(entry: dict, runs: Path = config.RUNS) -> None:
    runs.mkdir(parents=True, exist_ok=True)
    with open(runs / "registry.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def read(runs: Path = config.RUNS) -> list[dict]:
    path = runs / "registry.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
