"""Smoke run, grid search, seed training and the §8 sensitivity models for study 2, with a selection file.

The selection keeps every attempt (failed, retried, smaller batch) of every configuration and seed; a finished
run is reused only if it was trained on the current data cache.
"""
from __future__ import annotations

import json
import time

import torch

from . import config, runlog
from .train import BATCH, GRID, LEARNED, MAX_EPOCHS, context, fit, train_run

ZOO = config.RUNS / "zoo"
BUDGET_HOURS = 10.0
SENSITIVITY = {"kinds": ("gr_text", "gr_shuf"), "min_epochs": 5}  # protocol §8


def device() -> torch.device:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smoke() -> dict:
    """Largest configuration of each kind: one epoch and one full validation ranking; no metric is printed."""
    dev = device()
    with runlog.logged("smoke", {"configs": {k: GRID[k][-1] for k in LEARNED}}) as state:
        ctx = context()
        out = {}
        for kind in LEARNED:
            torch.cuda.reset_peak_memory_stats()
            _, _, log = fit(kind, GRID[kind][-1], 0, ctx, dev, 1, BATCH[kind], smoke=True)
            e = log["epochs"][0]
            out[kind] = {"train_seconds": e["train_seconds"], "rank_seconds": e["rank_seconds"],
                         "train_loss": e["train_loss"], "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        hours = sum(len(GRID[k]) * MAX_EPOCHS[k] * (out[k]["train_seconds"] + out[k]["rank_seconds"]) for k in LEARNED) / 3600
        state.update({"kinds": out, "projected_tuning_hours": round(hours, 2), "budget_hours": BUDGET_HOURS})
    return {"kinds": out, "projected_tuning_hours": round(hours, 2)}


def _smoke_ok() -> None:
    """§3.9: tuning runs only after a successful smoke run within the budget (no downsizing is implemented)."""
    runs = [e for e in runlog.entries() if e.get("kind") == "smoke" and e.get("status") == "ok"]
    if not runs or runs[-1]["projected_tuning_hours"] > BUDGET_HOURS:
        raise RuntimeError("no smoke run within the tuning budget; apply §3.9 and append it to the protocol")


def _matches(e: dict, kind: str, cfg: dict, seed: int, purpose: str) -> bool:
    c = e.get("config", {})
    return (e.get("kind") == "train" and c.get("kind") == kind and c.get("config") == cfg and c.get("seed") == seed
            and c.get("purpose") == purpose)


def attempts(kind: str, cfg: dict, seed: int, purpose: str) -> list[dict]:
    """Every finished attempt (ok or failed) for this specification, in registry order."""
    return [{"run_id": e["run_id"], "status": e["status"], "batch": e["config"].get("batch"),
             "data_hash": e.get("data_hash"), "code_hash": e.get("code_hash"), "error": e.get("error")}
            for e in runlog.entries() if _matches(e, kind, cfg, seed, purpose) and e.get("status") in ("ok", "failed")]


def _done(kind: str, cfg: dict, seed: int, purpose: str) -> dict | None:
    current = runlog.data_hash()
    hits = [e for e in runlog.entries() if _matches(e, kind, cfg, seed, purpose) and e.get("status") == "ok"
            and e.get("data_hash") == current]
    return hits[-1] if hits else None


def _run(kind: str, cfg: dict, seed: int, ctx, dev, purpose: str, min_epochs: int = 1) -> dict | None:
    """Reuse a finished run on the current cache or train; a failed run is retried once (§3.1)."""
    done = _done(kind, cfg, seed, purpose)
    if done is not None:
        return {**done, **done["config"]}
    for attempt in range(2):
        try:
            return train_run(kind, cfg, seed, ctx, dev, purpose, min_epochs=min_epochs)
        except torch.cuda.OutOfMemoryError:
            raise  # the smallest batch still does not fit: stop and append a remedy (§3.1)
        except Exception as exc:  # noqa: BLE001 - every ordinary failure is retried once, then reported
            if attempt == 1:
                print(f"{kind} {cfg} seed {seed} failed twice: {exc!r}")
    return None


KEEP = ("run_id", "path", "sha256", "valid_recall@100", "best_epoch", "epochs_run", "params", "seconds", "batch",
        "max_epochs")


def _write(selection: dict, run_id: str) -> str:
    ZOO.mkdir(parents=True, exist_ok=True)
    text = json.dumps(selection, indent=1)
    (ZOO / f"selection-{run_id}.json").write_text(text, encoding="utf-8")
    (ZOO / "selection.json").write_text(text, encoding="utf-8")
    return str(ZOO / f"selection-{run_id}.json")


def tune(kinds=LEARNED) -> dict:
    _smoke_ok()
    dev = device()
    with runlog.logged("tune", {"kinds": list(kinds), "grid": {k: GRID[k] for k in kinds}}) as state:
        ctx = context()
        selection = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}
        for kind in kinds:
            trials = []
            for cfg in GRID[kind]:
                r = _run(kind, cfg, 0, ctx, dev, "tune")
                trials.append({"config": cfg, "run_id": r["run_id"] if r else None,
                               "valid_recall@100": r["valid_recall@100"] if r else None, "status": "ok" if r else "failed",
                               "attempts": attempts(kind, cfg, 0, "tune")})
            ok = [t for t in trials if t["status"] == "ok"]
            if not ok:
                raise RuntimeError(f"every configuration of {kind} failed; append a remedy to the protocol (§3.1)")
            best = max(ok, key=lambda t: (t["valid_recall@100"], -GRID[kind].index(t["config"])))
            seeds, seed_attempts = {}, {}
            for seed in config.SEEDS:
                purpose = "tune" if seed == 0 else "seed"
                r = _run(kind, best["config"], seed, ctx, dev, purpose)
                if r is None:
                    raise RuntimeError(f"{kind} seed {seed} failed twice; append a remedy to the protocol (§3.1)")
                seeds[str(seed)] = {k: r[k] for k in KEEP}
                seed_attempts[str(seed)] = attempts(kind, best["config"], seed, purpose)
            selection["models"][kind] = {"config": best["config"], "winner_run_id": best["run_id"], "trials": trials,
                                         "seeds": seeds, "seed_attempts": seed_attempts}
        state["output"] = _write(selection, state["run_id"])
    return selection


def sensitivity() -> dict:
    """§8: the two semantic-ID models, selected configurations, trained with at least 5 epochs before selection."""
    dev = device()
    selection = json.loads((ZOO / "selection.json").read_text(encoding="utf-8"))
    with runlog.logged("sensitivity", SENSITIVITY) as state:
        ctx = context()
        block = {}
        for kind in SENSITIVITY["kinds"]:
            cfg = selection["models"][kind]["config"]
            seeds, seed_attempts = {}, {}
            for seed in config.SEEDS:
                r = _run(kind, cfg, seed, ctx, dev, "sensitivity", min_epochs=SENSITIVITY["min_epochs"])
                if r is None:
                    raise RuntimeError(f"sensitivity {kind} seed {seed} failed twice")
                seeds[str(seed)] = {k: r[k] for k in KEEP}
                seed_attempts[str(seed)] = attempts(kind, cfg, seed, "sensitivity")
            block[kind] = {"config": cfg, "min_epochs": SENSITIVITY["min_epochs"], "seeds": seeds,
                           "seed_attempts": seed_attempts}
        selection["sensitivity"] = block
        state["output"] = _write(selection, state["run_id"])
    return block
