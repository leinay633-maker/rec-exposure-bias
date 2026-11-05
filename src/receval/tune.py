"""Grid search, seed replicates and second-label retraining (protocol §5, §9); writes the zoo selection.

Every run is registered (config hash, data hash, code commit) and its artefact saved under
.runs/runs/<run_id>/; `.runs/zoo/selection.json` lists the chosen run per model, seed and label with
artefact hashes, which `configs/zoo_manifest.json` copies at the zoo freeze.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import torch

from . import config, registry
from .data import ZOO_MANIFEST, file_digest, verify_data, verify_manifest
from .pipeline import Prepared
from .retrieval import Pool, train_retriever
from .train import device, train_lightgbm, train_ranker

SEEDS = (0, 1, 2)
NEURAL = ("two_tower", "dcn", "din", "mmoe")
SECOND_LABEL_RETRAIN = ("logreg", "lightgbm", "two_tower", "dcn", "din")


def ranker_grid(name: str) -> list[dict]:
    """Grid in protocol order: the leftmost parameter varies slowest."""
    if name == "logreg":
        return [{"lr": 1e-3, "weight_decay": wd, "dim": 1} for wd in (1e-6, 1e-5, 1e-4, 1e-3)]
    if name == "lightgbm":
        return [{"num_leaves": nl, "learning_rate": lr, "min_child_samples": mc}
                for nl, lr, mc in itertools.product((31, 127), (0.05, 0.1), (50, 200))]
    return [{"lr": lr, "dim": dim, "weight_decay": wd}
            for lr, dim, wd in itertools.product((1e-3, 3e-3), (16, 32), (0.0, 1e-5))]


def retriever_grid() -> list[dict]:
    return [{"lr": lr, "out": out, "tau": tau, "dim": 16}
            for lr, out, tau in itertools.product((1e-3, 3e-3), (32, 64), (0.05, 0.1))]


def _save(model, path: Path) -> Path:
    if hasattr(model, "save_model"):
        target = path / "model.txt"
        model.save_model(str(target))
    else:
        target = path / "model.pt"
        torch.save(model.state_dict(), target)
    return target


def _run(kind: str, name: str, cfg: dict, seed: int, label: str, fn, attempts: list) -> dict:
    run_id, path = registry.new_run(f"{kind}-{name}")
    attempts.append(run_id)
    started = time.time()
    entry = {"run_id": run_id, "kind": kind, "model": name, "config": cfg, "config_hash": registry.config_hash(cfg),
             "seed": seed, "label": label, **registry.git_state(), "code_hash": registry.code_hash(),
             "data_hash": registry.data_hash(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    registry.record({**entry, "status": "started"})  # an interrupted run still shows up
    try:
        model, info = fn()
        artefact = _save(model, path)
        entry.update(status="ok", info=info, artefact=str(artefact.relative_to(config.ROOT)).replace("\\", "/"),
                     sha256=file_digest(artefact))
    except Exception as exc:  # recorded, never silently dropped (protocol §5.2)
        entry.update(status="failed", error=repr(exc))
    entry["seconds"] = round(time.time() - started, 1)
    (path / "run.json").write_text(json.dumps(entry, indent=2, default=str), encoding="utf-8")
    registry.record(entry)
    print({k: entry.get(k) for k in ("run_id", "status", "seconds")}, entry.get("info", {}).get("valid_uauc",
          entry.get("info", {}).get("valid_recall@100")), flush=True)
    return entry


def _metric(entry: dict) -> float:
    info = entry.get("info", {})
    return info.get("valid_uauc", info.get("valid_recall@100", float("-inf"))) if entry["status"] == "ok" else float("-inf")


def _best(entries: list[dict]) -> dict:
    """Highest validation metric; ties go to the earlier grid position (protocol §5.2)."""
    best = entries[0]
    for entry in entries[1:]:
        if _metric(entry) > _metric(best):
            best = entry
    if _metric(best) == float("-inf"):
        raise RuntimeError(f"every configuration failed for {best['model']}")
    return best


def _ranker_fn(name: str, cfg: dict, p: Prepared, seed: int, label: str, dev):
    if name == "lightgbm":
        return lambda: train_lightgbm(cfg, p.built["train"], p.built["valid"], seed=seed, label=label)
    return lambda: train_ranker(name, cfg, p.space, p.built["train"], p.built["valid"], seed=seed, label=label, dev=dev)


def _with_retry(kind, name, cfg, seed, label, fn, attempts: list) -> dict:
    entry = _run(kind, name, cfg, seed, label, fn, attempts)
    if entry["status"] != "ok":  # a failed final seed is rerun once (protocol §5.2)
        entry = _run(kind, name, cfg, seed, label, fn, attempts)
    return entry


def tune_rankers(p: Prepared, models=("logreg", "lightgbm", "two_tower", "dcn", "din", "mmoe")) -> dict:
    """`attempts` lists every run of the model, failed ones and retries included."""
    dev = device()
    selection = {}
    for name in models:
        attempts: list = []
        trials = [_run("tune", name, cfg, 0, "is_click", _ranker_fn(name, cfg, p, 0, "is_click", dev), attempts)
                  for cfg in ranker_grid(name)]
        best = _best(trials)
        seeds = {0: best}
        if name in NEURAL or name == "lightgbm":
            for seed in SEEDS[1:]:
                seeds[seed] = _with_retry("seed", name, best["config"], seed, "is_click",
                                          _ranker_fn(name, best["config"], p, seed, "is_click", dev), attempts)
        selection[name] = {"config": best["config"], "trials": [t["run_id"] for t in trials],
                           "is_click": {s: e["run_id"] for s, e in seeds.items()}, "attempts": attempts}
        if name in SECOND_LABEL_RETRAIN:
            seed_list = SEEDS if name != "logreg" else (0,)
            selection[name]["long_view"] = {
                s: _with_retry("second-label", name, best["config"], s, "long_view",
                               _ranker_fn(name, best["config"], p, s, "long_view", dev), attempts)["run_id"]
                for s in seed_list}
    return selection


def tune_retrievers(p: Prepared) -> dict:
    dev = device()
    pool = Pool(p.space, p.videos, p.train)
    targets = pool.targets(p.valid)
    selection = {}
    for name, logq in (("recall_two_tower", False), ("recall_two_tower_logq", True)):
        def fn(cfg, seed, logq=logq):
            return lambda: train_retriever(cfg, p.space, p.built["train"], p.train, p.users, pool, targets,
                                           seed=seed, logq=logq, dev=dev)
        attempts: list = []
        trials = [_run("tune", name, cfg, 0, "is_click", fn(cfg, 0), attempts) for cfg in retriever_grid()]
        best = _best(trials)
        seeds = {0: best}
        for seed in SEEDS[1:]:
            seeds[seed] = _with_retry("seed", name, best["config"], seed, "is_click", fn(best["config"], seed),
                                      attempts)
        selection[name] = {"config": best["config"], "trials": [t["run_id"] for t in trials],
                           "is_click": {s: e["run_id"] for s, e in seeds.items()}, "attempts": attempts}
    return selection


def write_selection(selection: dict, out: Path = config.RUNS / "zoo" / "selection.json") -> Path:
    """Attach artefact paths and hashes from the registry, so the zoo manifest can bind them."""
    runs = {e["run_id"]: e for e in registry.read() if "run_id" in e}  # the last entry per run is its outcome
    artefacts = []
    for name, model in selection.items():
        for label in ("is_click", "long_view"):
            for seed, run_id in model.get(label, {}).items():
                entry = runs[run_id]
                artefacts.append({"model": name, "label": label, "seed": int(seed), "run_id": run_id,
                                  "path": entry["artefact"], "sha256": entry["sha256"], "config": entry["config"],
                                  "valid_metric": _metric(entry)})
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), **registry.git_state(),
               "code_hash": registry.code_hash(), "data_manifest_sha256": file_digest(config.CACHE / "manifest.json"),
               "selection": selection, "artefacts": artefacts}
    verify_against_registry(payload)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def verify_against_registry(payload: dict, runs: Path = config.RUNS) -> None:
    """Each listed artefact is the saved output of a successful run with that very model, label, seed and config.

    Stops one run from standing in for several seeds or another model when a selection is edited or merged.
    """
    outcomes = {e["run_id"]: e for e in registry.read(runs) if "run_id" in e}  # a run's last entry is its outcome
    for a in payload.get("artefacts", []):
        entry = outcomes.get(a.get("run_id"))
        if entry is None:
            raise PermissionError(f"artefact {a['model']}/{a['label']}/seed {a['seed']}: run {a.get('run_id')} "
                                  "is not in the registry")
        expected = {"status": "ok", "model": a["model"], "label": a["label"], "seed": int(a["seed"]),
                    "config": payload["selection"][a["model"]]["config"], "artefact": a["path"], "sha256": a["sha256"]}
        wrong = [k for k, v in expected.items() if entry.get(k) != v]
        if a.get("config") != entry.get("config"):
            wrong.append("listed config")
        if a.get("valid_metric") != _metric(entry):
            wrong.append("valid_metric")
        if wrong:
            raise PermissionError(f"artefact {a['model']}/{a['label']}/seed {a['seed']} does not match the registered "
                                  f"run {a.get('run_id')}: {wrong}")


def write_zoo_manifest(selection: Path = config.RUNS / "zoo" / "selection.json", repo: Path = config.ROOT,
                       cache: Path = config.CACHE, runs: Path = config.RUNS, raw_dir: Path = config.PURE_DIR) -> Path:
    """Copy the selection to configs/zoo_manifest.json for the zoo-freeze commit (protocol §11).

    Refuses an incomplete or altered zoo, a data cache whose files no longer match its manifest, and a zoo
    whose proxy dry run (with the reload check) or power check has not succeeded for this very selection,
    data cache and code.
    """
    payload = json.loads(selection.read_text(encoding="utf-8"))
    verify_manifest(payload, repo)
    verify_against_registry(payload, runs)
    verify_data(cache, raw_dir)
    data_manifest = cache / "manifest.json"
    if payload["data_manifest_sha256"] != file_digest(data_manifest):
        raise PermissionError("the data cache changed after the zoo was trained")
    bound = {"selection_sha256": file_digest(selection), "data_hash": registry.data_hash(cache),
             "code_hash": registry.code_hash(repo)}
    for kind in ("analysis-proxy", "power-check"):
        if not registry.succeeded(kind, runs, **bound):
            raise PermissionError(f"no successful {kind} run for this selection, data and code; run it first")
    payload["data_manifest"] = json.loads(data_manifest.read_text(encoding="utf-8"))
    out = repo / ZOO_MANIFEST
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out
