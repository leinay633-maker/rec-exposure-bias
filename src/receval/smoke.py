"""Smoke run: every model end to end on validation data, with wall-clock timings (no test data).

Each item is registered as soon as it finishes or fails, so a crash midway still leaves a record.
"""
from __future__ import annotations

import time

import torch

from . import registry
from .pipeline import prepare
from .retrieval import Pool, evaluate_scorer, itemcf_scorer, popularity_scorer, train_retriever
from .train import device, popularity_scores, train_lightgbm, train_ranker, validation_uauc

RANKER_CFG = {"lr": 1e-3, "dim": 16, "weight_decay": 0.0}
LGB_CFG = {"learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 50}
RETRIEVER_CFG = {"lr": 1e-3, "dim": 16, "out": 64, "tau": 0.1}


def run(fraction: float, epochs: int) -> list[dict]:
    dev = device()
    common = {"kind": "smoke", "fraction": fraction, "epochs": epochs, **registry.git_state(),
              "code_hash": registry.code_hash(), "data_hash": registry.data_hash(),
              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    rows: list[dict] = []

    def timed(name, cfg, fn):
        started = time.time()
        entry = {**common, "model": name, "config": cfg, "config_hash": registry.config_hash(cfg),
                 "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        registry.record({**entry, "status": "started"})  # an interrupted item still shows up
        try:
            entry.update(status="ok", **fn())
        except Exception as exc:  # registered, then the smoke run continues with the next model
            entry.update(status="failed", error=repr(exc))
        entry["seconds"] = round(time.time() - started, 1)
        registry.record(entry)
        rows.append(entry)
        print({k: entry[k] for k in entry if k not in common}, flush=True)

    started = time.time()
    p = prepare(fraction)
    print({"model": "features", "seconds": round(time.time() - started, 1), "train_rows": len(p.train),
           "valid_rows": len(p.valid), "calib_rows": len(p.calib)}, flush=True)
    tr, va = p.built["train"], p.built["valid"]

    timed("popularity", {}, lambda: {"valid_uauc": validation_uauc(va, popularity_scores(p.space, va), "is_click")})
    for name in ("logreg", "two_tower", "dcn", "din", "mmoe"):
        timed(name, RANKER_CFG, lambda name=name: {"valid_uauc": train_ranker(
            name, RANKER_CFG, p.space, tr, va, seed=0, dev=dev, max_epochs=epochs)[1]["valid_uauc"]})
    timed("lightgbm", LGB_CFG, lambda: {"valid_uauc": train_lightgbm(LGB_CFG, tr, va, seed=0,
                                                                      rounds=50 * epochs)[1]["valid_uauc"]})

    pool = Pool(p.space, p.videos, p.train)
    targets = pool.targets(p.valid)
    users = sorted(targets)
    for name, factory in (("recall_popularity", popularity_scorer), ("recall_itemcf", itemcf_scorer)):
        timed(name, {}, lambda factory=factory: {
            "valid_recall@100": float(evaluate_scorer(factory(pool, p.train, dev), users, pool, targets)
                                      ["recall@100"].mean())})
    for name, logq in (("recall_two_tower", False), ("recall_two_tower_logq", True)):
        timed(name, RETRIEVER_CFG, lambda logq=logq: {"valid_recall@100": train_retriever(
            RETRIEVER_CFG, p.space, tr, p.train, p.users, pool, targets, seed=0, logq=logq, dev=dev,
            max_epochs=epochs)[1]["valid_recall@100"]})
    return rows
