"""Exploratory logQ x temperature ablation for the retrieval two-tower (protocol §15).

The two selected retrievers differ only in temperature (lr 1e-3, output 64; plain tau 0.05, logQ tau 0.1), so
the headline gap mixes logQ with temperature. This script fills the 2x2: the two frozen cells are loaded from
configs/zoo_manifest.json (hash-checked), the other two are trained here for seeds 0-2 and registered as kind
"ablation". Every cell is scored on the proxy pair (standard valid vs random calibration) and on the opened test
pair, on users with targets in both sets of the pair, each user's value averaged over seeds first. The test pair
was opened on 10-30, so everything here is exploratory and changes no protocol decision.

Usage: python scripts/logq_tau_ablation.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from receval import analysis as A
from receval import config, data, registry, stats, tune
from receval import orchestrate as O
from receval.pipeline import prepare
from receval.retrieval import Pool, evaluate_scorer, retriever_scorer, train_retriever
from receval.train import device

SEEDS = (0, 1, 2)
CELLS = {  # cell -> (registry model name, logq, tau); lr 1e-3, output 64, embedding 16 throughout
    "plain_t005": ("recall_two_tower", False, 0.05),
    "plain_t01": ("recall_two_tower", False, 0.1),
    "logq_t005": ("recall_two_tower_logq", True, 0.05),
    "logq_t01": ("recall_two_tower_logq", True, 0.1),
}
ORDER = tuple(CELLS)
FROZEN = ("plain_t005", "logq_t01")  # the two selected configs of §13.1
SIMPLE = {  # simple effects: (plus cell, minus cell)
    "logq_at_tau0.05": ("logq_t005", "plain_t005"),
    "logq_at_tau0.1": ("logq_t01", "plain_t01"),
    "tau0.1_vs_0.05_without_logq": ("plain_t01", "plain_t005"),
    "tau0.1_vs_0.05_with_logq": ("logq_t01", "logq_t005"),
}
ORIGINAL = ("logq_t01", "plain_t005")  # the gap reported in RESULTS §4
METRICS = ("recall@50", "recall@100", "recall@200", "ndcg@100")
FORMAL_RECALL = config.ROOT / "evidence/analysis/20261031-121508-analysis-926a49/recall.json"
TOLERANCE = 1e-9


def cfg_of(cell: str) -> dict:
    return {"lr": 1e-3, "out": 64, "tau": CELLS[cell][2], "dim": 16}  # key order as in tune.retriever_grid


def frozen_cells(payload: dict) -> dict[str, list[dict]]:
    index = A.artefact_index(payload["artefacts"])
    out = {}
    for cell in FROZEN:
        entries = index[(CELLS[cell][0], "is_click")]
        if [int(e["seed"]) for e in entries] != list(SEEDS) or any(e["config"] != cfg_of(cell) for e in entries):
            raise RuntimeError(f"frozen {cell} is not lr 1e-3 / out 64 / tau {CELLS[cell][2]} for seeds 0-2")
        out[cell] = entries
    return out


def tuned_seed0(cell: str) -> float | None:
    """Validation Recall@100 the tune run registered for this config and seed 0, if any."""
    model, cfg = CELLS[cell][0], cfg_of(cell)
    hits = [e for e in registry.read() if e.get("kind") == "tune" and e.get("status") == "ok"
            and e.get("model") == model and e.get("seed") == 0 and e.get("config") == cfg]
    return hits[-1]["info"]["valid_recall@100"] if hits else None


def train_cells(p, pool: Pool, valid_targets: dict, dev) -> tuple[dict[str, list[dict]], list[str], dict]:
    trained, attempts, repro = {}, [], {}
    for cell in ORDER:
        if cell in FROZEN:
            continue
        model, logq, _ = CELLS[cell]
        cfg = cfg_of(cell)
        entries = []
        for seed in SEEDS:
            def fn(cfg=cfg, seed=seed, logq=logq):
                return lambda: train_retriever(cfg, p.space, p.built["train"], p.train, p.users, pool, valid_targets,
                                               seed=seed, logq=logq, dev=dev)
            entry = tune._with_retry("ablation", model, cfg, seed, "is_click", fn(), attempts)
            if entry["status"] != "ok":
                raise RuntimeError(f"{cell} seed {seed} failed twice: {entry.get('error')}")
            entries.append({"model": model, "label": "is_click", "seed": seed, "run_id": entry["run_id"],
                            "path": entry["artefact"], "sha256": entry["sha256"], "config": cfg,
                            "valid_metric": tune._metric(entry)})
        trained[cell] = entries
        registered = tuned_seed0(cell)
        repro[cell] = {"tune_seed0": registered, "ablation_seed0": entries[0]["valid_metric"],
                       "difference": None if registered is None else entries[0]["valid_metric"] - registered}
    return trained, attempts, repro


def score_pair(frames, pool: Pool, p, models: dict[str, list], dev) -> tuple[list[int], dict[str, np.ndarray]]:
    """Per metric a (users, cells, sets, seeds) cube on the users with targets in both sets."""
    targets = [pool.targets(f) for f in frames]
    users = sorted(set(targets[0]) & set(targets[1]))
    cubes = {m: np.empty((len(users), len(ORDER), 2, len(SEEDS))) for m in METRICS}
    for c, cell in enumerate(ORDER):
        for k, model in enumerate(models[cell]):
            scorer = retriever_scorer(model, p.space, p.users, pool, dev)
            for s in (0, 1):
                table = evaluate_scorer(scorer, users, pool, targets[s]).loc[users]
                for m in METRICS:
                    cubes[m][:, c, s, k] = table[m].to_numpy()
    return users, cubes


def _summary(point: float, reps: np.ndarray) -> dict:
    return {"point": float(point), "ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
            "se": float(np.std(reps, ddof=1)), "p": stats.two_sided_p(reps)}


def contrasts(cube: np.ndarray, samples: int = 10_000, seed: int = 0) -> dict:
    """Simple effects, interaction and the original gap on each set and as random minus standard."""
    per_user = cube.mean(axis=3)  # (users, cells, sets)
    point = per_user.mean(0)
    reps = stats.joint_user_bootstrap(per_user, samples=samples, seed=seed)  # (samples, cells, sets)
    seed_means = cube.mean(0)  # (cells, sets, seeds)
    i = {cell: n for n, cell in enumerate(ORDER)}

    def diff(plus, minus, arr):
        return arr[..., i[plus], :] - arr[..., i[minus], :]

    def block(plus, minus):
        d_point, d_reps = diff(plus, minus, point), diff(plus, minus, reps)
        d_seed = seed_means[i[plus]] - seed_means[i[minus]]  # (sets, seeds)
        out = {}
        for s, name in enumerate(("standard", "random")):
            out[name] = {**_summary(d_point[s], d_reps[:, s]),
                         "seed_support": int(np.sum(np.sign(d_seed[s]) == np.sign(d_point[s])))}
        out["random_minus_standard"] = _summary(d_point[1] - d_point[0], d_reps[:, 1] - d_reps[:, 0])
        return out

    simple = {name: {"plus": plus, "minus": minus, **block(plus, minus)} for name, (plus, minus) in SIMPLE.items()}
    for key in ("standard", "random", "random_minus_standard"):
        reject = stats.holm([simple[n][key]["p"] for n in SIMPLE])
        for flag, name in zip(reject, SIMPLE):
            simple[name][key]["holm_significant"] = bool(flag)
    inter_point = (diff("logq_t01", "logq_t005", point) - diff("plain_t01", "plain_t005", point))
    inter_reps = (diff("logq_t01", "logq_t005", reps) - diff("plain_t01", "plain_t005", reps))
    interaction = {name: _summary(inter_point[s], inter_reps[:, s]) for s, name in enumerate(("standard", "random"))}
    interaction["random_minus_standard"] = _summary(inter_point[1] - inter_point[0],
                                                    inter_reps[:, 1] - inter_reps[:, 0])
    original = block(*ORIGINAL)
    shares = {}
    for s, name in enumerate(("standard", "random")):
        gap = diff(*ORIGINAL, point)[s]
        shares[name] = {"gap": float(gap),
                        "logq_first_then_tau": float(diff("logq_t005", "plain_t005", point)[s] / gap),
                        "tau_first_then_logq": float(diff("logq_t01", "plain_t01", point)[s] / gap)}
    return {"cells": {cell: [float(point[i[cell], 0]), float(point[i[cell], 1])] for cell in ORDER},
            "simple_effects": simple, "interaction": interaction, "original_gap": original,
            "logq_share_of_original_gap": shares}


def formal_check(cubes: dict, users: list[int]) -> dict:
    """On the test pair the frozen cells must reproduce the formal analysis's Recall@100 (RESULTS §4)."""
    formal = json.loads(FORMAL_RECALL.read_text(encoding="utf-8"))
    if formal["users"] != len(users):
        raise RuntimeError(f"R* differs from the formal analysis: {len(users)} vs {formal['users']}")
    rows, worst = [], 0.0
    for cell in FROZEN:
        mine = cubes["recall@100"][:, ORDER.index(cell)].mean(axis=2).mean(0)
        theirs = formal["recall@100"][CELLS[cell][0]]
        rows.append({"cell": cell, "here": [float(v) for v in mine], "formal": theirs})
        worst = max(worst, float(np.max(np.abs(np.asarray(mine) - np.asarray(theirs)))))
    if not worst < TOLERANCE:
        raise RuntimeError(f"frozen cells do not reproduce the formal Recall@100 (max difference {worst})")
    return {"users": len(users), "max_abs_difference": worst, "rows": rows}


def main() -> None:
    with registry.logged("ablation-logq-tau", {"cells": {c: cfg_of(c) for c in ORDER}, "seeds": list(SEEDS)}) as run:
        out_dir = config.RUNS / "ablation" / run["run_id"]
        run["output"] = str(out_dir)
        manifest_path = config.ROOT / data.ZOO_MANIFEST
        payload, run["zoo_manifest_sha256"] = O.load_selection(manifest_path)
        O._verify_inputs(payload)
        frozen = frozen_cells(payload)
        p = prepare(1.0)
        dev = device()
        pool = Pool(p.space, p.videos, p.train)
        trained, attempts, repro = train_cells(p, pool, pool.targets(p.valid), dev)
        artefacts = {**frozen, **trained}
        data.verify_artefacts(payload["artefacts"])  # frozen files unchanged after training
        models = {cell: [A.load_retriever(e, p.space, dev) for e in artefacts[cell]] for cell in ORDER}
        result = {"note": "exploratory, protocol §15; test pair opened on 2026-10-30", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "cells": {cell: {"model": CELLS[cell][0], "logq": CELLS[cell][1], "config": cfg_of(cell),
                                   "frozen": cell in FROZEN,
                                   "runs": [{k: e[k] for k in ("seed", "run_id", "path", "sha256", "valid_metric")}
                                            for e in artefacts[cell]]} for cell in ORDER},
                  "attempts": attempts, "reproduction_seed0": repro}
        pairs = {"proxy": ([p.valid, p.calib], O.PROXY_NOTE), "test": (O._test_sets(p), "standard test vs random test, 4/29-5/08, tab 1")}
        for name, (frames, note) in pairs.items():
            users, cubes = score_pair(frames, pool, p, models, dev)
            block = {"pair": note, "users": len(users), "recall@100": contrasts(cubes["recall@100"]),
                     "secondary": {m: {cell: [float(v) for v in cubes[m][:, c].mean(axis=2).mean(0)]
                                       for c, cell in enumerate(ORDER)} for m in METRICS if m != "recall@100"}}
            if name == "test":
                block["formal_check"] = formal_check(cubes, users)
            result[name] = block
        A.write(result, out_dir, "logq_tau")
    print(json.dumps({k: result[k]["recall@100"]["cells"] for k in ("proxy", "test")}, indent=1))


if __name__ == "__main__":
    main()
