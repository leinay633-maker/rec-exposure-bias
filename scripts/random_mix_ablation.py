"""Exploratory: does training with a slice of random-exposure log help the ranker on the random test? (§17)

LightGBM with the frozen configuration in five cells: A (the frozen models, hash-checked), A' (standard training
set, early stopping on random 4/26-4/28), B-standard (plus the standard 4/22-4/25 rows), B-random (plus the
random 4/22-4/25 rows, size-matched to B-standard's additions), C (random 4/22-4/25 only). All cells are scored
on the opened test pair (4/29-5/08, tab 1) on the common users of §7; per-user UAUC and mean log loss are
averaged over seeds first, then a joint user bootstrap gives the contrasts. Exploratory only.

Usage: python scripts/random_mix_ablation.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from receval import analysis as A
from receval import config, data, registry, stats
from receval import features as F
from receval import orchestrate as O
from receval.pipeline import prepare
from receval.train import lgb_matrix, train_lightgbm

SEEDS = (0, 1, 2)
CFG = {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50}
FIT, STOP = (20220422, 20220425), (20220426, 20220428)
CELLS = ("A", "A_prime", "B_standard", "B_random", "C")
CONTRASTS = {"B_random − B_standard": ("B_random", "B_standard"), "B_standard − A_prime": ("B_standard", "A_prime"),
             "C − A_prime": ("C", "A_prime"), "A_prime − A": ("A_prime", "A")}
SETS = ("standard_test", "random_test")
SAMPLE_SEED = 20261101


def concat(a: dict, b: dict) -> dict:
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def window(frame: pd.DataFrame, built: dict, span: tuple[int, int]) -> dict:
    return O.subset(built, frame["date"].between(*span).to_numpy())


def per_user_logloss(users: list, built: dict, p: np.ndarray) -> np.ndarray:
    y = built["labels"][:, F.LABELS.index("is_click")]
    q = np.clip(p, 1e-7, 1 - 1e-7)
    loss = pd.Series(-(y * np.log(q) + (1 - y) * np.log(1 - q)), index=built["user"])
    return loss.groupby(level=0).mean().loc[users].to_numpy()


def summary(rep: np.ndarray, point: float) -> dict:
    lo, hi = np.percentile(rep, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)], "se": float(np.std(rep, ddof=1)),
            "p": stats.two_sided_p(rep)}


def main() -> None:
    manifest = json.loads((config.ROOT / data.ZOO_MANIFEST).read_text(encoding="utf-8"))
    frozen = A.artefact_index(manifest["artefacts"])[("lightgbm", "is_click")]
    for e in frozen:
        if data.file_digest(config.ROOT / e["path"]) != e["sha256"] or e["config"] != CFG:
            raise PermissionError(f"frozen LightGBM {e['path']} does not match the manifest")
    with registry.logged("ablation-random-mix", {"config": CFG, "fit": FIT, "stop": STOP, "seeds": list(SEEDS)}) as run:
        p = prepare(1.0)
        rand_fit, rand_stop = window(p.calib, p.built["calib"], FIT), window(p.calib, p.built["calib"], STOP)
        std_fit = window(p.valid, p.built["valid"], FIT)
        n = min(len(std_fit["user"]), len(rand_fit["user"]))
        rng = np.random.default_rng(SAMPLE_SEED)
        pick = lambda b: b if len(b["user"]) == n else O.subset(b, np.sort(rng.choice(len(b["user"]), n, replace=False)))  # noqa: E731
        add_std, add_rand = pick(std_fit), pick(rand_fit)
        train_sets = {"A_prime": p.built["train"], "B_standard": concat(p.built["train"], add_std),
                      "B_random": concat(p.built["train"], add_rand), "C": rand_fit}
        frames = O._test_sets(p)
        sets = [F.build(f, p.space, p.users, p.videos, is_train=False) for f in frames]
        preds = [{} for _ in sets]
        fitted = {}
        for cell in CELLS:
            scores = [[] for _ in sets]
            fitted[cell] = []
            for seed in SEEDS:
                if cell == "A":
                    booster = A.load_ranker(frozen[seed], p.space, None)
                    info = {"best_iteration": booster.best_iteration, "source": frozen[seed]["path"]}
                else:
                    booster, info = train_lightgbm(CFG, train_sets[cell], rand_stop, seed=seed)
                fitted[cell].append({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in info.items()})
                for s, b in enumerate(sets):
                    scores[s].append(booster.predict(lgb_matrix(b), num_iteration=booster.best_iteration))
            for s in range(len(sets)):
                preds[s][cell] = np.vstack(scores[s])
        users = A.common_users(sets, "is_click")
        auc = A.per_user_auc_cube(users, sets, preds, "is_click", order=CELLS)  # (users, cells, sets, seeds)
        loss = np.empty_like(auc)
        for s, b in enumerate(sets):
            mask = np.isin(b["user"], users)
            sub = O.subset(b, mask)
            for c, cell in enumerate(CELLS):
                for k in range(len(SEEDS)):
                    loss[:, c, s, k] = per_user_logloss(users, sub, preds[s][cell][k][mask])
        cube = np.stack([auc, loss], 2)  # (users, cells, metric, sets, seeds)
        rep = stats.joint_user_bootstrap(cube.mean(-1), samples=10_000, seed=0)  # (samples, cells, metric, sets)
        point = cube.mean(-1).mean(0)
        seed_means = cube.mean(0)  # (cells, metric, sets, seeds)
        idx = {c: i for i, c in enumerate(CELLS)}
        out = {"cells": {}, "contrasts": {}, "users": len(users), "rows_added": n, "fitted": fitted,
               "rows": {"standard_fit": int(len(std_fit["user"])), "random_fit": int(len(rand_fit["user"])),
                        "random_stop": int(len(rand_stop["user"])), "train": int(len(p.built["train"]["user"]))}}
        for m, metric in enumerate(("uauc", "logloss")):
            for s, name in enumerate(SETS):
                out["cells"].setdefault(metric, {})[name] = {c: float(point[idx[c], m, s]) for c in CELLS}
                block = {}
                for label, (a, b) in CONTRASTS.items():
                    r = rep[:, idx[a], m, s] - rep[:, idx[b], m, s]
                    row = summary(r, point[idx[a], m, s] - point[idx[b], m, s])
                    per_seed = seed_means[idx[a], m, s] - seed_means[idx[b], m, s]
                    row["seed_support"] = int(np.sum(np.sign(per_seed) == np.sign(row["point"])))
                    row["per_seed"] = [float(v) for v in per_seed]
                    block[label] = row
                flags = stats.holm([v["p"] for v in block.values()])
                for v, f in zip(block.values(), flags):
                    v["holm_reject"] = bool(f)
                out["contrasts"].setdefault(metric, {})[name] = block
        out_dir = config.RUNS / "ablation" / run["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = A.write(out, out_dir, "random_mix")
        run["output"] = str(path)
    print(json.dumps({"users": out["users"], "rows_added": n, "cells": out["cells"],
                      "contrasts": {m: {s: {k: (round(v["point"], 4), [round(x, 4) for x in v["ci95"]], v["holm_reject"],
                                                v["seed_support"]) for k, v in blk.items()} for s, blk in ms.items()}
                                    for m, ms in out["contrasts"].items()}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
