"""Exploratory follow-up to §17 (protocol §18): the same cells trained for a fixed number of rounds.

§17's early stopping on the random log loss stopped the standard-data cells after 2-10 rounds while B-random ran
1,471-1,700, so the B contrasts mixed exposure with training length. Here every cell of seed s trains exactly as
many rounds as the frozen LightGBM of seed s has trees (A), with no early stopping; "A_refit" retrains A's data
the same way only to check that the pipeline reproduces A. Scored like §17. Exploratory only.

Usage: python scripts/random_mix_fixed_rounds.py
"""
from __future__ import annotations

import json

import numpy as np

from random_mix_ablation import CFG, FIT, SAMPLE_SEED, SEEDS, SETS, concat, per_user_logloss, summary, window
from receval import analysis as A
from receval import config, data, registry, stats
from receval import features as F
from receval import orchestrate as O
from receval.features import CAT_FIELDS, LABELS
from receval.pipeline import prepare
from receval.train import lgb_matrix

CELLS = ("A", "A_refit", "B_standard_fixed", "B_random_fixed", "C_fixed")
CONTRASTS = {"B_random_fixed − B_standard_fixed": ("B_random_fixed", "B_standard_fixed"),
             "B_standard_fixed − A": ("B_standard_fixed", "A"), "B_random_fixed − A": ("B_random_fixed", "A"),
             "C_fixed − A": ("C_fixed", "A")}
CHECK = ("A_refit", "A")


def train_fixed(cfg: dict, train: dict, rounds: int, seed: int, label: str = "is_click"):
    """The parameters of receval.train.train_lightgbm, trained for exactly `rounds` rounds, no validation set."""
    import lightgbm as lgb
    params = {"objective": "binary", "learning_rate": cfg["learning_rate"], "num_leaves": cfg["num_leaves"],
              "min_child_samples": cfg["min_child_samples"], "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "seed": seed, "verbose": -1, "num_threads": 0}
    dtrain = lgb.Dataset(lgb_matrix(train), train["labels"][:, LABELS.index(label)],
                         categorical_feature=list(range(len(CAT_FIELDS))), free_raw_data=False)
    return lgb.train(params, dtrain, num_boost_round=rounds)


def main() -> None:
    manifest = json.loads((config.ROOT / data.ZOO_MANIFEST).read_text(encoding="utf-8"))
    frozen = A.artefact_index(manifest["artefacts"])[("lightgbm", "is_click")]
    for e in frozen:
        if data.file_digest(config.ROOT / e["path"]) != e["sha256"] or e["config"] != CFG:
            raise PermissionError(f"frozen LightGBM {e['path']} does not match the manifest")
    with registry.logged("ablation-random-mix-fixed", {"config": CFG, "fit": FIT, "seeds": list(SEEDS)}) as run:
        p = prepare(1.0)
        rand_fit = window(p.calib, p.built["calib"], FIT)
        std_fit = window(p.valid, p.built["valid"], FIT)
        n = min(len(std_fit["user"]), len(rand_fit["user"]))
        rng = np.random.default_rng(SAMPLE_SEED)

        def pick(b: dict) -> dict:
            return b if len(b["user"]) == n else O.subset(b, np.sort(rng.choice(len(b["user"]), n, replace=False)))

        add_std, add_rand = pick(std_fit), pick(rand_fit)  # same draw order as §17
        train_sets = {"A_refit": p.built["train"], "B_standard_fixed": concat(p.built["train"], add_std),
                      "B_random_fixed": concat(p.built["train"], add_rand), "C_fixed": rand_fit}
        frames = O._test_sets(p)
        sets = [F.build(f, p.space, p.users, p.videos, is_train=False) for f in frames]
        boosters = [A.load_ranker(e, p.space, None) for e in frozen]
        rounds = [int(b.num_trees()) for b in boosters]
        preds = [{} for _ in sets]
        for cell in CELLS:
            scores = [[] for _ in sets]
            for k, seed in enumerate(SEEDS):
                booster = boosters[k] if cell == "A" else train_fixed(CFG, train_sets[cell], rounds[k], seed)
                for s, b in enumerate(sets):
                    scores[s].append(booster.predict(lgb_matrix(b)))
            for s in range(len(sets)):
                preds[s][cell] = np.vstack(scores[s])
        refit_diff = {name: float(np.abs(preds[s]["A_refit"] - preds[s]["A"]).max()) for s, name in enumerate(SETS)}
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
        out = {"cells": {}, "contrasts": {}, "check": {}, "users": len(users), "rows_added": n, "rounds": rounds,
               "refit_max_abs_prediction_diff": refit_diff,
               "rows": {"standard_fit": int(len(std_fit["user"])), "random_fit": int(len(rand_fit["user"])),
                        "train": int(len(p.built["train"]["user"]))}}

        def contrast(a: str, b: str, m: int, s: int) -> dict:
            row = summary(rep[:, idx[a], m, s] - rep[:, idx[b], m, s], point[idx[a], m, s] - point[idx[b], m, s])
            per_seed = seed_means[idx[a], m, s] - seed_means[idx[b], m, s]
            row["seed_support"] = int(np.sum(np.sign(per_seed) == np.sign(row["point"])))
            row["per_seed"] = [float(v) for v in per_seed]
            return row

        for m, metric in enumerate(("uauc", "logloss")):
            for s, name in enumerate(SETS):
                out["cells"].setdefault(metric, {})[name] = {c: float(point[idx[c], m, s]) for c in CELLS}
                block = {label: contrast(a, b, m, s) for label, (a, b) in CONTRASTS.items()}
                flags = stats.holm([v["p"] for v in block.values()])
                for v, f in zip(block.values(), flags):
                    v["holm_reject"] = bool(f)
                out["contrasts"].setdefault(metric, {})[name] = block
                out["check"].setdefault(metric, {})[name] = contrast(*CHECK, m, s)
        out_dir = config.RUNS / "ablation" / run["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = A.write(out, out_dir, "random_mix_fixed")
        run["output"] = str(path)
    print(json.dumps({"users": out["users"], "rounds": rounds, "refit_diff": refit_diff, "cells": out["cells"],
                      "contrasts": {m: {s: {k: (round(v["point"], 4), [round(x, 4) for x in v["ci95"]], v["holm_reject"],
                                                v["seed_support"]) for k, v in blk.items()} for s, blk in ms.items()}
                                    for m, ms in out["contrasts"].items()}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
