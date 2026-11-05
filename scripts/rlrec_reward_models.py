"""Study 3 reward models R4–R6 (docs/rlrec/PROTOCOL.md §5): retrain study 1's §18 cells, save them, then check them.

§18 kept no model files. B-standard, B-random and C are retrained for a fixed number of rounds exactly as
scripts/random_mix_fixed_rounds.py does (same rows, draw, rounds and parameters) and saved first. The checks then use
the saved files: the A refit must reproduce the frozen A's predictions exactly (no labels), and every cell's per-seed
UAUC and log loss on the two opened test sets, averaged over seeds as in §18, must equal random_mix_fixed.json to
1e-12. This is the only read of the test sets before study 3's model freeze: only "equal / not equal" and the largest
differences are written, never the metrics themselves. A mismatch stops the run; nothing is reselected.

Also records which 61,434 random FIT rows B-random added: R2 and R3 are fitted on the same rows.

Usage: python scripts/rlrec_reward_models.py
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np

from random_mix_ablation import CFG, FIT, SAMPLE_SEED, SEEDS, SETS, concat, per_user_logloss, window
from random_mix_fixed_rounds import CELLS, train_fixed
from receval import analysis as A
from receval import config, data, registry
from receval import features as F
from receval import orchestrate as O
from receval.pipeline import prepare
from receval.train import lgb_matrix

RUNS = config.ROOT / ".runs" / "rlrec"
REFERENCE = config.ROOT / "evidence/ablation/20261101-165659-ablation-random-mix-fixed-29339e/random_mix_fixed.json"
TOL = 1e-12
NEW = ("B_standard_fixed", "B_random_fixed", "C_fixed")


def main() -> None:
    manifest = json.loads((config.ROOT / data.ZOO_MANIFEST).read_text(encoding="utf-8"))
    frozen = A.artefact_index(manifest["artefacts"])[("lightgbm", "is_click")]
    for e in frozen:
        if data.file_digest(config.ROOT / e["path"]) != e["sha256"] or e["config"] != CFG:
            raise PermissionError(f"frozen LightGBM {e['path']} does not match the manifest")
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    cfg = {"config": CFG, "fit": list(FIT), "seeds": list(SEEDS), "sample_seed": SAMPLE_SEED,
           "reference": REFERENCE.relative_to(config.ROOT).as_posix(), "tolerance": TOL}
    with registry.logged("rlrec-reward-models", cfg, runs=RUNS) as run:
        started = time.time()
        out_dir = RUNS / "reward_models" / run["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        p = prepare(1.0)
        rand_fit = window(p.calib, p.built["calib"], FIT)
        std_fit = window(p.valid, p.built["valid"], FIT)
        n = min(len(std_fit["user"]), len(rand_fit["user"]))
        rng = np.random.default_rng(SAMPLE_SEED)
        picked = {}

        def pick(name: str, b: dict) -> dict:
            if len(b["user"]) == n:
                picked[name] = np.arange(n)
                return b
            picked[name] = np.sort(rng.choice(len(b["user"]), n, replace=False))
            return O.subset(b, picked[name])

        add_std, add_rand = pick("standard", std_fit), pick("random", rand_fit)  # same draw order as §17 and §18
        train_sets = {"A_refit": p.built["train"], "B_standard_fixed": concat(p.built["train"], add_std),
                      "B_random_fixed": concat(p.built["train"], add_rand), "C_fixed": rand_fit}
        boosters_a = [A.load_ranker(e, p.space, None) for e in frozen]
        rounds = [int(b.num_trees()) for b in boosters_a]
        if rounds != reference["rounds"]:
            raise RuntimeError(f"frozen A has {rounds} trees, §18 recorded {reference['rounds']}")

        # train and save first; every check below uses the files read back from disk
        models = []
        loaded = {"A": boosters_a}
        for cell in ("A_refit",) + NEW:
            loaded[cell] = []
            for k, seed in enumerate(SEEDS):
                booster = train_fixed(CFG, train_sets[cell], rounds[k], seed)
                path = out_dir / f"{cell}_seed{seed}.txt"
                booster.save_model(str(path))
                models.append({"cell": cell, "seed": seed, "trees": int(booster.num_trees()),
                               "path": path.relative_to(config.ROOT).as_posix(), "sha256": data.file_digest(path)})
                loaded[cell].append(lgb.Booster(model_file=str(path)))

        frames = O._test_sets(p)
        sets = [F.build(f, p.space, p.users, p.videos, is_train=False) for f in frames]
        preds = [{} for _ in sets]
        for cell in CELLS:
            for s, b in enumerate(sets):
                preds[s][cell] = np.vstack([booster.predict(lgb_matrix(b)) for booster in loaded[cell]])
        refit = {name: float(np.abs(preds[s]["A_refit"] - preds[s]["A"]).max()) for s, name in enumerate(SETS)}
        users = A.common_users(sets, "is_click")
        auc = A.per_user_auc_cube(users, sets, preds, "is_click", order=CELLS)  # (users, cells, sets, seeds)
        loss = np.empty_like(auc)
        for s, b in enumerate(sets):
            mask = np.isin(b["user"], users)
            sub = O.subset(b, mask)
            for c, cell in enumerate(CELLS):
                for k in range(len(SEEDS)):
                    loss[:, c, s, k] = per_user_logloss(users, sub, preds[s][cell][k][mask])
        point = np.stack([auc, loss], 2).mean(-1).mean(0)  # (cells, metric, sets), as §18
        diffs = {}
        for c, cell in enumerate(CELLS):
            for m, metric in enumerate(("uauc", "logloss")):
                for s, name in enumerate(SETS):
                    diffs[f"{cell}/{metric}/{name}"] = abs(float(point[c, m, s]) - reference["cells"][metric][name][cell])
        worst = max(diffs.values())
        equal = bool(worst <= TOL and max(refit.values()) == 0.0 and len(users) == reference["users"])

        fit_rows = {"standard_positions": picked["standard"].tolist(), "random_positions": picked["random"].tolist(),
                    "random_fit_rows": int(len(rand_fit["user"])), "added": int(n)}
        keys = p.calib[p.calib["date"].between(*FIT)].reset_index(drop=True).iloc[picked["random"]]
        fit_rows["random_keys"] = keys[["user_id", "video_id", "time_ms"]].to_numpy().tolist()
        rows_path = out_dir / "fit_rows.json"
        rows_path.write_text(json.dumps(fit_rows), encoding="utf-8")
        result = {"note": "docs/rlrec/PROTOCOL.md §5; only equality with §18 is reported",
                  "time": time.strftime("%Y-%m-%d %H:%M:%S"), "equal": equal, "tolerance": TOL,
                  "max_abs_metric_difference": worst, "refit_max_abs_prediction_difference": refit,
                  "users_match": len(users) == reference["users"], "rounds": rounds, "models": models,
                  "fit_rows": {"path": rows_path.relative_to(config.ROOT).as_posix(),
                               "sha256": data.file_digest(rows_path), "added": int(n),
                               "random_fit_rows": int(len(rand_fit["user"]))},
                  "seconds": round(time.time() - started, 1)}
        (out_dir / "reward_models.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        run.update({"output": (out_dir / "reward_models.json").relative_to(config.ROOT).as_posix(), "equal": equal})
        if not equal:
            raise RuntimeError(f"R4–R6 retraining does not reproduce §18 (worst metric difference {worst}, refit {refit})")
        (RUNS / "reward_models" / "latest.json").write_text(json.dumps(
            {"run_id": run["run_id"], "dir": out_dir.relative_to(config.ROOT).as_posix(), **result}, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("equal", "max_abs_metric_difference", "refit_max_abs_prediction_difference",
                                             "users_match", "rounds")}, indent=1))


if __name__ == "__main__":
    main()
