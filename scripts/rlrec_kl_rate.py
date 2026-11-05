"""Study 3: how fast the subset KL rises at the chosen learning rate (docs/rlrec/PROTOCOL.md §12.3).

The pilot stops at its first check past 1 nat, which came at step 100, so it has one point: the mean rise of the first
100 steps. Each run here repeats a formal run's seed and checks the KL every step up to 1.1 nat, then every 100 steps
until the formal stop. Only the KL and the time are written: no model is saved, no HOLD or test label is read, and no
policy value is computed. Nothing here goes into the results.

Usage: python scripts/rlrec_kl_rate.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from receval import registry
from rlrec import config
from rlrec import rl as RL
from rlrec import stages as S
from rlrec.data import load_base
from rlrec.policy import Tree

RUNS = (("R1", 0), ("R3", 1), ("R4", 2), ("R5", 0))
RL_SEED, DENSE_UNTIL, DENSE_MAX, SPARSE = 0, 1.1, 400, 100
COARSE_RISE, FINE_RISE = 0.1, 0.025  # §12.2


def dense_then_sparse():
    passed = []

    def schedule(step, kl, ahead, fine_for, check_every, fine_every):
        if kl >= DENSE_UNTIL or step >= DENSE_MAX:
            passed.append(step)
        return step + (SPARSE if passed else 1), None
    return schedule


def largest_window(kl: np.ndarray, limit: float) -> int:
    """The most consecutive steps over which the KL (one value per step, from step 0) never rises more than `limit`."""
    width = 0
    while width + 1 < len(kl) and (kl[width + 1:] - kl[:-(width + 1)]).max() <= limit:
        width += 1
    return width


def main() -> None:
    dev = S.device()
    base = load_base()
    lr = S.latest("rl_pilot")["lr"]
    cfg = {"runs": [list(r) for r in RUNS], "rl_seed": RL_SEED, "lr": lr, "dense_until": DENSE_UNTIL, "dense_max": DENSE_MAX,
           "sparse": SPARSE, "max_steps": RL.MAX_STEPS, "pilot_run": S.latest("rl_pilot")["run_id"],
           "inputs": S.inputs(*S.RL_INPUTS)}
    with registry.logged("rlrec-kl-rate", cfg, runs=config.RUNS) as run:
        out = S._dir("kl_rate", run["run_id"])
        curves = {}
        for reward, seed in RUNS:
            data = S._rl_data(base, reward, seed)
            model, codes = S.load_sft(seed, dev)
            ref, _ = S.load_sft(seed, dev)
            RL.schedule = dense_then_sparse()
            res = RL.train(model, ref, codes, Tree(codes), data, S.rl_lambda(reward, seed), lr, seed=100 * seed + RL_SEED,
                           dev=dev, levels=(2.0,), check_every=1, fine_every=1)
            rows = [{k: row[k] for k in ("step", "subset_kl", "seconds")} for row in res["log"]]
            dense = [row["subset_kl"] for i, row in enumerate(rows, 1) if row["step"] == i]
            kl = np.array([0.0] + dense)
            curves[f"{reward}/sft{seed}"] = {
                "log": rows, "last_step": rows[-1]["step"], "last_kl": rows[-1]["subset_kl"], "seconds": rows[-1]["seconds"],
                "dense_steps": len(dense), "max_step_rise": float(np.diff(kl).max()),
                "coarse": largest_window(kl, COARSE_RISE), "fine": largest_window(kl, FINE_RISE)}
            print(reward, seed, {k: v for k, v in curves[f"{reward}/sft{seed}"].items() if k != "log"}, flush=True)
            del model, ref, data
            torch.cuda.empty_cache()
        S._write({"config": cfg, "curves": curves, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, out / "kl_rate.json")
        run.update({"output": S.rel(out), "coarse": min(c["coarse"] for c in curves.values()),
                    "fine": min(c["fine"] for c in curves.values())})
    print(json.dumps({k: {x: v[x] for x in ("last_step", "last_kl", "dense_steps", "max_step_rise", "coarse", "fine", "seconds")}
                      for k, v in curves.items()}, indent=1))


if __name__ == "__main__":
    main()
