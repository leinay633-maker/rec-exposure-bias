"""Study 3, after the test: π0 sharpened to the same KL, with no reward at all (docs/rlrec/PROTOCOL.md §12.6).

The value of sampling from a policy (§7) rose for every policy while V_500 did not, and that value moves with how
concentrated a policy is as well as with where its mass sits. π_τ ∝ π0^(1/τ) keeps π0's order, so its V_K are π0's and
only the sampled value can move. It is O3 with r = log π0 (π ∝ π0^(1+1/β)), so the same bisection finds it on the KL
subset, without labels. Exploration: nothing here enters H1–H6 or changes a reported number.

Usage: python scripts/rlrec_sharpen_control.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from receval import data as rdata
from receval import features as F
from receval import registry, stats
from receval.data import file_digest
from rlrec import analytic, config
from rlrec import stages as S
from rlrec.data import hold, load_base, population
from rlrec.policy import Tree, exact_logprobs
from rlrec.report import mixture_support

CHUNK = 2_000  # as rlrec.report, so the same batches go through the models
TOLERANCE = 1e-6
O3_REWARDS = ("R1", "R3", "R4", "R5", "R6")
O4_REWARDS = ("R1", "R3", "R4", "R5")
MAIN = 1.0
REPORTS = {"hold": "report_hold/20261105-075700-rlrec-report-hold-2d1dcf/hold.json",
           "test": "report_test/20261105-100342-rlrec-report-test-5bb6ac/test.json"}
CHECKED = ("pi0", f"O3/R1/{MAIN}", f"O4/R1/{MAIN}/main")


def temperatures() -> dict:
    out = {}
    for s in config.SEEDS:
        logp0 = S.load_o3_array(f"pi0_kl_seed{s}").astype(np.float64)
        for level in config.KL_LEVELS:
            beta, kl = analytic.bisect_beta(logp0, logp0, level)
            out[f"{s}/{level}"] = {"beta": beta, "kl": kl, "tau": beta / (1 + beta)}
    return out


class Population:
    """One population's rows, π0 and reward tables, and the sampled-value sums of any policy on it."""

    def __init__(self, pop, base, dev: torch.device):
        self.pop, self.dev = pop, dev
        self.n = n = len(pop.users)
        self.allowed, self.hist = base.allowed(pop.users), base.histories(pop.users)
        self.chunks = [(a, min(a + CHUNK, n)) for a in range(0, n, CHUNK)]
        by_user = np.argsort(pop.pos, kind="stable")
        bounds = np.searchsorted(pop.pos[by_user], [a for a, _ in self.chunks] + [n])
        self.row_slices = [by_user[bounds[i]:bounds[i + 1]] for i in range(len(self.chunks))]
        self.tables = {}
        for r in O3_REWARDS:
            all_users, table = S.load_table(r)
            self.tables[r] = (table, S.rows_of(all_users, pop.users))
        self.logp0, self.trees = {}, {}
        for s in config.SEEDS:
            model, codes = S.load_sft(s, dev)
            self.trees[s] = Tree(codes)
            self.logp0[s] = exact_logprobs(model, codes, self.trees[s], self.hist, self.allowed, dev)
            del model
            torch.cuda.empty_cache()

    def reward(self, r: str, a: int, b: int) -> np.ndarray:
        table, idx = self.tables[r]
        return np.asarray(table[idx[a:b]], dtype=np.float64)

    def base_logp(self, s: int, a: int, b: int) -> np.ndarray:
        return self.logp0[s][a:b].astype(np.float64)

    def sums(self, logp_of, models: list, spread: bool = False, base_of=None) -> dict:
        """Mixture sums over `models` (sft seed, tag); logp_of(model, a, b) gives log π for users a:b."""
        pop, n = self.pop, self.n
        per_model = []
        prob_sum = np.zeros((n, self.allowed.shape[1]), dtype=np.float32) if spread else None
        kl_user = np.zeros(n)
        for m in models:
            w, wc, w_rows = np.zeros(n), np.zeros(n), np.zeros(pop.rows)
            handle = logp_of.open(m) if hasattr(logp_of, "open") else m
            for i, (a, b) in enumerate(self.chunks):
                logp = (logp_of.at if hasattr(logp_of, "at") else logp_of)(handle, a, b)
                rows = self.row_slices[i]
                pos = pop.pos[rows]
                v = logp.shape[1] * np.exp(logp[pos - a, pop.item[rows]])
                w_rows[rows] = v
                w += np.bincount(pos, weights=v, minlength=n)
                wc += np.bincount(pos, weights=v * pop.click[rows], minlength=n)
                if spread:
                    prob_sum[a:b] += np.exp(logp)
                    kl_user[a:b] += analytic.kl(logp, self.base_logp(m[0], a, b))
            if hasattr(logp_of, "close"):
                logp_of.close(handle)
            per_model.append((m[0], w, wc, w_rows))
        w_mix = np.mean([x[3] for x in per_model], 0)
        out = {"w": np.mean([x[1] for x in per_model], 0), "wc": np.mean([x[2] for x in per_model], 0),
               "ess": float(w_mix.sum() ** 2 / (w_mix ** 2).sum()),
               "seed_values": {str(s): float(sum(x[2].sum() for x in per_model if x[0] == s) / sum(x[1].sum() for x in per_model if x[0] == s))
                               for s in sorted({x[0] for x in per_model})}}
        if spread:
            out["support_median"] = float(np.median(mixture_support(prob_sum, len(models))))
            out["kl_mean"] = float((kl_user / len(models)).mean())
        return out


class RLModels:
    """log π of RLOO checkpoints, one model in memory at a time."""

    def __init__(self, p: Population, reward: str, level: float):
        self.p, self.reward, self.level = p, reward, level

    def open(self, m: tuple):
        s, k = m
        return S.load_rl(S.rl_name(self.reward, s, k, False), self.level, "main", self.p.dev)[:2] + (s,)

    def at(self, handle: tuple, a: int, b: int) -> np.ndarray:
        model, codes, s = handle
        p = self.p
        return exact_logprobs(model, codes, p.trees[s], p.hist[a:b], p.allowed[a:b], p.dev).astype(np.float64)

    def close(self, handle: tuple) -> None:
        del handle
        torch.cuda.empty_cache()


def policy_sums(p: Population, name: str, temps: dict, betas: dict) -> dict:
    sft = [(s, None) for s in config.SEEDS]
    if name == "pi0":
        return p.sums(lambda m, a, b: p.base_logp(m[0], a, b), sft, spread=True)
    if name.startswith("T/"):
        level = name.split("/")[1]

        def sharpened(m, a, b):
            lp0 = p.base_logp(m[0], a, b)
            return analytic.tilt(lp0, lp0, temps[f"{m[0]}/{level}"]["beta"])
        return p.sums(sharpened, sft, spread=True)
    if name.startswith("O3/"):
        _, reward, level = name.split("/")
        return p.sums(lambda m, a, b: analytic.tilt(p.base_logp(m[0], a, b), p.reward(reward, a, b), betas[f"{reward}/{m[0]}/{level}"]["beta"]), sft)
    _, reward, level, _ = name.split("/")
    return p.sums(RLModels(p, reward, float(level)), [(s, k) for s in config.SEEDS for k in config.SEEDS])


def check(res: dict, report: dict, which: str) -> None:
    values = report["statistics"]["values"]
    for name in CHECKED:
        expected = values[name]["sampled"]
        got = res[name]["wc"].sum() / res[name]["w"].sum()
        gaps = [abs(got - expected["point"]), abs(res[name]["ess"] / expected["ess"] - 1)]
        print(f"{which} {name}: sampled {got:.6f} (report {expected['point']:.6f}), ess {res[name]['ess']:.1f} (report {expected['ess']:.1f})", flush=True)
        if max(gaps) > TOLERANCE:
            raise SystemExit(f"{which} {name} does not reproduce the report: {gaps}")


def summary(rep: np.ndarray, point: float) -> dict:
    lo, hi = np.percentile(rep, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)], "se": float(np.std(rep, ddof=1))}


def statistics(res: dict) -> dict:
    names = list(res)
    values = np.stack([x for name in names for x in (res[name]["w"], res[name]["wc"])], 1)
    rep = stats.joint_user_bootstrap(values, samples=10_000, seed=0)
    point = values.mean(0)
    pos = {name: 2 * i for i, name in enumerate(names)}

    def value(name: str) -> tuple:
        i = pos[name]
        return rep[:, i + 1] / rep[:, i], point[i + 1] / point[i]

    out = {"values": {}, "differences": {}}
    for name in names:
        extra = {k: res[name][k] for k in ("ess", "seed_values", "support_median", "kl_mean") if k in res[name]}
        out["values"][name] = {**summary(*value(name)), **extra}

    def diff(a: str, b: str) -> None:
        (ra, pa), (rb, pb) = value(a), value(b)
        out["differences"][f"{a} − {b}"] = summary(ra - rb, pa - pb)

    for level in config.KL_LEVELS:
        diff(f"T/{level}", "pi0")
        for r in O3_REWARDS:
            diff(f"O3/{r}/{level}", f"T/{level}")
    for r in O4_REWARDS:
        diff(f"O4/{r}/{MAIN}/main", f"T/{MAIN}")
    return out


def run_population(which: str, pop, base, dev, temps: dict, betas: dict) -> dict:
    report = json.loads((config.RUNS / REPORTS[which]).read_text(encoding="utf-8"))
    p = Population(pop, base, dev)
    res = {name: policy_sums(p, name, temps, betas) for name in CHECKED}
    check(res, report, which)
    rest = [f"T/{lv}" for lv in config.KL_LEVELS] + [f"O3/{r}/{lv}" for r in O3_REWARDS for lv in config.KL_LEVELS]
    rest += [f"O4/{r}/{MAIN}/main" for r in O4_REWARDS]
    for name in rest:
        if name not in res:
            t0 = time.time()
            res[name] = policy_sums(p, name, temps, betas)
            print(f"{which} {name}: {res[name]['wc'].sum() / res[name]['w'].sum():.4f}, {time.time() - t0:.0f} s", flush=True)
    return {"users": int(p.n), "rows": int(pop.rows), **statistics(res)}


def main() -> None:
    S._check_freeze()
    dev = S.device()
    base = load_base()
    temps = temperatures()
    print(json.dumps(temps), flush=True)
    betas = S.latest("o3")["betas"]
    frame = F.dedup(rdata.load("random_test"))
    cfg = {"protocol": "docs/rlrec/PROTOCOL.md §12.6", "manifest_sha256": file_digest(S.MANIFEST), "levels": list(config.KL_LEVELS)}
    with registry.logged("rlrec-posthoc-sharpen", cfg, runs=config.RUNS) as run:
        t0 = time.time()
        out = S._dir("posthoc", run["run_id"])
        result = {"note": "docs/rlrec/PROTOCOL.md §12.6, exploration after the test", "temperatures": temps,
                  "hold": run_population("hold", hold(base), base, dev, temps, betas),
                  "test": run_population("test", population(base, frame[frame["date"].between(*config.TEST)]), base, dev, temps, betas)}
        result.update({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1)})
        path = S._write(result, out / "sharpen.json")
        run.update({"output": S.rel(path), "sha256": file_digest(path)})
    print(S.rel(path))


if __name__ == "__main__":
    main()
