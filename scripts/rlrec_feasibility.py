"""Feasibility audit for study 3 (docs/rlrec/FEASIBILITY.md §1–§2): how precisely random exposure can value a policy.

Stand-in policies are study 1's frozen retrievers, each turned into "recommend uniformly from the top K". Values are
SNIPS estimates on the random calibration hold-out (4/26–4/28, tab 1) under nominal uniform and estimated
propensities; a user bootstrap gives standard errors, extrapolated to the random test's size by rows and by users.
Neither test set is read.

Usage: python scripts/rlrec_feasibility.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from recall_decomposition import NOT_IN_TOP, rank_matrix
from receval import analysis as A
from receval import config, data, registry, stats
from receval import orchestrate as O
from receval.pipeline import prepare
from receval.retrieval import Pool, itemcf_scorer, popularity_scorer, retriever_scorer
from receval.train import device

KS = (10, 50, 100, 200, 500)
K_MAIN = 100
FIT, HOLD = (20220422, 20220425), (20220426, 20220428)
TEST_ROWS, TEST_USERS = 892_896, 26_907  # random test: tab-1 rows and all-tab users (docs/DATA.md §2, §4), metadata only
THRESHOLD = 0.05
SAMPLES = 10_000
RUNS = config.ROOT / ".runs" / "rlrec"
SEEDED = ("recall_two_tower", "recall_two_tower_logq")
NEAR_NULL = [("recall_two_tower/0", "recall_two_tower/1"), ("recall_two_tower_logq/0", "recall_two_tower_logq/1")]
LARGE = [(f"recall_two_tower_logq/{s}", f"recall_two_tower/{s}") for s in range(3)]
SUMS = ("rows", "clicks", "w", "wc", "w2")


def distribution(v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "p01": float(np.percentile(v, 1)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "cv": float(v.std() / v.mean()), "min": float(v.min())}


def interval(reps: np.ndarray, point: float) -> dict:
    return {"point": float(point), "ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
            "se": float(np.std(reps, ddof=1))}


def counts(ranks: np.ndarray, rows: dict, inv_q: np.ndarray, n: int) -> dict:
    """Per-user sums over hold-out rows inside each top K: rows, clicks, w = 1/q, w·click, w² (π = 1/K cancels)."""
    r = ranks[rows["pos"], rows["item"]]
    out = {}
    for k in KS:
        inside = (r < k).astype(float)
        w = inside * inv_q[rows["item"]]
        for name, weights in (("rows", inside), ("clicks", inside * rows["click"]), ("w", w), ("wc", w * rows["click"]),
                              ("w2", w ** 2)):
            out[(name, k)] = np.bincount(rows["pos"], weights=weights, minlength=n)
    return out


def main() -> None:
    cfg = {"ks": list(KS), "fit": list(FIT), "hold": list(HOLD), "test_rows": TEST_ROWS, "test_users": TEST_USERS,
           "threshold": THRESHOLD, "samples": SAMPLES}
    with registry.logged("rlrec-feasibility", cfg, runs=RUNS) as run:
        started = time.time()
        out_dir = RUNS / "feasibility" / run["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        run["output"] = str((out_dir / "feasibility.json").relative_to(config.ROOT).as_posix())
        payload, run["zoo_manifest_sha256"] = O.load_selection(config.ROOT / data.ZOO_MANIFEST)
        O._verify_inputs(payload)
        p = prepare(1.0)
        dev = device()
        pool = Pool(p.space, p.videos, p.train)
        n_pool = len(pool.video_ids)
        calib = p.calib
        fit, hold = calib[calib["date"].between(*FIT)], calib[calib["date"].between(*HOLD)]

        # propensities from the first four days; the hold-out is only scored
        n_fit = np.bincount(fit["video_id"].map(pool.row_of).to_numpy(), minlength=n_pool).astype(float)
        n_hold = np.bincount(hold["video_id"].map(pool.row_of).to_numpy(), minlength=n_pool).astype(float)
        q = (n_fit + 1) / (n_fit.sum() + n_pool)
        inv_q = 1 / q / n_pool  # nominal uniform q = 1 / n_pool gives weight 1, so the two scales are comparable

        part = hold.loc[:, ["user_id", "video_id", "is_click"]].copy()
        part["item"] = part["video_id"].map(pool.row_of)
        seen_sets = {u: set(v.tolist()) for u, v in pool.seen.items()}
        seen = np.array([int(i) in seen_sets.get(int(u), ()) for u, i in zip(part["user_id"], part["item"])])
        trained = part["user_id"].map(lambda u: int(u) in pool.seen).to_numpy()
        steps = {"hold_rows": int(len(part)), "hold_users": int(part["user_id"].nunique()),
                 "dropped_seen_rows": int(seen.sum()), "dropped_rows_of_users_without_training": int((~trained & ~seen).sum())}
        part = part[~seen & trained]
        users = sorted(int(u) for u in part["user_id"].unique())
        at = {u: i for i, u in enumerate(users)}
        n = len(users)
        rows = {"pos": part["user_id"].map(at).to_numpy(np.int64), "item": part["item"].to_numpy(np.int64),
                "click": part["is_click"].to_numpy(float)}
        steps.update({"users": n, "rows": int(len(part))})

        index = A.artefact_index(payload["artefacts"])
        scorers = {"recall_popularity/-": popularity_scorer(pool, p.train, dev),
                   "recall_itemcf/-": itemcf_scorer(pool, p.train, dev)}
        for name in SEEDED:
            for s, entry in enumerate(index[(name, "is_click")]):
                scorers[f"{name}/{s}"] = retriever_scorer(A.load_retriever(entry, p.space, dev), p.space, p.users, pool, dev)
        sums, tops = {}, {}
        for label, fn in scorers.items():
            ranks, top = rank_matrix(fn, users, pool, depth=max(KS))
            if (ranks[rows["pos"], rows["item"]] == NOT_IN_TOP).all():
                raise RuntimeError(f"{label}: no hold-out row inside the top {max(KS)}")
            sums[label] = counts(ranks, rows, inv_q, n)
            if not label.endswith("/-"):
                tops[label] = top
            print(label, "ranked", flush=True)
        data.verify_artefacts(payload["artefacts"])
        for name in SEEDED:  # three seeds' counts averaged per user, as in study 1 §20
            sums[f"{name}/mean"] = {key: np.mean([sums[f"{name}/{s}"][key] for s in range(3)], 0) for key in sums[f"{name}/0"]}

        labels = list(sums)
        cols = [(lab, name, k) for lab in labels for name in SUMS for k in KS]
        mat = np.stack([sums[lab][(name, k)] for lab, name, k in cols], 1)
        point = mat.mean(0)
        reps = stats.joint_user_bootstrap(mat, samples=SAMPLES, seed=0)
        col = {c: i for i, c in enumerate(cols)}

        def get(lab: str, name: str, k: int) -> tuple[float, np.ndarray]:
            i = col[(lab, name, k)]
            return point[i], reps[:, i]

        def value(lab: str, k: int, est: bool) -> tuple[float, np.ndarray]:
            (np_, nr), (dp, dr) = (get(lab, "wc", k), get(lab, "w", k)) if est else (get(lab, "clicks", k), get(lab, "rows", k))
            return np_ / dp, nr / dr

        policies = {}
        for lab in labels:
            policies[lab] = {}
            for k in KS:
                nom, est = value(lab, k, False), value(lab, k, True)
                w_sum, w2_sum = sums[lab][("w", k)].sum(), sums[lab][("w2", k)].sum()
                policies[lab][f"k{k}"] = {"rows_in_top_k": float(sums[lab][("rows", k)].sum()),
                                          "users_with_rows": int((sums[lab][("rows", k)] > 0).sum()),
                                          "nominal": interval(nom[1], nom[0]), "estimated": interval(est[1], est[0]),
                                          "estimated_minus_nominal": interval(est[1] - nom[1], est[0] - nom[0]),
                                          "ess_estimated": float(w_sum ** 2 / w2_sum) if w2_sum > 0 else 0.0}
        mult = stats.mde_multipliers(3)[0]
        scale_rows, scale_users = np.sqrt(steps["rows"] / TEST_ROWS), np.sqrt(n / TEST_USERS)

        def pair_block(a: str, b: str) -> dict:
            out = {}
            for k in KS:
                overlap = float(np.mean([len(np.intersect1d(x[:k], y[:k])) / k for x, y in zip(tops[a], tops[b])]))
                cell = {"top_k_overlap": overlap}
                for est in (False, True):
                    (pa, ra), (pb, rb) = value(a, k, est), value(b, k, est)
                    d = interval(ra - rb, pa - pb)
                    se_rows, se_users = d["se"] * scale_rows, d["se"] * scale_users
                    d.update({"se_test_by_rows": float(se_rows), "se_test_by_users": float(se_users),
                              "mde_test": float(mult * max(se_rows, se_users))})
                    cell["estimated" if est else "nominal"] = d
                out[f"k{k}"] = cell
            return out

        pairs = {"near_null": {f"{a} − {b}": pair_block(a, b) for a, b in NEAR_NULL},
                 "large": {f"{a} − {b}": pair_block(a, b) for a, b in LARGE}}
        base = policies["recall_two_tower_logq/mean"][f"k{K_MAIN}"]["nominal"]["point"]
        worst = max(c[f"k{K_MAIN}"]["nominal"]["mde_test"] for c in pairs["near_null"].values())
        gate = {"base_value": base, "threshold_abs": THRESHOLD * base, "worst_near_null_mde": worst,
                "mde_multiplier": mult, "passes": bool(worst < THRESHOLD * base)}
        result = {"note": "study 3 feasibility, docs/rlrec/FEASIBILITY.md §1–§2; no test data read",
                  "time": time.strftime("%Y-%m-%d %H:%M:%S"), "population": steps,
                  "propensity": {"fit_rows_per_video": distribution(n_fit), "hold_rows_per_video": distribution(n_hold),
                                 "fit_hold_pearson": float(np.corrcoef(n_fit, n_hold)[0, 1]),
                                 "videos_without_fit_rows": int((n_fit == 0).sum())},
                  "policies": policies, "pairs": pairs, "gate": gate,
                  "extrapolation": {"by_rows": float(scale_rows), "by_users": float(scale_users)},
                  "seconds": round(time.time() - started, 1)}
        (out_dir / "feasibility.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        run["gate_passes"] = gate["passes"]
    print(json.dumps({"population": steps, "gate": gate}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
