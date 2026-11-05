"""Analyses of protocol §7-§10, written before the zoo freeze and run unchanged after opening.

Each function takes two evaluation sets in a fixed order: index 0 is the standard (policy-exposed)
set and index 1 the random-exposure set. Before the freeze the same code runs on the proxy pair
(standard valid vs random calibration, both 4/22-4/28) for the power check.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import calibrate as C
from . import config, metrics, stats
from .features import LABELS, video_rate
from .models import RANKERS, TwoTowerRetriever
from .train import lgb_matrix, predict, to_tensors

RANKER_ORDER = ("popularity", "logreg", "lightgbm", "two_tower", "dcn", "din", "mmoe")
RECALL_ORDER = ("recall_popularity", "recall_itemcf", "recall_two_tower", "recall_two_tower_logq")
HOLM_FIRST_STEP, UNADJUSTED = stats.mde_multipliers(21)  # ~3.88 and ~2.80 for the 21 ranking pairs


# ---------- loading the zoo: selection.json before the freeze, configs/zoo_manifest.json after ----------

def artefact_index(artefacts: list[dict]) -> dict:
    """(model, label) -> artefact entries in seed order. Paths and configs come from this bound list only."""
    out: dict = {}
    for artefact in artefacts:
        out.setdefault((artefact["model"], artefact["label"]), []).append(artefact)
    return {key: sorted(entries, key=lambda a: int(a["seed"])) for key, entries in out.items()}


def load_ranker(entry: dict, space, dev):
    path = config.ROOT / entry["path"]
    if entry["model"] == "lightgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path))
    model = RANKERS[entry["model"]](space.cardinalities, len(space.numeric_names),
                                    numeric_names=space.numeric_names, dim=entry["config"].get("dim", 16)).to(dev)
    model.load_state_dict(torch.load(path, map_location=dev))
    return model


def load_retriever(entry: dict, space, dev) -> TwoTowerRetriever:
    model = TwoTowerRetriever(space.cardinalities, dim=entry["config"].get("dim", 16),
                              out=entry["config"]["out"]).to(dev)
    model.load_state_dict(torch.load(config.ROOT / entry["path"], map_location=dev))
    return model


def ranker_predictions(artefacts: list[dict], space, built: dict, label: str, dev) -> dict[str, np.ndarray]:
    """Model -> (seeds, rows) probabilities; deterministic models have one row of seeds."""
    index = artefact_index(artefacts)
    tensors = to_tensors(built, dev)
    out = {"popularity": video_rate(space, built["video"], label)[None, :]}
    for name in RANKER_ORDER[1:]:
        source = "is_click" if name == "mmoe" else label
        preds = []
        for entry in index[(name, source)]:
            model = load_ranker(entry, space, dev)
            if name == "lightgbm":
                preds.append(model.predict(lgb_matrix(built), num_iteration=model.best_iteration))
            else:
                preds.append(predict(model, tensors, LABELS.index(label) if name == "mmoe" else None))
        out[name] = np.vstack(preds)
    return out


# ---------- §6-§7: ranking comparison on common users ----------

def common_users(sets: list[dict], label: str) -> list:
    """Users with at least one positive and one negative of `label` in every set."""
    eligible = None
    for built in sets:
        y = built["labels"][:, LABELS.index(label)]
        frame = pd.DataFrame({"user": built["user"], "y": y}).groupby("user")["y"].agg(["min", "max"])
        users = set(frame.index[(frame["min"] == 0) & (frame["max"] == 1)])
        eligible = users if eligible is None else eligible & users
    return sorted(eligible)


def per_user_auc_cube(users: list, sets: list[dict], preds: list[dict[str, np.ndarray]], label: str,
                      order=RANKER_ORDER) -> np.ndarray:
    """(users, models, sets, seeds) per-user AUC; deterministic models repeat their single seed."""
    seeds = max(p.shape[0] for pred in preds for p in pred.values())
    cube = np.empty((len(users), len(order), len(sets), seeds))
    for s, (built, pred) in enumerate(zip(sets, preds)):
        y = built["labels"][:, LABELS.index(label)]
        mask = np.isin(built["user"], users)
        for m, name in enumerate(order):
            for k in range(seeds):
                scores = pred[name][min(k, pred[name].shape[0] - 1)]
                table = metrics.per_user_auc(built["user"][mask], y[mask], scores[mask])
                cube[:, m, s, k] = table.loc[users, "auc"].to_numpy()
    return cube


def _seed_support(cube: np.ndarray, i: int, j: int, signs: tuple[float, float]) -> int:
    """How many seeds reproduce the seed-mean flip direction on both sets."""
    count = 0
    for k in range(cube.shape[3]):
        delta = cube[:, i, :, k].mean(0) - cube[:, j, :, k].mean(0)
        if np.sign(delta[0]) == signs[0] and np.sign(delta[1]) == signs[1]:
            count += 1
    return count


def _did_support(cube: np.ndarray, i: int, j: int, sign: float) -> int:
    """How many seeds reproduce the sign of the seed-mean gap change D = delta_random - delta_standard."""
    count = 0
    for k in range(cube.shape[3]):
        delta = cube[:, i, :, k].mean(0) - cube[:, j, :, k].mean(0)
        if np.sign(delta[1] - delta[0]) == sign:
            count += 1
    return count


def _direction(delta: list[float]) -> str:
    if np.sign(delta[0]) * np.sign(delta[1]) < 0:
        return "reversed"
    if abs(delta[1]) == abs(delta[0]):
        return "unchanged"
    return "widened" if abs(delta[1]) > abs(delta[0]) else "narrowed"


def gap_changes(cube: np.ndarray, point: np.ndarray, reps: np.ndarray, order, seeded, alpha: float = 0.05,
                slope: float = float("nan")) -> dict:
    """Protocol §7.3: per pair D = delta_random - delta_standard, Holm over all pairs, then the seed rule.

    `residual` is delta_random - slope * delta_standard, the part of the pair's change that the common
    rescaling (gap_scaling) does not explain; report only.
    """
    rows = []
    for i, j in itertools.combinations(range(len(order)), 2):
        d_rep = (reps[:, i, 1] - reps[:, j, 1]) - (reps[:, i, 0] - reps[:, j, 0])
        delta = [float(point[i, s] - point[j, s]) for s in (0, 1)]
        d = delta[1] - delta[0]
        rows.append({"pair": [order[i], order[j]], "delta": delta, "did": d, "direction": _direction(delta),
                     "residual": float(delta[1] - slope * delta[0]), "se": float(np.std(d_rep, ddof=1)),
                     "ci": [float(np.percentile(d_rep, 2.5)), float(np.percentile(d_rep, 97.5))],
                     "p": stats.two_sided_p(d_rep), "seed_support": _did_support(cube, i, j, np.sign(d))})
    reject = stats.holm([r["p"] for r in rows], alpha)
    holm_first, unadjusted = stats.mde_multipliers(len(rows), alpha)
    for row, flag in zip(rows, reject):
        needs_seeds = row["pair"][0] in seeded or row["pair"][1] in seeded
        row["significant"] = bool(flag)
        row["change_final"] = bool(flag and row["did"] != 0 and (not needs_seeds or row["seed_support"] >= 2))
        row["mde_holm_first_step"] = holm_first * row["se"]
        row["mde_unadjusted"] = unadjusted * row["se"]
        row["p_report"] = "< 0.0002" if row["p"] == 0 else row["p"]
    return {"pairs": rows, "changes": int(sum(r["change_final"] for r in rows))}


def gap_scaling(point: np.ndarray, reps: np.ndarray) -> dict:
    """Slope (with intercept) and R^2 of random-set scores on standard-set scores across models; report only.

    The slope equals the least-squares slope through the origin of random-set gaps on standard-set gaps
    over all pairs. It is only an average rescaling: pairs can reverse or widen around it, which the
    per-pair residuals and R^2 show. Undefined (NaN) when every model scores the same on the standard set.
    """
    def fit(values):  # (..., models, 2)
        a = values[..., 0] - values[..., 0].mean(-1, keepdims=True)
        b = values[..., 1] - values[..., 1].mean(-1, keepdims=True)
        saa, sbb, sab = (a * a).sum(-1), (b * b).sum(-1), (a * b).sum(-1)
        # Ties are judged on the raw scores: the rounded mean can leave a tiny nonzero spread around equal values.
        spread_a = (np.ptp(values[..., 0], axis=-1) > 0) & (saa > 0)
        spread_b = (np.ptp(values[..., 1], axis=-1) > 0) & (sbb > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(spread_a, sab / saa, np.nan), np.where(spread_a & spread_b, sab ** 2 / (saa * sbb), np.nan)
    slope, r2 = fit(point)
    replicates = fit(reps)[0]
    finite = np.isfinite(replicates)
    ci = [float(np.percentile(replicates[finite], q)) for q in (2.5, 97.5)] if finite.any() else [float("nan")] * 2
    return {"slope": float(slope), "r2": float(r2), "ci": ci, "undefined_replicates": int((~finite).sum())}


def _kendall(point: np.ndarray, reps: np.ndarray) -> dict:
    """tau-b of the two sets' model orders, with a percentile interval; all-tied replicates are NaN and counted."""
    taus = np.asarray([stats.kendall(r[:, 0], r[:, 1]) for r in reps], float)
    finite = np.isfinite(taus)
    ci = [float(np.percentile(taus[finite], q)) for q in (2.5, 97.5)] if finite.any() else [float("nan")] * 2
    return {"kendall_tau_b": stats.kendall(point[:, 0], point[:, 1]), "kendall_ci": ci,
            "kendall_undefined_replicates": int((~finite).sum())}


def ranking_comparison(cube: np.ndarray, order=RANKER_ORDER, samples: int = 10_000, seed: int = 0,
                       seeded=("lightgbm", "two_tower", "dcn", "din", "mmoe")) -> dict:
    if cube.shape[0] == 0:
        return {"models": list(order), "users": 0, "status": "no common users; nothing to compare"}
    per_user = cube.mean(axis=3)
    point = per_user.mean(axis=0)
    reps = stats.joint_user_bootstrap(per_user, samples=samples, seed=seed)
    rows = stats.pairwise_flips(point, reps)
    for row in rows:
        i, j = row["i"], row["j"]
        row["pair"] = [order[i], order[j]]
        signs = (np.sign(row["delta"][0]), np.sign(row["delta"][1]))
        row["seed_support"] = _seed_support(cube, i, j, signs)
        needs_seeds = order[i] in seeded or order[j] in seeded
        row["flip_final"] = bool(row["flip"] and (not needs_seeds or row["seed_support"] >= 2))
        row["p_report"] = ["< 0.0002" if p == 0 else p for p in row["p"]]
    scaling = gap_scaling(point, reps)
    return {"models": list(order), "users": int(cube.shape[0]),
            "uauc": {order[m]: [float(point[m, 0]), float(point[m, 1])] for m in range(len(order))},
            "gap_changes": gap_changes(cube, point, reps, order, seeded, slope=scaling["slope"]),
            "gap_scaling": scaling, "pairs": rows, "flips": int(sum(r["flip_final"] for r in rows)),
            **_kendall(point, reps)}


def own_set_metrics(built: dict, pred: dict[str, np.ndarray], label: str, order=RANKER_ORDER) -> dict:
    """Secondary metrics on each set's own users: row-weighted GAUC, global AUC, LogLoss, Brier, ECE."""
    y = built["labels"][:, LABELS.index(label)]
    out = {}
    for name in order:
        rows = []
        for scores in pred[name]:
            table = metrics.per_user_auc(built["user"], y, scores)
            rows.append([metrics.gauc(table), metrics.uauc(table), metrics.auc(y, scores),
                         metrics.logloss(y, scores), metrics.brier(y, scores), metrics.ece(y, scores)])
        mean = np.mean(rows, axis=0)
        out[name] = dict(zip(["gauc", "uauc_own", "auc", "logloss", "brier", "ece"], map(float, mean)))
    return out


def se_summary(per_user: np.ndarray, order, samples: int = 2_000, seed: int = 0) -> dict:
    """Bootstrap SEs and MDEs of pairwise differences per set and of their D; no point estimates.

    `per_user` is (users, models, 2). Used by the pre-freeze pilot and the zoo power check, so neither
    shows a model-to-model difference.
    """
    if per_user.shape[0] < 2:
        return {"users": int(per_user.shape[0]), "status": "fewer than two users; standard errors cannot be estimated"}
    reps = stats.joint_user_bootstrap(per_user, samples=samples, seed=seed)
    pairs = list(itertools.combinations(range(len(order)), 2))
    holm_first, unadjusted = stats.mde_multipliers(len(pairs))
    se_set = [[float(np.std(reps[:, i, s] - reps[:, j, s], ddof=1)) for i, j in pairs] for s in (0, 1)]
    se_did = [float(np.std((reps[:, i, 1] - reps[:, j, 1]) - (reps[:, i, 0] - reps[:, j, 0]), ddof=1))
              for i, j in pairs]

    def summary(values):
        return {"median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values))}
    return {"users": int(per_user.shape[0]), "pairs": len(pairs), "holm_first_multiplier": holm_first,
            "unadjusted_multiplier": unadjusted,
            "se_standard": summary(se_set[0]), "se_random": summary(se_set[1]), "se_did": summary(se_did),
            "mde_holm_standard": summary([holm_first * x for x in se_set[0]]),
            "mde_holm_random": summary([holm_first * x for x in se_set[1]]),
            "mde_holm_did": summary([holm_first * x for x in se_did]),
            "se_by_pair": {f"{order[i]}|{order[j]}": {"standard": se_set[0][n], "random": se_set[1][n], "did": se_did[n]}
                           for n, (i, j) in enumerate(pairs)}}


def power_check(cube: np.ndarray, order=RANKER_ORDER, samples: int = 2_000, seed: int = 0) -> dict:
    """§7.3 zoo power check on the proxy pair: SEs and MDEs only."""
    return se_summary(cube.mean(axis=3), order, samples, seed)


# ---------- §8: calibration ----------

def calibration_study(calib: dict, test: dict, calib_pred: dict[str, np.ndarray], test_pred: dict[str, np.ndarray],
                      groups_of, label: str = "is_click", order=RANKER_ORDER, boot: int = 1_000) -> dict:
    """Fit each calibrator on nested calibration subsamples, score random-test rows, bootstrap users."""
    y_cal, y_test = calib["labels"][:, LABELS.index(label)], test["labels"][:, LABELS.index(label)]
    g_cal = np.array([groups_of.get(int(v), 0) for v in calib["video"]])
    g_test = np.array([groups_of.get(int(v), 0) for v in test["video"]])
    users = test["user"]
    user_codes, user_index = np.unique(users, return_inverse=True)
    results = {}

    def per_user_loss(p):
        p = np.clip(p, C.EPS, 1 - C.EPS)
        loss = -(y_test * np.log(p) + (1 - y_test) * np.log(1 - p))
        return np.bincount(user_index, loss, len(user_codes)) / np.bincount(user_index, minlength=len(user_codes))

    def fit(method, pc, yc, gc, pt):
        if method == "constant":  # the calibration sample's positive rate for every row (§8.2 baseline)
            return np.full(len(pt), float(np.clip(yc.mean(), C.EPS, 1 - C.EPS)))
        if method == "uncalibrated":
            return pt
        if method == "intercept":
            return C.apply_affine(C.fit_intercept(pc, yc), pt)
        if method == "platt":
            return C.apply_affine(C.fit_platt(pc, yc), pt)
        if method == "isotonic":
            return C.fit_isotonic(pc, yc).predict(pt)
        return C.apply_grouped(C.fit_grouped(pc, yc, gc), pt, g_test)

    methods = ("constant", "uncalibrated", "intercept", "platt", "isotonic", "grouped")
    pairs = {"intercept_minus_platt": ("intercept", "platt"), "platt_minus_grouped": ("platt", "grouped"),
             "uncalibrated_minus_intercept": ("uncalibrated", "intercept")}
    failures = []
    for name in order:
        seeds = test_pred[name].shape[0]
        for budget in C.BUDGETS:
            repeats = 1 if budget is None else C.REPEATS
            losses, brier, ece = ({m: [] for m in methods} for _ in range(3))
            uauc = {"uncalibrated": [], "grouped": []}
            for r in range(repeats):
                rows = C.subsample(len(y_cal), budget, r)
                for k in range(seeds):
                    pc, pt = calib_pred[name][k][rows], test_pred[name][k]
                    yc, gc = y_cal[rows], g_cal[rows]
                    for method in methods:
                        try:
                            p = fit(method, pc, yc, gc, pt)
                        except RuntimeError as exc:  # only this method is undecidable in this cell (§8.4)
                            failures.append({"model": name, "budget": budget, "repeat": r, "seed": k,
                                             "method": method, "error": str(exc)})
                            p = None
                        losses[method].append(per_user_loss(p) if p is not None else np.full(len(user_codes), np.nan))
                        brier[method].append(metrics.brier(y_test, p) if p is not None else np.nan)
                        ece[method].append(metrics.ece(y_test, p) if p is not None else np.nan)
                        if budget is None and method in uauc:  # grouped Platt can reorder rows within a user
                            uauc[method].append(metrics.uauc(metrics.per_user_auc(users, y_test, p))
                                                if p is not None else np.nan)
            # (repeats, users): each user's loss averaged over the seeds of one subsample
            by_repeat = {m: np.asarray(v, float).reshape(repeats, seeds, -1).mean(axis=1) for m, v in losses.items()}
            mean_loss = {m: v.mean(axis=0) for m, v in by_repeat.items()}
            comparisons = {c: mean_loss[a] - mean_loss[b] for c, (a, b) in pairs.items()}
            reps = stats.joint_user_bootstrap(np.column_stack(list(comparisons.values())), samples=boot, seed=0)
            spread = {**{m: v.mean(axis=1) for m, v in by_repeat.items()},
                      **{c: (by_repeat[a] - by_repeat[b]).mean(axis=1) for c, (a, b) in pairs.items()}}
            key = f"{name}@{'all' if budget is None else budget}"
            results[key] = {
                "logloss_user_mean": {m: float(v.mean()) for m, v in mean_loss.items()},
                "spread_across_repeats": {m: [float(np.min(v)), float(np.max(v))] for m, v in spread.items()},
                "brier": {m: float(np.mean(v)) for m, v in brier.items()},
                "ece": {m: float(np.mean(v)) for m, v in ece.items()},
                **({"uauc_random_test": {m: float(np.mean(v)) for m, v in uauc.items()}} if budget is None else {}),
                **{c: {"point": float(v.mean()), "ci": [float(np.percentile(reps[:, n], 2.5)),
                                                        float(np.percentile(reps[:, n], 97.5))]}
                   for n, (c, v) in enumerate(comparisons.items())}}
    # NaN (from a failed fit) never satisfies "> 0", so an undecidable cell counts as not holding.
    h2a = all(results[f"{n}@{b}"]["intercept_minus_platt"]["ci"][0] > 0 for n in order for b in ("16000", "64000", "all"))
    h2b = {n: bool(results[f"{n}@all"]["platt_minus_grouped"]["ci"][0] > 0) for n in order}
    return {"by_model_budget": results, "H2a": bool(h2a), "H2b_per_model": h2b,
            "H2b_overall": all(h2b.values()), "fit_failures": failures}


# ---------- §9: retrieval ----------

def recall_comparison(tables: list[dict[str, list[pd.DataFrame]]], order=RECALL_ORDER, samples: int = 10_000,
                      seed: int = 0, seeded=("recall_two_tower", "recall_two_tower_logq")) -> dict:
    """tables[s][model] is a list of per-seed per-user frames (index user; recall@50/100/200, ndcg@100) for set s.

    R* is the set of users present in every frame of both sets (each set's frames list the users with at
    least one target there). Flips use the §7.3 rules, including the seed-support rule.
    """
    users = sorted(set.intersection(*[set(f.index) for s in (0, 1) for m in order for f in tables[s][m]]))
    if not users:
        return {"models": list(order), "users": 0, "status": "no common users; nothing to compare"}
    seeds = max(len(tables[s][m]) for s in (0, 1) for m in order)
    cube = np.empty((len(users), len(order), 2, seeds))
    for m, name in enumerate(order):
        for s in (0, 1):
            frames = tables[s][name]
            for k in range(seeds):
                cube[:, m, s, k] = frames[min(k, len(frames) - 1)].loc[users, "recall@100"].to_numpy()
    per_user = cube.mean(axis=3)
    point = per_user.mean(0)
    reps = stats.joint_user_bootstrap(per_user, samples=samples, seed=seed)
    rows = stats.pairwise_flips(point, reps)
    for row in rows:
        i, j = row["i"], row["j"]
        row["pair"] = [order[i], order[j]]
        row["seed_support"] = _seed_support(cube, i, j, (np.sign(row["delta"][0]), np.sign(row["delta"][1])))
        needs_seeds = order[i] in seeded or order[j] in seeded
        row["flip_final"] = bool(row["flip"] and (not needs_seeds or row["seed_support"] >= 2))
    did = {}
    h3_first, h3_unadjusted = stats.mde_multipliers(2)  # H3a and H3b form their own family of two
    for key, (i, j) in {"H3a": (0, 3), "H3b": (3, 2)}.items():
        d_rep = (reps[:, i, 1] - reps[:, j, 1]) - (reps[:, i, 0] - reps[:, j, 0])
        se = float(np.std(d_rep, ddof=1))
        did[key] = {"pair": [order[i], order[j]],
                    "point": float((point[i, 1] - point[j, 1]) - (point[i, 0] - point[j, 0])),
                    "p": stats.two_sided_p(d_rep), "se": se, "mde_holm_first_step": h3_first * se,
                    "mde_unadjusted": h3_unadjusted * se,
                    "ci": [float(np.percentile(d_rep, 2.5)), float(np.percentile(d_rep, 97.5))]}
    reject = stats.holm([did["H3a"]["p"], did["H3b"]["p"]])
    did["H3a"]["holds"] = bool(reject[0] and did["H3a"]["point"] < 0)
    did["H3b"]["holds"] = bool(reject[1] and did["H3b"]["point"] > 0)
    # Secondary metrics on R*, each user's value averaged over seeds first; report only (§9.1).
    secondary = {metric: {name: [float(np.mean([f.loc[users, metric].mean() for f in tables[s][name]]))
                                 for s in (0, 1)] for name in order}
                 for metric in ("recall@50", "recall@200", "ndcg@100")}
    scaling = gap_scaling(point, reps)
    return {"models": list(order), "users": len(users),
            "recall@100": {order[m]: [float(point[m, 0]), float(point[m, 1])] for m in range(len(order))},
            "secondary": secondary,
            "gap_changes": gap_changes(cube, point, reps, order, seeded, slope=scaling["slope"]),
            "gap_scaling": scaling, "pairs": rows, "flips": int(sum(r["flip_final"] for r in rows)),
            **_kendall(point, reps), "did": did}


def write(result: dict, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False,
                               default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    return path
