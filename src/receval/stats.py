"""User-cluster bootstrap shared by both test sets, Holm correction, Kendall tau and flip rules."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import kendalltau, norm


def holm(pvalues, alpha: float = 0.05) -> np.ndarray:
    """Holm step-down: reject while p_(i) <= alpha / (m - i); flags returned in input order.

    A non-finite p value is treated as 1, so it can never be rejected.
    """
    p = np.nan_to_num(np.asarray(pvalues, float), nan=1.0, posinf=1.0, neginf=1.0)
    reject = np.zeros(len(p), bool)
    for step, i in enumerate(np.argsort(p, kind="stable")):
        if p[i] > alpha / (len(p) - step):
            break
        reject[i] = True
    return reject


def two_sided_p(replicates) -> float:
    """Twice the share of bootstrap replicates on the far side of zero, capped at 1; NaN if any is non-finite."""
    r = np.asarray(replicates, float)
    if not np.all(np.isfinite(r)):
        return float("nan")
    return float(min(1.0, 2 * min(np.mean(r <= 0), np.mean(r >= 0))))


def joint_user_bootstrap(per_user: np.ndarray, samples: int = 10_000, seed: int = 0,
                         chunk: int = 250) -> np.ndarray:
    """Equal-weight means under one user resampling applied to every model and test set at once.

    `per_user` has shape (users, ...): e.g. (users, models, sets) per-user AUCs on the common user
    set. Returns (samples, ...). Resampling counts come from a multinomial, which is the same as
    drawing users with replacement.
    """
    values = np.asarray(per_user, float)
    n = values.shape[0]
    flat = values.reshape(n, -1)
    rng = np.random.default_rng(seed)
    out = np.empty((samples, flat.shape[1]))
    for start in range(0, samples, chunk):
        size = min(chunk, samples - start)
        counts = rng.multinomial(n, np.full(n, 1 / n), size=size)
        out[start:start + size] = counts @ flat / n
    return out.reshape((samples,) + values.shape[1:])


def kendall(a, b) -> float:
    return float(kendalltau(a, b).statistic)


def pairwise_flips(point: np.ndarray, replicates: np.ndarray, alpha: float = 0.05) -> list[dict]:
    """Pairs whose difference is Holm-significant in both sets with opposite signs.

    `point` is (models, 2) and `replicates` (samples, models, 2), column 0 the standard test and
    column 1 the random test. The seed-agreement rule is applied by the caller.
    """
    models = point.shape[0]
    pairs = list(itertools.combinations(range(models), 2))
    rows = []
    for i, j in pairs:
        rows.append({"i": i, "j": j,
                     "delta": [float(point[i, s] - point[j, s]) for s in (0, 1)],
                     "se": [float(np.std(replicates[:, i, s] - replicates[:, j, s], ddof=1)) for s in (0, 1)],
                     "p": [two_sided_p(replicates[:, i, s] - replicates[:, j, s]) for s in (0, 1)]})
    for s in (0, 1):
        reject = holm([row["p"][s] for row in rows], alpha)
        for row, flag in zip(rows, reject):
            row.setdefault("significant", [False, False])[s] = bool(flag)
    holm_first, unadjusted = mde_multipliers(len(pairs), alpha)
    for row in rows:
        row["mde_holm_first_step"] = [float(holm_first * se) for se in row["se"]]
        row["mde_unadjusted"] = [float(unadjusted * se) for se in row["se"]]
        row["flip"] = bool(all(row["significant"]) and np.sign(row["delta"][0]) == -np.sign(row["delta"][1])
                           and row["delta"][0] != 0)
    return rows


def mde_multipliers(comparisons: int, alpha: float = 0.05, power: float = 0.8) -> tuple[float, float]:
    """(Holm first-step, unadjusted) multipliers of the standard error for a two-sided test."""
    z_power = norm.ppf(power)
    return float(norm.ppf(1 - alpha / (2 * comparisons)) + z_power), float(norm.ppf(1 - alpha / 2) + z_power)
