"""Calibration methods for protocol §8, fitted on random-exposure calibration rows.

intercept-only: p' = sigmoid(a + logit p), the pure level shift the base-rate gap would predict.
Platt: p' = sigmoid(a + b * logit p) with b >= 0, so within-user order is never reversed.
isotonic: monotone map of p, clipped to [EPS, 1 - EPS] outside the fitted range.
grouped Platt: one Platt per training-popularity quintile, shrunk towards the global Platt with
theta_g' = (n_g * theta_g + SHRINK * theta_global) / (n_g + SHRINK); degenerate groups use global.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

EPS = 1e-6
SHRINK = 2000.0
GROUPS = 5
BUDGETS = (1_000, 4_000, 16_000, 64_000, None)  # None = all calibration rows
REPEATS = 5


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(t) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(t, float)))


def _nll(a: float, b: float, z: np.ndarray, y: np.ndarray) -> float:
    t = a + b * z
    return float(np.mean(np.logaddexp(0, t) - y * t))


def _minimise(objective, x0, bounds=None) -> np.ndarray:
    """L-BFGS-B, then Powell from the same start if it does not report success; else raise."""
    res = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        res = minimize(objective, x0=x0, method="Powell", bounds=bounds)
    if not res.success or not np.all(np.isfinite(res.x)):
        raise RuntimeError(f"calibrator fit failed: {res.message}")
    return res.x


def fit_intercept(p, y) -> tuple[float, float]:
    z, y = logit(p), np.asarray(y, float)
    return float(_minimise(lambda x: _nll(x[0], 1.0, z, y), [0.0])[0]), 1.0


def fit_platt(p, y) -> tuple[float, float]:
    z, y = logit(p), np.asarray(y, float)
    a, b = _minimise(lambda x: _nll(x[0], x[1], z, y), [0.0, 1.0], bounds=[(None, None), (0.0, None)])
    return float(a), float(b)


def apply_affine(params: tuple[float, float], p) -> np.ndarray:
    a, b = params
    return np.clip(sigmoid(a + b * logit(p)), EPS, 1 - EPS)


def fit_isotonic(p, y) -> IsotonicRegression:
    return IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip", increasing=True).fit(
        np.asarray(p, float), np.asarray(y, float))


def popularity_groups(video_ids, train_counts: dict[int, int], groups: int = GROUPS) -> dict[int, int]:
    """Sort pool videos by (training exposures, video_id) and cut into equal-size groups 0..groups-1."""
    ordered = sorted(video_ids, key=lambda v: (train_counts.get(int(v), 0), int(v)))
    mapping = {}
    for g, chunk in enumerate(np.array_split(np.asarray(ordered), groups)):
        for v in chunk:
            mapping[int(v)] = g
    return mapping


def fit_grouped(p, y, group, groups: int = GROUPS) -> dict:
    p, y, group = np.asarray(p, float), np.asarray(y, float), np.asarray(group)
    try:
        overall = fit_platt(p, y)
    except RuntimeError as exc:
        raise RuntimeError(f"grouped Platt, global fit: {exc}") from exc
    params = {}
    for g in range(groups):
        mask = group == g
        n = int(mask.sum())
        if n == 0 or y[mask].min() == y[mask].max():
            params[g] = overall
            continue
        try:
            own = fit_platt(p[mask], y[mask])
        except RuntimeError as exc:
            raise RuntimeError(f"grouped Platt, group {g}: {exc}") from exc
        params[g] = tuple((n * o + SHRINK * w) / (n + SHRINK) for o, w in zip(own, overall))
    return {"global": overall, "groups": params}


def apply_grouped(fitted: dict, p, group) -> np.ndarray:
    p, group = np.asarray(p, float), np.asarray(group)
    out = apply_affine(fitted["global"], p)
    for g, params in fitted["groups"].items():
        mask = group == g
        if mask.any():
            out[mask] = apply_affine(params, p[mask])
    return out


def subsample(n: int, budget: int | None, repeat: int) -> np.ndarray:
    """Nested draws without replacement: budget N takes the first N of one fixed permutation."""
    order = np.random.default_rng(repeat).permutation(n)
    return order if budget is None or budget >= n else order[:budget]
