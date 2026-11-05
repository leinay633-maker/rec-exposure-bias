"""Ranking and probability metrics; per-user AUC is the building block of UAUC and GAUC."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

EPS = 1e-7


def auc(y, score) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    y = np.asarray(y, int)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(score, float))
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def per_user_auc(users, y, score) -> pd.DataFrame:
    """One row per user with both classes: auc, rows, pos, neg (ties take average ranks within user)."""
    frame = pd.DataFrame({"user": np.asarray(users), "y": np.asarray(y, int), "s": np.asarray(score, float)})
    frame["rank"] = frame.groupby("user")["s"].rank(method="average")
    grouped = frame.groupby("user")
    stats = pd.DataFrame({"rows": grouped.size(), "pos": grouped["y"].sum(),
                          "pos_rank_sum": frame["rank"].where(frame["y"] == 1, 0.0).groupby(frame["user"]).sum()})
    stats["neg"] = stats["rows"] - stats["pos"]
    stats = stats[(stats["pos"] > 0) & (stats["neg"] > 0)]
    stats["auc"] = (stats["pos_rank_sum"] - stats["pos"] * (stats["pos"] + 1) / 2) / (stats["pos"] * stats["neg"])
    return stats[["auc", "rows", "pos", "neg"]]


def uauc(per_user: pd.DataFrame, users=None) -> float:
    """Equal-weight mean of per-user AUC, optionally restricted to a fixed user set."""
    table = per_user if users is None else per_user.loc[list(users)]
    return float(table["auc"].mean())


def gauc(per_user: pd.DataFrame) -> float:
    """Conventional GAUC: per-user AUC weighted by the user's rows in this set."""
    return float(np.average(per_user["auc"], weights=per_user["rows"]))


def logloss(y, p) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y, p, bins: int = 15) -> float:
    """Expected calibration error over equal-mass bins of the predicted probability."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            total += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return float(total / len(y))
