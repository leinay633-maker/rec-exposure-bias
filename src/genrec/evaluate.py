"""Per-user metrics of study 2 (protocol §2.2, §4): one top-100 list per user, scored against several target sets."""
from __future__ import annotations

import numpy as np

from . import config

KS = (20, 50, 100)
DISCOUNT = 1 / np.log2(np.arange(2, max(KS) + 2))
IDEAL = np.concatenate([[0.0], np.cumsum(DISCOUNT)])  # IDEAL[m] = DCG of m hits at the top
NAMES = tuple(f"{m}@{k}" for m in ("recall", "ndcg") for k in KS)


def _head(row: np.ndarray) -> np.ndarray:
    """The first max(KS) entries, padded with -1 (no video) when the list is shorter."""
    out = np.full(max(KS), -1, dtype=np.int64)
    head = np.asarray(row[: max(KS)])
    out[: len(head)] = head
    return out


def per_user(top: np.ndarray, targets: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Recall@k and NDCG@k with binary gains; NaN where a user has no target."""
    out = {name: np.full(len(targets), np.nan) for name in NAMES}
    for i, (row, target) in enumerate(zip(top, targets)):
        if len(target) == 0:
            continue
        hits = np.isin(_head(row), target)
        for k in KS:
            out[f"recall@{k}"][i] = hits[:k].sum() / len(target)
            out[f"ndcg@{k}"][i] = (DISCOUNT[:k] * hits[:k]).sum() / IDEAL[min(len(target), k)]
    return out


def weighted(top: np.ndarray, targets: list[np.ndarray], w: np.ndarray) -> dict[str, np.ndarray]:
    """Exposure-weighted metrics (§2.2): each target counts w_i; NaN where the targets' total weight is 0."""
    out = {name: np.full(len(targets), np.nan) for name in NAMES}
    for i, (row, target) in enumerate(zip(top, targets)):
        tw = w[target].astype(float)
        total = tw.sum()
        if len(target) == 0 or total <= 0:
            continue
        head = _head(row)
        gain = np.where(np.isin(head, target), w[np.where(head >= 0, head, 0)], 0.0)
        ideal = np.sort(tw)[::-1]
        for k in KS:
            out[f"recall@{k}"][i] = gain[:k].sum() / total
            out[f"ndcg@{k}"][i] = (gain[:k] * DISCOUNT[:k]).sum() / (ideal[:k] * DISCOUNT[: min(k, len(ideal))]).sum()
    return out


def donor_draws(n_small: int, n_donors: int) -> np.ndarray:
    """(draws, small users) donor positions: draw r uses numpy.random.default_rng([SPLIT_SEED, 100 + r])."""
    return np.stack([np.random.default_rng([config.SPLIT_SEED, 100 + r]).integers(0, n_donors, size=n_small)
                     for r in range(config.BIASED_DRAWS)])


def borrowed(top: np.ndarray, targets: list[np.ndarray], rows: np.ndarray, assign: np.ndarray,
             donor_masks: np.ndarray, col_of: np.ndarray) -> dict[str, np.ndarray]:
    """Binary metrics against targets ∩ a donor's observed pool videos, per draw: (users, draws), NaN if no target.

    rows: each user's column in `assign`; donor_masks: (donors, pool) bool; col_of: video -> pool column.
    """
    draws = assign.shape[0]
    out = {name: np.full((len(targets), draws), np.nan) for name in NAMES}
    for i, (row, target) in enumerate(zip(top, targets)):
        if len(target) == 0:
            continue
        seen = donor_masks[assign[:, rows[i]]]  # (draws, pool)
        counts = seen[:, col_of[target]].sum(1)
        use = counts > 0
        if not use.any():
            continue
        head = _head(row)
        hit = np.isin(head, target)[None, :] & seen[:, col_of[np.where(head >= 0, head, 0)]] & (head >= 0)[None, :]
        for k in KS:
            rec = hit[:, :k].sum(1) / np.maximum(counts, 1)
            dcg = (hit[:, :k] * DISCOUNT[:k]).sum(1) / IDEAL[np.clip(np.minimum(counts, k), 1, None)]
            out[f"recall@{k}"][i, use] = rec[use]
            out[f"ndcg@{k}"][i, use] = dcg[use]
    return out


def borrowed_counts(targets: list[np.ndarray], rows: np.ndarray, assign: np.ndarray, donor_masks: np.ndarray,
                    col_of: np.ndarray) -> np.ndarray:
    """(users, draws) number of targets inside each draw's observed set, |T_u ∩ observed|."""
    out = np.zeros((len(targets), assign.shape[0]), dtype=np.int64)
    for i, target in enumerate(targets):
        if len(target):
            out[i] = donor_masks[assign[:, rows[i]]][:, col_of[target]].sum(1)
    return out


def group_recall(top: np.ndarray, targets: list[np.ndarray], group_of: np.ndarray, groups: int = 5) -> np.ndarray:
    """(users, groups) recall@100 within each heat group; NaN where the user has no target in the group."""
    out = np.full((len(targets), groups), np.nan)
    for i, (row, target) in enumerate(zip(top, targets)):
        if len(target) == 0:
            continue
        hit = np.isin(target, _head(row))
        g = group_of[target]
        for k in range(groups):
            sel = g == k
            if sel.any():
                out[i, k] = hit[sel].mean()
    return out


def slot_share(top: np.ndarray, group_of: np.ndarray, groups: int = 5) -> list[float]:
    head = top[:, :100]
    flat = group_of[head[head >= 0]]
    return [float((flat == k).mean()) for k in range(groups)]
