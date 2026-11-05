"""List values of study 3 (docs/rlrec/PROTOCOL.md §7): "recommend uniformly from the top K", nominal SNIPS.

A policy's top K for a user are its K best candidates by score (ties to the smaller pool row = smaller video_id).
For every random row of a population whose video is in that list, the user's row and click counts go up by one;
V_K = Σ clicks / Σ rows over users. Counts are per user so that seeds can be averaged and users resampled.
"""
from __future__ import annotations

import numpy as np

from .data import Population


def row_ranks(scores: np.ndarray, pop: Population, chunk: int = 1_000) -> np.ndarray:
    """0-based rank of each population row's video in its user's list; scores: (pop users, n_items)."""
    out = np.empty(pop.rows, dtype=np.int64)
    order_rows = np.argsort(pop.pos, kind="stable")
    pos_sorted = pop.pos[order_rows]
    bounds = np.searchsorted(pos_sorted, np.arange(0, len(pop.users) + chunk, chunk))
    for c, start in enumerate(range(0, len(pop.users), chunk)):
        block = scores[start:start + chunk]
        order = np.argsort(-block, axis=1, kind="stable")
        rank = np.empty_like(order)
        rank[np.arange(len(block))[:, None], order] = np.arange(block.shape[1])[None, :]
        rows = order_rows[bounds[c]:bounds[c + 1]]
        out[rows] = rank[pop.pos[rows] - start, pop.item[rows]]
    return out


def counts(ranks: np.ndarray, pop: Population, ks) -> dict:
    """k -> {'rows', 'clicks', 'long_view'}: per-user counts of the rows inside the top k."""
    n = len(pop.users)
    out = {}
    for k in ks:
        inside = (ranks < k).astype(float)
        out[k] = {"rows": np.bincount(pop.pos, weights=inside, minlength=n),
                  "clicks": np.bincount(pop.pos, weights=inside * pop.click, minlength=n),
                  "long_view": np.bincount(pop.pos, weights=inside * pop.long_view, minlength=n)}
    return out


def value(c: dict, label: str = "clicks") -> float:
    rows = c["rows"].sum()
    return float(c[label].sum() / rows) if rows else float("nan")
