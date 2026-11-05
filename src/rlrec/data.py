"""Study 3 data (docs/rlrec/PROTOCOL.md §2): eligible users, histories, candidates, SFT windows and populations.

Everything is built from study 1's prepared splits (receval.pipeline.prepare): the deduplicated standard training
window (all tabs), and the tab-1 standard validation and random calibration windows. Videos are pool rows (ascending
video_id, receval.retrieval.Pool); a user's candidates are the pool minus everything he touched in training.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

from genrec import data as gdata
from receval.pipeline import Prepared, prepare
from receval.retrieval import Pool

from . import config


@dataclass
class Base:
    p: Prepared
    pool: Pool
    events: dict[int, np.ndarray]  # user -> positive pool rows of the training window, in time order (all tabs)
    eligible: np.ndarray  # sorted users with at least one training positive (§2)

    @property
    def n_items(self) -> int:
        return len(self.pool.video_ids)

    def histories(self, users) -> np.ndarray:
        """(users, HISTORY) last training positives as pool rows, -1 padded on the left."""
        out = np.full((len(users), config.HISTORY), -1, dtype=np.int64)
        for row, u in enumerate(users):
            h = self.events[int(u)][-config.HISTORY:]
            out[row, config.HISTORY - len(h):] = h
        return out

    def allowed(self, users) -> np.ndarray:
        """(users, n_items) candidates: not touched in the training window (any tab)."""
        out = np.ones((len(users), self.n_items), dtype=bool)
        for row, u in enumerate(users):
            out[row, self.pool.seen.get(int(u), np.empty(0, np.int64))] = False
        return out

    def windows(self) -> np.ndarray:
        """SFT windows of up to HISTORY + 1 events cut as in study 2 (genrec.data.windows), pool rows."""
        return gdata.windows(SimpleNamespace(events=self.events), config.HISTORY)


def load_base(fraction: float = 1.0) -> Base:
    p = prepare(fraction)
    pool = Pool(p.space, p.videos, p.train)
    pos = p.train.loc[p.train["is_click"] == 1].sort_values(["user_id", "time_ms"], kind="stable")  # as features.fit
    events = {int(u): part["video_id"].map(pool.row_of).to_numpy(np.int64) for u, part in pos.groupby("user_id")}
    return Base(p, pool, events, np.array(sorted(events), dtype=np.int64))


def validation_subset(base: Base) -> tuple[np.ndarray, list[np.ndarray]]:
    """§4: 2,000 eligible users with a tab-1 validation positive among their candidates, and those targets."""
    targets = base.pool.targets(base.p.valid)  # tab-1 positives not touched in training
    users = np.array(sorted(u for u in targets if u in base.events), dtype=np.int64)
    pick = np.sort(np.random.default_rng(config.VALID_SUBSET_SEED).choice(users, config.VALID_SUBSET, replace=False))
    return pick, [targets[int(u)] for u in pick]


@dataclass
class Population:
    """Random tab-1 rows of one window whose video is among the user's candidates, for eligible users."""
    users: np.ndarray  # sorted user ids
    pos: np.ndarray  # row -> index into users
    item: np.ndarray  # row -> pool row
    click: np.ndarray
    long_view: np.ndarray
    dropped: dict

    @property
    def rows(self) -> int:
        return len(self.item)


def population(base: Base, frame: pd.DataFrame) -> Population:
    frame = frame[frame["tab"] == 1]
    part = frame.loc[:, ["user_id", "video_id", "is_click", "long_view"]].copy()
    part["item"] = part["video_id"].map(base.pool.row_of)
    if part["item"].isna().any():
        raise ValueError("random rows outside the pool")
    seen = {u: set(v.tolist()) for u, v in base.pool.seen.items()}
    touched = np.array([int(i) in seen.get(int(u), ()) for u, i in zip(part["user_id"], part["item"])])
    eligible = part["user_id"].isin(set(base.events)).to_numpy()
    dropped = {"rows": int(len(part)), "users": int(part["user_id"].nunique()), "touched_in_training": int(touched.sum()),
               "rows_of_ineligible_users": int((~eligible & ~touched).sum()),
               "ineligible_users": int(part.loc[~eligible, "user_id"].nunique())}
    part = part[~touched & eligible]
    users = np.array(sorted(int(u) for u in part["user_id"].unique()), dtype=np.int64)
    at = {u: i for i, u in enumerate(users)}
    return Population(users, part["user_id"].map(at).to_numpy(np.int64), part["item"].to_numpy(np.int64),
                      part["is_click"].to_numpy(float), part["long_view"].to_numpy(float), dropped)


def hold(base: Base) -> Population:
    calib = base.p.calib
    return population(base, calib[calib["date"].between(*config.HOLD)])


def fit_rows(base: Base) -> tuple[pd.DataFrame, dict]:
    """Random FIT rows (4/22–4/25, tab 1) with their built features in the actual context (§5)."""
    calib, built = base.p.calib, base.p.built["calib"]
    mask = calib["date"].between(*config.FIT).to_numpy()
    return calib[mask].reset_index(drop=True), {k: v[mask] for k, v in built.items()}


def reference_hour(base: Base) -> int:
    """Most common hour of the deduplicated standard training rows of tab 1, ties to the smaller hour (§5)."""
    hours = (base.p.train.loc[base.p.train["tab"] == 1, "hourmin"] // 100).astype(int)
    counts = hours.value_counts()
    return int(counts[counts == counts.max()].index.min())


def subset(users: np.ndarray, size: int, seed: int) -> np.ndarray:
    users = np.sort(np.asarray(users))
    return np.sort(np.random.default_rng(seed).choice(users, size, replace=False))

