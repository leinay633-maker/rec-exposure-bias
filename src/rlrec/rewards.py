"""Reward tables of study 3 (docs/rlrec/PROTOCOL.md §5).

LightGBM's input has no user-video interaction: user columns, video columns and context columns only. The table for
(user, video) pairs in the fixed reference context (tab 1, 2022-04-22, the modal training hour) is therefore built by
broadcasting per-user and per-video blocks, with the same encodings and standardisation as receval.features.build
(tested against it). R2 and R3 are Platt maps of R1 fitted on the random FIT rows in their actual context.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from receval import calibrate as C
from receval import features as F
from receval.features import CAT_FIELDS

CONTEXT = ("tab", "hour", "weekday")


class Broadcast:
    """Encoded per-user and per-video columns; `matrix(users)` gives LightGBM rows for users x all pool videos."""

    def __init__(self, space: F.FeatureSpace, users: pd.DataFrame, videos: pd.DataFrame, date: int, hour: int,
                 tab: int = 1):
        self.space = space
        self.video_ids, self.video_cat = F.item_table(space, videos)
        table = F.video_table(videos).set_index("video_id").loc[self.video_ids]
        self.video_num = {lvl: F._smoothed(space.stats[("full", lvl)], pd.Series(keys), space.priors)
                          for lvl, keys in (("video_id", self.video_ids), ("author_id", table["author_id"].to_numpy()))}
        weekday = int(pd.to_datetime(str(date), format="%Y%m%d").weekday())
        ctx = {"tab": tab, "hour": hour, "weekday": weekday}
        self.context = np.zeros(len(CAT_FIELDS), dtype=np.int64)
        for f in CONTEXT:
            self.context[CAT_FIELDS.index(f)] = space.vocab[f].get(str(ctx[f]), 0)
        self.users = users
        if space.numeric_mean is None:
            raise ValueError("the feature space has no numeric standardisation yet")

    def matrix(self, user_ids) -> np.ndarray:
        """(len(user_ids) * n_videos, columns) float64, user-major, videos in pool order."""
        user_ids = np.asarray(user_ids)
        user_cat, _ = F.user_table(self.space, self.users, user_ids)
        user_stats = F._smoothed(self.space.stats[("full", "user_id")], pd.Series(user_ids), self.space.priors)
        n_u, n_v = len(user_ids), len(self.video_ids)
        cat = user_cat[:, None, :] + self.video_cat[None, :, :] + self.context[None, None, :]
        rates = [self.video_num["video_id"][None, :, :2].repeat(n_u, 0), self.video_num["author_id"][None, :, :2].repeat(n_u, 0),
                 user_stats[:, None, :2].repeat(n_v, 1)]
        counts = [self.video_num["video_id"][None, :, 2:].repeat(n_u, 0), self.video_num["author_id"][None, :, 2:].repeat(n_u, 0),
                  user_stats[:, None, 2:].repeat(n_v, 1)]
        num = np.concatenate(rates + counts, axis=2)
        num = ((num - self.space.numeric_mean) / self.space.numeric_std).astype(np.float32)
        return np.concatenate([cat.reshape(n_u * n_v, -1).astype(np.float64), num.reshape(n_u * n_v, -1).astype(np.float64)], 1)


def predict_table(boosters: list, bc: Broadcast, user_ids, chunk: int = 256) -> np.ndarray:
    """(users, n_videos) float32: mean of the boosters' click probabilities in the reference context."""
    out = np.empty((len(user_ids), len(bc.video_ids)), dtype=np.float32)
    for s in range(0, len(user_ids), chunk):
        ids = user_ids[s:s + chunk]
        x = bc.matrix(ids)
        out[s:s + len(ids)] = np.mean([b.predict(x) for b in boosters], 0).reshape(len(ids), -1)
    return out


def popularity_group_of_items(train: pd.DataFrame, video_ids: np.ndarray) -> np.ndarray:
    """5 equal-size groups by deduplicated training exposures, all tabs (study 1 §8.2), per pool row."""
    mapping = C.popularity_groups(list(video_ids), train["video_id"].value_counts().to_dict())
    return np.array([mapping[int(v)] for v in video_ids], dtype=np.int64)


def fit_calibrations(p_fit: np.ndarray, y_fit: np.ndarray, group_fit: np.ndarray) -> dict:
    return {"platt": C.fit_platt(p_fit, y_fit), "grouped": C.fit_grouped(p_fit, y_fit, group_fit)}


def calibrated(kind: str, r1: np.ndarray, fitted: dict, item_group: np.ndarray) -> np.ndarray:
    """R2 ('platt') or R3 ('grouped') from an R1 table (users, n_videos)."""
    if kind == "platt":
        return C.apply_affine(fitted["platt"], r1).astype(np.float32)
    if kind == "grouped":
        out = np.empty_like(r1)
        for g, params in fitted["grouped"]["groups"].items():
            cols = item_group == g
            out[:, cols] = C.apply_affine(params, r1[:, cols])
        return out
    raise ValueError(kind)

