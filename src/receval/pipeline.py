"""Assemble the deduplicated splits and their features; never touches the sealed test sets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data
from . import features as F


@dataclass
class Prepared:
    space: F.FeatureSpace
    users: pd.DataFrame
    videos: pd.DataFrame
    train: pd.DataFrame
    valid: pd.DataFrame
    calib: pd.DataFrame
    built: dict[str, dict]


def subsample_users(frame: pd.DataFrame, fraction: float, seed: int = 0) -> pd.DataFrame:
    if fraction >= 1:
        return frame
    users = frame["user_id"].unique()
    keep = np.random.default_rng(seed).choice(users, max(1, int(len(users) * fraction)), replace=False)
    return frame[frame["user_id"].isin(keep)].reset_index(drop=True)


def prepare(fraction: float = 1.0) -> Prepared:
    users, videos = data.load("users"), data.load("videos")
    train = subsample_users(F.dedup(data.load("standard_train")), fraction)
    valid = F.dedup(data.load("standard_valid"))
    calib = F.dedup(data.load("random_calib"))
    valid = valid[valid["tab"] == 1].reset_index(drop=True)
    calib = calib[calib["tab"] == 1].reset_index(drop=True)
    if fraction < 1:
        keep = set(train["user_id"])
        valid = valid[valid["user_id"].isin(keep)].reset_index(drop=True)
        calib = calib[calib["user_id"].isin(keep)].reset_index(drop=True)
    space = F.fit(train, users, videos)
    built = {"train": F.build(train, space, users, videos, is_train=True)}
    built["valid"] = F.build(valid, space, users, videos, is_train=False)
    built["calib"] = F.build(calib, space, users, videos, is_train=False)
    return Prepared(space, users, videos, train, valid, calib, built)
