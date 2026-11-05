"""Feature building frozen at the end of the training window (protocol §4).

Every vocabulary, statistic and history is fitted on standard-log training rows (4/09-4/21) only.
Training rows get out-of-fold statistics (5 random row folds) and histories that stop strictly before
their own timestamp; evaluation rows use the full training window.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

LABELS = ("is_click", "long_view", "is_like")
# Activity and relationship counts are snapshots of unknown date (possibly after 4/21), so they are
# left out; the remaining profile fields are kept as a declared risk (protocol §4).
DROPPED_USER_FIELDS = ["user_active_degree", "is_lowactive_period", "follow_user_num_range",
                       "fans_user_num_range", "friend_user_num_range"]
USER_FIELDS = ["is_live_streamer", "is_video_author", "register_days_range"] + [f"onehot_feat{i}" for i in range(18)]
VIDEO_FIELDS = ["video_type", "upload_type", "music_type", "tag0", "duration_bucket", "aspect_bucket"]
ID_FIELDS = ["user_id", "video_id", "author_id"]
CONTEXT_FIELDS = ["tab", "hour", "weekday"]
CAT_FIELDS = ID_FIELDS + USER_FIELDS + VIDEO_FIELDS + CONTEXT_FIELDS
STAT_LEVELS = ("video_id", "author_id", "user_id")
STAT_LABELS = ("is_click", "long_view")
PRIOR_STRENGTH = 20.0
FOLDS = 5
HISTORY = 50
DURATION_EDGES = [7_000, 15_000, 30_000, 60_000, 120_000, 300_000]
ASPECT_EDGES = [0.6, 0.9, 1.1, 1.5]


def dedup(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one of fully identical rows; drop every (user, video, time) group that still has several rows.

    After exact duplicates are gone, rows sharing the key differ in some column (feedback or metadata),
    so there is no principled way to pick one of them.
    """
    out = frame.drop_duplicates()
    out = out[~out.duplicated(["user_id", "video_id", "time_ms"], keep=False)]
    return out.reset_index(drop=True)


def _check_training_window(frame: pd.DataFrame) -> None:
    if frame["date"].min() < config.TRAIN[0] or frame["date"].max() > config.TRAIN[1]:
        raise ValueError("feature fitting and training rows must only come from the training window")


def video_table(videos: pd.DataFrame) -> pd.DataFrame:
    table = videos[["video_id", "author_id", "video_type", "upload_type", "music_type"]].copy()
    table["tag0"] = videos["tag"].fillna("").astype(str).str.split(",").str[0]
    duration = videos["video_duration"]
    table["duration_bucket"] = np.where(duration.isna(), -1, np.digitize(duration.fillna(0), DURATION_EDGES))
    ratio = videos["server_width"] / videos["server_height"].replace(0, np.nan)
    table["aspect_bucket"] = np.where(ratio.isna(), -1, np.digitize(ratio.fillna(0), ASPECT_EDGES))
    return table


@dataclass
class FeatureSpace:
    vocab: dict[str, dict] = field(default_factory=dict)
    priors: dict[str, float] = field(default_factory=dict)
    stats: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)
    numeric_mean: np.ndarray | None = None
    numeric_std: np.ndarray | None = None
    histories: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.vocab[f]) + 1 for f in CAT_FIELDS]  # index 0 is "unknown"

    @property
    def numeric_names(self) -> list[str]:
        return [f"{level}_{label}_rate" for level in STAT_LEVELS for label in STAT_LABELS] + \
               [f"{level}_log_count" for level in STAT_LEVELS]


def _joined(frame: pd.DataFrame, users: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    out = frame.merge(video_table(videos), on="video_id", how="left", validate="many_to_one")
    out = out.merge(users[["user_id"] + USER_FIELDS], on="user_id", how="left", validate="many_to_one")
    out["hour"] = (out["hourmin"] // 100).astype(int)
    out["weekday"] = pd.to_datetime(out["date"].astype(str), format="%Y%m%d").dt.weekday
    return out


def _group_stats(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    grouped = frame.groupby(level)
    table = grouped[list(STAT_LABELS)].sum()
    table["n"] = grouped.size()
    return table


def _smoothed(table: pd.DataFrame, keys: pd.Series, priors: dict[str, float]) -> np.ndarray:
    aligned = table.reindex(keys.to_numpy())
    n = aligned["n"].fillna(0).to_numpy()
    cols = []
    for label in STAT_LABELS:
        s = aligned[label].fillna(0).to_numpy()
        cols.append((s + PRIOR_STRENGTH * priors[label]) / (n + PRIOR_STRENGTH))
    cols.append(np.log1p(n))
    return np.column_stack(cols)


def fit(train: pd.DataFrame, users: pd.DataFrame, videos: pd.DataFrame) -> FeatureSpace:
    _check_training_window(train)
    space = FeatureSpace()
    joined = _joined(train, users, videos)
    for f in CAT_FIELDS:
        values = pd.unique(joined[f].astype(str))
        space.vocab[f] = {v: i + 1 for i, v in enumerate(sorted(values))}
    space.priors = {label: float(train[label].mean()) for label in STAT_LABELS}
    for level in STAT_LEVELS:
        space.stats[("full", level)] = _group_stats(joined, level)
    positives = train.loc[train["is_click"] == 1].sort_values(["user_id", "time_ms"], kind="stable")
    for user, part in positives.groupby("user_id"):
        space.histories[int(user)] = part["video_id"].to_numpy()[-HISTORY:]
    return space


def _encode(joined: pd.DataFrame, space: FeatureSpace) -> np.ndarray:
    cols = [joined[f].astype(str).map(space.vocab[f]).fillna(0).astype(np.int64).to_numpy() for f in CAT_FIELDS]
    return np.column_stack(cols)


def _video_index(space: FeatureSpace, ids: np.ndarray) -> np.ndarray:
    vocab = space.vocab["video_id"]
    return np.fromiter((vocab.get(str(v), 0) for v in ids), dtype=np.int64, count=len(ids))


def _train_histories(train: pd.DataFrame, space: FeatureSpace) -> np.ndarray:
    """Positives strictly before each row's own timestamp, last HISTORY of them, left-padded with 0."""
    out = np.zeros((len(train), HISTORY), dtype=np.int64)
    order = train.sort_values(["user_id", "time_ms"], kind="stable")
    for _, part in order.groupby("user_id", sort=False):
        times = part["time_ms"].to_numpy()
        pos_mask = part["is_click"].to_numpy() == 1
        pos_times, pos_items = times[pos_mask], _video_index(space, part["video_id"].to_numpy()[pos_mask])
        counts = np.searchsorted(pos_times, times, side="left")  # positives strictly earlier
        for row, c in zip(part.index.to_numpy(), counts):
            items = pos_items[max(0, c - HISTORY):c]
            if len(items):
                out[row, HISTORY - len(items):] = items
    return out


def _eval_histories(frame: pd.DataFrame, space: FeatureSpace) -> np.ndarray:
    out = np.zeros((len(frame), HISTORY), dtype=np.int64)
    cache: dict[int, np.ndarray] = {}
    for row, user in enumerate(frame["user_id"].to_numpy()):
        if user not in cache:
            cache[user] = _video_index(space, space.histories.get(int(user), np.empty(0, dtype=np.int64)))
        items = cache[user]
        if len(items):
            out[row, HISTORY - len(items):] = items
    return out


def build(frame: pd.DataFrame, space: FeatureSpace, users: pd.DataFrame, videos: pd.DataFrame, *,
          is_train: bool, seed: int = 0) -> dict[str, np.ndarray]:
    frame = frame.reset_index(drop=True)
    if is_train:
        _check_training_window(frame)
    elif frame["date"].min() <= config.TRAIN[1]:
        raise ValueError("evaluation rows must come after the training window")
    joined = _joined(frame, users, videos)
    numeric = np.zeros((len(frame), len(space.numeric_names)), dtype=np.float64)
    if is_train:
        fold = np.random.default_rng(seed).integers(0, FOLDS, len(frame))
        for k in range(FOLDS):
            rest, here = joined[fold != k], fold == k
            priors = {label: float(rest[label].mean()) for label in STAT_LABELS}  # no own-fold labels
            numeric[here] = _stats_block(rest, joined[here], space, priors)
        histories = _train_histories(frame, space)
    else:
        numeric[:] = _stats_block(None, joined, space)
        histories = _eval_histories(frame, space)
    if is_train:
        space.numeric_mean, space.numeric_std = numeric.mean(axis=0), numeric.std(axis=0) + 1e-9
    elif space.numeric_mean is None:
        raise ValueError("build the training rows first: they define the numeric standardisation")
    numeric = (numeric - space.numeric_mean) / space.numeric_std
    labels = np.column_stack([frame[label].to_numpy() if label in frame else np.full(len(frame), -1)
                              for label in LABELS]).astype(np.float32)
    return {"cat": _encode(joined, space), "num": numeric.astype(np.float32), "hist": histories,
            "labels": labels, "user": frame["user_id"].to_numpy(), "video": frame["video_id"].to_numpy(),
            "tab": frame["tab"].to_numpy(), "date": frame["date"].to_numpy()}


def _codes(frame: pd.DataFrame, names: list[str], space: FeatureSpace) -> np.ndarray:
    cat = np.zeros((len(frame), len(CAT_FIELDS)), dtype=np.int64)
    for f in names:
        cat[:, CAT_FIELDS.index(f)] = frame[f].astype(str).map(space.vocab[f]).fillna(0).astype(np.int64).to_numpy()
    return cat


def item_table(space: FeatureSpace, videos: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Every pool video, in ascending video_id order (the retrieval tie-break order, protocol §9.1)."""
    table = video_table(videos).sort_values("video_id", kind="stable").reset_index(drop=True)
    return table["video_id"].to_numpy(), _codes(table, ["video_id", "author_id"] + VIDEO_FIELDS, space)


def user_table(space: FeatureSpace, users: pd.DataFrame, user_ids) -> tuple[np.ndarray, np.ndarray]:
    """Static user fields plus evaluation-time histories (last positives of the training window)."""
    frame = pd.DataFrame({"user_id": np.asarray(user_ids)}).merge(
        users[["user_id"] + USER_FIELDS], on="user_id", how="left", validate="one_to_one")
    return _codes(frame, ["user_id"] + USER_FIELDS, space), _eval_histories(frame, space)


def video_rate(space: FeatureSpace, video_ids, label: str = "is_click") -> np.ndarray:
    """Smoothed training-window rate of each video: the popularity ranker's score."""
    block = _smoothed(space.stats[("full", "video_id")], pd.Series(np.asarray(video_ids)), space.priors)
    return block[:, STAT_LABELS.index(label)]


def _stats_block(source: pd.DataFrame | None, target: pd.DataFrame, space: FeatureSpace,
                 priors: dict[str, float] | None = None) -> np.ndarray:
    rates, counts = [], []
    for level in STAT_LEVELS:
        table = space.stats[("full", level)] if source is None else _group_stats(source, level)
        block = _smoothed(table, target[level], priors or space.priors)
        rates.append(block[:, :len(STAT_LABELS)])
        counts.append(block[:, len(STAT_LABELS):])
    return np.column_stack(rates + counts)
