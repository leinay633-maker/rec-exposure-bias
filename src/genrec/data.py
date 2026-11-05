"""Study 2 cache: the KuaiRec matrices split per protocol §2, with the test rows sealed.

Before opening, the labels of sealed rows (small-matrix rows of test users, big-matrix rows dated 9/01-9/05)
never enter a DataFrame: `prepare` first reads only user, video, date and timestamp, finds the sealed line
numbers and skips them when it reads the full rows. `freeze.open_test` writes the sealed files once, and
`load` serves them only when they match the completed opening record.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from receval.data import file_digest

from . import config, runlog
from .text import load_captions

META = ["user_id", "video_id", "date", "timestamp"]
FILES = ("items", "train_events", "train_exposed_pool", "valid_targets", "valid_exposed_pool", "small_pairs", "proxy_dense")
SEALED = ("dense_test", "natural_test")
OPENINGS = "test_openings.jsonl"


def in_window(dates: pd.Series, window: tuple[int, int]) -> pd.Series:
    return dates.between(*window)


def sealed_lines(mask: np.ndarray) -> set[int]:
    return set((np.flatnonzero(mask) + 1).tolist())  # +1: line 0 is the header


def dedup(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, then every (user, video, timestamp) group whose rows still differ (DATA §4)."""
    frame = frame.drop_duplicates()
    return frame[~frame.duplicated(["user_id", "video_id", "timestamp"], keep=False)].reset_index(drop=True)


def read_big(raw: Path, skip_window: tuple[int, int] | None = config.NATURAL_TEST, only_window: bool = False):
    """Full big-matrix rows; rows inside `skip_window` are skipped unparsed (or, with only_window, all others)."""
    meta = pd.read_csv(raw / config.BIG, usecols=META)
    inside = in_window(meta["date"], skip_window).to_numpy()
    skip = sealed_lines(~inside if only_window else inside)
    full = pd.read_csv(raw / config.BIG, skiprows=skip)
    expected = meta[inside if only_window else ~inside].reset_index(drop=True)
    if len(full) != len(expected) or not full[META].reset_index(drop=True).equals(expected):
        raise RuntimeError("big matrix rows do not line up with their metadata")
    return full


def read_small(raw: Path, users: set[int]) -> pd.DataFrame:
    """Full small-matrix rows of the given users only; every other line is skipped unparsed."""
    meta = pd.read_csv(raw / config.SMALL, usecols=["user_id", "video_id"])
    keep = meta["user_id"].isin(users).to_numpy()
    full = pd.read_csv(raw / config.SMALL, skiprows=sealed_lines(~keep))
    if len(full) != int(keep.sum()) or not full[["user_id", "video_id"]].equals(meta[keep].reset_index(drop=True)):
        raise RuntimeError("small matrix rows do not line up with their metadata")
    return full


def label(frame: pd.DataFrame) -> pd.Series:
    return frame["watch_ratio"] > config.POSITIVE_RATIO


def first_positive_events(rows: pd.DataFrame) -> pd.DataFrame:
    pos = rows[label(rows)].sort_values(["user_id", "timestamp", "video_id"], kind="stable")
    return pos.drop_duplicates(["user_id", "video_id"], keep="first")[["user_id", "video_id", "timestamp"]]


def split_users(small_users: list[int], eligible: set[int]) -> tuple[list[int], list[int]]:
    pool = sorted(u for u in small_users if u in eligible)
    rng = np.random.default_rng(config.SPLIT_SEED)
    proxy = set(rng.choice(pool, size=round(config.PROXY_SHARE * len(pool)), replace=False).tolist())
    return sorted(proxy), sorted(u for u in pool if u not in proxy)


def prepare(raw: Path = config.RAW, cache: Path = config.CACHE) -> dict:
    with runlog.logged("prepare") as state:
        cache.mkdir(parents=True, exist_ok=True)
        small_meta = pd.read_csv(raw / config.SMALL, usecols=["user_id", "video_id"])
        small_users = sorted(small_meta["user_id"].unique().tolist())
        pool = sorted(small_meta["video_id"].unique().tolist())
        big = dedup(read_big(raw))
        train = big[in_window(big["date"], config.TRAIN)]
        valid = big[in_window(big["date"], config.VALID)]
        events = first_positive_events(train)
        with_history = set(events["user_id"].unique().tolist())
        proxy, test = split_users(small_users, with_history)
        others = sorted(set(big["user_id"].unique().tolist()) - set(small_users))
        small_set, pool_set = set(small_users), set(pool)

        captions, _ = load_captions()
        if captions["video_id"].tolist() != list(range(len(captions))):
            raise ValueError("video ids are expected to run 0..n-1")
        cats = captions["first_level_category_id"].astype(str)
        cat_index = {c: i for i, c in enumerate(sorted(cats.unique()))}
        exposed = train[["user_id", "video_id"]].drop_duplicates()
        other_rows = train[~train["user_id"].isin(small_set)]
        exposure = other_rows.groupby("video_id").size()
        reach = exposed[~exposed["user_id"].isin(small_set)].groupby("video_id").size()
        pos_users = events.groupby("video_id").size()
        items = pd.DataFrame({"video_id": captions["video_id"], "category": cats.map(cat_index).astype(int)})
        items["pool"] = items["video_id"].isin(pool_set)
        items["w"] = items["video_id"].map(exposure).fillna(0).astype(int)  # exposure rows by other users (§2)
        items["reach"] = items["video_id"].map(reach).fillna(0).astype(int)
        items["positive_users"] = items["video_id"].map(pos_users).fillna(0).astype(int)
        items["group"] = -1
        pool_items = items[items["pool"]].sort_values(["w", "video_id"], kind="stable")
        cold = pool_items["w"] == 0
        items.loc[pool_items.index[cold], "group"] = 0
        warm = pool_items.index[~cold]
        for g, part in enumerate(np.array_split(np.asarray(warm), 4), start=1):
            items.loc[part, "group"] = g

        exposed_pool = exposed[exposed["video_id"].isin(pool_set)].reset_index(drop=True)
        seen = set(zip(exposed_pool["user_id"], exposed_pool["video_id"]))
        vpos = valid[label(valid) & valid["video_id"].isin(pool_set) & valid["user_id"].isin(set(others))]
        vpos = vpos[["user_id", "video_id"]].drop_duplicates()
        vpos = vpos[[(u, v) not in seen for u, v in zip(vpos["user_id"], vpos["video_id"])]]
        vpos = vpos[vpos["user_id"].isin(with_history)].sort_values(["user_id", "video_id"]).reset_index(drop=True)
        valid_users = sorted(vpos["user_id"].unique().tolist())
        rng = np.random.default_rng(config.VALID_SUBSET_SEED)
        subset = valid_users if len(valid_users) <= config.VALID_SUBSET else sorted(
            rng.choice(valid_users, size=config.VALID_SUBSET, replace=False).tolist())

        vexp = valid[valid["video_id"].isin(pool_set) & valid["user_id"].isin(set(others))]
        vexp = vexp[["user_id", "video_id"]].drop_duplicates().sort_values(["user_id", "video_id"]).reset_index(drop=True)
        pairs_small = small_meta[small_meta["user_id"].isin(set(proxy) | set(test))]
        pairs_small = pairs_small.sort_values(["user_id", "video_id"]).reset_index(drop=True)  # no labels (§6.4)
        proxy_rows = read_small(raw, set(proxy))[["user_id", "video_id", "watch_ratio"]]
        frames = {"items": items, "train_events": events.reset_index(drop=True), "train_exposed_pool": exposed_pool,
                  "valid_targets": vpos, "valid_exposed_pool": vexp, "small_pairs": pairs_small,
                  "proxy_dense": proxy_rows.reset_index(drop=True)}
        manifest = {"files": {}, "raw_sha256": {n: file_digest(raw / n) for n in (config.BIG, config.SMALL, config.CAPTION)},
                    "archive_md5_expected": config.ARCHIVE_MD5}
        for name, frame in frames.items():
            path = cache / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            manifest["files"][name] = file_digest(path)
        donors = sorted(vexp["user_id"].unique().tolist())
        users = {"small": small_users, "proxy": proxy, "test": test, "others": others,
                 "valid_eligible": valid_users, "valid_subset": subset, "donors": donors, "pool": pool}
        (cache / "users.json").write_text(json.dumps(users), encoding="utf-8")
        manifest["files"]["users"] = file_digest(cache / "users.json")
        manifest["counts"] = {"train_rows": int(len(train)), "valid_rows": int(len(valid)), "events": int(len(events)),
                              "proxy": len(proxy), "test": len(test), "others": len(others), "pool": len(pool),
                              "valid_eligible": len(valid_users), "valid_subset": len(subset),
                              "valid_targets": int(len(vpos)), "cold_pool_items": int(cold.sum()),
                              "donors": len(donors), "small_pairs": int(len(pairs_small)),
                              "proxy_rows": int(len(proxy_rows))}
        (cache / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        keep = config.RUNS / "prepare" / state["run_id"]  # the cache is the current index (§6 step 0)
        keep.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "users.json"):
            shutil.copy2(cache / name, keep / name)
        state["counts"] = manifest["counts"]
        state["output"] = str(keep)
    return manifest


def manifest(cache: Path = config.CACHE) -> dict:
    return json.loads((cache / "manifest.json").read_text(encoding="utf-8"))


def openings(runs: Path = config.RUNS) -> list[dict]:
    path = runs / OPENINGS
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load(name: str, cache: Path = config.CACHE, runs: Path = config.RUNS):
    """A cached frame (or users.json) after checking its hash; sealed files need a completed opening."""
    if name in SEALED:
        done = [r for r in openings(runs) if r.get("status") == "completed"]
        if not done:
            raise PermissionError(f"{name} is sealed until `python -m genrec open-test` completes")
        path = cache / "sealed" / f"{name}.parquet"
        if file_digest(path) != done[-1]["files"][name]:
            raise PermissionError(f"{name} does not match the recorded opening")
        return pd.read_parquet(path)
    expected = manifest(cache)["files"][name]
    path = cache / ("users.json" if name == "users" else f"{name}.parquet")
    if file_digest(path) != expected:
        raise PermissionError(f"{path.name} does not match the cache manifest")
    return json.loads(path.read_text(encoding="utf-8")) if name == "users" else pd.read_parquet(path)


@dataclass
class Study:
    items: pd.DataFrame
    users: dict
    events: dict  # user -> video indices of positive events in time order
    exposed: dict  # user -> pool videos with any training row
    pool: np.ndarray
    n_items: int


def study(cache: Path = config.CACHE) -> Study:
    items = load("items", cache)
    events = load("train_events", cache)
    seq = {int(u): g["video_id"].to_numpy() for u, g in events.groupby("user_id", sort=True)}
    ex = load("train_exposed_pool", cache)
    exposed = {int(u): g["video_id"].to_numpy() for u, g in ex.groupby("user_id", sort=True)}
    users = load("users", cache)
    return Study(items, users, seq, exposed, np.asarray(users["pool"], dtype=np.int64), len(items))


def histories(s: Study, users: list[int], length: int = config.HISTORY) -> np.ndarray:
    """(users, length) last positive events, left-padded with -1."""
    out = np.full((len(users), length), -1, dtype=np.int64)
    for row, u in enumerate(users):
        h = s.events.get(int(u), np.zeros(0, dtype=np.int64))[-length:]
        if len(h):
            out[row, -len(h):] = h
    return out


def windows(s: Study, length: int = config.HISTORY) -> np.ndarray:
    """Training windows of up to length + 1 events, cut from the end with one event of overlap (§3.1)."""
    rows = []
    for u in sorted(s.events):
        e = s.events[u]
        end = len(e)
        while end >= 2:
            start = max(0, end - (length + 1))
            w = np.full(length + 1, -1, dtype=np.int64)
            w[-(end - start):] = e[start:end]
            rows.append(w)
            end = start + 1
    return np.stack(rows)


def window_users(s: Study, length: int = config.HISTORY) -> np.ndarray:
    """The user of each row of `windows`, in the same order."""
    out = []
    for u in sorted(s.events):
        end = len(s.events[u])
        while end >= 2:
            out.append(u)
            end = max(0, end - (length + 1)) + 1
    return np.asarray(out)


def pairs(win: np.ndarray, users: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-tower rows from the training windows: the same (context, target) pairs SASRec trains on (§3.1).

    Row for window w and position j (input w[j] and target w[j+1] both real): history = w[:j+1], left-padded.
    """
    length = win.shape[1] - 1
    hist, target, user = [], [], []
    for w, u in zip(win, users):
        for j in range(length):
            if w[j] >= 0 and w[j + 1] >= 0:
                h = np.full(length, -1, dtype=np.int64)
                h[length - (j + 1):] = w[:j + 1]
                hist.append(h)
                target.append(w[j + 1])
                user.append(u)
    return np.stack(hist), np.asarray(target), np.asarray(user)


def allowed(s: Study, users: list[int]) -> np.ndarray:
    """(users, pool) bool: pool videos each user may be recommended (training exposures removed)."""
    col = {int(v): j for j, v in enumerate(s.pool)}
    out = np.ones((len(users), len(s.pool)), dtype=bool)
    for row, u in enumerate(users):
        seen = s.exposed.get(int(u))
        if seen is not None and len(seen):
            out[row, [col[int(v)] for v in seen]] = False
    return out


def labelled(s_pairs: pd.DataFrame, users: list[int], pool: np.ndarray) -> np.ndarray:
    """(users, pool) bool: pool videos with a small-matrix row for each user (dense candidates, §2.2)."""
    col = {int(v): j for j, v in enumerate(pool)}
    out = np.zeros((len(users), len(pool)), dtype=bool)
    groups = {int(u): g["video_id"].to_numpy() for u, g in s_pairs.groupby("user_id", sort=True)}
    for row, u in enumerate(users):
        vids = groups.get(int(u))
        if vids is not None:
            out[row, [col[int(v)] for v in vids]] = True
    return out


def targets_from(frame: pd.DataFrame, users: list[int]) -> list[np.ndarray]:
    groups = {int(u): g["video_id"].to_numpy() for u, g in frame.groupby("user_id", sort=True)}
    return [groups.get(int(u), np.zeros(0, dtype=np.int64)) for u in users]


def dense_targets(rows: pd.DataFrame, users: list[int]) -> list[np.ndarray]:
    return targets_from(rows[label(rows)], users)
