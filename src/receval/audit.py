"""Data audit for KuaiRand-Pure.

Reads only the cache produced by `data.prepare`, so test-period feedback is never touched: for
4/29-5/08 the audit sees users, videos, dates and tabs, but no labels.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import load

LABELS = ("is_click", "long_view", "is_like")
LABELLED = ("standard_train", "standard_valid", "random_calib")
TEST_META = ("standard_test_meta", "random_test_meta")


def _describe(values: pd.Series) -> dict:
    q = values.quantile([0.01, 0.1, 0.5, 0.9, 0.99])
    return {"n": int(len(values)), "mean": float(values.mean()), "min": float(values.min()),
            "p01": float(q[0.01]), "p10": float(q[0.1]), "p50": float(q[0.5]), "p90": float(q[0.9]),
            "p99": float(q[0.99]), "max": float(values.max())}


def tab_table(frame: pd.DataFrame, labels: tuple[str, ...] = ()) -> list[dict]:
    grouped = frame.groupby("tab")
    table = grouped.size().rename("rows").to_frame()
    table["share"] = table["rows"] / table["rows"].sum()
    for label in labels:
        table[f"{label}_rate"] = grouped[label].mean()
    return table.reset_index().to_dict("records")


def eligible_users(frame: pd.DataFrame, label: str) -> set:
    """Users with at least one positive and one negative, i.e. those a per-user AUC can score."""
    bounds = frame.groupby("user_id")[label].agg(["min", "max"])
    return set(bounds.index[(bounds["min"] == 0) & (bounds["max"] == 1)])


def valid_play(frame: pd.DataFrame) -> pd.Series:
    """README's single-column definition of `is_click`: watched >7s, or all of a video of <=7s."""
    short = frame["duration_ms"] <= 7_000
    return ((short & (frame["play_time_ms"] >= frame["duration_ms"]))
            | (~short & (frame["play_time_ms"] > 7_000))).astype(int)


def long_view_rule(frame: pd.DataFrame) -> pd.Series:
    short = frame["duration_ms"] <= 18_000
    return ((short & (frame["play_time_ms"] >= frame["duration_ms"]))
            | (~short & (frame["play_time_ms"] >= 18_000))).astype(int)


def label_definition_check(frame: pd.DataFrame) -> list[dict]:
    """Per tab, how often the logged labels match the README's play-time formulas."""
    checked = frame.assign(valid_play=valid_play(frame), long_rule=long_view_rule(frame))
    rows = []
    for tab, part in checked.groupby("tab"):
        rows.append({"tab": int(tab), "rows": int(len(part)),
                     "is_click_eq_valid_play": float((part["is_click"] == part["valid_play"]).mean()),
                     "long_view_eq_rule": float((part["long_view"] == part["long_rule"]).mean())})
    return rows


def duplicate_detail(frame: pd.DataFrame) -> dict:
    keys = ["user_id", "video_id", "time_ms"]
    grouped = frame[frame.duplicated(keys, keep=False)].groupby(keys)
    differing = grouped[["is_click", "long_view", "play_time_ms"]].nunique().gt(1).any(axis=1)
    return {"groups": int(len(differing)), "groups_with_differing_feedback": int(differing.sum())}


def date_consistency(frame: pd.DataFrame) -> float:
    """Share of rows whose Beijing-time timestamp falls on a different day than `date`."""
    stamp = pd.to_datetime(frame["time_ms"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    return float((stamp.dt.strftime("%Y%m%d").astype(int) != frame["date"]).mean())


def audit(cache: Path = config.CACHE) -> dict:
    parts = {name: load(name, cache) for name in LABELLED + TEST_META}
    users, videos = load("users", cache), load("videos", cache)
    standard = pd.concat([parts["standard_train"], parts["standard_valid"], parts["standard_test_meta"]])
    random_log = pd.concat([parts["random_calib"], parts["random_test_meta"]])
    report: dict = {}

    report["totals"] = {
        "users_any_log": int(pd.concat([standard["user_id"], random_log["user_id"]]).nunique()),
        "standard_interactions": int(len(standard)), "standard_items": int(standard["video_id"].nunique()),
        "random_interactions": int(len(random_log)), "random_items": int(random_log["video_id"].nunique()),
        "user_feature_rows": int(len(users)), "video_feature_rows": int(len(videos)),
        "readme": config.README_TOTALS,
    }
    report["splits"] = {
        name: {"rows": int(len(f)), "users": int(f["user_id"].nunique()), "items": int(f["video_id"].nunique()),
               "dates": [int(f["date"].min()), int(f["date"].max())], "rows_per_date":
               {int(d): int(n) for d, n in f["date"].value_counts().sort_index().items()}}
        for name, f in parts.items()}
    report["is_rand"] = {name: {int(k): int(v) for k, v in f["is_rand"].value_counts().items()}
                         for name, f in parts.items()}
    report["tab"] = {name: tab_table(f, LABELS if name in LABELLED else ()) for name, f in parts.items()}
    report["label_rates"] = {name: {label: float(parts[name][label].mean()) for label in LABELS}
                             for name in LABELLED}

    train = parts["standard_train"]
    train_users, train_items = set(train["user_id"]), set(train["video_id"])
    report["overlap"] = {
        name: {"users_seen_in_train": float(pd.Series(f["user_id"].unique()).isin(train_users).mean()),
               "rows_with_train_user": float(f["user_id"].isin(train_users).mean()),
               "items_seen_in_train": float(pd.Series(f["video_id"].unique()).isin(train_items).mean()),
               "rows_with_train_item": float(f["video_id"].isin(train_items).mean())}
        for name, f in parts.items() if name != "standard_train"}
    report["rows_per_user"] = {name: _describe(parts[name].groupby("user_id").size()) for name in parts}
    # Test labels stay sealed, so the 4/22-4/28 pair (standard valid vs random calib) stands in for
    # how many users both test sets will share once each side needs a positive and a negative.
    report["common_eligible_proxy"] = {}
    for label in LABELS:
        valid = eligible_users(parts["standard_valid"], label)
        calib = eligible_users(parts["random_calib"], label)
        report["common_eligible_proxy"][label] = {"standard_valid": len(valid), "random_calib": len(calib),
                                                  "both": len(valid & calib)}
    both_meta = set(parts["standard_test_meta"]["user_id"]) & set(parts["random_test_meta"]["user_id"])
    report["test_users_in_both_logs"] = int(len(both_meta))
    tab1 = {name: parts[name][parts[name]["tab"] == 1] for name in parts}
    report["common_eligible_proxy_tab1"] = {
        label: {"standard_valid": len(eligible_users(tab1["standard_valid"], label)),
                "random_calib": len(eligible_users(tab1["random_calib"], label)),
                "both": len(eligible_users(tab1["standard_valid"], label)
                            & eligible_users(tab1["random_calib"], label))}
        for label in LABELS}
    report["test_users_in_both_logs_tab1"] = int(len(set(tab1["standard_test_meta"]["user_id"])
                                                     & set(tab1["random_test_meta"]["user_id"])))
    report["label_definition_check"] = {name: label_definition_check(parts[name]) for name in LABELLED}
    report["duplicate_detail"] = {name: duplicate_detail(parts[name]) for name in LABELLED}

    counts = random_log["video_id"].value_counts()
    mean = float(counts.mean())
    report["random_item_frequency"] = {
        **_describe(counts), "pool_items_seen": int(len(counts)),
        "cv": float(counts.std() / mean), "poisson_cv_if_uniform": 1 / math.sqrt(mean)}

    report["duplicates"] = {
        name: {"exact": int(f.duplicated().sum()),
               "user_video_time": int(f.duplicated(["user_id", "video_id", "time_ms"]).sum())}
        for name, f in parts.items()}
    report["date_mismatch_share"] = {name: date_consistency(f) for name, f in parts.items()}

    known = set(videos["video_id"])
    report["side_coverage"] = {
        "log_items_with_video_features": float(np.mean([v in known for v in
                                                        pd.concat([standard, random_log])["video_id"].unique()])),
        "log_users_with_user_features": float(pd.Series(pd.concat([standard, random_log])["user_id"].unique())
                                              .isin(set(users["user_id"])).mean()),
        "video_type": {str(k): int(v) for k, v in videos["video_type"].value_counts(dropna=False).items()},
        "upload_dt_range": [str(videos["upload_dt"].min()), str(videos["upload_dt"].max())],
    }
    return report
