"""Study 2 data audit (docs/kuairec/DATA.md). Reads labels only inside the big matrix's training window.

The small matrix is the sealed dense test: only its user, video and date columns are read here. The big
matrix's validation window is counted but its labels are not summarised either.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import config, data, runlog
from .text import load_captions

META = ["user_id", "video_id", "date", "timestamp"]


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def in_window(frame: pd.DataFrame, window: tuple[int, int]) -> pd.Series:
    return frame["date"].between(*window)


def dedup_report(frame: pd.DataFrame) -> dict:
    exact = int(frame.duplicated().sum())
    rest = frame.drop_duplicates()
    key = rest.duplicated(["user_id", "video_id", "timestamp"], keep=False)
    return {"rows": int(len(frame)), "exact_duplicate_rows": exact,
            "rows_in_conflicting_key_groups_after_exact_dedup": int(key.sum()),
            "rows_after_rule": int(len(rest) - key.sum())}


def quantiles(x: pd.Series) -> dict:
    qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    return {f"p{int(q * 100)}": float(v) for q, v in zip(qs, x.quantile(qs))}


def run() -> dict:
    raw = config.RAW
    out: dict = {"archive_md5": md5(config.ARCHIVE), "archive_md5_expected": config.ARCHIVE_MD5,
                 "caption_md5": md5(raw / config.CAPTION), "caption_md5_expected": config.CAPTION_MD5}
    big_meta = pd.read_csv(raw / config.BIG, usecols=META)
    small_meta = pd.read_csv(raw / config.SMALL, usecols=META)
    for name, frame, totals in (("big", big_meta, config.BIG_TOTALS), ("small", small_meta, config.SMALL_TOTALS)):
        dates = frame["date"].dropna().astype(int)
        out[name] = {"rows": int(len(frame)), "users": int(frame["user_id"].nunique()),
                     "items": int(frame["video_id"].nunique()), "published": totals,
                     "date_min": int(dates.min()), "date_max": int(dates.max()),
                     "dates_present": sorted(int(d) for d in dates.unique()),
                     "rows_missing_date": int(frame["date"].isna().sum()),
                     "rows_per_date": {int(k): int(v) for k, v in dates.value_counts().sort_index().items()}}
    small_users, small_items = set(small_meta["user_id"]), set(small_meta["video_id"])
    out["overlap"] = {
        "small_users_in_big": int(len(small_users & set(big_meta["user_id"]))),
        "small_items_in_big": int(len(small_items & set(big_meta["video_id"]))),
        "big_rows_on_small_user_small_item_pairs": int((big_meta["user_id"].isin(small_users)
                                                         & big_meta["video_id"].isin(small_items)).sum()),
        "big_rows_of_small_users": int(big_meta["user_id"].isin(small_users).sum()),
        "big_rows_on_small_items": int(big_meta["video_id"].isin(small_items).sum()),
        "small_pairs_per_user": quantiles(small_meta.groupby("user_id").size()),
    }
    big = data.read_big(raw)  # the natural-test window stays unparsed (§6.4)
    out["big"]["dedup"] = dedup_report(big)
    big = big.drop_duplicates()
    big = big[~big.duplicated(["user_id", "video_id", "timestamp"], keep=False)]
    train = big[in_window(big, config.TRAIN)]
    valid_rows = big[in_window(big, config.VALID)]
    out["windows"] = {"train": list(config.TRAIN), "valid": list(config.VALID),
                      "train_rows": int(len(train)), "valid_rows": int(len(valid_rows)),
                      "train_users": int(train["user_id"].nunique()), "valid_users": int(valid_rows["user_id"].nunique())}
    wr = train["watch_ratio"]
    out["train_labels"] = {"watch_ratio": quantiles(wr),
                           "positive_rate": {str(t): float((wr > t).mean()) for t in (0.5, 1.0, 2.0)},
                           "positives_gt2": int((wr > config.POSITIVE_RATIO).sum())}
    pos = train[train["watch_ratio"] > config.POSITIVE_RATIO]
    is_small = pos["user_id"].isin(small_users)
    per_user = pos.groupby("user_id").size()
    out["train_positives_per_user"] = {
        "small_users_with_any": int(per_user.index.isin(list(small_users)).sum()),
        "small_users": quantiles(per_user[per_user.index.isin(list(small_users))]),
        "other_users": quantiles(per_user[~per_user.index.isin(list(small_users))]),
        "positive_rows_small_users": int(is_small.sum()), "positive_rows_other_users": int((~is_small).sum()),
        "positives_on_pool_items": int(pos["video_id"].isin(small_items).sum())}
    others = train[~train["user_id"].isin(small_users)]
    pool_counts = others[others["video_id"].isin(small_items)].groupby("video_id").size()
    pool_counts = pool_counts.reindex(sorted(small_items), fill_value=0)
    distinct_pool = others[others["video_id"].isin(small_items)].groupby("user_id")["video_id"].nunique()
    shares = np.sort(pool_counts.to_numpy())[::-1] / pool_counts.sum()
    out["pool_exposure_train_other_users"] = {
        "items_with_zero": int((pool_counts == 0).sum()), "counts": quantiles(pool_counts),
        "top_20pct_items_share": float(shares[: len(shares) // 5].sum()),
        "distinct_pool_items_per_user": quantiles(distinct_pool),
        "share_of_pool_per_user": quantiles(distinct_pool / len(small_items))}
    cap, repaired = load_captions()
    has_caption = cap["caption"].str.strip().ne("")
    has_cover = cap["manual_cover_text"].str.strip().ne("") & cap["manual_cover_text"].ne("UNKNOWN")
    has_cat = cap["first_level_category_name"].str.strip().ne("") & cap["first_level_category_name"].ne("UNKNOWN")
    any_text = has_caption | has_cover | has_cat
    pool = cap["video_id"].isin(small_items)
    out["text"] = {"caption_rows": int(len(cap)), "records_repaired": repaired, "distinct_videos": int(cap["video_id"].nunique()),
                   "with_caption": int(has_caption.sum()), "with_cover_text": int(has_cover.sum()),
                   "with_category": int(has_cat.sum()), "with_any_text": int(any_text.sum()),
                   "pool_videos_with_caption": int((has_caption & pool).sum()),
                   "pool_videos_with_any_text": int((any_text & pool).sum()),
                   "big_items_missing_from_caption_file": int(len(set(big_meta["video_id"]) - set(cap["video_id"])))}
    return out


def main() -> None:
    with runlog.logged("audit") as state:
        out_dir = config.RUNS / "audit" / state["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run()
        (out_dir / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        state["output"] = str(out_dir / "audit.json")
    print(json.dumps({k: v for k, v in result.items() if k not in ("big", "small")}, indent=1, ensure_ascii=False))
    for name in ("big", "small"):
        print(name, {k: v for k, v in result[name].items() if k not in ("rows_per_date", "dates_present")})
        print(name, "dates", result[name]["dates_present"][0], "...", result[name]["dates_present"][-1],
              len(result[name]["dates_present"]), "days")


if __name__ == "__main__":
    main()
