"""Coverage of the KuaiRand caption and category supplement (Zenodo 18159199) on KuaiRand-Pure's videos (docs/rlrec/DATA.md §2).

The downloaded files are treated as untrusted data: run with `python -I`, paths passed as arguments.

Usage: python -I scripts/kuairand_text_coverage.py <captions.csv> <categories.csv>
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from receval import config, registry

RUNS = config.ROOT / ".runs" / "rlrec"
PURE_VIDEOS = config.ROOT / ".data/raw/KuaiRand-Pure/data/video_features_basic_pure.csv"
EXPECTED_MD5 = {"kuairand_video_captions.csv": "9fd4e7587ffd0c8dc289d013858fb229",
                "kuairand_video_categories.csv": "8a4c772d3d8f8cf65fa1e1eb896f4177"}
CHUNK = 2_000_000
MATCH_BAR = 0.99
USABLE_BAR = 0.95
TAGS = re.compile(r"#[^#\s]*|@\S+")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def pure_rows(path: Path, wanted: set, usecols: list[str], dtype: dict) -> tuple[pd.DataFrame, int]:
    """Rows whose final_video_id is one of Pure's videos; also the total row count of the file."""
    keep, total = [], 0
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=CHUNK, keep_default_na=True):
        total += len(chunk)
        hit = chunk[chunk["final_video_id"].isin(wanted)]
        if len(hit):
            keep.append(hit)
    return (pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=usecols)), total


def share(x) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def main(captions: Path, categories: Path) -> None:
    cfg = {"captions": captions.name, "categories": categories.name, "match_bar": MATCH_BAR, "usable_bar": USABLE_BAR}
    with registry.logged("rlrec-text-coverage", cfg, runs=RUNS) as run:
        started = time.time()
        out_dir = RUNS / "text-coverage" / run["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for p in (captions, categories):
            files[p.name] = {"bytes": p.stat().st_size, "md5": md5(p)}
            if files[p.name]["md5"] != EXPECTED_MD5[p.name]:
                raise RuntimeError(f"{p.name}: md5 {files[p.name]['md5']} differs from the Zenodo record")
        pure = pd.read_csv(PURE_VIDEOS, usecols=["video_id", "video_duration", "tag"])
        ids = set(int(v) for v in pure["video_id"])
        n = len(ids)

        cap, cap_total = pure_rows(captions, ids, ["final_video_id", "caption", "show_cover_text", "duration"],
                                   {"final_video_id": np.int64, "caption": str, "show_cover_text": str})
        cat_cols = ["final_video_id"] + [f"{lvl}_level_category_{f}" for lvl in ("first", "second", "third", "fourth")
                                         for f in ("id", "name", "prob")]
        cat, cat_total = pure_rows(categories, ids, cat_cols, {"final_video_id": np.int64})

        result = {"note": "docs/rlrec/DATA.md §2; KuaiRand supplement on KuaiRand-Pure videos",
                  "time": time.strftime("%Y-%m-%d %H:%M:%S"), "files": files, "pure_videos": n,
                  "file_rows": {"captions": cap_total, "categories": cat_total}}

        # captions
        dup = int(cap["final_video_id"].duplicated().sum())
        cap = cap.drop_duplicates("final_video_id", keep="first")
        merged = pure.merge(cap, left_on="video_id", right_on="final_video_id", how="left")
        present = merged["final_video_id"].notna().to_numpy()
        # some lines are broken (caption text pushed into the duration column): their fields are unreliable
        raw_dur = merged["duration"]
        merged["duration"] = pd.to_numeric(raw_dur, errors="coerce")
        malformed = present & (raw_dur.notna() & merged["duration"].isna()).to_numpy()
        text = merged["caption"].fillna("").astype(str).str.strip()
        stripped = text.str.replace(TAGS, "", regex=True).str.strip()
        lengths = text[present & (text != "")].str.len().to_numpy()
        dur_ok = (merged["duration"] == merged["video_duration"]).to_numpy()
        both_dur = present & merged["duration"].notna().to_numpy() & merged["video_duration"].notna().to_numpy()
        result["captions"] = {
            "videos_present": int(present.sum()), "duplicate_ids": dup, "malformed_rows": int(malformed.sum()),
            "malformed_ids": [int(v) for v in merged.loc[malformed, "video_id"]],
            "empty_caption": int((present & (text == "")).sum()),
            "empty_after_removing_hashtags_and_mentions": int((present & (stripped == "")).sum()),
            "show_cover_text_nonempty": int((present & merged["show_cover_text"].fillna("").str.strip().ne("")).sum()),
            "caption_length": ({"mean": float(lengths.mean()), "p10": float(np.percentile(lengths, 10)),
                                "p50": float(np.percentile(lengths, 50)), "p90": float(np.percentile(lengths, 90)),
                                "max": int(lengths.max())} if len(lengths) else None),
            "duration_compared": int(both_dur.sum()), "duration_equal": int((dur_ok & both_dur).sum()),
            "duration_equal_share": share(dur_ok[both_dur]),
        }
        usable = present & ~malformed & (text != "")

        # categories
        dup_c = int(cat["final_video_id"].duplicated().sum())
        cat = cat.drop_duplicates("final_video_id", keep="first")
        mc = pure.merge(cat, left_on="video_id", right_on="final_video_id", how="left")
        present_c = mc["final_video_id"].notna().to_numpy()
        tags = [None if pd.isna(x) else [int(v) for v in str(x).split(",") if v.strip()] for x in mc["tag"]]
        first = mc["first_level_category_id"].to_numpy()
        both_tag = present_c & np.array([t is not None for t in tags])
        tag_in = np.array([t is not None and not np.isnan(f) and int(f) in t for t, f in zip(tags, first)])
        tag_first = np.array([t is not None and not np.isnan(f) and int(f) == t[0] for t, f in zip(tags, first)])
        levels = {}
        for lvl in ("first", "second", "third", "fourth"):
            name = mc[f"{lvl}_level_category_name"]
            known = present_c & name.notna().to_numpy() & (name != "UNKNOWN").to_numpy()
            levels[lvl] = {"known": int(known.sum()), "distinct": int(mc.loc[known, f"{lvl}_level_category_id"].nunique()),
                           "prob_p50_when_known": float(np.median(mc.loc[known, f"{lvl}_level_category_prob"]))
                           if known.any() else None}
        result["categories"] = {"videos_present": int(present_c.sum()), "duplicate_ids": dup_c,
                                "tag_compared": int(both_tag.sum()), "multi_tag_videos": int(sum(t is not None and len(t) > 1 for t in tags)),
                                "first_level_in_tags_share": share(tag_in[both_tag]),
                                "first_level_equals_first_tag_share": share(tag_first[both_tag]), "levels": levels}

        matched = result["captions"]["duration_equal_share"] >= MATCH_BAR  # categories are supporting evidence only
        result["decision"] = {"ids_match": bool(matched), "usable_captions": int(usable.sum()),
                              "usable_share": float(usable.sum() / n),
                              "use_text": bool(matched and usable.sum() / n >= USABLE_BAR)}
        result["seconds"] = round(time.time() - started, 1)
        (out_dir / "coverage.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        run["output"] = str((out_dir / "coverage.json").relative_to(config.ROOT).as_posix())
        run["use_text"] = result["decision"]["use_text"]
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
