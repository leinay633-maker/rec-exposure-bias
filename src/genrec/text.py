"""Video text for study 2: read kuairec_caption_category.csv, repairing captions split by raw line breaks.

Four records of the published file contain an unquoted line break inside the caption, so they arrive as a
3-field line followed by an 8-field line. They are joined back; every other row must have 10 fields.
"""
from __future__ import annotations

import csv

import pandas as pd

from . import config

FIELDS = 10


def load_captions(path=None) -> tuple[pd.DataFrame, int]:
    path = path or config.RAW / config.CAPTION
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header, body, out, repaired, i = rows[0], rows[1:], [], 0, 0
    while i < len(body):
        row = body[i]
        if len(row) == FIELDS:
            out.append(row)
            i += 1
        elif len(row) == 3 and i + 1 < len(body) and len(body[i + 1]) == FIELDS - 2:
            nxt = body[i + 1]
            out.append(row[:2] + [row[2] + "\n" + nxt[0]] + nxt[1:])
            repaired += 1
            i += 2
        else:
            raise ValueError(f"caption file row {i + 2} has {len(row)} fields and cannot be repaired")
    frame = pd.DataFrame(out, columns=header)
    frame["video_id"] = frame["video_id"].astype(int)
    if frame["video_id"].duplicated().any():
        raise ValueError("caption file lists a video twice")
    return frame.sort_values("video_id").reset_index(drop=True), repaired


def describe(row: pd.Series) -> str:
    """The text fed to the frozen encoder: caption, cover text, topic tags and the category path."""
    parts = []
    caption = row["caption"].strip()
    if caption:
        parts.append(f"标题：{caption}")
    cover = row["manual_cover_text"].strip()
    if cover and cover != "UNKNOWN":
        parts.append(f"封面文字：{cover}")
    tags = row["topic_tag"].strip().strip("[]")
    if tags:
        parts.append(f"话题：{tags}")
    cats = [row[c].strip() for c in ("first_level_category_name", "second_level_category_name", "third_level_category_name")]
    cats = [c for c in cats if c and c != "UNKNOWN"]
    if cats:
        parts.append("类目：" + " / ".join(cats))
    return "；".join(parts) if parts else "无文本"
