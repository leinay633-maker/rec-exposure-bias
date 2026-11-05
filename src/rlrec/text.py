"""Video text, embeddings and semantic IDs of study 3 (docs/rlrec/PROTOCOL.md §3).

The KuaiRand supplement (Zenodo 18159199) is read in chunks and only Pure's videos are kept. Rows whose `duration`
is not numeric are malformed (caption text pushed into later columns, docs/rlrec/DATA.md §1): their caption and
cover text are dropped. Texts follow genrec.text.describe; embeddings use study 2's pinned Qwen3 encoder and
settings; the RQ-VAE, the retry rule and the fourth code are study 2's (genrec.sids, genrec.semantic_id).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from genrec import encode as genc
from genrec.semantic_id import collision_stats, disambiguate, train_rqvae
from genrec.sids import MIN_CODES, RQ
from receval.data import file_digest

from . import config

CHUNK = 2_000_000
LEVELS = ("first", "second", "third", "fourth")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _rows(path: Path, wanted: set, usecols: list[str], dtype: dict) -> pd.DataFrame:
    keep = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=CHUNK):
        hit = chunk[chunk["final_video_id"].isin(wanted)]
        if len(hit):
            keep.append(hit)
    out = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=usecols)
    return out.drop_duplicates("final_video_id", keep="first")


def read_supplement(video_ids: np.ndarray, captions: Path = config.CAPTIONS,
                    categories: Path = config.CATEGORIES, check_md5: bool = True) -> pd.DataFrame:
    """One row per video (in the given order): caption, cover text, malformed flag and the four category names."""
    if check_md5:
        for p in (captions, categories):
            if md5(p) != config.TEXT_MD5[p.name]:
                raise PermissionError(f"{p.name} differs from the Zenodo record")
    wanted = set(int(v) for v in video_ids)
    cap = _rows(captions, wanted, ["final_video_id", "caption", "show_cover_text", "duration"],
                {"final_video_id": np.int64, "caption": str, "show_cover_text": str, "duration": str})
    names = [f"{lvl}_level_category_name" for lvl in LEVELS]
    cat = _rows(categories, wanted, ["final_video_id"] + names, {"final_video_id": np.int64})
    frame = pd.DataFrame({"video_id": np.asarray(video_ids, dtype=np.int64)})
    frame = frame.merge(cap, left_on="video_id", right_on="final_video_id", how="left").drop(columns="final_video_id")
    frame = frame.merge(cat, left_on="video_id", right_on="final_video_id", how="left").drop(columns="final_video_id")
    raw = frame["duration"]
    frame["malformed"] = raw.notna() & pd.to_numeric(raw, errors="coerce").isna()
    return frame


def _clean(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def describe(row) -> str:
    """标题：…；封面文字：…；类目：a / b / c / d — "无文本" when nothing is usable (§3)."""
    parts = []
    if not bool(row["malformed"]):
        caption, cover = _clean(row["caption"]), _clean(row["show_cover_text"])
        if caption:
            parts.append(f"标题：{caption}")
        if cover and cover != "UNKNOWN":
            parts.append(f"封面文字：{cover}")
    cats = [c for c in (_clean(row[f"{lvl}_level_category_name"]) for lvl in LEVELS) if c and c != "UNKNOWN"]
    if cats:
        parts.append("类目：" + " / ".join(cats))
    return "；".join(parts) if parts else "无文本"


@torch.no_grad()
def embed(texts: list[str], batch: int = 64) -> tuple[np.ndarray, int]:
    """Study 2's encoder and settings (pinned revision, fp16 on CUDA, last token, left padding, L2 norm)."""
    import json

    from transformers import AutoModel, AutoTokenizer
    info = json.loads(genc.REVISION_FILE.read_text(encoding="utf-8"))
    if info["revision"] != genc.REVISION:
        raise PermissionError(f"encoder revision {info['revision']} is not the pinned {genc.REVISION}")
    for name, digest in info["files"].items():
        if file_digest(genc.MODEL_DIR / name) != digest:
            raise PermissionError(f"encoder file {name} changed since it was fetched")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(genc.MODEL_DIR, padding_side="left", local_files_only=True)
    model = AutoModel.from_pretrained(genc.MODEL_DIR, torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32,
                                      local_files_only=True).to(dev).eval()
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    truncated = 0
    for start in range(0, len(texts), batch):
        idx = order[start:start + batch]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=genc.MAX_TOKENS, return_tensors="pt")
        truncated += int((enc["attention_mask"].sum(1) >= genc.MAX_TOKENS).sum())
        enc = {k: v.to(dev) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state
        out[idx] = torch.nn.functional.normalize(genc.last_token_pool(hidden, enc["attention_mask"]).float(), dim=-1).cpu().numpy()
    return out, truncated


def semantic_ids(emb: np.ndarray, seed: int, dev: torch.device) -> tuple[np.ndarray, dict]:
    """Study 2's RQ-VAE (3 x 64) with its retry rule, then the fourth code."""
    cfg, retried = dict(RQ), False
    _, info = train_rqvae(emb, seed=seed, cfg=cfg, dev=dev)
    if min(info["codes_used_per_level"]) < MIN_CODES:
        cfg["steps"] *= 2
        retried = True
        _, info = train_rqvae(emb, seed=seed, cfg=cfg, dev=dev)
    codes = disambiguate(info["codes"], seed)
    report = {"seed": seed, "config": cfg, "retried_with_double_steps": retried,
              "codes_used_per_level": info["codes_used_per_level"], "three_code_collisions": collision_stats(info["codes"]),
              "fourth_code_max": int(codes[:, -1].max()), "final_loss": info["log"][-1]}
    return codes, report
