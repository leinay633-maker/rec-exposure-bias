"""Frozen text embeddings for every KuaiRec video (protocol §3.5), computed once and cached with hashes.

`fetch` downloads one pinned commit of Qwen/Qwen3-Embedding-0.6B (safetensors only, no remote code);
`encode` loads it from disk, embeds `text.describe(row)` for all 10,728 videos with last-token pooling and
left padding, L2-normalises and writes `text_emb.npy` plus a metadata file with the revision and hashes.
"""
from __future__ import annotations

import json
import shutil
import time

import numpy as np
import torch

from receval.data import file_digest

from . import config, runlog
from .text import describe, load_captions

REPO = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"  # protocol §3.5
MODEL_DIR = config.ROOT / ".data" / "models" / "Qwen3-Embedding-0.6B"
REVISION_FILE = MODEL_DIR / "revision.json"
ALLOW = ["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json", "tokenizer*"]
MAX_TOKENS = 256
EMB_FILE, META_FILE = config.CACHE / "text_emb.npy", config.CACHE / "text_emb.json"


def fetch(revision: str = REVISION) -> dict:
    from huggingface_hub import snapshot_download

    with runlog.logged("fetch-encoder", {"repo": REPO, "revision": revision}) as state:
        sha = revision
        path = snapshot_download(REPO, revision=sha, local_dir=MODEL_DIR, allow_patterns=ALLOW)
        files = {p.name: file_digest(p) for p in sorted(MODEL_DIR.glob("*")) if p.is_file() and p.name != "revision.json"}
        info = {"repo": REPO, "revision": sha, "path": str(path), "files": files}
        REVISION_FILE.write_text(json.dumps(info, indent=2), encoding="utf-8")
        state.update({"revision": sha, "files": len(files)})
    return info


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask[:, -1].all()):  # left padding: the last position is every row's last real token
        return hidden[:, -1]
    lengths = mask.sum(1) - 1
    return hidden[torch.arange(len(hidden), device=hidden.device), lengths]


@torch.no_grad()
def encode(batch: int = 64) -> dict:
    from transformers import AutoModel, AutoTokenizer

    info = json.loads(REVISION_FILE.read_text(encoding="utf-8"))
    if info["revision"] != REVISION:
        raise PermissionError(f"encoder revision {info['revision']} is not the pinned {REVISION}")
    for name, digest in info["files"].items():
        if file_digest(MODEL_DIR / name) != digest:
            raise PermissionError(f"encoder file {name} changed since it was fetched")
    with runlog.logged("encode-text", {"revision": info["revision"], "max_tokens": MAX_TOKENS}) as state:
        captions, repaired = load_captions()
        texts = [describe(row) for _, row in captions.iterrows()]
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="left", local_files_only=True)
        model = AutoModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32,
                                          local_files_only=True).to(dev).eval()
        order = np.argsort([len(t) for t in texts], kind="stable")  # similar lengths per batch, less padding
        out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
        truncated, started = 0, time.time()
        for start in range(0, len(texts), batch):
            idx = order[start:start + batch]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
            truncated += int((enc["attention_mask"].sum(1) >= MAX_TOKENS).sum())
            enc = {k: v.to(dev) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            emb = torch.nn.functional.normalize(last_token_pool(hidden, enc["attention_mask"]).float(), dim=-1)
            out[idx] = emb.cpu().numpy()
        config.CACHE.mkdir(parents=True, exist_ok=True)
        np.save(EMB_FILE, out)
        meta = {"revision": info["revision"], "videos": int(len(texts)), "dim": int(out.shape[1]),
                "video_ids": captions["video_id"].tolist(), "records_repaired": repaired,
                "texts_hitting_max_tokens": truncated, "seconds": round(time.time() - started, 1),
                "sha256": file_digest(EMB_FILE), "dtype": "float16 on cuda" if dev.type == "cuda" else "float32"}
        META_FILE.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        keep = config.RUNS / "encode-text" / state["run_id"]  # the cache file is the current index (§6 step 0)
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(EMB_FILE, keep / EMB_FILE.name)
        shutil.copy2(META_FILE, keep / META_FILE.name)
        state.update({"output": str(keep), "sha256": meta["sha256"]})
    return meta


def load() -> tuple[np.ndarray, list[int]]:
    """Embeddings in video_id order, checked against the hash written by `encode`."""
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    if file_digest(EMB_FILE) != meta["sha256"]:
        raise PermissionError("text_emb.npy does not match its recorded hash")
    return np.load(EMB_FILE), meta["video_ids"]
