"""Train the RQ-VAE for each seed and cache the 4-position semantic IDs (protocol §3.5-§3.6)."""
from __future__ import annotations

import json
import shutil

import numpy as np
import torch

from receval.data import file_digest

from . import config, encode, runlog
from .semantic_id import collision_stats, disambiguate, shuffled, train_rqvae

RQ = {"latent": 32, "levels": 3, "codebook": 64, "lr": 1e-3, "steps": 3000, "reset_every": 200}
MIN_CODES = 32
DIR = config.CACHE / "sids"


def path(seed: int) -> tuple:
    return DIR / f"text_seed{seed}.npy", DIR / f"text_seed{seed}.json"


def build(seeds=config.SEEDS) -> dict:
    emb, video_ids = encode.load()
    if video_ids != sorted(video_ids):
        raise ValueError("embeddings must be in video_id order")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    with runlog.logged("sids", {"rqvae": RQ, "seeds": list(seeds), "min_codes": MIN_CODES}) as state:
        for seed in seeds:
            cfg, retried = dict(RQ), False
            _, info = train_rqvae(emb, seed=seed, cfg=cfg, dev=dev)
            if min(info["codes_used_per_level"]) < MIN_CODES:
                cfg["steps"] *= 2
                retried = True
                _, info = train_rqvae(emb, seed=seed, cfg=cfg, dev=dev)
            codes = disambiguate(info["codes"], seed)
            npy, meta = path(seed)
            np.save(npy, codes)
            report = {"seed": seed, "config": cfg, "retried_with_double_steps": retried,
                      "codes_used_per_level": info["codes_used_per_level"],
                      "three_code_collisions": collision_stats(info["codes"]),
                      "fourth_code_max": int(codes[:, -1].max()), "final_loss": info["log"][-1],
                      "sha256": file_digest(npy), "embedding_sha256": file_digest(encode.EMB_FILE)}
            meta.write_text(json.dumps(report, indent=1), encoding="utf-8")
            out[seed] = report
        keep = config.RUNS / "sids" / state["run_id"]  # the cache files are the current index (§6 step 0)
        keep.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            for f in path(seed):
                shutil.copy2(f, keep / f.name)
        state["reports"] = {s: {k: r[k] for k in ("codes_used_per_level", "three_code_collisions", "sha256")}
                            for s, r in out.items()}
        state["output"] = str(keep)
    return out


def load(seed: int, shuffle: bool, pool: np.ndarray | None = None) -> np.ndarray:
    """Text semantic IDs of a seed (rows in video index order), or the same IDs shuffled within pool / non-pool."""
    npy, meta = path(seed)
    if file_digest(npy) != json.loads(meta.read_text(encoding="utf-8"))["sha256"]:
        raise PermissionError(f"{npy.name} does not match its recorded hash")
    codes = np.load(npy)
    if not shuffle:
        return codes
    if pool is None:
        raise ValueError("the shuffled control needs the candidate pool")
    return shuffled(codes, pool, seed)
