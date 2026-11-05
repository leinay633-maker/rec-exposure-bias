"""SFT policy π0 of study 3 (docs/rlrec/PROTOCOL.md §4).

Study 2's generator, windows and loss on the training window's positives (all tabs), with study 2's optimiser,
warm-up, clipping and patience. Early stopping looks at the Recall@100 of the exact masked top 100 on 2,000 fixed
validation users; the first five epochs are never chosen and never count towards the patience.
"""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
from torch import nn

from genrec.evaluate import per_user
from genrec.generative import Codes, Generator
from genrec.generative import loss_fn as gr_loss
from genrec.train import CLIP, PATIENCE, WARMUP, EarlyStop, _batches
from receval import registry
from receval.data import file_digest

from . import config
from .policy import Tree, exact_logprobs

GRID = [{"d": d, "layers": n} for d in (128, 256) for n in (2, 4)]
FIXED = {"lr": 1e-3, "heads": 4, "dropout": 0.1}
BATCH, MAX_EPOCHS, MIN_EPOCHS, TOP = 128, 40, 5, 100


def build(codes: Codes, cfg: dict, seed: int) -> Generator:
    torch.manual_seed(seed)
    return Generator(codes, config.HISTORY, cfg["d"], cfg["layers"], FIXED["heads"], FIXED["dropout"])


def top_k(logp: np.ndarray, k: int) -> np.ndarray:
    """Best k pool rows per user by log-probability, ties to the smaller row (= smaller video_id); -1 if fewer."""
    order = np.argsort(-logp, axis=1, kind="stable")[:, :k]
    ok = np.isfinite(np.take_along_axis(logp, order, 1))
    return np.where(ok, order, -1)


def validate(model: Generator, codes: Codes, tree: Tree, valid: dict, dev: torch.device) -> float:
    logp = exact_logprobs(model, codes, tree, valid["hist"], valid["allowed"], dev)
    return float(np.nanmean(per_user(top_k(logp, TOP), valid["targets"])["recall@100"]))


def fit(cfg: dict, seed: int, codes: Codes, windows: np.ndarray, valid: dict, dev: torch.device,
        max_epochs: int = MAX_EPOCHS, batch: int = BATCH, smoke: bool = False) -> tuple:
    random.seed(seed)
    np.random.seed(seed)
    model = build(codes, cfg, seed).to(dev)
    tree = Tree(codes)
    opt = torch.optim.AdamW(model.parameters(), lr=FIXED["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, (step + 1) / WARMUP))
    gen = torch.Generator().manual_seed(seed)
    tokens = torch.as_tensor(codes.window_tokens(windows), device=dev)
    log, best_state, stop = [], None, EarlyStop(PATIENCE, MIN_EPOCHS)
    for epoch in range(1 if smoke else max_epochs):
        model.train()
        started, total, steps = time.time(), 0.0, 0
        for idx in _batches(len(tokens), batch, gen):
            loss = gr_loss(model, tokens[idx.to(dev)])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, step {steps}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            sched.step()
            total += float(loss.detach())
            steps += 1
        train_seconds, started = time.time() - started, time.time()
        val = validate(model, codes, tree, valid, dev)
        log.append({"epoch": epoch, "train_loss": total / steps, "valid_recall@100": val,
                    "train_seconds": round(train_seconds, 1), "valid_seconds": round(time.time() - started, 1)})
        if smoke:
            break
        improved, done = stop.step(epoch, val)
        if improved:
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if done:
            break
    return best_state, {"epochs": log, "best_epoch": stop.best_epoch, "valid_recall@100": stop.best,
                        "epochs_run": len(log)}


def train_run(cfg: dict, seed: int, codes: Codes, codes_sha256: str, windows: np.ndarray, valid: dict,
              dev: torch.device, purpose: str) -> dict:
    """One registered SFT run; halves the batch on CUDA out-of-memory (study 2's rule, minimum 16)."""
    batch = BATCH
    while True:
        spec = {"kind": "sft", "config": cfg, "fixed": FIXED, "seed": seed, "max_epochs": MAX_EPOCHS,
                "min_epochs": MIN_EPOCHS, "batch": batch, "purpose": purpose, "codes_sha256": codes_sha256}
        try:
            with registry.logged("rlrec-sft", spec, runs=config.RUNS) as state:
                started = time.time()
                best_state, log = fit(cfg, seed, codes, windows, valid, dev, batch=batch)
                config.MODELS.mkdir(parents=True, exist_ok=True)
                path = config.MODELS / f"{state['run_id']}.pt"
                torch.save({"kind": "sft", "config": cfg, "seed": seed, "codes_sha256": codes_sha256,
                            "state_dict": best_state, "log": log}, path)
                state.update({"path": str(path.relative_to(config.ROOT).as_posix()), "sha256": file_digest(path),
                              "valid_recall@100": log["valid_recall@100"], "best_epoch": log["best_epoch"],
                              "epochs_run": log["epochs_run"], "seconds": round(time.time() - started, 1),
                              "epoch_log": log["epochs"]})
                return dict(state, **spec)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch // 2 < 16:
                raise
            batch //= 2


def load(entry: dict, codes: Codes, dev: torch.device) -> Generator:
    path = config.ROOT / entry["path"]
    if file_digest(path) != entry["sha256"]:
        raise PermissionError(f"{entry['path']} does not match its recorded sha256")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["config"] != entry["config"] or saved["seed"] != entry["seed"]:
        raise PermissionError(f"{entry['path']} does not carry the recorded configuration and seed")
    model = build(codes, saved["config"], saved["seed"])
    model.load_state_dict(saved["state_dict"])
    return model.to(dev).eval()
