"""Training and ranking for study 2 (protocol §3): one entry point per model kind, early stopping on the
validation subset, and model files saved with their configuration and validation result.
"""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from receval.data import file_digest

from . import config, data, encode, runlog, sids
from .evaluate import per_user
from .generative import Codes, Generator, Trie, beam_search
from .generative import loss_fn as gr_loss
from .seqmodels import SASRec, TwoTower, sasrec_loss, two_tower_loss

KINDS = ("pop", "two_tower", "sasrec", "gr_text", "gr_shuf")
LEARNED = KINDS[1:]
BATCH = {"two_tower": 1024, "sasrec": 128, "gr_text": 128, "gr_shuf": 128}
MIN_BATCH = {"two_tower": 128, "sasrec": 16, "gr_text": 16, "gr_shuf": 16}
MAX_EPOCHS = {"two_tower": 20, "sasrec": 40, "gr_text": 40, "gr_shuf": 40}
FIXED = {"two_tower": {"tau": 0.1, "out": 64}, "sasrec": {"lr": 1e-3, "layers": 2, "heads": 2},
         "gr_text": {"lr": 1e-3, "heads": 4, "dropout": 0.1}, "gr_shuf": {"lr": 1e-3, "heads": 4, "dropout": 0.1}}
GRID = {
    "two_tower": [{"lr": lr, "dim": d} for lr in (1e-3, 3e-3) for d in (32, 64)],
    "sasrec": [{"d": d, "dropout": p} for d in (64, 128) for p in (0.2, 0.5)],
    "gr_text": [{"d": d, "layers": n} for d in (128, 256) for n in (2, 4)],
    "gr_shuf": [{"d": d, "layers": n} for d in (128, 256) for n in (2, 4)],
}
WARMUP, CLIP, PATIENCE, BEAM, TOP = 200, 1.0, 3, 100, 100
MODELS = config.RUNS / "models"


@dataclass
class Context:
    s: data.Study
    windows: np.ndarray
    tt: tuple  # (history, target, user) rows
    log_q: np.ndarray
    text: np.ndarray
    item_cat: np.ndarray
    n_cats: int
    col_of: np.ndarray  # video -> pool column, -1 outside the pool
    valid_users: list
    valid_hist: np.ndarray
    valid_allowed: np.ndarray
    valid_targets: list


def context() -> Context:
    s = data.study()
    win = data.windows(s)
    tt = data.pairs(win, data.window_users(s))
    counts = np.bincount(tt[1], minlength=s.n_items).astype(float)
    log_q = np.log(np.maximum(counts / counts.sum(), 1e-13))
    text, vids = encode.load()
    if vids != list(range(s.n_items)):
        raise ValueError("text embeddings must cover every video in index order")
    col_of = np.full(s.n_items, -1, dtype=np.int64)
    col_of[s.pool] = np.arange(len(s.pool))
    users = s.users["valid_subset"]
    vt = data.targets_from(data.load("valid_targets"), users)
    return Context(s, win, tt, log_q, text, s.items["category"].to_numpy(), int(s.items["category"].max()) + 1,
                   col_of, users, data.histories(s, users), data.allowed(s, users), vt)


def init_embeddings(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()


def codes_for(kind: str, seed: int, ctx: Context) -> Codes:
    codes = sids.load(seed, shuffle=(kind == "gr_shuf"), pool=ctx.s.pool)
    return Codes(codes, sizes=[sids.RQ["codebook"]] * sids.RQ["levels"] + [int(codes[:, -1].max()) + 1])


def build(kind: str, cfg: dict, seed: int, ctx: Context):
    """(model, extra) for a kind; extra holds the codes and trie of the semantic-ID models."""
    torch.manual_seed(seed)
    if kind == "two_tower":
        model = TwoTower(ctx.s.n_items, ctx.n_cats, ctx.text, ctx.item_cat, dim=cfg["dim"], out=FIXED[kind]["out"])
        extra = None
    elif kind == "sasrec":
        f = FIXED[kind]
        model = SASRec(ctx.s.n_items, config.HISTORY, cfg["d"], f["layers"], f["heads"], cfg["dropout"])
        extra = None
    else:
        f = FIXED[kind]
        codes = codes_for(kind, seed, ctx)
        model = Generator(codes, config.HISTORY, cfg["d"], cfg["layers"], f["heads"], f["dropout"])
        extra = {"codes": codes, "trie": Trie(codes, ctx.s.pool)}
    init_embeddings(model)
    return model, extra


def lr_of(kind: str, cfg: dict) -> float:
    return cfg["lr"] if "lr" in cfg else FIXED[kind]["lr"]


@torch.no_grad()
def top_from_scores(scores: torch.Tensor, allowed: torch.Tensor, pool: np.ndarray, k: int = TOP) -> np.ndarray:
    scores = scores.float().masked_fill(~allowed, float("-inf"))
    order = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :k]
    ok = torch.isfinite(scores.gather(1, order))
    out = np.asarray(pool)[order.cpu().numpy()]
    out[~ok.cpu().numpy()] = -1
    return out


@torch.no_grad()
def rank(kind: str, model, extra, hist: np.ndarray, allowed: np.ndarray, ctx: Context, dev: torch.device,
         chunk: int = 1024) -> np.ndarray:
    """Top-100 video indices per user among the allowed pool videos (-1 where fewer exist)."""
    pool = ctx.s.pool
    if kind in ("gr_text", "gr_shuf"):
        items, _ = beam_search(model, extra["codes"], extra["trie"], hist, allowed, BEAM, dev)
        out = np.where(items >= 0, pool[np.maximum(items, 0)], -1)
        return out[:, :TOP]
    out = []
    if kind == "pop":
        base = torch.as_tensor(ctx.s.items["positive_users"].to_numpy()[pool], dtype=torch.float32, device=dev)
    else:
        model.eval()
        pool_t = torch.as_tensor(pool, device=dev)
        table = model.items(pool_t) if kind == "two_tower" else model.table()[pool_t]
    for start in range(0, len(hist), chunk):
        h = torch.as_tensor(hist[start:start + chunk], device=dev)
        ok = torch.as_tensor(allowed[start:start + chunk], device=dev)
        if kind == "pop":
            scores = base.expand(len(h), -1)
        elif kind == "two_tower":
            scores = model.users(h) @ table.T
        else:
            scores = model.user(h) @ table.T
        out.append(top_from_scores(scores, ok, pool))
    return np.concatenate(out)


def validate(kind: str, model, extra, ctx: Context, dev: torch.device) -> float:
    top = rank(kind, model, extra, ctx.valid_hist, ctx.valid_allowed, ctx, dev)
    return float(np.nanmean(per_user(top, ctx.valid_targets)["recall@100"]))


class EarlyStop:
    """§3.1: keep the best validation epoch, stop after `patience` epochs without a strict improvement.

    With min_epochs > 1 (§8 sensitivity) epochs before index min_epochs - 1 are neither candidates nor counted.
    """

    def __init__(self, patience: int = PATIENCE, min_epochs: int = 1):
        self.patience, self.min_epochs = patience, min_epochs
        self.best, self.best_epoch, self.stale = -math.inf, -1, 0

    def step(self, epoch: int, val: float) -> tuple[bool, bool]:
        """(this epoch is the new best, stop now)."""
        if epoch < self.min_epochs - 1:
            return False, False
        if val > self.best:
            self.best, self.best_epoch, self.stale = val, epoch, 0
            return True, False
        self.stale += 1
        return False, self.stale >= self.patience


def _batches(n: int, size: int, gen: torch.Generator):
    perm = torch.randperm(n, generator=gen)
    for start in range(0, n, size):
        yield perm[start:start + size]


def fit(kind: str, cfg: dict, seed: int, ctx: Context, dev: torch.device, max_epochs: int, batch: int,
        smoke: bool = False, min_epochs: int = 1) -> tuple:
    """Train with early stopping (§3.1). Returns (best state, extra, log); smoke trains one epoch, no metric."""
    random.seed(seed)
    np.random.seed(seed)
    model, extra = build(kind, cfg, seed, ctx)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_of(kind, cfg), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, (step + 1) / WARMUP))
    gen = torch.Generator().manual_seed(seed)
    if kind == "two_tower":
        hist = torch.as_tensor(ctx.tt[0], device=dev)
        target = torch.as_tensor(ctx.tt[1], device=dev)
        user = torch.as_tensor(ctx.tt[2], device=dev)
        log_q = torch.as_tensor(ctx.log_q, dtype=torch.float32, device=dev)
        n = len(target)
    elif kind == "sasrec":
        windows = torch.as_tensor(ctx.windows, device=dev)
        n = len(windows)
    else:
        tokens = torch.as_tensor(extra["codes"].window_tokens(ctx.windows), device=dev)
        n = len(tokens)
    log, best_state, stop = [], None, EarlyStop(PATIENCE, min_epochs)
    for epoch in range(1 if smoke else max_epochs):
        model.train()
        started, total, steps = time.time(), 0.0, 0
        for idx in _batches(n, batch, gen):
            idx = idx.to(dev)
            if kind == "two_tower":
                loss = two_tower_loss(model, hist[idx], target[idx], user[idx], log_q, FIXED[kind]["tau"])
            elif kind == "sasrec":
                loss = sasrec_loss(model, windows[idx])
            else:
                loss = gr_loss(model, tokens[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, step {steps}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            sched.step()
            total += float(loss.detach())
            steps += 1
        train_seconds = time.time() - started
        started = time.time()
        if smoke:
            rank(kind, model, extra, ctx.valid_hist, ctx.valid_allowed, ctx, dev)
            log.append({"epoch": epoch, "train_loss": total / steps, "train_seconds": round(train_seconds, 1),
                        "rank_seconds": round(time.time() - started, 1)})
            break
        val = validate(kind, model, extra, ctx, dev)
        log.append({"epoch": epoch, "train_loss": total / steps, "valid_recall@100": val,
                    "train_seconds": round(train_seconds, 1), "valid_seconds": round(time.time() - started, 1)})
        improved, done = stop.step(epoch, val)
        if improved:
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if done:
            break
    return best_state, extra, {"epochs": log, "best_epoch": stop.best_epoch, "valid_recall@100": stop.best,
                               "epochs_run": len(log)}


def params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def train_run(kind: str, cfg: dict, seed: int, ctx: Context, dev: torch.device, purpose: str,
              max_epochs: int | None = None, min_epochs: int = 1) -> dict:
    """One registered training run; retries with half the batch on CUDA out-of-memory (§3.1)."""
    max_epochs = max_epochs or MAX_EPOCHS[kind]
    batch = BATCH[kind]
    while True:
        spec = {"kind": kind, "config": cfg, "fixed": FIXED[kind], "seed": seed, "max_epochs": max_epochs,
                "batch": batch, "purpose": purpose}
        if min_epochs > 1:
            spec["min_epochs"] = min_epochs
        try:
            with runlog.logged("train", spec) as state:
                started = time.time()
                best_state, extra, log = fit(kind, cfg, seed, ctx, dev, max_epochs, batch, min_epochs=min_epochs)
                MODELS.mkdir(parents=True, exist_ok=True)
                path = MODELS / f"{state['run_id']}.pt"
                torch.save({"kind": kind, "config": cfg, "seed": seed, "state_dict": best_state, "log": log}, path)
                model, _ = build(kind, cfg, seed, ctx)
                state.update({"path": str(path.relative_to(config.ROOT).as_posix()), "sha256": file_digest(path),
                              "valid_recall@100": log["valid_recall@100"], "best_epoch": log["best_epoch"],
                              "epochs_run": log["epochs_run"], "params": params(model),
                              "seconds": round(time.time() - started, 1), "epoch_log": log["epochs"]})
                return dict(state, **spec)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch // 2 < MIN_BATCH[kind]:
                raise
            batch //= 2


def load_model(entry: dict, ctx: Context, dev: torch.device):
    path = config.ROOT / entry["path"]
    if file_digest(path) != entry["sha256"]:
        raise PermissionError(f"{entry['path']} does not match its recorded sha256")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["kind"] != entry["kind"] or saved["config"] != entry["config"] or saved["seed"] != entry["seed"]:
        raise PermissionError(f"{entry['path']} does not carry the recorded kind, configuration and seed")
    model, extra = build(entry["kind"], entry["config"], entry["seed"], ctx)
    model.load_state_dict(saved["state_dict"])
    return model.to(dev).eval(), extra
