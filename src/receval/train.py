"""Training loops for the ranking zoo (protocol §5): neural rankers, LightGBM and popularity."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn

from . import metrics
from .features import CAT_FIELDS, LABELS, FeatureSpace, video_rate
from .models import RANKERS

BATCH = 4096
MAX_EPOCHS = 15
PATIENCE = 2


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensors(built: dict, dev: torch.device) -> dict[str, torch.Tensor]:
    return {"cat": torch.as_tensor(built["cat"], device=dev), "num": torch.as_tensor(built["num"], device=dev),
            "hist": torch.as_tensor(built["hist"], device=dev),
            "labels": torch.as_tensor(built["labels"], device=dev)}


def validation_uauc(built: dict, scores: np.ndarray, label: str) -> float:
    """UAUC on the users of this set who have both classes (validation has only one set)."""
    y = built["labels"][:, LABELS.index(label)]
    return metrics.uauc(metrics.per_user_auc(built["user"], y, scores))


@torch.no_grad()
def predict(model: nn.Module, tensors: dict[str, torch.Tensor], task: int | None = None,
            batch: int = 16_384) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, tensors["cat"].shape[0], batch):
        sl = slice(start, start + batch)
        logits = model(tensors["cat"][sl], tensors["num"][sl], tensors["hist"][sl])
        if logits.dim() == 2:
            logits = logits[:, task]
        out.append(torch.sigmoid(logits).float().cpu())
    return torch.cat(out).numpy()


def train_ranker(name: str, cfg: dict, space: FeatureSpace, train: dict, valid: dict, *, seed: int,
                 label: str = "is_click", dev: torch.device | None = None,
                 max_epochs: int = MAX_EPOCHS) -> tuple[nn.Module, dict]:
    """Protocol §5.2: on CUDA out-of-memory, restart once with 2 x 2048 gradient accumulation."""
    oom = None
    try:
        return _train_ranker(name, cfg, space, train, valid, seed=seed, label=label, dev=dev,
                             max_epochs=max_epochs, accumulation=1)
    except torch.cuda.OutOfMemoryError as exc:
        oom = repr(exc)
    # Retry outside the except block so the failed attempt's tensors can be freed first.
    torch.cuda.empty_cache()
    model, info = _train_ranker(name, cfg, space, train, valid, seed=seed, label=label, dev=dev,
                                max_epochs=max_epochs, accumulation=2)
    return model, {**info, "oom_retry": True, "first_attempt_error": oom}


def _train_ranker(name: str, cfg: dict, space: FeatureSpace, train: dict, valid: dict, *, seed: int,
                  label: str, dev: torch.device | None, max_epochs: int, accumulation: int) -> tuple[nn.Module, dict]:
    dev = dev or device()
    seed_everything(seed)
    model = RANKERS[name](space.cardinalities, len(space.numeric_names), numeric_names=space.numeric_names,
                          dim=cfg.get("dim", 16)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    tr, va = to_tensors(train, dev), to_tensors(valid, dev)
    task = LABELS.index(label)
    multitask = name == "mmoe"
    best, best_state, stale, log = -np.inf, None, 0, []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        started, losses = time.time(), []
        for idx in torch.randperm(tr["cat"].shape[0], generator=generator).split(BATCH):
            opt.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for part in idx.to(dev).chunk(accumulation):
                logits = model(tr["cat"][part], tr["num"][part], tr["hist"][part])
                target = tr["labels"][part]
                loss = (nn.functional.binary_cross_entropy_with_logits(logits, target) if multitask else
                        nn.functional.binary_cross_entropy_with_logits(logits, target[:, task]))
                loss = loss * len(part) / len(idx)  # same gradient as one batch of BATCH rows
                loss.backward()
                batch_loss += loss.item()
            opt.step()
            losses.append(batch_loss)
        score = validation_uauc(valid, predict(model, va, task if multitask else None), label)
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), "valid_uauc": score,
                    "seconds": round(time.time() - started, 1)})
        if score > best:
            best, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {"valid_uauc": best, "epochs": log, "best_epoch": int(np.argmax([e["valid_uauc"] for e in log]))}


def lgb_matrix(built: dict) -> np.ndarray:
    return np.hstack([built["cat"].astype(np.float64), built["num"].astype(np.float64)])


def train_lightgbm(cfg: dict, train: dict, valid: dict, *, seed: int, label: str = "is_click",
                   rounds: int = 2000):
    import lightgbm as lgb
    task = LABELS.index(label)
    cats = list(range(len(CAT_FIELDS)))
    params = {"objective": "binary", "learning_rate": cfg["learning_rate"], "num_leaves": cfg["num_leaves"],
              "min_child_samples": cfg["min_child_samples"], "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "seed": seed, "verbose": -1, "num_threads": 0}
    dtrain = lgb.Dataset(lgb_matrix(train), train["labels"][:, task], categorical_feature=cats, free_raw_data=False)
    dvalid = lgb.Dataset(lgb_matrix(valid), valid["labels"][:, task], categorical_feature=cats, reference=dtrain)
    booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dvalid],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    scores = booster.predict(lgb_matrix(valid), num_iteration=booster.best_iteration)
    return booster, {"valid_uauc": validation_uauc(valid, scores, label), "best_iteration": booster.best_iteration}


def popularity_scores(space: FeatureSpace, built: dict, label: str = "is_click") -> np.ndarray:
    return video_rate(space, built["video"], label)
