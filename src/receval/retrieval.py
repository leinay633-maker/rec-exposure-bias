"""Full-pool retrieval over the 7,583 videos (protocol §9): popularity, ItemCF, two-tower retrievers."""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .features import CAT_FIELDS, FeatureSpace, item_table, user_table
from .models import TwoTowerRetriever
from .train import BATCH, MAX_EPOCHS, PATIENCE, device, seed_everything

K_MAIN = 100
KS = (50, 100, 200)


class Pool:
    """Pool videos, per-user seen sets from the training window, and lookup between ids and rows."""

    def __init__(self, space: FeatureSpace, videos: pd.DataFrame, train: pd.DataFrame):
        self.video_ids, self.item_cat = item_table(space, videos)
        self.row_of = {int(v): i for i, v in enumerate(self.video_ids)}
        self.seen = {int(u): np.array(sorted({self.row_of[int(v)] for v in part}), dtype=np.int64)
                     for u, part in train.groupby("user_id")["video_id"]}

    def targets(self, frame: pd.DataFrame) -> dict[int, np.ndarray]:
        """Per user: positive pool rows of this set that the user did not touch in training."""
        out = {}
        for user, part in frame.loc[frame["is_click"] == 1].groupby("user_id")["video_id"]:
            rows = {self.row_of[int(v)] for v in part} - set(self.seen.get(int(user), ()))
            if rows:
                out[int(user)] = np.array(sorted(rows), dtype=np.int64)
        return out


def recall_ndcg(scores: torch.Tensor, users: list[int], pool: Pool, targets: dict[int, np.ndarray],
                ks=KS) -> pd.DataFrame:
    """Per-user Recall@k and NDCG@100 from a (users x items) score block; seen items are excluded."""
    scores = scores.clone()
    for row, user in enumerate(users):
        seen = pool.seen.get(user)
        if seen is not None and len(seen):
            scores[row, torch.as_tensor(seen, device=scores.device)] = -torch.inf
    # Stable descending sort: ties keep pool order (video_id order), so popularity ties are reproducible.
    top = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :max(ks)].cpu().numpy()
    discounts = 1 / np.log2(np.arange(2, K_MAIN + 2))
    rows = []
    for row, user in enumerate(users):
        target = targets[user]
        hits = np.isin(top[row], target)
        record = {"user": user, "targets": len(target)}
        for k in ks:
            record[f"recall@{k}"] = hits[:k].sum() / len(target)
        ideal = discounts[:min(len(target), K_MAIN)].sum()
        record["ndcg@100"] = (discounts * hits[:K_MAIN]).sum() / ideal
        rows.append(record)
    return pd.DataFrame(rows).set_index("user")


def evaluate_scorer(score_fn, users: list[int], pool: Pool, targets: dict[int, np.ndarray],
                    chunk: int = 2048) -> pd.DataFrame:
    if not users:  # an empty population keeps its columns, so callers report it instead of crashing
        return pd.DataFrame(columns=["targets", *(f"recall@{k}" for k in KS), "ndcg@100"],
                            index=pd.Index([], name="user"), dtype=float)
    parts = [recall_ndcg(score_fn(users[i:i + chunk]), users[i:i + chunk], pool, targets)
             for i in range(0, len(users), chunk)]
    return pd.concat(parts)


def popularity_scorer(pool: Pool, train: pd.DataFrame, dev: torch.device):
    counts = train.loc[train["is_click"] == 1, "video_id"].value_counts()
    vector = torch.as_tensor(counts.reindex(pool.video_ids).fillna(0).to_numpy(np.float32), device=dev)
    return lambda users: vector.expand(len(users), -1)


def itemcf_scorer(pool: Pool, train: pd.DataFrame, dev: torch.device):
    """Cosine item-item similarity from co-positive users; user score = sum of similarities."""
    positives = train.loc[train["is_click"] == 1, ["user_id", "video_id"]].drop_duplicates()
    user_index = {u: i for i, u in enumerate(positives["user_id"].unique())}
    rows = torch.as_tensor(positives["user_id"].map(user_index).to_numpy(), device=dev)
    cols = torch.as_tensor(positives["video_id"].map(pool.row_of).to_numpy(), device=dev)
    matrix = torch.zeros(len(user_index), len(pool.video_ids), device=dev)
    matrix[rows, cols] = 1.0
    co = matrix.T @ matrix
    norms = torch.sqrt(torch.diag(co)).clamp(min=1.0)
    similarity = co / norms[:, None] / norms[None, :]
    similarity.fill_diagonal_(0.0)

    def score(users):
        index = [user_index.get(u) for u in users]
        block = torch.zeros(len(users), len(pool.video_ids), device=dev)
        known = [i for i, x in enumerate(index) if x is not None]
        if known:
            block[known] = matrix[[index[i] for i in known]] @ similarity
        return block
    return score


def retriever_scorer(model: TwoTowerRetriever, space: FeatureSpace, users_frame: pd.DataFrame, pool: Pool,
                     dev: torch.device):
    model.eval()
    with torch.no_grad():
        items = model.items(torch.as_tensor(pool.item_cat, device=dev))

    def score(users):
        cat, hist = user_table(space, users_frame, users)
        with torch.no_grad():
            u = model.users(torch.as_tensor(cat, device=dev), torch.as_tensor(hist, device=dev))
        return u @ items.T
    return score


def train_retriever(cfg: dict, space: FeatureSpace, train_built: dict, train: pd.DataFrame,
                    users_frame: pd.DataFrame, pool: Pool, valid_targets: dict[int, np.ndarray], *, seed: int,
                    logq: bool, dev: torch.device | None = None, max_epochs: int = MAX_EPOCHS):
    """In-batch softmax over the batch's positive videos; logQ subtracts log sampling frequency."""
    dev = dev or device()
    seed_everything(seed)
    model = TwoTowerRetriever(space.cardinalities, dim=cfg.get("dim", 16), out=cfg["out"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    positive = train_built["labels"][:, 0] == 1
    cat = torch.as_tensor(train_built["cat"][positive], device=dev)
    hist = torch.as_tensor(train_built["hist"][positive], device=dev)
    item_col = CAT_FIELDS.index("video_id")
    item_ids = cat[:, item_col]
    user_ids = cat[:, CAT_FIELDS.index("user_id")]
    freq = torch.bincount(item_ids, minlength=space.cardinalities[item_col]).float()
    log_q = torch.log(freq / freq.sum()).clamp(min=-30.0)
    users = sorted(valid_targets)
    best, best_state, stale, log = -np.inf, None, 0, []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        started, losses = time.time(), []
        for idx in torch.randperm(cat.shape[0], generator=generator).split(BATCH):
            idx = idx.to(dev)
            u = model.users(cat[idx], hist[idx])
            v = model.items(cat[idx])
            logits = u @ v.T / cfg["tau"]
            if logq:
                logits = logits - log_q[item_ids[idx]][None, :]
            # Neither a repeated video nor another positive of the same user counts as a negative.
            same = (item_ids[idx][:, None] == item_ids[idx][None, :]) | (user_ids[idx][:, None] == user_ids[idx][None, :])
            same.fill_diagonal_(False)
            logits = logits.masked_fill(same, -torch.inf)
            loss = nn.functional.cross_entropy(logits, torch.arange(len(idx), device=dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        table = evaluate_scorer(retriever_scorer(model, space, users_frame, pool, dev), users, pool, valid_targets)
        score = float(table[f"recall@{K_MAIN}"].mean())
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), f"valid_recall@{K_MAIN}": score,
                    "seconds": round(time.time() - started, 1)})
        if score > best:
            best, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {f"valid_recall@{K_MAIN}": best, "epochs": log}
