"""Score-based models of study 2 (protocol §3.2-§3.4): popularity, logQ two-tower with text, SASRec.

Video indices run 0..n-1 and histories are left-padded with -1.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .transformer import Stack


class SASRec(nn.Module):
    def __init__(self, n_items: int, history: int, d: int, layers: int = 2, heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.d = d
        self.item = nn.Embedding(n_items + 1, d, padding_idx=0)  # video i is row i + 1
        self.pos = nn.Embedding(history, d)
        self.stack = Stack(d, layers, heads, dropout)
        self.drop = nn.Dropout(dropout)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(seq.shape[1], device=seq.device)
        x = self.drop(self.item(seq + 1) * math.sqrt(self.d) + self.pos(pos))
        return self.stack(x, seq >= 0)

    def user(self, hist: torch.Tensor) -> torch.Tensor:
        return self.encode(hist)[:, -1]

    def table(self) -> torch.Tensor:
        return self.item.weight[1:]


def sasrec_loss(model: SASRec, windows: torch.Tensor) -> torch.Tensor:
    """Full softmax over all videos at every position whose input event is real (the first is input only)."""
    inp, tgt = windows[:, :-1], windows[:, 1:]
    h = model.encode(inp)
    keep = (tgt >= 0) & (inp >= 0)
    return nn.functional.cross_entropy(h[keep] @ model.table().T, tgt[keep])


class TwoTower(nn.Module):
    """User tower: mean of history ID embeddings. Video tower: ID, first-level category and frozen text."""

    def __init__(self, n_items: int, n_cats: int, text: np.ndarray, item_cat: np.ndarray, dim: int, out: int = 64):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.cat_emb = nn.Embedding(n_cats, dim)
        self.text_proj = nn.Linear(text.shape[1], dim)
        self.register_buffer("text", torch.as_tensor(text, dtype=torch.float32), persistent=False)
        self.register_buffer("item_cat", torch.as_tensor(item_cat, dtype=torch.long), persistent=False)
        self.user_tower = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, out))
        self.item_tower = nn.Sequential(nn.Linear(3 * dim, 256), nn.ReLU(), nn.Linear(256, out))

    def users(self, hist: torch.Tensor) -> torch.Tensor:
        e = self.item_emb(hist + 1)
        mask = (hist >= 0).unsqueeze(-1).float()
        mean = (e * mask).sum(1) / mask.sum(1).clamp(min=1)
        return nn.functional.normalize(self.user_tower(mean), dim=-1)

    def items(self, idx: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.item_emb(idx + 1), self.cat_emb(self.item_cat[idx]), self.text_proj(self.text[idx])], -1)
        return nn.functional.normalize(self.item_tower(x), dim=-1)


def two_tower_loss(model: TwoTower, hist: torch.Tensor, target: torch.Tensor, user: torch.Tensor,
                   log_q: torch.Tensor, tau: float) -> torch.Tensor:
    """In-batch softmax with logQ; a repeated video or another row of the same user is not a negative."""
    logits = model.users(hist) @ model.items(target).T / tau - log_q[target][None, :]
    same = (target[:, None] == target[None, :]) | (user[:, None] == user[None, :])
    same.fill_diagonal_(False)
    logits = logits.masked_fill(same, float("-inf"))
    return nn.functional.cross_entropy(logits, torch.arange(len(target), device=target.device))
