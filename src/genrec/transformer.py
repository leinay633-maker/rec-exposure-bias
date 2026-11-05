"""Pre-norm causal Transformer shared by SASRec and the semantic-ID generator (protocol §3.1).

Padding positions never serve as keys; every position may attend to itself, so a padded query row is never
empty (its output is unused). `Stack.prefix` encodes a batch of histories once and returns their keys and
values; `Stack.step` then advances many beams per history by one token, attending to the cached history
and to the beam's own earlier tokens. The step reproduces the full forward pass (tested).
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def attention_mask(valid: torch.Tensor) -> torch.Tensor:
    """(n, L) bool -> (n, 1, L, L) bool: causal, padding keys hidden, diagonal always open."""
    length = valid.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool, device=valid.device).tril()
    eye = torch.eye(length, dtype=torch.bool, device=valid.device)
    return (causal & (valid[:, None, :] | eye))[:, None]


class Block(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        if d % heads:
            raise ValueError("hidden size must be divisible by the number of heads")
        self.heads, self.p = heads, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.heads, -1)).transpose(-3, -2)  # (..., L, d) -> (..., h, L, dh)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        q, k, v = (self._heads(t) for t in self.qkv(self.ln1(x)).chunk(3, -1))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(-3, -2).flatten(-2)))
        return x + self.drop(self.ff(self.ln2(x))), (k, v)

    def step(self, x: torch.Tensor, hist: tuple, hist_valid: torch.Tensor, past: tuple | None):
        """x: (U, B, d) one new token per beam; hist k/v: (U, h, Lh, dh); past k/v: (U, B, h, s, dh) or None."""
        q, k, v = (self._heads(t.unsqueeze(-2)) for t in self.qkv(self.ln1(x)).chunk(3, -1))  # (U, B, h, 1, dh)
        k_s = k if past is None else torch.cat([past[0], k], -2)
        v_s = v if past is None else torch.cat([past[1], v], -2)
        scale = 1 / math.sqrt(q.shape[-1])
        s_hist = torch.einsum("ubhqd,uhld->ubhql", q, hist[0]) * scale
        s_hist = s_hist.masked_fill(~hist_valid[:, None, None, None, :], float("-inf"))
        s_self = torch.einsum("ubhqd,ubhsd->ubhqs", q, k_s) * scale
        att = torch.softmax(torch.cat([s_hist, s_self], -1), -1)
        lh = hist[0].shape[-2]
        out = torch.einsum("ubhql,uhld->ubhqd", att[..., :lh], hist[1]) + torch.einsum("ubhqs,ubhsd->ubhqd", att[..., lh:], v_s)
        x = x + self.proj(out.squeeze(-2).flatten(-2))  # (U, B, h, dh) -> heads concatenated, as in forward
        return x + self.ff(self.ln2(x)), (k_s, v_s)


class Stack(nn.Module):
    def __init__(self, d: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = attention_mask(valid)
        for block in self.blocks:
            x, _ = block(x, mask)
        return self.norm(x)

    def prefix(self, x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, list[tuple]]:
        mask, cache = attention_mask(valid), []
        for block in self.blocks:
            x, kv = block(x, mask)
            cache.append(kv)
        return self.norm(x), cache

    def step(self, x: torch.Tensor, cache: list[tuple], hist_valid: torch.Tensor, past: list | None):
        new = []
        for i, block in enumerate(self.blocks):
            x, kv = block.step(x, cache[i], hist_valid, None if past is None else past[i])
            new.append(kv)
        return self.norm(x), new
