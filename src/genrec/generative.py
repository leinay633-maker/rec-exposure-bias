"""Semantic-ID generative retrieval (protocol §3.5): a decoder-only Transformer over 4-token video codes and a
prefix-trie constrained beam search.

Token 0 is padding and every code position (level) has its own token range. A training window of up to L + 1
positive events is left-padded to L + 1 slots of 4 tokens; every token of every event except the window's
first is predicted from all tokens before it, with a softmax over that level's tokens only. At inference the
history fills slots 0..L-1 and the next video is generated in slot L, one level at a time.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .transformer import Stack

PAD = 0


class Codes:
    """4-position codes of every video (row = video index) and their token ids."""

    def __init__(self, codes: np.ndarray, sizes: list[int] | None = None):
        """sizes: codebook size per position; inferred as max + 1 when not given (unit tests)."""
        self.codes = np.asarray(codes, dtype=np.int64)
        self.levels = self.codes.shape[1]
        self.sizes = [int(self.codes[:, level].max()) + 1 for level in range(self.levels)] if sizes is None else list(sizes)
        if len(self.sizes) != self.levels or any(
                self.codes[:, level].min() < 0 or self.codes[:, level].max() >= size for level, size in enumerate(self.sizes)):
            raise ValueError("every code must lie in [0, size) of its position")
        self.offsets = np.cumsum([1] + self.sizes[:-1]).astype(np.int64)
        self.vocab = 1 + sum(self.sizes)
        self.tokens = self.codes + self.offsets[None, :]
        if len(np.unique(self.codes, axis=0)) != len(self.codes):
            raise ValueError("codes must be unique per video")

    def level_bias(self) -> torch.Tensor:
        """(levels, vocab) additive bias: 0 inside a level's token range, -inf elsewhere (padding included)."""
        bias = torch.full((self.levels, self.vocab), float("-inf"))
        for level, (off, size) in enumerate(zip(self.offsets, self.sizes)):
            bias[level, off:off + size] = 0.0
        return bias

    def window_tokens(self, windows: np.ndarray) -> np.ndarray:
        """(n, slots) video indices with -1 padding -> (n, slots * levels) tokens."""
        windows = np.asarray(windows)
        toks = self.tokens[np.where(windows >= 0, windows, 0)]
        toks[windows < 0] = PAD
        return toks.reshape(len(windows), -1)


class Generator(nn.Module):
    def __init__(self, codes: Codes, history: int, d: int, layers: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.levels, self.history = codes.levels, history
        self.tok = nn.Embedding(codes.vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(codes.levels * (history + 1), d)
        self.lvl = nn.Embedding(codes.levels, d)
        self.stack = Stack(d, layers, heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("level_bias", codes.level_bias(), persistent=False)

    def embed(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.tok(tokens) + self.pos(positions) + self.lvl(positions % self.levels)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.tok.weight.T

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        return self.logits(self.stack(self.drop(self.embed(tokens, pos)), tokens != PAD))


def loss_fn(model: Generator, tokens: torch.Tensor) -> torch.Tensor:
    """tokens: (n, 4 (L + 1)) from `Codes.window_tokens`; the first real slot of each row is input only."""
    levels = model.levels
    inp, tgt = tokens[:, :-1], tokens[:, 1:]
    logits = model(inp)
    target_pos = torch.arange(1, tokens.shape[1], device=tokens.device)
    first_slot = (tokens.view(len(tokens), -1, levels)[..., 0] == PAD).sum(1)  # left padding: leading empty slots
    keep = (tgt != PAD) & ((target_pos // levels)[None, :] > first_slot[:, None])
    logits = logits + model.level_bias[target_pos % levels][None]
    return nn.functional.cross_entropy(logits[keep], tgt[keep])


class Trie:
    """Prefix ids of the candidate videos at every depth, for vectorised constrained beam search."""

    def __init__(self, codes: Codes, candidates: np.ndarray):
        self.candidates = np.asarray(candidates, dtype=np.int64)
        toks = codes.tokens[self.candidates]  # (n_cand, levels)
        self.levels = codes.levels
        self.pid = [np.zeros(len(toks), dtype=np.int64)]  # depth 0: the empty prefix
        self.parent, self.token = [None], [None]
        for depth in range(1, self.levels + 1):
            keys, pid = np.unique(toks[:, :depth], axis=0, return_inverse=True)
            pid = pid.reshape(-1)
            self.pid.append(pid)
            first = np.zeros(len(keys), dtype=np.int64)
            first[pid[::-1]] = np.arange(len(pid))[::-1]  # a candidate carrying each prefix
            self.parent.append(self.pid[depth - 1][first])
            self.token.append(keys[:, -1])
        self.item_of_leaf = np.empty(len(self.pid[-1]), dtype=np.int64)
        self.item_of_leaf[self.pid[-1]] = np.arange(len(toks))  # leaf id -> candidate position

    def to(self, dev: torch.device) -> dict:
        t = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
        return {"pid": [t(p) for p in self.pid], "parent": [None] + [t(p) for p in self.parent[1:]],
                "token": [None] + [t(p) for p in self.token[1:]], "leaf": t(self.item_of_leaf),
                "sizes": [int(p.max()) + 1 for p in self.pid]}


@torch.no_grad()
def beam_search(model: Generator, codes: Codes, trie: Trie, history: np.ndarray, allowed: np.ndarray, beam: int,
                dev: torch.device, chunk: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Top-`beam` candidates per user, best first.

    history: (users, L) video indices, -1 padded on the left; allowed: (users, n_cand) bool over trie.candidates.
    Returns candidate positions (users, beam; -1 when fewer exist) and summed log-probabilities.
    """
    model.eval()
    t = trie.to(dev)
    users, length = history.shape
    out_items = np.full((users, beam), -1, dtype=np.int64)
    out_scores = np.full((users, beam), -np.inf)
    hist_tokens = codes.window_tokens(history)
    for start in range(0, users, chunk):
        hist = torch.as_tensor(hist_tokens[start:start + chunk], device=dev)
        ok = torch.as_tensor(allowed[start:start + chunk], device=dev)
        n = len(hist)
        rows = torch.arange(n, device=dev)[:, None]
        h, cache = model.stack.prefix(model.embed(hist, torch.arange(hist.shape[1], device=dev)), hist != PAD)
        logp = torch.log_softmax(model.logits(h[:, -1]).float() + model.level_bias[0], -1)[:, None]  # (n, 1, V)
        beam_pid = torch.zeros(n, 1, dtype=torch.long, device=dev)
        beam_score = torch.zeros(n, 1, device=dev)
        beam_ok = torch.ones(n, 1, dtype=torch.bool, device=dev)
        past = None
        hist_valid = hist != PAD
        for depth in range(trie.levels):
            # which beam (if any) holds each candidate's prefix at this depth
            table = torch.full((n, t["sizes"][depth]), -1, dtype=torch.long, device=dev)
            b_idx = torch.arange(beam_pid.shape[1], device=dev).expand(n, -1)
            table[rows.expand_as(beam_pid)[beam_ok], beam_pid[beam_ok]] = b_idx[beam_ok]
            owner = table[:, t["pid"][depth]]  # (n, n_cand)
            alive = ok & (owner >= 0)
            own = owner.clamp(min=0)
            tok = t["token"][depth + 1][t["pid"][depth + 1]]  # token each candidate takes at this depth
            cand = beam_score.gather(1, own) + logp[rows, own, tok[None, :].expand(n, -1)]
            cand = cand.masked_fill(~alive, float("-inf"))
            nxt = torch.full((n, t["sizes"][depth + 1]), float("-inf"), device=dev)
            nxt.scatter_reduce_(1, t["pid"][depth + 1].expand(n, -1), cand, reduce="amax")
            order = torch.sort(nxt, dim=1, descending=True, stable=True).indices[:, :beam]
            new_score = nxt.gather(1, order)
            new_ok = torch.isfinite(new_score)
            parent_beam = table.gather(1, t["parent"][depth + 1][order]).clamp(min=0)
            beam_pid, beam_score, beam_ok = order, new_score, new_ok
            if depth == trie.levels - 1:
                break
            if past is not None:
                past = [tuple(x[rows, parent_beam] for x in kv) for kv in past]
            token = t["token"][depth + 1][order].masked_fill(~new_ok, PAD)
            pos = torch.full_like(token, length * trie.levels + depth)
            x = model.embed(token, pos)
            h, past = model.stack.step(x, cache, hist_valid, past)
            logp = torch.log_softmax(model.logits(h).float() + model.level_bias[depth + 1], -1)
        items = t["leaf"][beam_pid].masked_fill(~beam_ok, -1)
        k = items.shape[1]
        out_items[start:start + n, :k] = items.cpu().numpy()
        out_scores[start:start + n, :k] = beam_score.masked_fill(~beam_ok, float("-inf")).cpu().numpy()
    return out_items, out_scores


@torch.no_grad()
def exact_scores(model: Generator, codes: Codes, trie: Trie, history: np.ndarray, dev: torch.device) -> np.ndarray:
    """Summed log-probability of every candidate by teacher forcing (for tests on small cases)."""
    model.eval()
    users, length = history.shape
    cand_tokens = torch.as_tensor(codes.tokens[trie.candidates], device=dev)
    hist = torch.as_tensor(codes.window_tokens(history), device=dev)
    out = np.zeros((users, len(trie.candidates)))
    for u in range(users):
        seq = torch.cat([hist[u].expand(len(cand_tokens), -1), cand_tokens], 1)
        logits = model(seq[:, :-1])[:, -trie.levels:].float()
        logits = logits + model.level_bias[None]
        logp = torch.log_softmax(logits, -1).gather(2, cand_tokens[..., None]).squeeze(-1)
        out[u] = logp.sum(1).cpu().numpy()
    return out
