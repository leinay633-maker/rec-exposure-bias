"""Candidate-masked semantic-ID policy of study 3 (docs/rlrec/PROTOCOL.md §4, §6.2).

π(i|u) = Π_l p_l(c_l(i) | prefix): level l's softmax is renormalised over the children of the prefix that still hold
one of the user's candidates, so π is a distribution over the user's candidates. `exact_logprobs` expands the whole
prefix tree (no pruning); `sample` walks the same masked distributions one level at a time; `sequence_logprob`
recomputes the masked log-probability of given videos by teacher forcing, with gradients for RLOO.

Videos are the rows of `Codes` (the pool); histories are video rows, -1 padded on the left, as in genrec.
"""
from __future__ import annotations

import numpy as np
import torch

from genrec.generative import PAD, Codes, Generator, Trie

NEG = float("-inf")


class Tree:
    """The pool's prefix tree as dense tables; depth 0 is the empty prefix, depth `levels` the videos."""

    def __init__(self, codes: Codes):
        self.levels = codes.levels
        self.n_items = len(codes.codes)
        trie = Trie(codes, np.arange(self.n_items))
        self.pid = trie.pid  # per depth: (n_items,) prefix id of every video
        self.size = [int(p.max()) + 1 for p in trie.pid]  # prefixes per depth
        self.parent, self.token = trie.parent, trie.token  # per depth >= 1: parent id at depth - 1, global token id
        self.offsets, self.code_sizes = codes.offsets, codes.sizes
        self.child = [None]  # per depth >= 1: (prefixes at depth - 1, codes of level depth - 1) -> prefix id or -1
        for d in range(1, self.levels + 1):
            table = np.full((self.size[d - 1], codes.sizes[d - 1]), -1, dtype=np.int64)
            table[self.parent[d], self.token[d] - codes.offsets[d - 1]] = np.arange(self.size[d])
            self.child.append(table)
        self.item_of_leaf = trie.item_of_leaf
        last = np.bincount(self.parent[self.levels], minlength=self.size[self.levels - 1])
        self.multi = np.flatnonzero(last >= 2)  # depth levels-1 prefixes whose last level is a real choice
        self.slot = np.full(self.size[self.levels - 1], -1, dtype=np.int64)
        self.slot[self.multi] = np.arange(len(self.multi))

    def to(self, dev: torch.device) -> dict:
        t = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
        return {"pid": [t(p) for p in self.pid], "parent": [None] + [t(p) for p in self.parent[1:]],
                "token": [None] + [t(p) for p in self.token[1:]], "child": [None] + [t(c) for c in self.child[1:]],
                "leaf": t(self.item_of_leaf), "multi": t(self.multi), "slot": t(self.slot)}


def alive(tree: Tree, t: dict, allowed: torch.Tensor) -> list[torch.Tensor]:
    """allowed: (n, n_items) bool -> per depth (n, prefixes) bool: the prefix holds an allowed video."""
    n = len(allowed)
    out = [allowed.any(1, keepdim=True)]
    a = allowed.float()
    for d in range(1, tree.levels + 1):
        cnt = torch.zeros(n, tree.size[d], device=allowed.device).scatter_add_(1, t["pid"][d].expand(n, -1), a)
        out.append(cnt > 0)
    return out


def _group_log_softmax(z: torch.Tensor, ok: torch.Tensor, parent: torch.Tensor, n_parents: int) -> torch.Tensor:
    """Log-softmax of z (n, m) among the ok siblings sharing a parent; -inf where not ok."""
    n = len(z)
    z = z.masked_fill(~ok, NEG)
    idx = parent.expand(n, -1)
    m = torch.full((n, n_parents), NEG, device=z.device).scatter_reduce(1, idx, z, reduce="amax", include_self=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    s = torch.zeros(n, n_parents, device=z.device).scatter_add_(1, idx, torch.exp(z - m[:, parent]))
    out = z - (m + torch.log(s))[:, parent]
    return out.masked_fill(~ok, NEG)


def _masked_log_softmax(logits: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~ok, NEG).log_softmax(-1).masked_fill(~ok, NEG)


def _encode(model: Generator, codes: Codes, history: np.ndarray, dev: torch.device):
    hist = torch.as_tensor(codes.window_tokens(history), device=dev)
    valid = hist != PAD
    h, cache = model.stack.prefix(model.embed(hist, torch.arange(hist.shape[1], device=dev)), valid)
    return h[:, -1], cache, valid, hist.shape[1]


def _step(model: Generator, tokens: torch.Tensor, pos: int, cache, valid, past):
    x = model.embed(tokens, torch.full_like(tokens, pos))
    h, past = model.stack.step(x, cache, valid, past)
    return model.logits(h).float(), past


def exact_logprobs(model: Generator, codes: Codes, tree: Tree, history: np.ndarray, allowed: np.ndarray,
                   dev: torch.device, chunk: int = 32, t: dict | None = None) -> np.ndarray:
    """(users, n_items) float32 log π(i|u); -inf outside the user's candidates. Every user needs a candidate.
    Halves the users per batch on CUDA out-of-memory (the result does not depend on it)."""
    if not np.asarray(allowed).any(1).all():
        raise ValueError("every user needs at least one candidate")
    while True:
        try:
            return _exact(model, codes, tree, history, allowed, dev, chunk, t)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk == 1:
                raise
            chunk //= 2


@torch.no_grad()
def _exact(model, codes, tree, history, allowed, dev, chunk, t) -> np.ndarray:
    model.eval()
    t = tree.to(dev) if t is None else t
    L = tree.levels
    out = np.empty((len(history), tree.n_items), dtype=np.float32)
    for start in range(0, len(history), chunk):
        ok = torch.as_tensor(allowed[start:start + chunk], device=dev)
        n = len(ok)
        rows = torch.arange(n, device=dev)[:, None]
        al = alive(tree, t, ok)
        h, cache, valid, base = _encode(model, codes, history[start:start + chunk], dev)
        logits = model.logits(h).float()  # (n, V): the first level
        lp = [None, _group_log_softmax(logits[:, t["token"][1]], al[1], t["parent"][1], 1)]
        past = None
        for d in range(1, L):
            # advance the depth-d prefixes by their last token; all of them, except that at the last level only
            # prefixes with several children need it (a single child has conditional probability 1)
            keep = t["multi"] if d == L - 1 else torch.arange(tree.size[d], device=dev)
            if len(keep) == 0:  # no prefix with several children: the last level is certain everywhere
                lp.append(torch.zeros(n, tree.size[d + 1], device=dev).masked_fill(~al[d + 1], NEG))
                continue
            if past is not None:  # the previous step advanced every depth-(d-1) prefix, in id order
                idx = t["parent"][d][keep]
                past = [tuple(x[:, idx] for x in kv) for kv in past]
            logits, past = _step(model, t["token"][d][keep].expand(n, -1), base + d - 1, cache, valid, past)
            kid_parent, kid_token = t["parent"][d + 1], t["token"][d + 1]
            if d < L - 1:  # keep is every depth-d prefix, so a parent's id is its row in logits
                z = logits[rows, kid_parent[None, :], kid_token[None, :]]
                lp.append(_group_log_softmax(z, al[d + 1], kid_parent, tree.size[d]))
            else:
                slot = t["slot"][kid_parent]
                has = slot >= 0
                z = logits[rows, slot.clamp(min=0)[None, :], kid_token[None, :]]
                last = _group_log_softmax(z, al[d + 1] & has[None, :], kid_parent, tree.size[d])
                lp.append(torch.where(has[None, :], last, torch.zeros_like(last).masked_fill(~al[d + 1], NEG)))
        total = sum(lp[d][:, t["pid"][d]] for d in range(1, L + 1))
        out[start:start + n] = total.masked_fill(~ok, NEG).cpu().numpy()
    return out


def _paths(tree: Tree, t: dict, items: torch.Tensor) -> list[torch.Tensor]:
    """Prefix ids at every depth of the given videos."""
    return [t["pid"][d][items] for d in range(tree.levels + 1)]


def _level_logp(tree: Tree, t: dict, al: list, logits: torch.Tensor, prefix: torch.Tensor, d: int) -> torch.Tensor:
    """Masked log-probabilities over the children of `prefix` (depth d - 1) at level d - 1.

    logits: (n, S, V) or (n, V); prefix: (n, S) or (n,) -> (n, S, codes of the level) with -inf for absent children.
    """
    off, size = int(tree.offsets[d - 1]), int(tree.code_sizes[d - 1])
    local = logits[..., off:off + size]
    kids = t["child"][d][prefix]  # (..., size)
    n = len(kids)
    flat = kids.reshape(n, -1)
    ok = (flat >= 0) & torch.gather(al[d], 1, flat.clamp(min=0))
    return _masked_log_softmax(local, ok.reshape(kids.shape)), kids


@torch.no_grad()
def sample(model: Generator, codes: Codes, tree: Tree, history: np.ndarray, allowed: np.ndarray, n_samples: int,
           dev: torch.device, generator: torch.Generator, t: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Independent draws from π(.|u): (users, n_samples) video rows and their log-probabilities."""
    model.eval()
    t = tree.to(dev) if t is None else t
    ok = torch.as_tensor(allowed, device=dev)
    if not bool(ok.any(1).all()):
        raise ValueError("every user needs at least one candidate")
    n = len(ok)
    al = alive(tree, t, ok)
    h, cache, valid, base = _encode(model, codes, history, dev)
    logits = model.logits(h).float()[:, None, :].expand(n, n_samples, -1)
    prefix = torch.zeros(n, n_samples, dtype=torch.long, device=dev)
    total = torch.zeros(n, n_samples, device=dev)
    past = None
    for d in range(1, tree.levels + 1):
        logp, kids = _level_logp(tree, t, al, logits, prefix, d)
        pick = torch.multinomial(logp.exp().reshape(n * n_samples, -1), 1, generator=generator).reshape(n, n_samples)
        total = total + logp.gather(2, pick[..., None]).squeeze(-1)
        prefix = kids.gather(2, pick[..., None]).squeeze(-1)
        if d < tree.levels:
            logits, past = _step(model, t["token"][d][prefix], base + d - 1, cache, valid, past)
    items = t["leaf"][prefix]
    return items.cpu().numpy(), total.cpu().numpy()


def sequence_logprob(model: Generator, codes: Codes, tree: Tree, t: dict, hist_tokens: torch.Tensor,
                     al: list, items: torch.Tensor) -> torch.Tensor:
    """Masked log π(item|u) by teacher forcing, differentiable. hist_tokens: (n, L*levels); items: (n, S)."""
    valid = hist_tokens != PAD
    dev = hist_tokens.device
    h, cache = model.stack.prefix(model.embed(hist_tokens, torch.arange(hist_tokens.shape[1], device=dev)), valid)
    base = hist_tokens.shape[1]
    n, s = items.shape
    paths = _paths(tree, t, items)
    logits = model.logits(h[:, -1]).float()[:, None, :].expand(n, s, -1)
    total = torch.zeros(n, s, device=dev)
    past = None
    for d in range(1, tree.levels + 1):
        logp, _ = _level_logp(tree, t, al, logits, paths[d - 1], d)
        local = t["token"][d][paths[d]] - int(tree.offsets[d - 1])
        total = total + logp.gather(2, local[..., None]).squeeze(-1)
        if d < tree.levels:
            logits, past = _step(model, t["token"][d][paths[d]], base + d - 1, cache, valid, past)
    return total
