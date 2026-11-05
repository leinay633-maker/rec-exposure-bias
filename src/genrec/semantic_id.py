"""Semantic IDs for study 2 (protocol §3.5-§3.6): an RQ-VAE over frozen text embeddings, plus the shuffled control.

Each video gets three residual codes; videos that share all three get a fourth code 0, 1, 2, ... in the order of
a seeded random key, so every video has a unique tuple. The control shuffles the finished tuples among the pool
videos and, separately, among the other videos, so both prefix trees keep their shape.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-2], sizes[1:-1]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


class RQVAE(nn.Module):
    def __init__(self, in_dim: int, latent: int = 32, hidden=(512, 256, 128), levels: int = 3,
                 codebook: int = 64, beta: float = 0.25):
        super().__init__()
        self.encoder = mlp([in_dim, *hidden, latent])
        self.decoder = mlp([latent, *reversed(hidden), in_dim])
        self.codebooks = nn.Parameter(torch.randn(levels, codebook, latent) * 0.1)
        self.levels, self.size, self.beta = levels, codebook, beta

    def quantise(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Greedy residual quantisation: returns the summed codewords, the codes and each level's residual."""
        residual, total, codes, residuals = z, torch.zeros_like(z), [], []
        for level in range(self.levels):
            book = self.codebooks[level]
            idx = torch.cdist(residual, book).argmin(1)
            residuals.append(residual)
            word = book[idx]
            codes.append(idx)
            total = total + word
            residual = residual - word
        return total, torch.stack(codes, 1), residuals

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        zq, codes, residuals = self.quantise(z)
        recon = self.decoder(z + (zq - z).detach())  # straight-through estimator
        rq_loss = 0.0
        for level, r in enumerate(residuals):
            word = self.codebooks[level][codes[:, level]]
            rq_loss = rq_loss + ((r.detach() - word) ** 2).sum(-1).mean() + self.beta * ((r - word.detach()) ** 2).sum(-1).mean()
        return recon, codes, rq_loss


def _kmeans(x: torch.Tensor, k: int, iters: int, gen: torch.Generator) -> torch.Tensor:
    centres = x[torch.randperm(len(x), generator=gen, device="cpu")[:k].to(x.device)].clone()
    for _ in range(iters):
        assign = torch.cdist(x, centres).argmin(1)
        for c in range(k):
            members = x[assign == c]
            if len(members):
                centres[c] = members.mean(0)
    return centres


def train_rqvae(emb: np.ndarray, *, seed: int, cfg: dict, dev: torch.device) -> tuple[RQVAE, dict]:
    """Full-batch training; codebooks start from k-means on each level's residuals; unused codes are reset."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(emb, dtype=torch.float32, device=dev)
    model = RQVAE(x.shape[1], latent=cfg["latent"], levels=cfg["levels"], codebook=cfg["codebook"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    with torch.no_grad():  # k-means init, level by level, on the untrained encoder's output
        residual = model.encoder(x)
        for level in range(model.levels):
            model.codebooks[level] = _kmeans(residual, model.size, 10, gen)
            residual = residual - model.codebooks[level][torch.cdist(residual, model.codebooks[level]).argmin(1)]
    log = []
    for step in range(cfg["steps"]):
        recon, codes, rq_loss = model(x)
        rec_loss = ((recon - x) ** 2).sum(-1).mean()
        loss = rec_loss + rq_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % cfg["reset_every"] == 0 and step + 1 < cfg["steps"]:
            with torch.no_grad():  # dead-code reset: unused codewords jump to random current residuals
                _, codes, residuals = model.quantise(model.encoder(x))
                for level in range(model.levels):
                    used = torch.bincount(codes[:, level], minlength=model.size) > 0
                    dead = (~used).nonzero().flatten()
                    if len(dead):
                        pick = torch.randint(len(x), (len(dead),), generator=gen).to(dev)
                        model.codebooks.data[level, dead] = residuals[level][pick]
        if step % 200 == 0 or step == cfg["steps"] - 1:
            log.append({"step": step, "recon": float(rec_loss.detach()), "rq": float(rq_loss.detach())})
    model.eval()
    with torch.no_grad():
        _, codes, _ = model.quantise(model.encoder(x))
    codes = codes.cpu().numpy()
    usage = [int(len(np.unique(codes[:, level]))) for level in range(model.levels)]
    return model, {"log": log, "codes_used_per_level": usage, "codes": codes}


def disambiguate(codes: np.ndarray, seed: int) -> np.ndarray:
    """Append a fourth code numbering the videos that share all three codes, in random-key order (then video index)."""
    key = np.random.default_rng(seed).random(len(codes))
    order = np.lexsort((np.arange(len(codes)), key))
    out = np.zeros((len(codes), codes.shape[1] + 1), dtype=np.int64)
    out[:, :-1] = codes
    seen: dict[tuple, int] = {}
    for i in order:
        row = tuple(codes[i])
        out[i, -1] = seen.get(row, 0)
        seen[row] = out[i, -1] + 1
    return out


def shuffled(codes: np.ndarray, pool: np.ndarray, seed: int) -> np.ndarray:
    """The same tuples permuted among pool videos, then among the other videos (numpy.random.default_rng(seed))."""
    rng = np.random.default_rng(seed)
    pool = np.sort(np.asarray(pool))
    others = np.setdiff1d(np.arange(len(codes)), pool)
    out = codes.copy()
    for ids in (pool, others):
        out[ids] = codes[ids[rng.permutation(len(ids))]]
    return out


def collision_stats(codes: np.ndarray) -> dict:
    _, counts = np.unique(codes, axis=0, return_counts=True)
    return {"items": int(len(codes)), "distinct_prefixes": int(len(counts)),
            "items_sharing_a_prefix": int(counts[counts > 1].sum()), "max_group": int(counts.max())}
