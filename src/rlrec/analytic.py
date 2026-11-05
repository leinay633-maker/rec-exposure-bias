"""The analytic optimum of the KL-regularised objective (docs/rlrec/PROTOCOL.md §6.1).

π_β(i|u) ∝ π0(i|u)·exp(r(u,i)/β) maximises E_π[r] − β·KL(π‖π0) for every user. KL(π_β‖π0) falls as β grows, so
β for a KL level is found by bisection on log β. Inputs are log-probabilities with -inf outside the candidates.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

NEG = -np.inf


def tilt(logp0: np.ndarray, r: np.ndarray, beta: float) -> np.ndarray:
    """log π_β, rows = users; -inf stays -inf."""
    s = np.asarray(logp0, np.float64) + np.asarray(r, np.float64) / beta
    return s - logsumexp(s, axis=1, keepdims=True)


def kl(logp: np.ndarray, logp0: np.ndarray) -> np.ndarray:
    """KL(π‖π0) per user, over the candidates (where π > 0)."""
    logp, logp0 = np.asarray(logp, np.float64), np.asarray(logp0, np.float64)
    p = np.exp(logp)
    diff = np.zeros_like(logp)
    np.subtract(logp, logp0, out=diff, where=p > 0)
    return (p * diff).sum(1)


def mean_kl(logp0: np.ndarray, r: np.ndarray, beta: float, chunk: int = 2_000) -> float:
    return float(np.concatenate([kl(tilt(logp0[s:s + chunk], r[s:s + chunk], beta), logp0[s:s + chunk])
                                 for s in range(0, len(logp0), chunk)]).mean())


def kl_limit(logp0: np.ndarray, r: np.ndarray) -> float:
    """Mean KL as β -> 0: π_β puts all mass on each user's best candidates, KL = -log π0(best set)."""
    out = []
    for u in range(len(logp0)):
        ok = np.isfinite(logp0[u])
        best = ok & (r[u] == r[u][ok].max())
        out.append(-logsumexp(logp0[u][best]))
    return float(np.mean(out))


class Unreachable(ValueError):
    """A KL level the tilt cannot reach on the β bracket; `info` says why (§6.1)."""

    def __init__(self, info: dict):
        super().__init__(f"KL level {info['target']} unreachable: {info['reason']}")
        self.info = info


def bisect_beta(logp0: np.ndarray, r: np.ndarray, target: float, lo: float = 1e-6, hi: float = 1e2,
                rtol: float = 1e-3, max_iter: int = 200) -> tuple[float, float]:
    """β with mean KL within rtol·target of target; returns (β, mean KL). Raises Unreachable otherwise."""
    k_lo, k_hi = mean_kl(logp0, r, lo), mean_kl(logp0, r, hi)
    if not k_hi <= target <= k_lo:
        limit = kl_limit(logp0, r)
        reason = "theoretical limit below the level" if limit < target else (
            "level below the KL at the largest β" if target < k_hi else "not reached numerically on the bracket")
        raise Unreachable({"target": target, "kl_at_beta_lo": k_lo, "kl_at_beta_hi": k_hi, "kl_limit": limit,
                           "beta_bracket": [lo, hi], "reason": reason})
    a, b = np.log(lo), np.log(hi)
    for _ in range(max_iter):
        mid = (a + b) / 2
        k = mean_kl(logp0, r, float(np.exp(mid)))
        if abs(k - target) <= rtol * target:
            return float(np.exp(mid)), k
        if k > target:  # too far from π0: weaker tilt, larger β
            a = mid
        else:
            b = mid
    raise RuntimeError("bisection did not converge")
