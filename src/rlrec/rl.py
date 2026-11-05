"""RLOO with the KL folded into the reward (docs/rlrec/PROTOCOL.md §6.2).

For S draws i_1..i_S of one user, r'_j = r_j − λ·[log π_θ(i_j) − log π0(i_j)] (no gradient through the bracket) and
A_j = r'_j − mean of the other S − 1 values. The loss −mean_j(A_j · log π_θ(i_j)) then has the gradient of
−(E_π[r] − λ·KL(π‖π0)) in expectation, whose maximiser is π_β with β = λ. The GRPO-style variant divides A by the
group's standard deviation of r'.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import analytic, policy


def shaped_reward(r: torch.Tensor, logp: torch.Tensor, logp0: torch.Tensor, lam: float) -> torch.Tensor:
    return r - lam * (logp.detach() - logp0)


def advantages(shaped: torch.Tensor, normalise: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """Leave-one-out advantages along the last axis (the S draws of a user)."""
    s = shaped.shape[-1]
    if s < 2:
        raise ValueError("RLOO needs at least two draws per user")
    adv = shaped - (shaped.sum(-1, keepdim=True) - shaped) / (s - 1)
    if normalise:
        adv = adv / (shaped.std(-1, keepdim=True) + eps)
    return adv


def rloo_loss(logp: torch.Tensor, r: torch.Tensor, logp0: torch.Tensor, lam: float,
              normalise: bool = False) -> torch.Tensor:
    """logp: (users, S) log π_θ of the draws, with gradient; r, logp0: (users, S) without."""
    adv = advantages(shaped_reward(r, logp, logp0, lam), normalise).detach()
    return -(adv * logp).mean()


USERS_PER_STEP, SAMPLES, MAX_STEPS, CLIP = 256, 8, 3_000, 1.0
CHECK_EVERY, FINE_EVERY, NEAR, TOLERANCE = 100, 10, 0.85, 0.05  # §6.2
EVERY_STEP_UNTIL = 1.0  # §12.3: before the main level the KL rises up to 0.06 a step, the width of its [1, 1.05] band


@dataclass
class RLData:
    hist: np.ndarray  # (training users, HISTORY) pool rows, -1 padded
    allowed: np.ndarray  # (training users, n_items) bool
    rewards: np.ndarray  # (training users, n_items) float32, rows aligned with hist
    kl_hist: np.ndarray  # the KL subset (§6.1)
    kl_allowed: np.ndarray
    kl_logp0: np.ndarray  # exact log π0 on the KL subset


def schedule(step: int, kl: float, ahead: list, fine_for, check_every: int = CHECK_EVERY,
             fine_every: int = FINE_EVERY, every_step_until: float | None = None) -> tuple[int, float | None]:
    """§6.2: the step of the next KL check, and the level being approached in fine mode (None when coarse).
    Fine mode starts once the KL reaches NEAR of the next level not yet passed and lasts until that level is
    passed, even if the KL falls back in between. While `every_step_until` is not yet passed, every step is checked
    (§12.3)."""
    if fine_for is not None and fine_for not in ahead:
        fine_for = None
    if fine_for is None and ahead and kl >= NEAR * min(ahead):
        fine_for = min(ahead)
    if every_step_until in ahead:
        return step + 1, fine_for
    return step + (fine_every if fine_for is not None else check_every), fine_for


def user_batches(n: int, size: int, rng: np.random.Generator):
    """Endless sorted batches of users (§6.2): each round is one shuffle of all n users, read without replacement; a
    round's tail is completed by the head of the next round, so nobody is skipped."""
    order = rng.permutation(n)
    while True:
        while len(order) < size:
            order = np.concatenate([order, rng.permutation(n)])
        yield np.sort(order[:size])
        order = order[size:]


def max_rise(log: list[dict], window: int = 100) -> float:
    """Largest KL rise over any `window` steps (§12.2), the KL read off linearly between checks, from 0 at step 0."""
    steps = np.array([0] + [row["step"] for row in log], dtype=float)
    line = np.interp(np.arange(steps[-1] + 1), steps, [0.0] + [row["subset_kl"] for row in log])
    k = min(window, len(line) - 1)
    return float((line[k:] - line[:-k]).max())


def max_slope(log: list[dict]) -> float:
    """Largest KL rise per step between two checks (§12.2 sets the check intervals from it), from 0 at step 0."""
    steps = np.array([0] + [row["step"] for row in log], dtype=float)
    kl = np.array([0.0] + [row["subset_kl"] for row in log])
    return float((np.diff(kl) / np.diff(steps)).max())


def subset_kl(model, codes, tree, t, data: RLData, dev) -> float:
    logp = policy.exact_logprobs(model, codes, tree, data.kl_hist, data.kl_allowed, dev, t=t)
    return float(analytic.kl(logp, data.kl_logp0).mean())


def train(model, ref, codes, tree, data: RLData, lam: float, lr: float, seed: int, dev,
          levels=(0.5, 1.0, 2.0), normalise: bool = False, max_steps: int = MAX_STEPS,
          check_every: int = CHECK_EVERY, fine_every: int = FINE_EVERY, users_per_step: int = USERS_PER_STEP,
          samples: int = SAMPLES, every_step_until: float | None = None) -> dict:
    """RLOO from the SFT weights in `model` (eval mode throughout, so dropout is off). The subset KL is checked every
    `check_every` steps, and every `fine_every` steps once it reaches NEAR of the next level not yet passed; every step
    from the first until `every_step_until` is passed, when given. Returns, per KL level, the first checked state whose
    KL reaches it (or the largest-KL state, marked not reached), and the log."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    model.eval()
    ref.eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    t = tree.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    batches = user_batches(len(data.hist), users_per_step, rng)
    log, selected, best = [], {}, {"kl": -np.inf}
    first_check = 1 if every_step_until in levels else check_every
    started, next_check, fine_for = time.time(), first_check, None  # fine_for: the level being approached
    for step in range(1, max_steps + 1):
        idx = next(batches)
        hist, allowed = data.hist[idx], data.allowed[idx]
        items_np, _ = policy.sample(model, codes, tree, hist, allowed, samples, dev, gen, t=t)
        al = policy.alive(tree, t, torch.as_tensor(allowed, device=dev))
        ht = torch.as_tensor(codes.window_tokens(hist), device=dev)
        items = torch.as_tensor(items_np, device=dev)
        with torch.no_grad():
            lp0 = policy.sequence_logprob(ref, codes, tree, t, ht, al, items)
        lp = policy.sequence_logprob(model, codes, tree, t, ht, al, items)
        r = torch.as_tensor(data.rewards[idx[:, None], items_np], dtype=torch.float32, device=dev)
        loss = rloo_loss(lp, r, lp0, lam, normalise)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite RL loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if step == next_check or step == max_steps:
            kl = subset_kl(model, codes, tree, t, data, dev)
            row = {"step": step, "subset_kl": kl, "reward": float(r.mean()), "sample_kl": float((lp.detach() - lp0).mean()),
                   "loss": float(loss.detach()), "seconds": round(time.time() - started, 1)}
            log.append(row)
            state = None
            if kl > best["kl"]:
                state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                best = {"kl": kl, "step": step, "state": state}
            for level in levels:
                if level not in selected and kl >= level:
                    state = state or copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                    selected[level] = {"step": step, "kl": kl, "reached": True,
                                       "aligned": kl <= (1 + TOLERANCE) * level, "state": state}
            if kl >= max(levels):
                break
            ahead = [level for level in levels if level not in selected]
            next_check, fine_for = schedule(step, kl, ahead, fine_for, check_every, fine_every, every_step_until)
    for level in levels:
        if level not in selected:
            selected[level] = {"step": best["step"], "kl": best["kl"], "reached": False, "aligned": False,
                               "state": best["state"]}
    return {"selected": selected, "log": log, "steps": log[-1]["step"] if log else 0}
