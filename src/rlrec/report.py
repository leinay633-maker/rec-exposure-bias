"""Evaluation of every frozen policy on one population (docs/rlrec/PROTOCOL.md §7, §8).

Run on HOLD after the model freeze (descriptive, and a dry run of this code), then once on the random test.
Policies and their models:
  pi0            the SFT policy, one model per SFT seed
  O2  / O2p      rerank π0's top 2,000 by r / rank all candidates by r (β -> 0); O2p is seed-free
  O3             π_β per reward, KL level and SFT seed
  O4 / O4g       RLOO checkpoints (main runs) / the GRPO-style variant (secondary runs)
Counts are per user and averaged over a policy's models (the seed-mixture policy, §7); per-SFT-seed counts give
the seed values. One joint user bootstrap covers every policy.
"""
from __future__ import annotations

import numpy as np
import torch

from receval import stats

from . import analytic, config
from . import stages as S
from .evaluate import counts, row_ranks

O2_DEPTH = 2_000
HOT_GROUP = 4
TOP_LIST = 500
BASE_REWARDS = ("R1", "R2", "R3", "R4", "R5", "R6")


def policy_list() -> list[tuple]:
    out = [("pi0", None, None, None)]
    out += [("O2", r, None, None) for r in BASE_REWARDS] + [("O2p", r, None, None) for r in BASE_REWARDS]
    out += [("O3", r, lv, None) for r in S.REWARDS for lv in config.KL_LEVELS]
    out += [("O4", r, lv, "main") for r in ("R1", "R3", "R4", "R5") for lv in config.KL_LEVELS]
    out += [("O4g", r, lv, "secondary") for r in ("R1", "R3") for lv in config.KL_LEVELS]
    return out


def key(p: tuple) -> str:
    kind, reward, level, which = p
    return "/".join(str(x) for x in (kind, reward, level, which) if x is not None)


def _models(p: tuple) -> list[tuple[int, int | None]]:
    kind, _, _, which = p
    if kind == "O2p":
        return [(0, None)]
    if kind in ("O4", "O4g"):
        rl_seeds = config.SEEDS if which == "main" else (0,)
        return [(s, k) for s in config.SEEDS for k in rl_seeds]
    return [(s, None) for s in config.SEEDS]


CHUNK = 2_000


KEEP = {"pi0", "O3/R1/1.0", "O3/R3/1.0", "O3/R4/1.0", "O3/R5/1.0", "O4/R1/1.0/main", "O4/R3/1.0/main", "O4/R4/1.0/main",
        "O4/R5/1.0/main"}


def _rank(scores: np.ndarray, dev: torch.device) -> torch.Tensor:
    """Descending order per row, ties to the smaller column (= smaller video_id); -inf last."""
    return torch.sort(-torch.as_tensor(scores, device=dev), dim=1, stable=True).indices


def evaluate(pop, base, std_targets: dict, dev: torch.device, diagnostics: bool = False) -> dict:
    """Per-policy per-user counts, distribution summaries and overlaps; users are processed in chunks."""
    from .policy import Tree, exact_logprobs
    users, n = pop.users, len(pop.users)
    n_items = base.n_items
    tables = {}
    for r in S.REWARDS:
        all_users, table = S.load_table(r)
        tables[r] = (table, S.rows_of(all_users, users))
    item_group = np.load(config.ROOT / S.latest("tables")["files"]["item_group"]["file"])
    allowed = base.allowed(users)
    hist = base.histories(users)
    chunks = [(a, min(a + CHUNK, n)) for a in range(0, n, CHUNK)]
    by_user = np.argsort(pop.pos, kind="stable")
    bounds = np.searchsorted(pop.pos[by_user], [a for a, _ in chunks] + [n])
    row_slices = [by_user[bounds[i]:bounds[i + 1]] for i in range(len(chunks))]
    rows_per_user = np.bincount(pop.pos, minlength=n)
    std_rows = [std_targets.get(int(u)) for u in users]

    def reward(r: str, a: int, b: int) -> np.ndarray:
        table, idx = tables[r]
        return np.asarray(table[idx[a:b]], dtype=np.float64)

    logp0, trees = {}, {}
    for s in config.SEEDS:
        model, codes = S.load_sft(s, dev)
        trees[s] = Tree(codes)
        logp0[s] = exact_logprobs(model, codes, trees[s], hist, allowed, dev)  # float32 (users, items)
        del model
        torch.cuda.empty_cache()
    ks = tuple(sorted(set(config.KS)))
    keep_lists, results, unreachable = {}, {}, []
    betas = S.latest("o3")["betas"]

    for p in policy_list():
        kind, rname, level, which = p
        name = key(p)
        if kind == "O3" and not all(betas[f"{rname}/{s}/{level}"]["reachable"] for s in config.SEEDS):
            unreachable.append(name)
            continue
        if kind in ("O4", "O4g"):
            done = S.rl_index(which)
            if any(S.rl_name(rname, s, k, kind == "O4g") not in done for s, k in _models(p)):
                unreachable.append(name)  # a reward that reaches no KL level gets no RL runs (§6.2)
                continue
        per_model = []
        has_dist = kind in ("pi0", "O3", "O4", "O4g")
        prob_sum = np.zeros((n, n_items), dtype=np.float32) if has_dist else None  # Σ_m π_m: the mixture's support
        for s, k in _models(p):
            model = codes = checkpoint = None
            if kind in ("O4", "O4g"):
                model, codes, checkpoint = S.load_rl(S.rl_name(rname, s, k, kind == "O4g"), level, which, dev)
            c = {kk: {m: np.zeros(n) for m in ("rows", "clicks", "long_view")} for kk in ks}
            hot, covered, recall = 0.0, np.zeros(n_items, dtype=bool), []
            dist = {m: np.zeros(n) for m in ("kl_user", "proxy_R1", "proxy_R3", "w", "wc")}
            dist["w_rows"] = np.zeros(pop.rows)
            top_all = np.empty((n, TOP_LIST), dtype=np.int16) if name in KEEP else None
            for i, (a, b) in enumerate(chunks):
                lp0 = logp0[s][a:b].astype(np.float64)
                logp = None
                if kind == "pi0":
                    scores = logp = lp0
                elif kind == "O2":
                    first = _rank(lp0, dev)[:, :O2_DEPTH].cpu().numpy()
                    inside = np.zeros_like(lp0, dtype=bool)
                    np.put_along_axis(inside, first, True, 1)
                    scores = np.where(inside & np.isfinite(lp0), reward(rname, a, b), -np.inf)
                elif kind == "O2p":
                    scores = np.where(allowed[a:b], reward(rname, a, b), -np.inf)
                elif kind == "O3":
                    r_, beta = reward(rname, a, b), betas[f"{rname}/{s}/{level}"]["beta"]
                    scores = lp0 + r_ / beta  # §7, as stages.policy_scores; -inf stays -inf
                    logp = analytic.tilt(lp0, r_, beta)
                else:
                    scores = logp = exact_logprobs(model, codes, trees[s], hist[a:b], allowed[a:b], dev).astype(np.float64)
                order = _rank(scores, dev)
                rank = torch.empty_like(order)
                rank.scatter_(1, order, torch.arange(n_items, device=dev).expand(b - a, -1).contiguous())
                rows = row_slices[i]
                pos = pop.pos[rows]
                rr = rank[torch.as_tensor(pos - a, device=dev), torch.as_tensor(pop.item[rows], device=dev)].cpu().numpy()
                for kk in ks:
                    inside = (rr < kk).astype(float)
                    c[kk]["rows"] += np.bincount(pos, weights=inside, minlength=n)
                    c[kk]["clicks"] += np.bincount(pos, weights=inside * pop.click[rows], minlength=n)
                    c[kk]["long_view"] += np.bincount(pos, weights=inside * pop.long_view[rows], minlength=n)
                top = order[:, :TOP_LIST].cpu().numpy()
                hot += float((item_group[top] == HOT_GROUP).sum())
                covered[np.unique(top)] = True
                for j in range(b - a):
                    t_ = std_rows[a + j]
                    if t_ is not None:
                        recall.append(float(np.isin(t_, top[j]).mean()))
                if top_all is not None:
                    top_all[a:b] = top
                if has_dist:
                    prob = np.exp(logp)
                    w = config_pool_size(logp) * prob[pos - a, pop.item[rows]]
                    dist["kl_user"][a:b] = analytic.kl(logp, lp0)
                    dist["proxy_R1"][a:b] = (prob * reward("R1", a, b)).sum(1)
                    dist["proxy_R3"][a:b] = (prob * reward("R3", a, b)).sum(1)
                    dist["w_rows"][rows] = w
                    dist["w"] += np.bincount(pos, weights=w, minlength=n)
                    dist["wc"] += np.bincount(pos, weights=w * pop.click[rows], minlength=n)
                    prob_sum[a:b] += prob
                del order, rank
            if model is not None:
                del model
                torch.cuda.empty_cache()
            e = {"hot_share": hot / (n * TOP_LIST), "coverage": int(covered.sum()),
                 "regular_recall500": float(np.mean(recall)) if recall else float("nan")}
            if has_dist:
                e.update(dist)
            if checkpoint is not None:
                e["checkpoint"] = {x: checkpoint[x] for x in ("step", "kl", "reached", "aligned")}
            per_model.append((s, k, c, e))
            if top_all is not None:
                keep_lists[(name, s, k)] = top_all
        results[name] = _aggregate(per_model, ks, n, rows_per_user, mixture_support(prob_sum, len(per_model)) if has_dist else None)
    out = {"users": int(n), "rows": int(pop.rows), "dropped": pop.dropped, "policies": results,
           "unreachable": unreachable, "overlaps": _overlaps(keep_lists)}
    if diagnostics:  # §7: label-based calibration of the rewards on these rows, in the reference context
        y = pop.click
        out["reward_calibration"] = {}
        for r in ("R1", "R2", "R3"):
            table, idx = tables[r]
            prob = np.clip(np.asarray(table[idx[pop.pos], pop.item], dtype=np.float64), 1e-7, 1 - 1e-7)
            out["reward_calibration"][r] = {"logloss": float(-(y * np.log(prob) + (1 - y) * np.log(1 - prob)).mean()),
                                            "mean_prediction": float(prob.mean()), "click_rate": float(y.mean())}
    return out


def config_pool_size(logp: np.ndarray) -> float:
    """Nominal uniform propensity over the whole pool (§7): q = 1 / 7,583."""
    return float(logp.shape[1])


def mixture_support(prob_sum: np.ndarray, models: int, chunk: int = 2_000) -> np.ndarray:
    """1/Σπ² per user for the mixture, whose π is the models' average (§7); prob_sum holds Σ_m π_m."""
    out = np.empty(len(prob_sum))
    for a in range(0, len(prob_sum), chunk):
        out[a:a + chunk] = 1.0 / ((prob_sum[a:a + chunk].astype(np.float64) / models) ** 2).sum(1)
    return out


def _ratio(c: dict) -> float | None:
    return float(c["clicks"].sum() / c["rows"].sum()) if c["rows"].sum() else None


def _aggregate(per_model: list, ks, n: int, rows_per_user: np.ndarray, support: np.ndarray | None = None) -> dict:
    mix = {k: {m: np.mean([c[k][m] for _, _, c, _ in per_model], 0) for m in ("rows", "clicks", "long_view")} for k in ks}
    seeds = sorted({s for s, _, _, _ in per_model})
    per_seed = {s: {k: {m: np.mean([c[k][m] for s2, _, c, _ in per_model if s2 == s], 0) for m in ("rows", "clicks")}
                    for k in ks} for s in seeds}
    per_model_values = [{"sft_seed": s, "rl_seed": k, **({"checkpoint": e["checkpoint"]} if "checkpoint" in e else {}),
                         "rows": {str(kk): float(c[kk]["rows"].sum()) for kk in ks},
                         "clicks": {str(kk): float(c[kk]["clicks"].sum()) for kk in ks},
                         "V": {str(kk): _ratio(c[kk]) for kk in ks}} for s, k, c, e in per_model]
    out = {"models": len(per_model), "mix": mix, "per_seed": per_seed, "per_model": per_model_values,
           "seed_values": {str(s): {str(k): _ratio(v[k]) for k in ks} for s, v in per_seed.items()},
           "hot_share": float(np.mean([e["hot_share"] for *_, e in per_model])),
           "coverage": float(np.mean([e["coverage"] for *_, e in per_model])),
           "regular_recall500": float(np.mean([e["regular_recall500"] for *_, e in per_model]))}
    marks = [e["checkpoint"] for *_, e in per_model if "checkpoint" in e]
    if marks:  # §6.2: a level's checkpoints are reported as they are, with what they reached
        out["checkpoints"] = {"models": len(marks), "aligned": sum(m["aligned"] for m in marks),
                              "reached": sum(m["reached"] for m in marks)}
    if "kl_user" in per_model[0][3]:
        kl = np.mean([e["kl_user"] for *_, e in per_model], 0)
        out["kl"] = {"mean": float(kl.mean()), "p10": float(np.percentile(kl, 10)), "p50": float(np.percentile(kl, 50)),
                     "p90": float(np.percentile(kl, 90)),
                     "row_weighted_mean": float((kl * rows_per_user).sum() / rows_per_user.sum())}
        out["proxy"] = {r: float(np.mean([e[f"proxy_{r}"].mean() for *_, e in per_model])) for r in ("R1", "R3")}
        out["sampled"] = {m: np.mean([e[m] for *_, e in per_model], 0) for m in ("w", "wc")}
        w_mix = np.mean([e["w_rows"] for *_, e in per_model], 0)  # the mixture's π is the models' average
        out["ess_mix"] = float(w_mix.sum() ** 2 / (w_mix ** 2).sum()) if (w_mix ** 2).sum() > 0 else float("nan")
        out["support_median"] = float(np.median(support))
    return out


def _overlaps(lists: dict) -> dict:
    def mean_overlap(a, b) -> float:
        return float(np.mean([len(np.intersect1d(x, y)) / TOP_LIST for x, y in zip(a, b)]))

    out = {}
    pairs = {"H1": ("O3/R1/1.0", "pi0"), "H2": ("O3/R3/1.0", "O3/R1/1.0"), "H3": ("O3/R5/1.0", "O3/R4/1.0"),
             "H4": ("O4/R1/1.0/main", "pi0"), "H5": ("O4/R3/1.0/main", "O4/R1/1.0/main"),
             "H6": ("O4/R5/1.0/main", "O4/R4/1.0/main")}
    for h, (a, b) in pairs.items():
        vals = []
        for (name, s, k), top in lists.items():
            if name != a:
                continue
            other = lists.get((b, s, k)) if b != "pi0" else lists.get((b, s, None))
            if other is not None:
                vals.append(mean_overlap(top, other))
        out[h] = float(np.mean(vals)) if vals else None
    return out


HYPOTHESES = {"H1": ("O3/R1/1.0", "pi0", "two-sided"), "H2": ("O3/R3/1.0", "O3/R1/1.0", "greater"),
              "H3": ("O3/R5/1.0", "O3/R4/1.0", "greater"), "H4": ("O4/R1/1.0/main", "pi0", "two-sided"),
              "H5": ("O4/R3/1.0/main", "O4/R1/1.0/main", "greater"),
              "H6": ("O4/R5/1.0/main", "O4/R4/1.0/main", "greater")}
MECHANISM = {"R2 − R1": ("O3/R2/1.0", "O3/R1/1.0"), "R3 − R2": ("O3/R3/1.0", "O3/R2/1.0"),
             "R4 − R1": ("O3/R4/1.0", "O3/R1/1.0"), "R6 − R1": ("O3/R6/1.0", "O3/R1/1.0"),
             "R3（FIT 全部行）− R1": ("O3/R3_all_rows/1.0", "O3/R1/1.0"),
             "R3（实际语境拟合）− R1": ("O3/R3_actual_context/1.0", "O3/R1/1.0")}


def _summary(rep: np.ndarray, point: float, with_p: bool = False) -> dict:
    lo, hi = np.percentile(rep, [2.5, 97.5])
    out = {"point": float(point), "ci95": [float(lo), float(hi)], "se": float(np.std(rep, ddof=1))}
    return {**out, "p": stats.two_sided_p(rep)} if with_p else out


def statistics(ev: dict, k_main: int, family: list[str], status: dict | None = None, descriptive: bool = False) -> dict:
    """Bootstrap every policy's V_K (all K, both labels) and the sampled value; hypotheses at k_main. Only testable
    hypotheses of the family get a p, Holm and a verdict (§8); `descriptive` (HOLD, §7) gives none of them, and an
    untestable hypothesis keeps its point and interval, for description only (§1)."""
    pol = ev["policies"]
    names = sorted(pol)
    ks = sorted(pol[names[0]]["mix"])
    cols, index = [], {}
    for name in names:
        for k in ks:
            for m in ("rows", "clicks", "long_view"):
                index[(name, k, m)] = len(cols)
                cols.append(pol[name]["mix"][k][m])
        if "sampled" in pol[name]:
            for m in ("w", "wc"):
                index[(name, "sampled", m)] = len(cols)
                cols.append(pol[name]["sampled"][m])
    values = np.stack(cols, 1)
    rep = stats.joint_user_bootstrap(values, samples=10_000, seed=0)
    point = values.mean(0)

    def v(name, k, label="clicks"):
        a, b = index[(name, k, "rows")], index[(name, k, label)]
        return rep[:, b] / rep[:, a], point[b] / point[a]

    out = {"values": {}, "hypotheses": {}, "mechanism": {}, "o4_vs_o3": {}, "k_main": k_main, "family": family}
    for name in names:
        row = {}
        for k in ks:
            r_, p_ = v(name, k)
            row[str(k)] = _summary(r_, p_)
            rl, pl = v(name, k, "long_view")
            row[f"{k}_long_view"] = _summary(rl, pl)
        if (name, "sampled", "w") in index:
            w, wc = point[index[(name, "sampled", "w")]], point[index[(name, "sampled", "wc")]]
            row["sampled"] = {**_summary(rep[:, index[(name, "sampled", "wc")]] / rep[:, index[(name, "sampled", "w")]], wc / w),
                              "ess": pol[name]["ess_mix"], "support_median": pol[name]["support_median"]}
        out["values"][name] = row

    def diff(a, b, k, with_p: bool = False) -> dict:
        ra, pa = v(a, k)
        rb, pb = v(b, k)
        return _summary(ra - rb, pa - pb, with_p)

    def seed_values(a, b, k) -> list[float]:
        vals = []
        for s in config.SEEDS:
            sa, sb = pol[a]["per_seed"][s][k], pol[b]["per_seed"][s][k]
            vals.append(float(sa["clicks"].sum() / sa["rows"].sum() - sb["clicks"].sum() / sb["rows"].sum()))
        return vals

    mult = stats.mde_multipliers(max(len(family), 1))[0]
    rows = {}
    for h, (a, b, side) in HYPOTHESES.items():
        reasons = (status or {}).get(h, {}).get("reasons", [])
        if a not in pol or b not in pol:
            rows[h] = {"testable": False, "reasons": reasons or ["policy missing"], "in_family": h in family}
            continue
        testable = status is None or status[h]["testable"]
        d = diff(a, b, k_main, with_p=testable and h in family and not descriptive)
        per_seed = seed_values(a, b, k_main)
        d.update({"testable": testable, "side": side, "per_seed": per_seed,
                  "seed_support": int(sum(np.sign(x) == np.sign(d["point"]) for x in per_seed)),
                  "in_family": h in family, "other_k": {str(k): diff(a, b, k) for k in ks if k != k_main}})
        if not testable:
            d["reasons"] = reasons
        rows[h] = d
    # an untestable hypothesis stays in the family with p = 1 (§1): Holm keeps the family's size
    confirmed = family if not descriptive else []
    flags = stats.holm([rows[h]["p"] if rows[h].get("testable") else 1.0 for h in confirmed]) if confirmed else []
    for h, f in zip(confirmed, flags):
        r = rows[h]
        if not r.get("testable"):
            r.update({"holm_reject": False, "holds": False})
            continue
        direction = r["side"] == "two-sided" or r["point"] > 0
        r.update({"holm_reject": bool(f), "holds": bool(f and direction and r["seed_support"] >= 2),
                  "mde": float(mult * r["se"])})
    out["hypotheses"] = rows
    for label, (a, b) in MECHANISM.items():
        if a in pol and b in pol:
            out["mechanism"][label] = {str(k): diff(a, b, k) for k in ks}
    for r_ in ("R1", "R3", "R4", "R5"):
        for lv in config.KL_LEVELS:
            a, b = f"O4/{r_}/{lv}/main", f"O3/{r_}/{lv}"
            if a not in pol or b not in pol:
                continue
            o4_o3, o4_p0, o3_p0 = diff(a, b, k_main), diff(a, "pi0", k_main), diff(b, "pi0", k_main)
            entry = {"o4_minus_o3": o4_o3, "o4_minus_pi0": o4_p0, "o3_minus_pi0": o3_p0}
            if o3_p0["ci95"][0] > 0:
                entry["catch_up_ratio"] = o4_p0["point"] / o3_p0["point"]
            out["o4_vs_o3"][f"{r_}/{lv}"] = entry
    return out
