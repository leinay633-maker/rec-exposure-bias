"""Proxy run, power check and formal analysis of study 2 (protocol §5).

Every model ranks each user once (top 100 among that user's candidates); the dense, exposure-weighted and
borrowed-exposure metrics are all computed from that one list. Per-user metrics are averaged over the three
seeds, then a joint user bootstrap gives intervals, standard errors and two-sided p values. The §8 sensitivity
models stand in for the two semantic-ID models in a separate block that does not enter the H1–H3 decisions.
"""
from __future__ import annotations

import itertools
import json
import math
import time

import numpy as np
from scipy.stats import kendalltau

from receval.data import file_digest
from receval.stats import holm, joint_user_bootstrap, mde_multipliers, two_sided_p

from . import config, data, runlog, sids
from .evaluate import NAMES, borrowed, borrowed_counts, donor_draws, group_recall, per_user, slot_share, weighted
from .train import KINDS, LEARNED, context, load_model, rank, validate
from .tune import SENSITIVITY, ZOO, device

PAIRS = list(itertools.combinations(KINDS, 2))
PRIMARY = (("H1", "D", "gr_text", "sasrec", 1), ("H2", "D", "gr_text", "two_tower", 0),
           ("H3", "dense", "gr_text", "gr_shuf", 1))
SAMPLES, BOOT_SEED, ALPHA, RELOAD_TOL = 10_000, 0, 0.05, 1e-6
P_FLOOR = 2 / SAMPLES


def _json(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False, default=float)


def nanmean(x: np.ndarray, axis: int) -> np.ndarray:
    """Mean over the finite entries along an axis; NaN, without a warning, where there are none."""
    ok = np.isfinite(x)
    n = ok.sum(axis)
    total = np.where(ok, x, 0.0).sum(axis)
    return np.divide(total, n, out=np.full(np.shape(total), np.nan), where=n > 0)


def selection_file() -> tuple[dict, str]:
    path = ZOO / "selection.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    if set(selection.get("sensitivity", {})) != set(SENSITIVITY["kinds"]):
        raise RuntimeError("the selection has no §8 sensitivity models yet; run `python -m genrec sensitivity`")
    return selection, file_digest(path)


def entries_of(selection: dict, sensitivity: bool = False) -> dict:
    """(kind, seed) -> model entry; with sensitivity, the §8 models stand in for the two semantic-ID models."""
    out = {}
    for kind in LEARNED:
        use = sensitivity and kind in SENSITIVITY["kinds"]
        block = selection["sensitivity"][kind] if use else selection["models"][kind]
        for seed, e in block["seeds"].items():
            out[(kind, int(seed))] = {"kind": kind, "config": block["config"], "seed": int(seed), **e}
    return out


def reload_check(models: dict, ctx, dev) -> dict:
    """label -> registered vs reproduced validation Recall@100 of every model file the analysis uses."""
    out = {}
    for label, e in sorted(models.items()):
        model, extra = load_model(e, ctx, dev)
        got = validate(e["kind"], model, extra, ctx, dev)
        diff = abs(got - e["valid_recall@100"])
        out[label] = {"registered": e["valid_recall@100"], "reproduced": got, "abs_diff": diff}
        if not diff < RELOAD_TOL:
            raise RuntimeError(f"{label}: reloaded model gives {got}, registered {e['valid_recall@100']}")
    return out


def tops(entries: dict, hist: np.ndarray, allowed: np.ndarray, ctx, dev, seeds=config.SEEDS, kinds=KINDS,
         timing: dict | None = None) -> dict:
    """kind -> [top-100 array per seed]; popularity has no seed, so its one list stands for every seed."""
    out = {}
    for kind in kinds:
        if kind == "pop":
            out[kind] = [rank("pop", None, None, hist, allowed, ctx, dev)] * len(seeds)
            continue
        out[kind] = []
        for seed in seeds:
            model, extra = load_model(entries[(kind, seed)], ctx, dev)
            started = time.time()
            out[kind].append(rank(kind, model, extra, hist, allowed, ctx, dev))
            if timing is not None:
                timing[f"{kind}/{seed}"] = round(time.time() - started, 1)
    return out


def seed_stack(per_seed: list[dict], name: str) -> np.ndarray:
    return np.stack([m[name] for m in per_seed], -1)  # (users, seeds)


def summary(rep: np.ndarray, point: float) -> dict:
    lo, hi = np.percentile(rep, [2.5, 97.5])
    p = two_sided_p(rep)
    return {"point": float(point), "ci95": [float(lo), float(hi)], "se": float(np.std(rep, ddof=1)), "p": p,
            "p_text": "< 0.0002" if p == 0 else f"{p:.4f}"}


def seed_support(per_seed: np.ndarray, point: float) -> int:
    return int(np.sum(np.sign(per_seed) == np.sign(point)))


def block_pairs(values: np.ndarray, tests: list[str], per_seed: dict | None = None) -> dict:
    """values: (users, models, tests) seed means. Differences per test and D = first test − second, with Holm."""
    rep = joint_user_bootstrap(values, samples=SAMPLES, seed=BOOT_SEED)
    point = values.mean(0)
    idx = {k: i for i, k in enumerate(KINDS)}
    rows = []
    for a, b in PAIRS:
        row = {"pair": f"{a} − {b}"}
        for t, name in enumerate(tests):
            row[name] = summary(rep[:, idx[a], t] - rep[:, idx[b], t], point[idx[a], t] - point[idx[b], t])
        if len(tests) == 2:
            d_rep = (rep[:, idx[a], 0] - rep[:, idx[b], 0]) - (rep[:, idx[a], 1] - rep[:, idx[b], 1])
            d_pt = (point[idx[a], 0] - point[idx[b], 0]) - (point[idx[a], 1] - point[idx[b], 1])
            row["D"] = summary(d_rep, d_pt)
        if per_seed is not None:
            for col in [*tests, "D"] if len(tests) == 2 else tests:
                s_vals = per_seed[(a, b, col)]
                row[col]["seed_support"] = seed_support(s_vals, row[col]["point"])
                row[col]["per_seed"] = [float(v) for v in s_vals]
        rows.append(row)
    for col in [*tests, "D"] if len(tests) == 2 else tests:
        flags = holm([r[col]["p"] for r in rows], ALPHA)
        for r, f in zip(rows, flags):
            r[col]["holm_reject"] = bool(f)
    return {"pairs": rows, "models": {k: {t: float(point[idx[k], i]) for i, t in enumerate(tests)} for k in KINDS}}


def per_seed_contrasts(metric: dict, users_mask: np.ndarray, tests: list[str]) -> dict:
    """(a, b, column) -> per-seed contrast values on the population, columns as in block_pairs."""
    out = {}
    for a, b in PAIRS:
        diffs = {t: (metric[a][t][users_mask] - metric[b][t][users_mask]).mean(0) for t in tests}  # (seeds,)
        for t in tests:
            out[(a, b, t)] = diffs[t]
        if len(tests) == 2:
            out[(a, b, "D")] = diffs[tests[0]] - diffs[tests[1]]
    return out


def primary(pop_d: dict, pop_dense: dict) -> dict:
    rows = []
    for name, stat, a, b, direction in PRIMARY:
        block = pop_d if stat == "D" else pop_dense
        row = next(r for r in block["pairs"] if r["pair"] in (f"{a} − {b}", f"{b} − {a}"))
        sign = 1 if row["pair"] == f"{a} − {b}" else -1
        cell = dict(row["D" if stat == "D" else "dense"])
        if sign < 0:
            cell = {**cell, "point": -cell["point"], "ci95": [-cell["ci95"][1], -cell["ci95"][0]],
                    "per_seed": [-v for v in cell.get("per_seed", [])]}
        rows.append({"hypothesis": name, "statistic": f"{stat}({a} − {b})", "expected": direction, **cell})
    flags = holm([r["p"] for r in rows], ALPHA)
    mult = mde_multipliers(len(rows), ALPHA)[0]
    for r, f in zip(rows, flags):
        r["holm_reject"] = bool(f)
        r["mde_holm_first_step"] = float(mult * r["se"])
        support = int(np.sum(np.sign(r["per_seed"]) == np.sign(r["point"])))
        r["seed_support"] = support
        r["holds"] = bool(f and support >= 2 and (r["expected"] == 0 or np.sign(r["point"]) == r["expected"]))
    return {"rows": rows, "mde_multiplier": mult}


def evaluate_population(t: dict, targets: list, s, ctx, borrow: dict | None) -> dict:
    """All per-user metrics for the top lists `t` (kind -> per-seed arrays)."""
    w = s.items["w"].to_numpy()
    group_of = s.items["group"].to_numpy()
    dense_pop = np.array([len(x) > 0 for x in targets])
    out = {"dense": {}, "weighted": {}, "borrowed": {}, "groups": {}, "slots": {}}
    for kind, per_seed in t.items():
        d = [per_user(top, targets) for top in per_seed]
        wt = [weighted(top, targets, w) for top in per_seed]
        out["dense"][kind] = {n: seed_stack(d, n) for n in NAMES}
        out["weighted"][kind] = {n: seed_stack(wt, n) for n in NAMES}
        if borrow is not None:
            b = [borrowed(top, targets, borrow["rows"], borrow["assign"], borrow["masks"], ctx.col_of) for top in per_seed]
            out["borrowed"][kind] = {n: np.stack([x[n] for x in b], -1) for n in NAMES}  # (users, draws, seeds)
        out["groups"][kind] = np.stack([group_recall(top, targets, group_of) for top in per_seed], -1)
        out["slots"][kind] = np.mean([slot_share(top[dense_pop], group_of) for top in per_seed], 0).tolist()
    return out


def hypothesis_blocks(m: dict, dense_pop: np.ndarray, d_pop: np.ndarray, names=NAMES) -> dict:
    """Per metric: the 10 pairs on the H1 population (dense, weighted, D), on the H3 population (dense), H1–H3."""
    metrics = {}
    for name in names:
        dense = {k: m["dense"][k][name] for k in KINDS}
        wt = {k: m["weighted"][k][name] for k in KINDS}
        both = np.stack([np.stack([dense[k][d_pop].mean(-1), wt[k][d_pop].mean(-1)], -1) for k in KINDS], 1)
        only = np.stack([dense[k][dense_pop].mean(-1)[:, None] for k in KINDS], 1)
        metric = {k: {"dense": dense[k], "weighted": wt[k]} for k in KINDS}
        block_d = block_pairs(both, ["dense", "weighted"], per_seed_contrasts(metric, d_pop, ["dense", "weighted"]))
        block_dense = block_pairs(only, ["dense"], per_seed_contrasts(metric, dense_pop, ["dense"]))
        metrics[name] = {"weighted_population": block_d, "dense_population": block_dense,
                         "primary": primary(block_d, block_dense)}
    return metrics


def borrow_inputs(s, users: list) -> dict:
    small = s.users["small"]
    donors = s.users["donors"]
    vexp = data.load("valid_exposed_pool")
    col = {int(v): j for j, v in enumerate(s.pool)}
    masks = np.zeros((len(donors), len(s.pool)), dtype=bool)
    pos = {u: i for i, u in enumerate(donors)}
    for u, v in zip(vexp["user_id"], vexp["video_id"]):
        masks[pos[int(u)], col[int(v)]] = True
    row_of = {u: i for i, u in enumerate(small)}
    return {"assign": donor_draws(len(small), len(donors)), "masks": masks,
            "rows": np.asarray([row_of[u] for u in users])}


def analyse(entries: dict, sens: dict, users: list, dense_rows, natural: dict, ctx, dev, label: str) -> dict:
    s = ctx.s
    targets = data.dense_targets(dense_rows, users)
    allowed = data.labelled(data.load("small_pairs"), users, s.pool)
    hist = data.histories(s, users)
    borrow = borrow_inputs(s, users)
    timing: dict = {}
    t = tops(entries, hist, allowed, ctx, dev, timing=timing)
    m = evaluate_population(t, targets, s, ctx, borrow)
    w = s.items["w"].to_numpy()
    n_t = np.array([len(x) for x in targets])
    w_t = np.array([w[x].sum() if len(x) else 0 for x in targets])
    dense_pop = n_t > 0
    d_pop = dense_pop & (w_t > 0)
    counts = borrowed_counts(targets, borrow["rows"], borrow["assign"], borrow["masks"], ctx.col_of)
    draws_used = (counts > 0).sum(1)  # the draws used depend on the targets only, not on the model
    b_pop = d_pop & (draws_used > 0)
    if not (dense_pop.any() and d_pop.any() and b_pop.any()):
        raise RuntimeError(f"{label}: an analysis population is empty (§6 step 2)")
    per_draw = counts.sum(1) / np.maximum(draws_used, 1)  # mean |T_u ∩ observed| over the used draws
    result = {"label": label, "users": len(users),
              "population": {"dense": int(dense_pop.sum()), "weighted_and_D": int(d_pop.sum()), "borrowed": int(b_pop.sum()),
                             "excluded_no_target": int((~dense_pop).sum()), "excluded_zero_weight": int((dense_pop & ~d_pop).sum()),
                             "excluded_no_borrowed_draw": int((d_pop & ~b_pop).sum()),
                             "targets_per_user": np.percentile(n_t[dense_pop], [10, 50, 90]).tolist(),
                             "borrowed_targets_per_user": np.percentile(per_draw[b_pop], [10, 50, 90]).tolist(),
                             "mean_borrowed_draws_used": float(draws_used[b_pop].mean())}}
    metrics = hypothesis_blocks(m, dense_pop, d_pop)
    result["primary"] = metrics["recall@100"]["primary"]
    result["metrics"] = metrics
    result["borrowed"] = borrowed_block(m, b_pop)
    result["groups"] = {k: {"recall_by_group": nanmean(nanmean(m["groups"][k], -1)[dense_pop], 0).tolist(),
                            "users_with_group_targets": np.sum(np.isfinite(m["groups"][k][dense_pop][..., 0]), 0).tolist(),
                            "slot_share": m["slots"][k]} for k in KINDS}
    result["natural"] = natural_block(entries, natural, ctx, dev)
    vecs = {"dense": [metrics["recall@100"]["dense_population"]["models"][k]["dense"] for k in KINDS],
            "weighted": [metrics["recall@100"]["weighted_population"]["models"][k]["weighted"] for k in KINDS],
            "borrowed": [result["borrowed"]["models"][k]["borrowed"] for k in KINDS],
            "natural": [result["natural"]["models"][k]["natural"] for k in KINDS]}
    result["kendall_tau"] = {f"{a} vs {b}": float(kendalltau(vecs[a], vecs[b]).statistic)
                             for a, b in itertools.combinations(vecs, 2)}
    sens_timing: dict = {}
    t_sens = {**t, **tops(sens, hist, allowed, ctx, dev, kinds=SENSITIVITY["kinds"], timing=sens_timing)}
    result["sensitivity"] = sensitivity_block(evaluate_population(t_sens, targets, s, ctx, None), dense_pop, d_pop)
    result["ranking_seconds"] = {**timing, **{f"{k}/sensitivity": v for k, v in sens_timing.items()}}
    return result


def sensitivity_block(m: dict, dense_pop: np.ndarray, d_pop: np.ndarray) -> dict:
    """§8: H1–H3 with the sensitivity models standing in; point, interval, p and seed support, no decision."""
    name = "recall@100"
    blocks = hypothesis_blocks(m, dense_pop, d_pop, names=(name,))[name]
    keep = ("hypothesis", "statistic", "expected", "point", "ci95", "se", "p", "p_text", "per_seed", "seed_support")
    return {"note": "§8 sensitivity analysis; not part of the H1–H3 decisions",
            "rows": [{k: r[k] for k in keep} for r in blocks["primary"]["rows"]],
            "models": {k: {"dense_h3_population": blocks["dense_population"]["models"][k]["dense"],
                           **blocks["weighted_population"]["models"][k]} for k in SENSITIVITY["kinds"]}}


def borrowed_block(m: dict, b_pop: np.ndarray) -> dict:
    """Dense vs borrowed-exposure differences on the borrowed population, plus the Monte Carlo error of 20 draws."""
    name = "recall@100"
    per_user_b = {k: nanmean(m["borrowed"][k][name], 1) for k in KINDS}  # (users, seeds) mean over used draws
    dense = {k: m["dense"][k][name] for k in KINDS}
    values = np.stack([np.stack([dense[k][b_pop].mean(-1), per_user_b[k][b_pop].mean(-1)], -1) for k in KINDS], 1)
    metric = {k: {"dense": dense[k], "borrowed": per_user_b[k]} for k in KINDS}
    block = block_pairs(values, ["dense", "borrowed"], per_seed_contrasts(metric, b_pop, ["dense", "borrowed"]))

    def mc_se(y: np.ndarray) -> float:
        """sqrt(sum over users with >= 2 used draws of s_u^2 / m_u) / population size (protocol §5.3)."""
        used = np.isfinite(y)
        var = [np.var(row[ok], ddof=1) / ok.sum() for row, ok in zip(y, used) if ok.sum() >= 2]
        return float(math.sqrt(sum(var)) / len(y))

    block["monte_carlo_se"] = {
        "models": {k: mc_se(m["borrowed"][k][name].mean(-1)[b_pop]) for k in KINDS},
        "pairs": {f"{a} − {b}": mc_se((m["borrowed"][a][name] - m["borrowed"][b][name]).mean(-1)[b_pop]) for a, b in PAIRS}}
    return block


def natural_block(entries: dict, natural: dict, ctx, dev) -> dict:
    s = ctx.s
    targets = data.targets_from(natural["frame"], natural["users"])
    keep = [i for i, x in enumerate(targets) if len(x)]
    users = [natural["users"][i] for i in keep]
    targets = [targets[i] for i in keep]
    if not users:
        raise RuntimeError("the natural-test population is empty (§6 step 2)")
    hist = data.histories(s, users)
    t = tops(entries, hist, data.allowed(s, users), ctx, dev)
    metric = {k: {"natural": np.stack([per_user(top, targets)["recall@100"] for top in t[k]], -1)} for k in KINDS}
    vals = np.stack([metric[k]["natural"].mean(-1) for k in KINDS], 1)[:, :, None]
    block = block_pairs(vals, ["natural"], per_seed_contrasts(metric, np.ones(len(users), dtype=bool), ["natural"]))
    others = len(s.users["others"])
    block.update({"label": natural["label"], "users": len(users), "other_users": others,
                  "excluded": others - len(users), "steps": natural.get("steps")})
    return block


def natural_targets(frame, s) -> tuple:
    """Natural-window targets: positive pool videos of other users not exposed to them in training (§2.2)."""
    others = set(s.users["others"])
    rows = frame[frame["user_id"].isin(others)]
    pos = rows[data.label(rows) & rows["video_id"].isin(set(s.pool.tolist()))][["user_id", "video_id"]].drop_duplicates()
    steps = {"other_users_in_window": int(rows["user_id"].nunique()), "with_positive_pool_row": int(pos["user_id"].nunique())}
    seen = {(int(u), int(v)) for u, vs in s.exposed.items() for v in vs}
    pos = pos[[(int(u), int(v)) not in seen for u, v in zip(pos["user_id"], pos["video_id"])]]
    steps["with_unexposed_target"] = int(pos["user_id"].nunique())
    pos = pos[pos["user_id"].isin(set(s.events))]
    steps["with_positive_event_in_training"] = int(pos["user_id"].nunique())
    return pos, steps


def run(proxy: bool) -> dict:
    """The proxy run (proxy users, validation window as the natural stand-in) or the formal analysis."""
    from .freeze import fingerprint, load_manifest

    dev = device()
    kind = "analysis-proxy" if proxy else "analysis"
    with runlog.logged(kind) as state:
        ctx = context()
        s = ctx.s
        if proxy:
            selection, sel_sha = selection_file()
            inputs_sha = fingerprint()
            users, rows = s.users["proxy"], data.load("proxy_dense")
            valid = data.load("valid_targets")
            natural = {"users": sorted(valid["user_id"].unique().tolist()), "frame": valid,
                       "label": "validation window (code-path check on the proxy run)"}
        else:
            manifest, man_sha = load_manifest()
            selection, sel_sha, inputs_sha = manifest["selection"], manifest["selection_sha256"], manifest["inputs_sha256"]
            state["manifest_sha256"] = man_sha
            users, rows = s.users["test"], data.load("dense_test")
            frame, steps = natural_targets(data.load("natural_test"), s)
            natural = {"users": sorted(frame["user_id"].unique().tolist()), "frame": frame,
                       "label": "natural test 9/01–9/05", "steps": steps}
        state.update({"selection_sha256": sel_sha, "inputs_sha256": inputs_sha})
        out_dir = config.RUNS / kind / state["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        entries, sens = entries_of(selection), entries_of(selection, sensitivity=True)
        models = {f"{k}/{seed}": e for (k, seed), e in entries.items()}
        models.update({f"{k}/{seed}/sensitivity": sens[(k, seed)] for k in SENSITIVITY["kinds"] for seed in config.SEEDS})
        result = {"selection_sha256": sel_sha, "inputs_sha256": inputs_sha, "reload_check": reload_check(models, ctx, dev)}
        result.update(analyse(entries, sens, users, rows, natural, ctx, dev, kind))
        result["diagnostics"] = diagnostics(selection)
        (out_dir / "result.json").write_text(_json(result), encoding="utf-8")
        state["output"] = str(out_dir / "result.json")
    return result


def diagnostics(selection: dict) -> dict:
    """§5.3: codes used per level and three-code collisions; parameters, epochs and training time per model."""
    codes = {}
    for seed in config.SEEDS:
        meta = json.loads(sids.path(seed)[1].read_text(encoding="utf-8"))
        codes[str(seed)] = {k: meta[k] for k in ("codes_used_per_level", "three_code_collisions", "fourth_code_max",
                                                 "retried_with_double_steps")}
    models = {}
    for group, blocks in (("main", selection["models"]), ("sensitivity", selection["sensitivity"])):
        for kind, block in blocks.items():
            for seed, e in block["seeds"].items():
                models[f"{kind}/{seed}" + ("/sensitivity" if group == "sensitivity" else "")] = {
                    k: e[k] for k in ("params", "epochs_run", "best_epoch", "seconds", "batch")}
    return {"semantic_ids": codes, "models": models}


def power_check() -> dict:
    """Standard errors of the three primary statistics on proxy users with seed-0 models; no point estimates."""
    from .freeze import fingerprint

    dev = device()
    with runlog.logged("power-check") as state:
        ctx = context()
        s = ctx.s
        selection, sel_sha = selection_file()
        state.update({"selection_sha256": sel_sha, "inputs_sha256": fingerprint()})
        entries = entries_of(selection)
        users = s.users["proxy"]
        targets = data.dense_targets(data.load("proxy_dense"), users)
        allowed = data.labelled(data.load("small_pairs"), users, s.pool)
        hist = data.histories(s, users)
        t = tops(entries, hist, allowed, ctx, dev, seeds=(0,))
        w = s.items["w"].to_numpy()
        n_t = np.array([len(x) for x in targets])
        w_t = np.array([w[x].sum() if len(x) else 0 for x in targets])
        dense_pop, d_pop = n_t > 0, (n_t > 0) & (w_t > 0)
        if not (dense_pop.any() and d_pop.any()):
            raise RuntimeError("power check: an analysis population is empty")
        r = {k: (per_user(t[k][0], targets)["recall@100"], weighted(t[k][0], targets, w)["recall@100"]) for k in KINDS}
        idx = {k: i for i, k in enumerate(KINDS)}
        both = np.stack([np.stack([r[k][0][d_pop], r[k][1][d_pop]], -1) for k in KINDS], 1)
        only = np.stack([r[k][0][dense_pop][:, None] for k in KINDS], 1)
        rep_d = joint_user_bootstrap(both, samples=SAMPLES, seed=BOOT_SEED)
        rep_o = joint_user_bootstrap(only, samples=SAMPLES, seed=BOOT_SEED)
        scale = math.sqrt(len(s.users["proxy"]) / len(s.users["test"]))
        mult = mde_multipliers(3, ALPHA)[0]
        out = {}
        for name, stat, a, b, _ in PRIMARY:
            if stat == "D":
                rep = (rep_d[:, idx[a], 0] - rep_d[:, idx[b], 0]) - (rep_d[:, idx[a], 1] - rep_d[:, idx[b], 1])
            else:
                rep = rep_o[:, idx[a], 0] - rep_o[:, idx[b], 0]
            se = float(np.std(rep, ddof=1))
            if not (math.isfinite(se) and se > 0):
                raise RuntimeError(f"power check: the standard error of {name} is not finite and positive")
            out[name] = {"se_proxy": se, "se_scaled_to_test": se * scale, "mde_holm_first_step": mult * se * scale}
        result = {"users": {"proxy_assigned": len(users), "dense": int(dense_pop.sum()), "weighted_and_D": int(d_pop.sum())},
                  "scale": scale, "multiplier": mult, "standard_errors": out}
        out_dir = config.RUNS / "power-check" / state["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "power.json").write_text(_json(result), encoding="utf-8")
        state.update({"output": str(out_dir / "power.json"), "standard_errors": out})
    return result
