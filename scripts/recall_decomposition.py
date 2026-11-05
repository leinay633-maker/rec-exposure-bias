"""Exploratory recall decomposition on the opened test pair (protocol §20).

Absolute Recall@100 gaps from the standard and the random test sit on different target distributions, so their
shrinkage is not by itself a shrinkage of model differences. This script re-scores the four frozen retrievers on R*
and reports per-user denominators and random baselines, relative lifts, the click rate of a "recommend uniformly
from the top K" policy estimated on random-exposure rows (SNIPS under nominal uniform logging), and a popularity
decomposition of each pair's gap. The test pair was opened on 2026-10-30, so everything here is exploratory and
changes no protocol decision.

Usage: python scripts/recall_decomposition.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from receval import analysis as A
from receval import calibrate as C
from receval import config, data, registry, stats
from receval import orchestrate as O
from receval.pipeline import prepare
from receval.retrieval import Pool, itemcf_scorer, popularity_scorer, retriever_scorer
from receval.train import device

MODELS = ("recall_popularity", "recall_itemcf", "recall_two_tower", "recall_two_tower_logq")
SEEDED = ("recall_two_tower", "recall_two_tower_logq")
SETS = ("standard", "random")
KS = (50, 100, 200)
K_MAIN = 100
GROUPS = C.GROUPS
PAIRS = [(a, b) for i, a in enumerate(MODELS) for b in MODELS[i + 1:]]  # same 6 pairs and order as RESULTS §4
FORMAL_RECALL = config.ROOT / "evidence/analysis/20261031-121508-analysis-926a49/recall.json"
TOLERANCE = 1e-9
SAMPLES = 10_000
NOT_IN_TOP = np.iinfo(np.int16).max


def rank_matrix(score_fn, users: list[int], pool: Pool, depth: int = max(KS),
                chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Top `depth` pool rows per user, and the (users, pool) rank of each of them (0 = best; NOT_IN_TOP elsewhere).

    Same ordering as retrieval.recall_ndcg: training-seen videos excluded, stable descending sort.
    """
    ranks = np.full((len(users), len(pool.video_ids)), NOT_IN_TOP, dtype=np.int16)
    tops = np.empty((len(users), depth), dtype=np.int64)
    positions = np.arange(depth, dtype=np.int16)
    for start in range(0, len(users), chunk):
        block = users[start:start + chunk]
        scores = score_fn(block).clone()
        for row, user in enumerate(block):
            seen = pool.seen.get(user)
            if seen is not None and len(seen):
                scores[row, torch.as_tensor(seen, device=scores.device)] = -torch.inf
        top = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :depth].cpu().numpy()
        rows = np.arange(start, start + len(block))[:, None]
        ranks[rows, top] = positions[None, :]
        tops[start:start + len(block)] = top
    return ranks, tops


def target_table(targets: dict[int, np.ndarray], users: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flattened (user position, pool row) of every target, and each user's target count."""
    pos, item, count = [], [], np.zeros(len(users))
    for i, u in enumerate(users):
        t = targets[u]
        pos.append(np.full(len(t), i))
        item.append(t)
        count[i] = len(t)
    return np.concatenate(pos), np.concatenate(item), count


def row_table(frame, users: list[int], pool: Pool, seen_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Test rows of R* users whose video is not in the user's training-seen set: (user position, pool row, click)."""
    at = {u: i for i, u in enumerate(users)}
    part = frame.loc[frame["user_id"].isin(at), ["user_id", "video_id", "is_click"]]
    pos = part["user_id"].map(at).to_numpy(np.int64)
    item = part["video_id"].map(pool.row_of).to_numpy()
    keep = ~np.isnan(item.astype(float))
    pos, item, click = pos[keep], item[keep].astype(np.int64), part["is_click"].to_numpy(float)[keep]
    eligible = ~seen_mask[pos, item]
    return {"pos": pos[eligible], "item": item[eligible], "click": click[eligible],
            "dropped_unmapped": int((~keep).sum()), "dropped_seen": int((~eligible).sum())}


def per_user_block(ranked: tuple[np.ndarray, np.ndarray], n: int, tables: list, rows: list,
                   group_of_row: np.ndarray) -> dict:
    """Per-user arrays for one model and seed: recall, hit groups, top-K slot groups, rows and clicks in the top K."""
    ranks, tops = ranked
    out = {}
    for s, (t_pos, t_item, t_count) in enumerate(tables):
        t_rank = ranks[t_pos, t_item]
        for k in KS:
            hit = (t_rank < k).astype(float)
            out[("recall", s, k)] = np.bincount(t_pos, weights=hit, minlength=n) / t_count
            r = rows[s]
            inside = (ranks[r["pos"], r["item"]] < k).astype(float)
            out[("rows_in", s, k)] = np.bincount(r["pos"], weights=inside, minlength=n)
            out[("clicks_in", s, k)] = np.bincount(r["pos"], weights=inside * r["click"], minlength=n)
        hit = (t_rank < K_MAIN).astype(float)
        cell = t_pos * GROUPS + group_of_row[t_item]
        out[("hit_group", s)] = (np.bincount(cell, weights=hit, minlength=n * GROUPS).reshape(n, GROUPS)
                                 / t_count[:, None])
    slot_groups = group_of_row[tops[:, :K_MAIN]]
    out[("slots",)] = np.stack([(slot_groups == g).mean(axis=1) for g in range(GROUPS)], axis=1)
    return out


def averaged(blocks: list[dict]) -> dict:
    return {key: np.mean([b[key] for b in blocks], axis=0) for key in blocks[0]}


class Columns:
    """Packs named per-user arrays into one (users, columns) matrix for a single joint bootstrap."""

    def __init__(self):
        self.parts, self.index, self.width = [], {}, 0

    def add(self, name, values: np.ndarray) -> None:
        values = np.asarray(values, float)
        flat = values.reshape(values.shape[0], -1)
        self.index[name] = (self.width, self.width + flat.shape[1], values.shape[1:])
        self.parts.append(flat)
        self.width += flat.shape[1]

    def matrix(self) -> np.ndarray:
        return np.concatenate(self.parts, axis=1)

    def get(self, means: np.ndarray, name) -> np.ndarray:
        """means: (..., columns) -> (..., *shape) for one name."""
        lo, hi, shape = self.index[name]
        return means[..., lo:hi].reshape(means.shape[:-1] + shape)


def summary(point: float, reps: np.ndarray, test: bool = True) -> dict:
    """Point, percentile interval and standard error; a two-sided p against 0 only for differences (test=True).

    Ratios and rates have no meaningful test against 0, so they carry no p (protocol §21).
    """
    reps = np.asarray(reps, float)
    out = {"point": float(point), "ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
           "se": float(np.std(reps, ddof=1))}
    if test:
        out["p"] = stats.two_sided_p(reps)
    return out


def distribution(values: np.ndarray) -> dict:
    v = np.asarray(values, float)
    return {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90))}


def holm_flags(blocks: list[dict]) -> None:
    for flag, b in zip(stats.holm([b["p"] for b in blocks]), blocks):
        b["holm_significant"] = bool(flag)


def analyse(cols: Columns, mat: np.ndarray, n_users: int) -> dict:
    point = mat.mean(axis=0)
    reps = stats.joint_user_bootstrap(mat, samples=SAMPLES, seed=0)
    both = (point, reps)  # every statistic is computed on the point estimate and on each replicate

    def get(name):
        return tuple(cols.get(x, name) for x in both)

    out = {"recall": {}, "relative": {}, "policy": {}, "decomposition": {}}
    # recall and baseline-relative recall, per model, set and K
    for m in MODELS:
        out["recall"][m] = {f"recall@{k}": [float(cols.get(point, (m, "recall", s, k))) for s in range(2)] for k in KS}

    # 2. relative lifts and their change from standard to random
    for k in KS:
        rows = []
        for a, b in PAIRS:
            entry = {"pair": [a, b]}
            for kind in ("raw", "lift"):
                ratio = []
                for s in range(2):
                    if kind == "raw":
                        num, den = get((a, "recall", s, k)), get((b, "recall", s, k))
                    else:
                        num = tuple(x - 1 for x in get((a, "over_base", s, k)))
                        den = tuple(x - 1 for x in get((b, "over_base", s, k)))
                    ratio.append(tuple(nu / de for nu, de in zip(num, den)))
                entry[kind] = {name: summary(r[0], r[1], test=False) for name, r in zip(SETS, ratio)}
                change = tuple(np.log(r_rand) - np.log(r_std) for r_std, r_rand in zip(*ratio))
                entry[kind]["log_ratio_change"] = summary(change[0], change[1])
            rows.append(entry)
        if k == K_MAIN:
            holm_flags([e["raw"]["log_ratio_change"] for e in rows])
        out["relative"][f"k{k}"] = rows

    # 3. click rate of "recommend uniformly from the top K" on test rows (SNIPS, nominal uniform logging)
    for s, name in enumerate(SETS):
        base_rows, base_clicks = get(("base_rows", s)), get(("base_clicks", s))
        base_rate = tuple(c / r for c, r in zip(base_clicks, base_rows))
        block = {"base_click_rate": summary(base_rate[0], base_rate[1], test=False),
                 "eligible_rows": float(base_rows[0] * n_users), "by_k": {}}
        for k in KS:
            models, rates = {}, {}
            for m in MODELS:
                rows_in, clicks_in = get((m, "rows_in", s, k)), get((m, "clicks_in", s, k))
                rate = tuple(c / r for c, r in zip(clicks_in, rows_in))
                rates[m] = rate
                models[m] = {"rows_in_top_k": float(rows_in[0] * n_users), "click_rate": summary(rate[0], rate[1], test=False),
                             "lift_over_base": summary(rate[0] / base_rate[0], rate[1] / base_rate[1], test=False)}
            pairs = []
            for a, b in PAIRS:
                diff = tuple(x - y for x, y in zip(rates[a], rates[b]))
                ratio = tuple(x / y for x, y in zip(rates[a], rates[b]))
                pairs.append({"pair": [a, b], "difference": summary(diff[0], diff[1]),
                              "ratio": summary(ratio[0], ratio[1], test=False)})
            if k == K_MAIN and name == "random":
                holm_flags([p["difference"] for p in pairs])
            block["by_k"][f"k{k}"] = {"models": models, "pairs": pairs}
        out["policy"][name] = block

    # 4. popularity decomposition at K = 100: R = sum_g W_g R_g on each set
    weights = [get(("target_share", s)) for s in range(2)]  # each: (point (G,), reps (S, G))
    contrib = {m: [get((m, "hit_group", s)) for s in range(2)] for m in MODELS}
    support = [bool(weights[0][0][g] > 0 and weights[1][0][g] > 0) for g in range(GROUPS)]
    sup = np.array(support)
    dec = {"target_share": {name: [float(x) for x in weights[s][0]] for s, name in enumerate(SETS)},
           "common_support_groups": [g for g in range(GROUPS) if support[g]],
           "group_recall": {}, "slot_share": {}, "pairs": []}
    reweighted = {}
    for m in MODELS:
        dec["group_recall"][m] = {name: [float(c / w) if w > 0 else None
                                         for c, w in zip(contrib[m][s][0], weights[s][0])]
                                  for s, name in enumerate(SETS)}
        dec["slot_share"][m] = [float(x) for x in cols.get(point, (m, "slots"))]
        # standard-test group recall re-weighted with the random test's target shares, on common support
        reweighted[m] = tuple(
            (w_rand[..., sup] / w_rand[..., sup].sum(-1, keepdims=True) * (c_std[..., sup] / w_std[..., sup])).sum(-1)
            for w_std, w_rand, c_std in zip(weights[0], weights[1], contrib[m][0]))
    dec["reweighted_standard_recall"] = {m: summary(reweighted[m][0], reweighted[m][1], test=False) for m in MODELS}
    for a, b in PAIRS:
        std = tuple(x - y for x, y in zip(get((a, "recall", 0, K_MAIN)), get((b, "recall", 0, K_MAIN))))
        rand = tuple(x - y for x, y in zip(get((a, "recall", 1, K_MAIN)), get((b, "recall", 1, K_MAIN))))
        rew = tuple(x - y for x, y in zip(reweighted[a], reweighted[b]))
        dec["pairs"].append({"pair": [a, b], "standard_gap": summary(std[0], std[1]),
                             "reweighted_standard_gap": summary(rew[0], rew[1]),
                             "random_gap": summary(rand[0], rand[1]),
                             "composition_part": summary(rew[0] - std[0], rew[1] - std[1]),
                             "within_group_part": summary(rand[0] - rew[0], rand[1] - rew[1])})
    out["decomposition"] = dec
    return out


def main() -> None:
    cfg = {"models": list(MODELS), "ks": list(KS), "k_main": K_MAIN, "groups": GROUPS, "samples": SAMPLES}
    with registry.logged("posthoc-recall-decomposition", cfg) as run:
        out_dir = config.RUNS / "posthoc" / run["run_id"]
        run["output"] = str(out_dir)
        started = time.time()
        payload, run["zoo_manifest_sha256"] = O.load_selection(config.ROOT / data.ZOO_MANIFEST)
        O._verify_inputs(payload)
        p = prepare(1.0)
        dev = device()
        pool = Pool(p.space, p.videos, p.train)
        frames = O._test_sets(p)
        targets = [pool.targets(f) for f in frames]
        users = sorted(set(targets[0]) & set(targets[1]))
        n = len(users)
        n_pool = len(pool.video_ids)
        seen_mask = np.zeros((n, n_pool), dtype=bool)
        for i, u in enumerate(users):
            seen = pool.seen.get(u)
            if seen is not None and len(seen):
                seen_mask[i, seen] = True
        candidates = n_pool - seen_mask.sum(axis=1)
        tables = [target_table(t, users) for t in targets]
        rows = [row_table(f, users, pool, seen_mask) for f in frames]
        train_counts = p.train["video_id"].value_counts().to_dict()
        groups_of = C.popularity_groups(p.videos["video_id"].tolist(), train_counts)
        group_of_row = np.array([groups_of[int(v)] for v in pool.video_ids], dtype=np.int64)

        index = A.artefact_index(payload["artefacts"])
        scorers = {"recall_popularity": [popularity_scorer(pool, p.train, dev)],
                   "recall_itemcf": [itemcf_scorer(pool, p.train, dev)]}
        for name in SEEDED:
            scorers[name] = [retriever_scorer(A.load_retriever(entry, p.space, dev), p.space, p.users, pool, dev)
                             for entry in index[(name, "is_click")]]
        per_seed, blocks = {}, {}
        for m in MODELS:
            seed_blocks = [per_user_block(rank_matrix(fn, users, pool), n, tables, rows, group_of_row)
                           for fn in scorers[m]]
            blocks[m] = averaged(seed_blocks)
            if m in SEEDED:
                per_seed[m] = [{"recall@100": [float(b[("recall", s, K_MAIN)].mean()) for s in range(2)],
                                "random_click_rate@100": float(b[("clicks_in", 1, K_MAIN)].sum()
                                                                / b[("rows_in", 1, K_MAIN)].sum())}
                               for b in seed_blocks]
        data.verify_artefacts(payload["artefacts"])

        formal = json.loads(FORMAL_RECALL.read_text(encoding="utf-8"))
        if formal["users"] != n:
            raise RuntimeError(f"R* differs from the formal analysis: {n} vs {formal['users']}")
        worst = max(abs(float(blocks[m][("recall", s, K_MAIN)].mean()) - formal["recall@100"][m][s])
                    for m in MODELS for s in range(2))
        if not worst < TOLERANCE:
            raise RuntimeError(f"Recall@100 does not reproduce the formal analysis (max difference {worst})")

        cols = Columns()
        for s in range(2):
            t_pos, t_item, t_count = tables[s]
            cols.add(("target_share", s), np.bincount(t_pos * GROUPS + group_of_row[t_item],
                                                      minlength=n * GROUPS).reshape(n, GROUPS) / t_count[:, None])
            cols.add(("base_rows", s), np.bincount(rows[s]["pos"], minlength=n))
            cols.add(("base_clicks", s), np.bincount(rows[s]["pos"], weights=rows[s]["click"], minlength=n))
        for m in MODELS:
            for s in range(2):
                for k in KS:
                    cols.add((m, "recall", s, k), blocks[m][("recall", s, k)])
                    cols.add((m, "over_base", s, k), blocks[m][("recall", s, k)] / (k / candidates))
                    cols.add((m, "rows_in", s, k), blocks[m][("rows_in", s, k)])
                    cols.add((m, "clicks_in", s, k), blocks[m][("clicks_in", s, k)])
                cols.add((m, "hit_group", s), blocks[m][("hit_group", s)])
            cols.add((m, "slots"), blocks[m][("slots",)])

        result = {"note": "exploratory, protocol §20; test pair opened on 2026-10-30",
                  "time": time.strftime("%Y-%m-%d %H:%M:%S"), "users": n,
                  "formal_check": {"max_abs_difference": worst, "tolerance": TOLERANCE},
                  "denominators": {
                      "candidates": distribution(candidates),
                      "targets": {name: distribution(tables[s][2]) for s, name in enumerate(SETS)},
                      "share_targets_at_most_k": {name: {f"k{k}": float(np.mean(tables[s][2] <= k)) for k in KS}
                                                  for s, name in enumerate(SETS)},
                      "random_baseline": {f"k{k}": distribution(k / candidates) for k in KS},
                      "ideal_upper_bound": {name: {f"k{k}": float(np.mean(np.minimum(k, tables[s][2]) / tables[s][2]))
                                                   for k in KS} for s, name in enumerate(SETS)},
                      "test_rows": {name: {"eligible": int(len(rows[s]["pos"])), "dropped_seen": rows[s]["dropped_seen"],
                                           "dropped_unmapped": rows[s]["dropped_unmapped"]}
                                    for s, name in enumerate(SETS)}},
                  **analyse(cols, cols.matrix(), n),
                  "per_seed": per_seed,
                  "seconds": round(time.time() - started, 1)}
        A.write(result, out_dir, "recall_decomposition")
    main_pairs = result["relative"][f"k{K_MAIN}"]
    print(json.dumps({"users": n, "formal_max_diff": worst,
                      "raw_ratio_plain_over_logq": main_pairs[-1]["raw"],
                      "random_click_rate@100": {m: result["policy"]["random"]["by_k"][f"k{K_MAIN}"]["models"][m]
                                                ["click_rate"]["point"] for m in MODELS}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
