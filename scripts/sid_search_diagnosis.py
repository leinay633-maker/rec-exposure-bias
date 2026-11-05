"""Exploratory beam-search diagnosis of study 2's semantic-ID models (docs/kuairec/PROTOCOL.md §10.1).

The frozen analysis ranks with a prefix-trie beam search of width 100. This script re-ranks every semantic-ID model
file with widths 100, 200 and 400 and with exact scoring (width = pool size, so no prefix is ever pruned), checks the
exact scores against teacher forcing, and reports the overlap with the exact top 100, Recall@100 on the four tests,
H1–H3 and the pair differences under each search, and the ranking time and peak GPU memory. The test data was
opened once on 2026-11-01; nothing here changes a protocol decision.

Usage: python scripts/sid_search_diagnosis.py [--smoke]
  --smoke  time each search on the validation subset with one model file; no test data, no metrics.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import torch
from scipy.stats import kendalltau

from genrec import analysis as A
from genrec import config, data, runlog
from genrec.evaluate import borrowed_counts, per_user
from genrec.freeze import load_manifest
from genrec.generative import Trie, beam_search, exact_scores
from genrec.train import KINDS, TOP, context, load_model
from genrec.tune import device
from receval.stats import joint_user_bootstrap

SID = ("gr_text", "gr_shuf")
GROUPS = ("main", "sensitivity")
SEARCHES = ("beam100", "beam200", "beam400", "exact")
WIDTH = {"beam100": 100, "beam200": 200, "beam400": 400}  # exact: the pool size
CHUNK = {"beam100": 128, "beam200": 128, "beam400": 128, "exact": 16}
NOISE = "beam100_chunk16"  # the formal width with the exact search's batch size (§10.1 item 2)
TESTS = ("dense", "weighted", "borrowed", "natural")
FORMAL = config.ROOT / "evidence/kuairec/analysis/20261101-165505-analysis-c8c25f/result.json"
REPRO_TOL, EXACT_TOL = 1e-12, 1e-4
CHECK_USERS, CHECK_OTHERS = 8, 200
GATE_SHIFT = 0.005
SAMPLES, BOOT_SEED = 10_000, 0


def set_tf32(on: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def search(model, extra, hist: np.ndarray, allowed: np.ndarray, pool: np.ndarray, width: int, chunk: int,
           dev: torch.device) -> tuple[np.ndarray, dict]:
    """Top-100 video indices per user (-1 where fewer), as genrec.train.rank maps them, plus time and peak memory."""
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter()
    items, _ = beam_search(model, extra["codes"], extra["trie"], hist, allowed, width, dev, chunk=chunk)
    torch.cuda.synchronize(dev)
    cost = {"seconds": round(time.perf_counter() - started, 2), "peak_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 3),
            "width": width, "chunk": chunk}
    top = np.where(items >= 0, pool[np.maximum(items, 0)], -1)[:, :TOP]
    return top, cost


def exactness(model, extra, hist: np.ndarray, allowed: np.ndarray, pool: np.ndarray, dev: torch.device) -> dict:
    """§10.1 check 2: exact-search scores of 8 users' top 100 plus 200 other allowed videos against teacher forcing."""
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(hist), CHECK_USERS, replace=False))
    n_pool = len(pool)
    set_tf32(False)
    items, _ = beam_search(model, extra["codes"], extra["trie"], hist[pick], allowed[pick], n_pool, dev, chunk=CHECK_USERS)
    subsets = []
    for r, u in enumerate(pick):
        top = items[r, :TOP]
        top = top[top >= 0]
        rest = np.setdiff1d(np.flatnonzero(allowed[u]), top)
        subsets.append(np.concatenate([top, np.sort(rng.choice(rest, min(CHECK_OTHERS, len(rest)), replace=False))]))
    worst = {}
    for on in (False, True):
        set_tf32(on)
        items, scores = beam_search(model, extra["codes"], extra["trie"], hist[pick], allowed[pick], n_pool, dev,
                                    chunk=CHECK_USERS)
        diff = 0.0
        for r, sub in enumerate(subsets):
            score_of = np.full(n_pool, np.nan)
            ok = items[r] >= 0
            score_of[items[r][ok]] = scores[r][ok]
            forced = exact_scores(model, extra["codes"], Trie(extra["codes"], pool[sub]), hist[pick[r]][None], dev)[0]
            diff = max(diff, float(np.max(np.abs(score_of[sub] - forced))))
        worst["tf32_on" if on else "tf32_off"] = diff
    set_tf32(True)  # back to tune.device()'s setting, which the formal analysis used
    if not worst["tf32_off"] <= EXACT_TOL:
        raise RuntimeError(f"exact search differs from teacher forcing by {worst['tf32_off']} (> {EXACT_TOL})")
    return {"users": [int(i) for i in pick], "videos_per_user": [int(len(x)) for x in subsets],
            "max_abs_diff_tf32_off": worst["tf32_off"], "max_abs_diff_tf32_on": worst["tf32_on"]}


def overlap(a: np.ndarray, b: np.ndarray) -> dict:
    """Per user |a ∩ b| / |b| over real entries (b = the reference list), summarised."""
    share = np.array([len(np.intersect1d(x[x >= 0], y[y >= 0])) / max(1, int((y >= 0).sum())) for x, y in zip(a, b)])
    return {"mean": float(share.mean()), "p10": float(np.percentile(share, 10)), "identical_share": float(np.mean(share == 1.0))}


def interval(rep: np.ndarray, point: float) -> dict:
    lo, hi = np.percentile(rep, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)]}


def natural_pairs(t: dict, targets: list) -> tuple[dict, dict]:
    """genrec.analysis.natural_block's statistics on precomputed top lists."""
    metric = {k: {"natural": np.stack([per_user(top, targets)["recall@100"] for top in t[k]], -1)} for k in KINDS}
    vals = np.stack([metric[k]["natural"].mean(-1) for k in KINDS], 1)[:, :, None]
    block = A.block_pairs(vals, ["natural"], A.per_seed_contrasts(metric, np.ones(len(targets), dtype=bool), ["natural"]))
    return block, metric


def smoke() -> None:
    dev = device()
    with runlog.logged("posthoc-sid-search-smoke", {"searches": list(SEARCHES), "chunk": CHUNK}) as state:
        manifest, state["manifest_sha256"] = load_manifest()
        ctx = context()
        entry = A.entries_of(manifest["selection"])[("gr_text", 0)]
        model, extra = load_model(entry, ctx, dev)
        pool = ctx.s.pool
        cost = {}
        for name in SEARCHES:
            width = len(pool) if name == "exact" else WIDTH[name]
            _, cost[name] = search(model, extra, ctx.valid_hist, ctx.valid_allowed, pool, width, CHUNK[name], dev)
        state["users"] = len(ctx.valid_users)
        state["cost"] = cost
    print(json.dumps(cost, indent=1))


def main() -> None:
    dev = device()
    if dev.type != "cuda":
        raise RuntimeError("the timing and memory figures of §10.1 need CUDA")
    cfg = {"searches": list(SEARCHES), "width": WIDTH, "chunk": CHUNK, "check": [CHECK_USERS, CHECK_OTHERS, EXACT_TOL],
           "samples": SAMPLES, "gate_shift": GATE_SHIFT}
    with runlog.logged("posthoc-sid-search", cfg) as state:
        started = time.time()
        out_dir = config.RUNS / "posthoc" / state["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        state["output"] = str((out_dir / "sid_search.json").relative_to(config.ROOT).as_posix())
        manifest, state["manifest_sha256"] = load_manifest()
        selection = manifest["selection"]
        ctx = context()
        s, pool = ctx.s, ctx.s.pool
        entries = {"main": A.entries_of(selection), "sensitivity": A.entries_of(selection, sensitivity=True)}

        # populations exactly as genrec.analysis.run(proxy=False) and analyse()
        users = s.users["test"]
        targets = data.dense_targets(data.load("dense_test"), users)
        allowed = data.labelled(data.load("small_pairs"), users, pool)
        hist = data.histories(s, users)
        borrow = A.borrow_inputs(s, users)
        frame, _ = A.natural_targets(data.load("natural_test"), s)
        nat_all = sorted(frame["user_id"].unique().tolist())
        nat_t_all = data.targets_from(frame, nat_all)
        keep = [i for i, x in enumerate(nat_t_all) if len(x)]
        nat_users, nat_targets = [nat_all[i] for i in keep], [nat_t_all[i] for i in keep]
        nat_hist, nat_allowed = data.histories(s, nat_users), data.allowed(s, nat_users)
        w = s.items["w"].to_numpy()
        n_t = np.array([len(x) for x in targets])
        w_t = np.array([w[x].sum() if len(x) else 0 for x in targets])
        dense_pop = n_t > 0
        d_pop = dense_pop & (w_t > 0)
        counts = borrowed_counts(targets, borrow["rows"], borrow["assign"], borrow["masks"], ctx.col_of)
        b_pop = d_pop & ((counts > 0).sum(1) > 0)
        pops = {"dense": (hist, allowed), "natural": (nat_hist, nat_allowed)}

        base = {p: {} for p in pops}  # popularity, two-tower and SASRec need no beam search
        for kind in ("pop", "two_tower", "sasrec"):
            for p, (h, a) in pops.items():
                base[p][kind] = A.tops(entries["main"], h, a, ctx, dev, kinds=(kind,))[kind]

        tops = {g: {p: {name: {k: [] for k in SID} for name in (*SEARCHES, NOISE)} for p in pops} for g in GROUPS}
        cost, checks = {}, {}
        for g in GROUPS:
            for kind in SID:
                for seed in config.SEEDS:
                    label = f"{kind}/{seed}" + ("/sensitivity" if g == "sensitivity" else "")
                    model, extra = load_model(entries[g][(kind, seed)], ctx, dev)
                    checks[label] = exactness(model, extra, hist, allowed, pool, dev)
                    for p, (h, a) in pops.items():
                        for name in (*SEARCHES, NOISE):
                            width = len(pool) if name == "exact" else 100 if name == NOISE else WIDTH[name]
                            chunk = CHUNK["exact"] if name == NOISE else CHUNK[name]
                            top, c = search(model, extra, h, a, pool, width, chunk, dev)
                            tops[g][p][name][kind].append(top)
                            cost.setdefault(p, {}).setdefault(label, {})[name] = c
                    del model, extra
                    torch.cuda.empty_cache()
                    print(label, "ranked", flush=True)
                if g == "main":  # early form of check 1, so a mismatch stops the run before the long part
                    got = np.stack([per_user(t, targets)["recall@100"] for t in tops[g]["dense"]["beam100"][kind]],
                                   -1)[d_pop].mean(-1).mean()
                    ref = json.loads(FORMAL.read_text(encoding="utf-8"))["metrics"]["recall@100"][
                        "weighted_population"]["models"][kind]["dense"]
                    if not abs(got - ref) < REPRO_TOL:
                        raise RuntimeError(f"{kind}: width-100 dense Recall@100 {got} vs formal {ref}")

        def lists(g: str, p: str, name: str) -> dict:
            return {**base[p], **tops[g][p][name]}

        overlaps = {p: {f"{k}/{seed}" + ("/sensitivity" if g == "sensitivity" else ""): {
            **{name: overlap(tops[g][p][name][k][i], tops[g][p]["exact"][k][i]) for name in SEARCHES[:-1]},
            "noise_floor": overlap(tops[g][p][NOISE][k][i], tops[g][p]["beam100"][k][i])}
            for g in GROUPS for k in SID for i, seed in enumerate(config.SEEDS)} for p in pops}

        formal = json.loads(FORMAL.read_text(encoding="utf-8"))
        per_search, per_user_vals, worst = {}, {}, 0.0
        for name in SEARCHES:
            per_search[name] = {}
            for g in GROUPS:
                m = A.evaluate_population(lists(g, "dense", name), targets, s, ctx, borrow)
                metrics = A.hypothesis_blocks(m, dense_pop, d_pop, names=("recall@100",))["recall@100"]
                borrowed = A.borrowed_block(m, b_pop)
                natural, nat_metric = natural_pairs(lists(g, "natural", name), nat_targets)
                vecs = {"dense": [metrics["dense_population"]["models"][k]["dense"] for k in KINDS],
                        "weighted": [metrics["weighted_population"]["models"][k]["weighted"] for k in KINDS],
                        "borrowed": [borrowed["models"][k]["borrowed"] for k in KINDS],
                        "natural": [natural["models"][k]["natural"] for k in KINDS]}
                per_search[name][g] = {
                    "models": {k: {t: vecs[t][i] for t in TESTS} for i, k in enumerate(KINDS)},
                    "primary": metrics["primary"]["rows"],
                    "pairs": {"dense_weighted_D": metrics["weighted_population"]["pairs"],
                              "dense_h3_population": metrics["dense_population"]["pairs"],
                              "borrowed": borrowed["pairs"], "natural": natural["pairs"]},
                    "kendall_tau": {f"{a} vs {b}": float(kendalltau(vecs[a], vecs[b]).statistic)
                                    for a, b in itertools.combinations(vecs, 2)}}
                for k in SID:
                    per_user_vals[(g, k, "dense", name)] = m["dense"][k]["recall@100"][d_pop].mean(-1)
                    per_user_vals[(g, k, "weighted", name)] = m["weighted"][k]["recall@100"][d_pop].mean(-1)
                    per_user_vals[(g, k, "borrowed", name)] = A.nanmean(m["borrowed"][k]["recall@100"], 1)[b_pop].mean(-1)
                    per_user_vals[(g, k, "natural", name)] = nat_metric[k]["natural"].mean(-1)
                if name == "beam100":  # §10.1 check 1: the formal analysis, value for value
                    f = formal["metrics"]["recall@100"]
                    got_ref = []
                    for k in KINDS:
                        if g == "main" or k in SID:
                            ref = (f["weighted_population"]["models"][k] if g == "main" else
                                   {t: formal["sensitivity"]["models"][k][t] for t in ("dense", "weighted")})
                            got_ref += [(metrics["weighted_population"]["models"][k][t], ref[t]) for t in ("dense", "weighted")]
                        if g == "main":
                            got_ref += [(metrics["dense_population"]["models"][k]["dense"], f["dense_population"]["models"][k]["dense"]),
                                        (borrowed["models"][k]["borrowed"], formal["borrowed"]["models"][k]["borrowed"]),
                                        (natural["models"][k]["natural"], formal["natural"]["models"][k]["natural"])]
                    if g == "main":
                        got_ref += [(a["point"], b["point"]) for a, b in zip(metrics["primary"]["rows"], formal["primary"]["rows"])]
                    worst = max([worst] + [abs(a - b) for a, b in got_ref])
            print(name, "scored", flush=True)
        if not worst < REPRO_TOL:
            raise RuntimeError(f"width-100 search does not reproduce the formal analysis (max difference {worst})")

        recall = {}
        for pop_name, tests in (("d_pop", ("dense", "weighted")), ("b_pop", ("borrowed",)), ("natural", ("natural",))):
            cols = [(g, k, t, name) for g in GROUPS for k in SID for t in tests for name in SEARCHES]
            values = np.stack([per_user_vals[c] for c in cols], 1)
            rep = joint_user_bootstrap(values, samples=SAMPLES, seed=BOOT_SEED)
            at = {c: i for i, c in enumerate(cols)}
            for g, k, t, name in cols:
                cell = recall.setdefault(g, {}).setdefault(k, {}).setdefault(t, {})
                cell[name] = {"value": float(values[:, at[(g, k, t, name)]].mean())}
                if name != "exact":
                    i, j = at[(g, k, t, name)], at[(g, k, t, "exact")]
                    cell[name]["minus_exact"] = interval(rep[:, i] - rep[:, j], values[:, i].mean() - values[:, j].mean())

        def excludes_zero(c: dict) -> bool:
            return c["ci95"][0] > 0 or c["ci95"][1] < 0

        shifts = [{"group": g, "model": k, "test": t, **recall[g][k][t]["beam100"]["minus_exact"]}
                  for g in GROUPS for k in SID for t in TESTS
                  if excludes_zero(recall[g][k][t]["beam100"]["minus_exact"])
                  and abs(recall[g][k][t]["beam100"]["minus_exact"]["point"]) >= GATE_SHIFT]
        flips = []
        for g in GROUPS:
            for block, col in (("dense_weighted_D", "dense"), ("dense_weighted_D", "weighted"), ("borrowed", "borrowed"),
                               ("natural", "natural")):
                for r100, rex in zip(per_search["beam100"][g]["pairs"][block], per_search["exact"][g]["pairs"][block]):
                    if not any(k in r100["pair"] for k in SID):
                        continue
                    a, b = r100[col], rex[col]
                    if np.sign(a["point"]) != np.sign(b["point"]) and excludes_zero(a) and excludes_zero(b):
                        flips.append({"group": g, "pair": r100["pair"], "test": col, "beam100": a["point"], "exact": b["point"]})
        changes = [{"hypothesis": a["hypothesis"], "beam100": a["holds"], "exact": b["holds"]}
                   for a, b in zip(per_search["beam100"]["main"]["primary"], per_search["exact"]["main"]["primary"])
                   if a["holds"] != b["holds"]]

        result = {"note": "exploratory, docs/kuairec/PROTOCOL.md §10.1; test data opened on 2026-11-01",
                  "time": time.strftime("%Y-%m-%d %H:%M:%S"), "manifest_sha256": state["manifest_sha256"],
                  "formal_analysis": FORMAL.relative_to(config.ROOT).as_posix(),
                  "populations": {"dense_and_weighted": int(d_pop.sum()), "borrowed": int(b_pop.sum()),
                                  "natural": len(nat_users), "pool": int(len(pool))},
                  "reproduction": {"max_abs_difference": worst, "tolerance": REPRO_TOL},
                  "exactness": checks, "overlap": overlaps, "recall": recall, "searches": per_search,
                  "gate": {"recall_shifts": shifts, "pair_flips": flips, "decision_changes": changes,
                           "triggered": bool(shifts or flips or changes)},
                  "cost": cost, "seconds": round(time.time() - started, 1)}
        (out_dir / "sid_search.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float),
                                                 encoding="utf-8")
        state["gate_triggered"] = result["gate"]["triggered"]
    print(json.dumps({"reproduction": worst, "gate": result["gate"],
                      "overlap_dense_beam100": {k: v["beam100"]["mean"] for k, v in overlaps["dense"].items()}},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    if parser.parse_args().smoke:
        smoke()
    else:
        main()
