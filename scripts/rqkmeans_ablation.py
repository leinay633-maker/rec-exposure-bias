"""Exploratory residual k-means code tables for study 2 (docs/kuairec/PROTOCOL.md §10.2).

Builds plain and balanced residual k-means codes (3 levels × 64, plus the disambiguation code) on the frozen Qwen3
text vectors, trains the text semantic-ID generator's selected configuration on each table for seeds 0–2 with the
frozen training code (only the code table is swapped), and compares the result with the RQ-VAE text semantic IDs:
first on the validation subset, then, as exploration only, on the four opened tests with width-100 and exact search.

Usage: python scripts/rqkmeans_ablation.py [--search-json <.runs/kuairec/posthoc/<run>/sid_search.json>]
  --search-json  cross-check the RQ-VAE model's re-ranked Recall@100 against the §10.1 run.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import sid_search_diagnosis as D
from genrec import analysis as A
from genrec import config, data, encode, runlog, sids, train
from genrec.evaluate import borrowed_counts, per_user
from genrec.freeze import load_manifest
from genrec.generative import Codes
from genrec.semantic_id import _kmeans, collision_stats, disambiguate
from genrec.tune import device
from receval.data import file_digest
from receval.stats import joint_user_bootstrap

VARIANTS = ("plain", "balanced")
SID_MODELS = ("rqvae", *VARIANTS)
BASE = ("pop", "two_tower", "sasrec")
OTHERS = ("rqvae", "sasrec", "two_tower")
SEARCHES = ("beam100", "exact")
LEVELS, CODEBOOK, ITERS = sids.RQ["levels"], sids.RQ["codebook"], 10
RELOAD_TOL, CROSS_TOL = 1e-6, 1e-12


def balanced_assign(dist: np.ndarray, cap: int) -> np.ndarray:
    """(point, centre) pairs by distance, then point, then centre; each point goes to the first centre not yet full."""
    n, k = dist.shape
    order = np.lexsort((np.tile(np.arange(k), n), np.repeat(np.arange(n), k), dist.ravel()))
    assign, load, left = [-1] * n, [0] * k, n
    for flat in order.tolist():
        i, c = divmod(flat, k)
        if assign[i] < 0 and load[c] < cap:
            assign[i] = c
            load[c] += 1
            left -= 1
            if not left:
                break
    return np.asarray(assign, dtype=np.int64)


def balanced_kmeans(x: torch.Tensor, k: int, iters: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """semantic_id._kmeans with the same initial centres, but each centre takes at most ceil(n / k) points."""
    centres = x[torch.randperm(len(x), generator=gen, device="cpu")[:k].to(x.device)].clone()
    cap = math.ceil(len(x) / k)
    for _ in range(iters):
        assign = torch.as_tensor(balanced_assign(torch.cdist(x, centres).cpu().numpy(), cap), device=x.device)
        for c in range(k):
            members = x[assign == c]
            if len(members):
                centres[c] = members.mean(0)
    return centres, assign  # the last assignment is the level's code


def residual_codes(emb: np.ndarray, seed: int, variant: str, dev: torch.device) -> tuple[np.ndarray, dict]:
    gen = torch.Generator().manual_seed(seed)  # shared by the three levels, as in semantic_id.train_rqvae
    x = torch.as_tensor(emb, dtype=torch.float32, device=dev)
    residual, codes = x.clone(), []
    for _ in range(LEVELS):
        if variant == "plain":
            centres = _kmeans(residual, CODEBOOK, ITERS, gen)
            code = torch.cdist(residual, centres).argmin(1)
        else:
            centres, code = balanced_kmeans(residual, CODEBOOK, ITERS, gen)
        residual = residual - centres[code]
        codes.append(code.cpu().numpy())
    codes = np.stack(codes, 1)
    info = {"codes_used_per_level": [int(len(np.unique(codes[:, level]))) for level in range(LEVELS)],
            "three_code_collisions": collision_stats(codes),
            "mean_input_norm": float(x.norm(dim=1).mean()), "mean_residual_norm": float(residual.norm(dim=1).mean())}
    return disambiguate(codes, seed), info


@contextlib.contextmanager
def code_table(tables: dict):
    """Swap genrec.train.codes_for, so build(), train_run() and load_model() use these codes (seed -> array)."""
    original = train.codes_for

    def codes_for(kind: str, seed: int, ctx) -> Codes:
        c = tables[seed]
        return Codes(c, sizes=[CODEBOOK] * LEVELS + [int(c[:, -1].max()) + 1])

    train.codes_for = codes_for
    try:
        yield
    finally:
        train.codes_for = original


def bootstrap_diffs(values: dict, pairs: list, samples: int = D.SAMPLES) -> dict:
    """values: column -> per-user array (same users); pairs: (label, column a, column b) -> point and 95% interval of a − b."""
    cols = list(values)
    rep = joint_user_bootstrap(np.stack([values[c] for c in cols], 1), samples=samples, seed=D.BOOT_SEED)
    at = {c: i for i, c in enumerate(cols)}
    return {label: D.interval(rep[:, at[a]] - rep[:, at[b]], float(values[a].mean() - values[b].mean()))
            for label, a, b in pairs}


def main(search_json: Path | None) -> None:
    dev = device()  # TF32 on, as in the zoo's training and the formal analysis
    if dev.type != "cuda":
        raise RuntimeError("needs CUDA")
    cfg = {"variants": list(VARIANTS), "levels": LEVELS, "codebook": CODEBOOK, "iters": ITERS,
           "searches": list(SEARCHES), "chunk": {k: D.CHUNK[k] for k in SEARCHES}, "samples": D.SAMPLES,
           "search_json": None if search_json is None else search_json.as_posix()}
    with runlog.logged("posthoc-rqkmeans", cfg) as state:
        started = time.time()
        out_dir = config.RUNS / "posthoc" / state["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        state["output"] = str((out_dir / "rqkmeans.json").relative_to(config.ROOT).as_posix())
        manifest, state["manifest_sha256"] = load_manifest()
        selection = manifest["selection"]
        ctx = train.context()
        s, pool = ctx.s, ctx.s.pool
        entries = A.entries_of(selection)
        gr_cfg = selection["models"]["gr_text"]["config"]
        emb, _ = encode.load()

        # code tables, with TF32 off as in sids.build (which never calls tune.device())
        D.set_tf32(False)
        tables, tables_info = {v: {} for v in VARIANTS}, {v: {} for v in VARIANTS}
        for variant in VARIANTS:
            for seed in config.SEEDS:
                codes, info = residual_codes(emb, seed, variant, dev)
                path = out_dir / f"rqkmeans_{variant}_seed{seed}.npy"
                np.save(path, codes)
                info.update({"fourth_code_max": int(codes[:, -1].max()), "pool_three_code_collisions":
                             collision_stats(codes[pool, :3]), "file": path.name, "sha256": file_digest(path)})
                tables[variant][seed], tables_info[variant][str(seed)] = codes, info
                print(variant, seed, info["codes_used_per_level"], info["three_code_collisions"], flush=True)
        D.set_tf32(True)
        rqvae_info = {}
        for seed in config.SEEDS:
            c = sids.load(seed, shuffle=False)
            rqvae_info[str(seed)] = {"codes_used_per_level": [int(len(np.unique(c[:, level]))) for level in range(LEVELS)],
                                     "three_code_collisions": collision_stats(c[:, :3]),
                                     "pool_three_code_collisions": collision_stats(c[pool, :3]),
                                     "fourth_code_max": int(c[:, -1].max())}

        runs = {}
        for variant in VARIANTS:
            with code_table(tables[variant]):
                runs[variant] = [train.train_run("gr_text", gr_cfg, seed, ctx, dev, purpose=f"posthoc-rqkmeans-{variant}")
                                 for seed in config.SEEDS]
            print(variant, "trained", [r["valid_recall@100"] for r in runs[variant]], flush=True)

        def loaded(name: str, seed: int):
            if name == "rqvae":
                return train.load_model(entries[("gr_text", seed)], ctx, dev)
            with code_table(tables[name]):
                return train.load_model(runs[name][config.SEEDS.index(seed)], ctx, dev)

        def registered(name: str, seed: int) -> float:
            e = entries[("gr_text", seed)] if name == "rqvae" else runs[name][config.SEEDS.index(seed)]
            return e["valid_recall@100"]

        # validation subset, per user: the quantity models are selected on
        valid, reload = {}, {}
        for name in SID_MODELS:
            per_seed = []
            for seed in config.SEEDS:
                model, extra = loaded(name, seed)
                top = train.rank("gr_text", model, extra, ctx.valid_hist, ctx.valid_allowed, ctx, dev)
                v = per_user(top, ctx.valid_targets)["recall@100"]
                got = float(np.nanmean(v))
                reload[f"{name}/{seed}"] = {"registered": registered(name, seed), "reproduced": got}
                if not abs(got - registered(name, seed)) < RELOAD_TOL:
                    raise RuntimeError(f"{name}/{seed}: validation Recall@100 {got} vs registered {registered(name, seed)}")
                per_seed.append(v)
            valid[name] = np.stack(per_seed, -1)
        has_target = np.isfinite(valid["rqvae"][:, 0])
        valid_means = {n: valid[n][has_target].mean(-1) for n in SID_MODELS}
        validation = {"users": int(has_target.sum()),
                      "models": {n: {"mean": float(valid_means[n].mean()),
                                     "per_seed": [float(x) for x in valid[n][has_target].mean(0)]} for n in SID_MODELS},
                      "differences": bootstrap_diffs(valid_means, [(f"{v} − rqvae", v, "rqvae") for v in VARIANTS])}

        # opened tests, exploration only: populations exactly as genrec.analysis
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
        pops = {"dense": (hist, allowed), "natural": (data.histories(s, nat_users), data.allowed(s, nat_users))}
        w = s.items["w"].to_numpy()
        n_t = np.array([len(x) for x in targets])
        w_t = np.array([w[x].sum() if len(x) else 0 for x in targets])
        d_pop = (n_t > 0) & (w_t > 0)
        counts = borrowed_counts(targets, borrow["rows"], borrow["assign"], borrow["masks"], ctx.col_of)
        b_pop = d_pop & ((counts > 0).sum(1) > 0)

        tops = {(p, srch): {} for p in pops for srch in SEARCHES}
        for kind in BASE:  # no beam search: one ranking serves both searches
            for p, (h, a) in pops.items():
                t = A.tops(entries, h, a, ctx, dev, kinds=(kind,))[kind]
                for srch in SEARCHES:
                    tops[(p, srch)][kind] = t
        cost = {}
        for name in SID_MODELS:
            for seed in config.SEEDS:
                model, extra = loaded(name, seed)
                for p, (h, a) in pops.items():
                    for srch in SEARCHES:
                        width = len(pool) if srch == "exact" else D.WIDTH[srch]
                        top, c = D.search(model, extra, h, a, pool, width, D.CHUNK[srch], dev)
                        tops[(p, srch)].setdefault(name, []).append(top)
                        cost[f"{name}/{seed}/{p}/{srch}"] = c
                del model, extra
                torch.cuda.empty_cache()
                print(name, seed, "ranked", flush=True)

        models = (*BASE, *SID_MODELS)
        per_user_vals, per_seed_vals = {}, {}
        for srch in SEARCHES:
            m = A.evaluate_population(tops[("dense", srch)], targets, s, ctx, borrow)
            for n in models:
                stacks = {"dense": m["dense"][n]["recall@100"][d_pop], "weighted": m["weighted"][n]["recall@100"][d_pop],
                          "borrowed": A.nanmean(m["borrowed"][n]["recall@100"], 1)[b_pop],
                          "natural": np.stack([per_user(t, nat_targets)["recall@100"] for t in tops[("natural", srch)][n]], -1)}
                for test, arr in stacks.items():
                    per_user_vals[(n, test, srch)] = arr.mean(-1)
                    per_seed_vals[(n, test, srch)] = [float(x) for x in arr.mean(0)]

        test_block = {}
        for pop_name, tests in (("d_pop", ("dense", "weighted")), ("b_pop", ("borrowed",)), ("natural", ("natural",))):
            values = {f"{n}|{t}|{srch}": per_user_vals[(n, t, srch)] for n in models for t in tests for srch in SEARCHES}
            pairs = [(f"{v} − {o}|{t}|{srch}", f"{v}|{t}|{srch}", f"{o}|{t}|{srch}")
                     for v in VARIANTS for o in OTHERS for t in tests for srch in SEARCHES]
            diffs = bootstrap_diffs(values, pairs)
            for t in tests:
                for srch in SEARCHES:
                    test_block.setdefault(t, {})[srch] = {
                        "models": {n: {"mean": float(per_user_vals[(n, t, srch)].mean()),
                                       "per_seed": per_seed_vals[(n, t, srch)]} for n in models},
                        "differences": {f"{v} − {o}": diffs[f"{v} − {o}|{t}|{srch}"] for v in VARIANTS for o in OTHERS}}

        cross = None
        if search_json is not None:  # the RQ-VAE model ranked here must match the §10.1 run value for value
            ref = json.loads(search_json.read_text(encoding="utf-8"))["recall"]["main"]["gr_text"]
            cross = max(abs(test_block[t][srch]["models"]["rqvae"]["mean"] - ref[t][srch]["value"])
                        for t in D.TESTS for srch in SEARCHES)
            if not cross < CROSS_TOL:
                raise RuntimeError(f"RQ-VAE Recall@100 differs from the §10.1 run by {cross}")

        keep_keys = ("run_id", "path", "sha256", "valid_recall@100", "best_epoch", "epochs_run", "params", "seconds", "batch")
        result = {"note": "exploratory, docs/kuairec/PROTOCOL.md §10.2; test data opened on 2026-11-01",
                  "time": time.strftime("%Y-%m-%d %H:%M:%S"), "manifest_sha256": state["manifest_sha256"],
                  "config": gr_cfg, "tables": tables_info, "rqvae_tables": rqvae_info,
                  "training": {v: [{k: r[k] for k in keep_keys} | {"purpose": r["purpose"]} for r in runs[v]] for v in VARIANTS},
                  "rqvae_training": {str(seed): {k: entries[("gr_text", seed)][k] for k in
                                                 ("run_id", "valid_recall@100", "best_epoch", "epochs_run", "params")}
                                     for seed in config.SEEDS},
                  "reload_check": reload, "validation": validation,
                  "populations": {"dense_and_weighted": int(d_pop.sum()), "borrowed": int(b_pop.sum()), "natural": len(nat_users)},
                  "test": test_block, "cross_check_with_search_run": cross, "cost": cost,
                  "seconds": round(time.time() - started, 1)}
        (out_dir / "rqkmeans.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float),
                                               encoding="utf-8")
    print(json.dumps({"validation": validation, "dense_beam100": test_block["dense"]["beam100"]["differences"]},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-json", type=Path)
    main(parser.parse_args().search_json)
