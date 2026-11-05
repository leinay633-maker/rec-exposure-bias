"""TF32 on/off check of the exact scoring (docs/kuairec/PROTOCOL.md §10.3): only reports, no gate of its own.

Every semantic-ID model file of §10.1 is ranked three times on the dense and natural populations: width 100 as in the
formal analysis (TF32 on), and the exact search (width = pool, 16 users per batch) with TF32 on and with TF32 off.
The §10.1 gate is then recomputed against the TF32-off exact search.

Usage: python scripts/sid_tf32_check.py
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import torch
from scipy.stats import kendalltau

import sid_search_diagnosis as D
from genrec import analysis as A
from genrec import config, data, runlog
from genrec.evaluate import borrowed_counts
from genrec.freeze import load_manifest
from genrec.train import KINDS, context, load_model
from genrec.tune import device
from receval.stats import joint_user_bootstrap

MODES = ("beam100", "exact_on", "exact_off")
SETTING = {"beam100": (100, D.CHUNK["beam100"], True), "exact_on": (None, D.CHUNK["exact"], True),
           "exact_off": (None, D.CHUNK["exact"], False)}  # width (None = pool), users per batch, TF32
REFERENCE = config.ROOT / "evidence/kuairec/posthoc/20261103-193144-posthoc-sid-search-4f5fb8/sid_search.json"
REPRO_TOL = 1e-12


def excludes_zero(c: dict) -> bool:
    return c["ci95"][0] > 0 or c["ci95"][1] < 0


def main() -> None:
    dev = device()
    if dev.type != "cuda":
        raise RuntimeError("§10.3 is about TF32 on CUDA")
    cfg = {"modes": list(MODES), "setting": {m: list(v) for m, v in SETTING.items()}, "samples": D.SAMPLES,
           "reference": REFERENCE.relative_to(config.ROOT).as_posix()}
    with runlog.logged("posthoc-sid-tf32", cfg) as state:
        started = time.time()
        out_dir = config.RUNS / "posthoc" / state["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        state["output"] = str((out_dir / "tf32_check.json").relative_to(config.ROOT).as_posix())
        manifest, state["manifest_sha256"] = load_manifest()
        selection = manifest["selection"]
        ctx = context()
        s, pool = ctx.s, ctx.s.pool
        entries = {"main": A.entries_of(selection), "sensitivity": A.entries_of(selection, sensitivity=True)}

        # populations exactly as scripts/sid_search_diagnosis.py (and genrec.analysis)
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

        base = {p: {} for p in pops}
        for kind in ("pop", "two_tower", "sasrec"):
            for p, (h, a) in pops.items():
                base[p][kind] = A.tops(entries["main"], h, a, ctx, dev, kinds=(kind,))[kind]

        tops = {g: {p: {m: {k: [] for k in D.SID} for m in MODES} for p in pops} for g in D.GROUPS}
        for g in D.GROUPS:
            for kind in D.SID:
                for seed in config.SEEDS:
                    model, extra = load_model(entries[g][(kind, seed)], ctx, dev)
                    for p, (h, a) in pops.items():
                        for m in MODES:
                            width, chunk, tf32 = SETTING[m]
                            D.set_tf32(tf32)
                            top, _ = D.search(model, extra, h, a, pool, width or len(pool), chunk, dev)
                            tops[g][p][m][kind].append(top)
                    D.set_tf32(True)
                    del model, extra
                    torch.cuda.empty_cache()
                    print(g, kind, seed, "ranked", flush=True)

        def lists(g: str, p: str, m: str) -> dict:
            return {**base[p], **tops[g][p][m]}

        per_mode, per_user_vals = {}, {}
        for m in MODES:
            per_mode[m] = {}
            for g in D.GROUPS:
                res = A.evaluate_population(lists(g, "dense", m), targets, s, ctx, borrow)
                metrics = A.hypothesis_blocks(res, dense_pop, d_pop, names=("recall@100",))["recall@100"]
                borrowed = A.borrowed_block(res, b_pop)
                natural, nat_metric = D.natural_pairs(lists(g, "natural", m), nat_targets)
                vecs = {"dense": [metrics["dense_population"]["models"][k]["dense"] for k in KINDS],
                        "weighted": [metrics["weighted_population"]["models"][k]["weighted"] for k in KINDS],
                        "borrowed": [borrowed["models"][k]["borrowed"] for k in KINDS],
                        "natural": [natural["models"][k]["natural"] for k in KINDS]}
                per_mode[m][g] = {
                    "models": {k: {t: vecs[t][i] for t in D.TESTS} for i, k in enumerate(KINDS)},
                    "primary": metrics["primary"]["rows"],
                    "pairs": {"dense_weighted_D": metrics["weighted_population"]["pairs"],
                              "borrowed": borrowed["pairs"], "natural": natural["pairs"]},
                    "kendall_tau": {f"{a} vs {b}": float(kendalltau(vecs[a], vecs[b]).statistic)
                                    for a, b in itertools.combinations(vecs, 2)}}
                for k in D.SID:
                    per_user_vals[(g, k, "dense", m)] = res["dense"][k]["recall@100"][d_pop].mean(-1)
                    per_user_vals[(g, k, "weighted", m)] = res["weighted"][k]["recall@100"][d_pop].mean(-1)
                    per_user_vals[(g, k, "borrowed", m)] = A.nanmean(res["borrowed"][k]["recall@100"], 1)[b_pop].mean(-1)
                    per_user_vals[(g, k, "natural", m)] = nat_metric[k]["natural"].mean(-1)
            print(m, "scored", flush=True)

        # check: the TF32-on runs reproduce the §10.1 run (reported, not enforced)
        ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
        repro = {m: max(abs(per_mode[m][g]["models"][k][t] - ref["searches"]["exact" if m == "exact_on" else "beam100"][g]
                                ["models"][k][t]) for g in D.GROUPS for k in KINDS for t in D.TESTS)
                 for m in ("beam100", "exact_on")}

        recall = {}
        for tests in (("dense", "weighted"), ("borrowed",), ("natural",)):
            cols = [(g, k, t, m) for g in D.GROUPS for k in D.SID for t in tests for m in MODES]
            values = np.stack([per_user_vals[c] for c in cols], 1)
            rep = joint_user_bootstrap(values, samples=D.SAMPLES, seed=D.BOOT_SEED)
            at = {c: i for i, c in enumerate(cols)}
            for g, k, t, m in cols:
                cell = recall.setdefault(g, {}).setdefault(k, {}).setdefault(t, {})
                cell[m] = float(values[:, at[(g, k, t, m)]].mean())
            for g, k, t in {(g, k, t) for g, k, t, _ in cols}:
                cell = recall[g][k][t]
                for name, (a, b) in {"on_minus_off": ("exact_on", "exact_off"),
                                     "beam100_minus_off": ("beam100", "exact_off")}.items():
                    i, j = at[(g, k, t, a)], at[(g, k, t, b)]
                    cell[name] = D.interval(rep[:, i] - rep[:, j], values[:, i].mean() - values[:, j].mean())

        overlaps = {p: {f"{k}/{seed}" + ("/sensitivity" if g == "sensitivity" else ""):
                        D.overlap(tops[g][p]["exact_on"][k][i], tops[g][p]["exact_off"][k][i])
                        for g in D.GROUPS for k in D.SID for i, seed in enumerate(config.SEEDS)} for p in pops}

        blocks = (("dense_weighted_D", "dense"), ("dense_weighted_D", "weighted"), ("borrowed", "borrowed"),
                  ("natural", "natural"))
        shifts = [{"group": g, "model": k, "test": t, **recall[g][k][t]["beam100_minus_off"]}
                  for g in D.GROUPS for k in D.SID for t in D.TESTS
                  if excludes_zero(recall[g][k][t]["beam100_minus_off"])
                  and abs(recall[g][k][t]["beam100_minus_off"]["point"]) >= D.GATE_SHIFT]
        flips = []
        for g in D.GROUPS:
            for block, col in blocks:
                for r100, roff in zip(per_mode["beam100"][g]["pairs"][block], per_mode["exact_off"][g]["pairs"][block]):
                    if not any(k in r100["pair"] for k in D.SID):
                        continue
                    a, b = r100[col], roff[col]
                    if np.sign(a["point"]) != np.sign(b["point"]) and excludes_zero(a) and excludes_zero(b):
                        flips.append({"group": g, "pair": r100["pair"], "test": col, "beam100": a["point"], "exact_off": b["point"]})
        changes = [{"hypothesis": a["hypothesis"], "beam100": a["holds"], "exact_off": b["holds"]}
                   for a, b in zip(per_mode["beam100"]["main"]["primary"], per_mode["exact_off"]["main"]["primary"])
                   if a["holds"] != b["holds"]]
        reference_flips = []
        for x in ref["gate"]["pair_flips"]:
            names = [row["pair"] for row in per_mode["exact_off"][x["group"]]["pairs"]["dense_weighted_D"]]
            block = dict((c, b) for b, c in blocks)[x["test"]]
            i = names.index(x["pair"])
            reference_flips.append({**x, "exact_on_now": per_mode["exact_on"][x["group"]]["pairs"][block][i][x["test"]],
                                    "exact_off": per_mode["exact_off"][x["group"]]["pairs"][block][i][x["test"]]})

        result = {"note": "exploratory, docs/kuairec/PROTOCOL.md §10.3; test data opened on 2026-11-01",
                  "time": time.strftime("%Y-%m-%d %H:%M:%S"), "manifest_sha256": state["manifest_sha256"],
                  "reference": REFERENCE.relative_to(config.ROOT).as_posix(),
                  "populations": {"dense_and_weighted": int(d_pop.sum()), "borrowed": int(b_pop.sum()),
                                  "natural": len(nat_users), "pool": int(len(pool))},
                  "reproduction_of_reference": {"max_abs_difference": repro, "tolerance": REPRO_TOL,
                                                "within": {m: bool(v < REPRO_TOL) for m, v in repro.items()}},
                  "overlap_on_off": overlaps, "recall": recall, "modes": per_mode,
                  "gate_against_tf32_off": {"recall_shifts": shifts, "pair_flips": flips, "decision_changes": changes},
                  "reference_flips": reference_flips, "seconds": round(time.time() - started, 1)}
        (out_dir / "tf32_check.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float),
                                                 encoding="utf-8")
    print(json.dumps({"reproduction": repro, "shifts": len(shifts), "flips": len(flips), "changes": changes,
                      "reference_flips": [(f["pair"], f["group"], f["test"], f["exact_off"]["point"], f["exact_off"]["ci95"])
                                          for f in reference_flips]}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
