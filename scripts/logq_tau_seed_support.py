"""Seed support for every contrast of the logQ x temperature ablation (protocol §15, deviation §16).

The ablation script counted seed support only on the standard and random columns. This re-scores the twelve
models it recorded (hash-checked, nothing retrained) and counts, for every contrast and column (standard,
random, random minus standard), how many of the 3 seed-paired contrasts share the sign of the seed-mean point.
It also checks that the seed-mean points reproduce the ablation output exactly.

Usage: python scripts/logq_tau_seed_support.py <ablation run id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logq_tau_ablation as L  # noqa: E402

from receval import analysis as A  # noqa: E402
from receval import config, data, registry  # noqa: E402
from receval import orchestrate as O  # noqa: E402
from receval.pipeline import prepare  # noqa: E402
from receval.retrieval import Pool  # noqa: E402
from receval.train import device  # noqa: E402

COLUMNS = ("standard", "random", "random_minus_standard")


def columns(per_seed: np.ndarray) -> np.ndarray:
    """(sets, seeds) -> (3 columns, seeds): standard, random and random minus standard."""
    return np.stack([per_seed[0], per_seed[1], per_seed[1] - per_seed[0]])


def support(cube: np.ndarray) -> dict:
    seed_means = cube.mean(0)  # (cells, sets, seeds): each seed's mean over users
    point = cube.mean(axis=3).mean(0)  # (cells, sets)
    i = {cell: n for n, cell in enumerate(L.ORDER)}

    def contrast(plus, minus):
        return seed_means[i[plus]] - seed_means[i[minus]], point[i[plus]] - point[i[minus]]

    def count(per_seed, pt):
        cols, pts = columns(per_seed), np.array([pt[0], pt[1], pt[1] - pt[0]])
        return {name: {"point": float(pts[c]), "per_seed": [float(v) for v in cols[c]],
                       "seed_support": int(np.sum(np.sign(cols[c]) == np.sign(pts[c])))}
                for c, name in enumerate(COLUMNS)}

    out = {name: count(*contrast(plus, minus)) for name, (plus, minus) in L.SIMPLE.items()}
    tw, tw_pt = contrast("logq_t01", "logq_t005")
    tp, tp_pt = contrast("plain_t01", "plain_t005")
    out["interaction"] = count(tw - tp, tw_pt - tp_pt)
    out["original_gap"] = count(*contrast(*L.ORIGINAL))
    return out


def main() -> None:
    run_id = sys.argv[1]
    source = config.RUNS / "ablation" / run_id / "logq_tau.json"
    ablation = json.loads(source.read_text(encoding="utf-8"))
    with registry.logged("ablation-seed-support", {"ablation_run": run_id}) as run:
        out_dir = config.RUNS / "ablation" / run_id
        run["output"] = str(out_dir / "seed_support.json")
        run["ablation_sha256"] = data.file_digest(source)
        entries = {}
        for cell in L.ORDER:
            entries[cell] = []
            for r in ablation["cells"][cell]["runs"]:
                path = config.ROOT / r["path"]
                if data.file_digest(path) != r["sha256"]:
                    raise PermissionError(f"{r['path']} changed since the ablation run")
                entries[cell].append({"path": r["path"], "config": L.cfg_of(cell)})
        p = prepare(1.0)
        dev = device()
        pool = Pool(p.space, p.videos, p.train)
        models = {cell: [A.load_retriever(e, p.space, dev) for e in entries[cell]] for cell in L.ORDER}
        result = {"ablation_run": run_id, "note": "exploratory, protocol §15; supplement recorded in §16"}
        for name, frames in {"proxy": [p.valid, p.calib], "test": O._test_sets(p)}.items():
            users, cubes = L.score_pair(frames, pool, p, models, dev)
            cube = cubes["recall@100"]
            recorded = ablation[name]["recall@100"]["cells"]
            worst = max(abs(float(v) - recorded[cell][s]) for c, cell in enumerate(L.ORDER)
                        for s, v in enumerate(cube[:, c].mean(axis=2).mean(0)))
            if not worst < L.TOLERANCE or len(users) != ablation[name]["users"]:
                raise RuntimeError(f"{name}: re-scored cells differ from the ablation output (max {worst})")
            result[name] = {"users": len(users), "max_abs_difference_to_ablation": worst, **support(cube)}
        A.write(result, out_dir, "seed_support")
    print(json.dumps({k: {c: {col: v[col]["seed_support"] for col in COLUMNS} for c, v in result[k].items()
                          if isinstance(v, dict) and "standard" in v} for k in ("proxy", "test")}, indent=1))


if __name__ == "__main__":
    main()
