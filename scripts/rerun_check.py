"""Retrain one selected ranker configuration from scratch and compare its validation UAUC with the registered run.

The retrained model is registered (kind "rerun-check") like any other run but is never part of the zoo.
Usage: python scripts/rerun_check.py <model> [seed]
"""
from __future__ import annotations

import json
import sys

from receval import tune
from receval.orchestrate import SELECTION, load_selection
from receval.pipeline import prepare
from receval.train import device


def main() -> None:
    model, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
    payload, digest = load_selection(SELECTION)
    entry = next(a for a in payload["artefacts"]
                 if a["model"] == model and a["label"] == "is_click" and int(a["seed"]) == seed)
    p = prepare(1.0)
    rerun = tune._run("rerun-check", model, entry["config"], seed, "is_click",
                      tune._ranker_fn(model, entry["config"], p, seed, "is_click", device()), [])
    seeds = [a["valid_metric"] for a in payload["artefacts"] if a["model"] == model and a["label"] == "is_click"]
    print(json.dumps({"model": model, "seed": seed, "selection_sha256": digest, "original_run": entry["run_id"],
                      "rerun_id": rerun["run_id"], "registered": entry["valid_metric"], "rerun": tune._metric(rerun),
                      "difference": tune._metric(rerun) - entry["valid_metric"],
                      "seed_range": max(seeds) - min(seeds)}, indent=2))


if __name__ == "__main__":
    main()
