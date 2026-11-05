"""End-to-end commands: tune the zoo, dry-run and power-check it on the proxy pair, analyse the opened test sets.

Each command registers a start entry and then ok or failed (`registry.logged`) with code and data hashes,
plus the hash of the zoo list it used; its output goes to a directory named after the run id. Models are
loaded only through the artefact list (path, sha256, config) of `.runs/zoo/selection.json` before the zoo
freeze and of `configs/zoo_manifest.json` after it. Each tune run keeps its own selection under
`.runs/zoo/<run_id>/`; `.runs/zoo/selection.json` is a copy of the latest.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as A
from . import calibrate as C
from . import config, data, registry, stats
from . import features as F
from .pipeline import Prepared, prepare
from .retrieval import Pool, evaluate_scorer, itemcf_scorer, popularity_scorer, retriever_scorer
from .train import device

ZOO = config.RUNS / "zoo"
SELECTION = ZOO / "selection.json"
PROXY_NOTE = "standard valid vs random calibration, 4/22-4/28, tab 1"
PROXY_CALIB_SPLIT = 20220425  # dry-run calibration: fit on random calibration rows up to this date, score the rest
RELOAD_TOLERANCE = 1e-5


def load_selection(path: Path) -> tuple[dict, str]:
    """The zoo payload (run ids per model, label and seed, plus the artefact list) and the sha256 of those bytes."""
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _verify_inputs(payload: dict) -> None:
    """The zoo list matches its registered runs and files, and the cache is the one the zoo was trained on."""
    from .tune import verify_against_registry
    data.verify_manifest(payload)
    verify_against_registry(payload)
    data.verify_data()  # features are rebuilt from the cache files, so their content is checked, not only the manifest
    if payload["data_manifest_sha256"] != data.file_digest(config.CACHE / "manifest.json"):
        raise PermissionError("data cache differs from the one the zoo was trained on")


def require_estimates(block: dict, what: str) -> None:
    """A gate run must produce real numbers: an explicit "cannot estimate" status or a non-finite SE fails it."""
    values = list(block.get("h3_mde_did", {}).values())
    if "status" not in block and "se_did" in block:
        values += [block[k]["median"] for k in ("se_standard", "se_random", "se_did")]
    if "status" in block or not values or not np.all(np.isfinite(values)):
        raise RuntimeError(f"{what}: no usable estimates ({block.get('status', 'non-finite or missing values')})")


def require_population(block: dict, what: str) -> None:
    """The dry run must exercise the analysis on real users, not report an empty population."""
    if "status" in block or not block.get("users"):
        raise RuntimeError(f"{what}: {block.get('status', 'no users')}")


def seed0(artefacts: list[dict]) -> list[dict]:
    return [a for a in artefacts if int(a["seed"]) == 0]


def subset(built: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in built.items()}


def run_tune(models=None) -> Path:
    from .tune import tune_rankers, tune_retrievers, write_selection
    with registry.logged("tune", {"models": list(models) if models else "all"}) as run:
        snapshot = ZOO / run["run_id"]  # never overwritten by a later tune run
        run["output"] = str(snapshot)
        p = prepare(1.0)
        selection = tune_rankers(p) if models is None else tune_rankers(p, models)
        write_selection(selection, out=snapshot / "selection_rankers.json")  # kept if the retriever part crashes
        selection.update(tune_retrievers(p))
        shutil.copyfile(write_selection(selection, out=snapshot / "selection.json"), SELECTION)
        run["selection_sha256"] = data.file_digest(SELECTION)
    return SELECTION


def _recall_se(scorers: dict, pool: Pool, targets: list[dict]) -> dict:
    """SE-only summary of Recall@100 on users with targets in both proxy sets, plus the H3 thresholds."""
    users = sorted(set(targets[0]) & set(targets[1]))
    if len(users) < 2:
        return {"users": len(users), "status": "fewer than two users with targets in both sets; no SEs"}
    cube = np.array([[evaluate_scorer(scorers[m], users, pool, targets[s])["recall@100"].to_numpy()
                      for s in (0, 1)] for m in A.RECALL_ORDER]).transpose(2, 0, 1)  # (users, models, sets)
    recall = A.se_summary(cube, A.RECALL_ORDER)
    # H3a / H3b are only two DiD tests (Holm over two): report their own thresholds.
    h3_multiplier = stats.mde_multipliers(2)[0]
    recall["h3_mde_did"] = {key: h3_multiplier * recall["se_by_pair"][pair]["did"]
                            for key, pair in (("H3a", "recall_popularity|recall_two_tower_logq"),
                                              ("H3b", "recall_two_tower|recall_two_tower_logq"))}
    return recall


def run_power_check() -> Path:
    """§7.3 power check of the selected zoo (seed 0) on the proxy pair: standard errors and MDEs only."""
    with registry.logged("power-check", source=SELECTION.relative_to(config.ROOT).as_posix()) as run:
        out_dir = config.RUNS / "power_check" / run["run_id"]
        run["output"] = str(out_dir)
        payload, run["selection_sha256"] = load_selection(SELECTION)
        _verify_inputs(payload)
        p = prepare(1.0)
        dev = device()
        artefacts = seed0(payload["artefacts"])
        sets = [p.built["valid"], p.built["calib"]]
        preds = [A.ranker_predictions(artefacts, p.space, b, "is_click", dev) for b in sets]
        cube = A.per_user_auc_cube(A.common_users(sets, "is_click"), sets, preds, "is_click")
        pool = Pool(p.space, p.videos, p.train)
        index = A.artefact_index(artefacts)
        scorers = {"recall_popularity": popularity_scorer(pool, p.train, dev),
                   "recall_itemcf": itemcf_scorer(pool, p.train, dev),
                   **{name: retriever_scorer(A.load_retriever(index[(name, "is_click")][0], p.space, dev),
                                             p.space, p.users, pool, dev)
                      for name in ("recall_two_tower", "recall_two_tower_logq")}}
        result = {"proxy": PROXY_NOTE + ", seed 0", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "ranking_is_click": A.power_check(cube),
                  "recall@100": _recall_se(scorers, pool, [pool.targets(f) for f in (p.valid, p.calib)])}
        out = A.write(result, out_dir, "power_check")
        for key in ("ranking_is_click", "recall@100"):  # freeze-zoo accepts only a check that produced SEs
            require_estimates(result[key], f"power check, {key}")
    return out


def run_pilot_power(epochs: int = 3) -> Path:
    """Pre-freeze pilot: default configs, short training, proxy window; reports standard errors only.

    Nothing here prints or stores model-to-model differences or levels, so the choice of the primary
    analysis can lean on detection thresholds without seeing effects. Pilot models are not kept.
    """
    from .retrieval import train_retriever
    from .smoke import LGB_CFG, RANKER_CFG, RETRIEVER_CFG
    from .train import lgb_matrix, predict, to_tensors, train_lightgbm, train_ranker
    cfg = {"ranker": RANKER_CFG, "lightgbm": {**LGB_CFG, "rounds": 200}, "retriever": RETRIEVER_CFG, "epochs": epochs}
    with registry.logged("pilot-power", cfg) as run:
        p = prepare(1.0)
        dev = device()
        sets = [p.built["valid"], p.built["calib"]]
        preds = [{"popularity": F.video_rate(p.space, b["video"], "is_click")[None, :]} for b in sets]
        for name in ("logreg", "lightgbm", "two_tower", "dcn", "din", "mmoe"):
            if name == "lightgbm":
                model, _ = train_lightgbm(LGB_CFG, p.built["train"], p.built["valid"], seed=0, rounds=200)
                for s, b in enumerate(sets):
                    preds[s][name] = model.predict(lgb_matrix(b), num_iteration=model.best_iteration)[None, :]
            else:
                model, _ = train_ranker(name, RANKER_CFG, p.space, p.built["train"], p.built["valid"], seed=0,
                                        dev=dev, max_epochs=epochs)
                for s, b in enumerate(sets):
                    preds[s][name] = predict(model, to_tensors(b, dev), 0 if name == "mmoe" else None)[None, :]
        cube = A.per_user_auc_cube(A.common_users(sets, "is_click"), sets, preds, "is_click")
        pool = Pool(p.space, p.videos, p.train)
        targets = [pool.targets(f) for f in (p.valid, p.calib)]
        scorers = {"recall_popularity": popularity_scorer(pool, p.train, dev),
                   "recall_itemcf": itemcf_scorer(pool, p.train, dev)}
        for name, logq in (("recall_two_tower", False), ("recall_two_tower_logq", True)):
            model, _ = train_retriever(RETRIEVER_CFG, p.space, p.built["train"], p.train, p.users, pool, targets[0],
                                       seed=0, logq=logq, dev=dev, max_epochs=epochs)
            scorers[name] = retriever_scorer(model, p.space, p.users, pool, dev)
        result = {"note": "pilot for design only: default configs, short training, SEs and thresholds, no effects",
                  "time": time.strftime("%Y-%m-%d %H:%M:%S"), "epochs": epochs,
                  "ranking_is_click": A.power_check(cube), "recall@100": _recall_se(scorers, pool, targets)}
        out = A.write(result, config.RUNS / "pilot" / run["run_id"], "power_pilot")
        run["output"] = str(out)
    return out


def _test_sets(p: Prepared) -> list[pd.DataFrame]:
    frames = []
    for name in ("standard_test", "random_test"):
        frame = F.dedup(data.load(name))
        frames.append(frame[frame["tab"] == 1].reset_index(drop=True))
    return frames


def _coverage(sets: list[dict], users: list, label: str) -> dict:
    both = set(sets[0]["user"]) & set(sets[1]["user"])
    return {"common_users": len(users), "users_in_both_sets": len(both),
            "share": len(users) / max(len(both), 1),
            "rows_excluded": [int((~np.isin(b["user"], users)).sum()) for b in sets]}


def _recall_coverage(frames: list[pd.DataFrame], targets: list[dict], users: list) -> dict:
    """R* against the users who appear in both sets (§9.1), and each set's users with at least one target."""
    both = set(frames[0]["user_id"]) & set(frames[1]["user_id"])
    return {"common_users": len(users), "users_in_both_sets": len(both), "share": len(users) / max(len(both), 1),
            "users_with_targets": [len(t) for t in targets]}


def ranking_block(sets: list[dict], preds: list[dict], label: str) -> dict:
    users = A.common_users(sets, label)
    cube = A.per_user_auc_cube(users, sets, preds, label)
    return {"coverage": _coverage(sets, users, label), **A.ranking_comparison(cube)}


def mechanism(built: dict, pred: dict, groups_of: dict, label: str) -> dict:
    """Per training-popularity group: rows, positive rate, mean prediction and global AUC (§10).

    Each seed is scored on its own and the results averaged; predictions are never averaged (no ensemble).
    """
    y = built["labels"][:, F.LABELS.index(label)]
    group = np.array([groups_of.get(int(v), 0) for v in built["video"]])
    out = {}
    for name, seeds in pred.items():
        rows = []
        for g in range(C.GROUPS):
            mask = group == g
            if not mask.any():
                rows.append({"group": g, "rows": 0, "rate": None, "mean_pred": None, "auc": None})
                continue
            rows.append({"group": g, "rows": int(mask.sum()), "rate": float(y[mask].mean()),
                         "mean_pred": float(np.mean([s[mask].mean() for s in seeds])),
                         "auc": float(np.mean([A.metrics.auc(y[mask], s[mask]) for s in seeds]))})
        out[name] = rows
    return out


def reload_check(artefacts: list[dict], p: Prepared, dev) -> dict:
    """Every saved model, loaded through the artefact list, must reproduce its registered validation metric."""
    from .train import validation_uauc
    index = A.artefact_index(artefacts)
    rows = []
    for label in ("is_click", "long_view"):
        preds = A.ranker_predictions(artefacts, p.space, p.built["valid"], label, dev)
        for name in A.RANKER_ORDER[1:]:
            for n, entry in enumerate(index.get((name, label), [])):  # MMoE's long_view head has no artefact
                rows.append({"model": name, "label": label, "seed": int(entry["seed"]),
                             "registered": entry["valid_metric"],
                             "reloaded": validation_uauc(p.built["valid"], preds[name][n], label)})
    pool = Pool(p.space, p.videos, p.train)
    targets = pool.targets(p.valid)
    for name in ("recall_two_tower", "recall_two_tower_logq"):
        for entry in index[(name, "is_click")]:
            model = A.load_retriever(entry, p.space, dev)
            table = evaluate_scorer(retriever_scorer(model, p.space, p.users, pool, dev), sorted(targets), pool, targets)
            rows.append({"model": name, "label": "is_click", "seed": int(entry["seed"]),
                         "registered": entry["valid_metric"], "reloaded": float(table["recall@100"].mean())})
    worst = float(np.max([abs(r["registered"] - r["reloaded"]) for r in rows]))  # NaN propagates and fails
    return {"tolerance": RELOAD_TOLERANCE, "max_abs_difference": worst, "passed": bool(worst < RELOAD_TOLERANCE),
            "rows": rows}


def run_analysis(proxy: bool = False) -> Path:
    """All of §7-§10. With proxy=True it is the dry run on standard valid vs random calibration (§11).

    The dry run first requires every saved model to reproduce its registered validation metric; its
    calibration study fits on random calibration rows up to 4/25 and scores those of 4/26-4/28. The
    formal run reads models only from configs/zoo_manifest.json and re-verifies the data cache first.
    """
    source = SELECTION if proxy else config.ROOT / data.ZOO_MANIFEST
    with registry.logged("analysis-proxy" if proxy else "analysis",
                         source=source.relative_to(config.ROOT).as_posix()) as run:
        out_dir = config.RUNS / "analysis" / run["run_id"]
        run["output"] = str(out_dir)
        payload, run["selection_sha256" if proxy else "zoo_manifest_sha256"] = load_selection(source)
        _analysis(payload, proxy, out_dir)
    return out_dir


def _analysis(payload: dict, proxy: bool, out_dir: Path) -> None:
    _verify_inputs(payload)
    artefacts = payload["artefacts"]
    p = prepare(1.0)
    dev = device()
    if proxy:
        check = reload_check(artefacts, p, dev)
        A.write(check, out_dir, "reload_check")
        if not check["passed"]:
            raise RuntimeError(f"reload check failed: max difference {check['max_abs_difference']}")
        frames = [p.valid, p.calib]
        sets = [p.built["valid"], p.built["calib"]]
    else:
        frames = _test_sets(p)
        sets = [F.build(f, p.space, p.users, p.videos, is_train=False) for f in frames]
    train_counts = p.train["video_id"].value_counts().to_dict()
    groups_of = C.popularity_groups(p.videos["video_id"].tolist(), train_counts)

    for label in ("is_click", "long_view"):
        preds = [A.ranker_predictions(artefacts, p.space, b, label, dev) for b in sets]
        result = {"label": label, "proxy": proxy, "main": ranking_block(sets, preds, label),
                  "own_set": [A.own_set_metrics(b, pr, label) for b, pr in zip(sets, preds)],
                  "mechanism": [mechanism(b, pr, groups_of, label) for b, pr in zip(sets, preds)]}
        if not proxy:
            halves = {}
            for name, window in {"first": (20220429, 20220503), "second": (20220504, 20220508)}.items():
                masks = [(f["date"] >= window[0]).to_numpy() & (f["date"] <= window[1]).to_numpy() for f in frames]
                halves[name] = ranking_block([subset(b, m) for b, m in zip(sets, masks)],
                                             [{k: v[:, m] for k, v in pr.items()} for pr, m in zip(preds, masks)],
                                             label)
            result["time_halves"] = halves
        A.write(result, out_dir, f"ranking_{label}")
        if proxy:
            require_population(result["main"], f"dry run, ranking {label}")

    if proxy:
        early = p.built["calib"]["date"] <= PROXY_CALIB_SPLIT
        fit_on, score_on = subset(p.built["calib"], early), subset(p.built["calib"], ~early)
    else:
        fit_on, score_on = p.built["calib"], sets[1]
    calibration = A.calibration_study(fit_on, score_on, A.ranker_predictions(artefacts, p.space, fit_on, "is_click", dev),
                                      A.ranker_predictions(artefacts, p.space, score_on, "is_click", dev), groups_of)
    A.write({"proxy": proxy, **calibration}, out_dir, "calibration")

    pool = Pool(p.space, p.videos, p.train)
    targets = [pool.targets(f) for f in frames]
    users = sorted(set(targets[0]) & set(targets[1]))
    index = A.artefact_index(artefacts)
    tables: list[dict] = [{}, {}]
    for s in (0, 1):
        tables[s]["recall_popularity"] = [evaluate_scorer(popularity_scorer(pool, p.train, dev), users, pool, targets[s])]
        tables[s]["recall_itemcf"] = [evaluate_scorer(itemcf_scorer(pool, p.train, dev), users, pool, targets[s])]
        for name in ("recall_two_tower", "recall_two_tower_logq"):
            tables[s][name] = [evaluate_scorer(retriever_scorer(A.load_retriever(entry, p.space, dev), p.space,
                                                                p.users, pool, dev), users, pool, targets[s])
                               for entry in index[(name, "is_click")]]
    recall = A.recall_comparison(tables)
    recall["coverage"] = _recall_coverage(frames, targets, users)
    A.write({"proxy": proxy, **recall}, out_dir, "recall")
    if proxy:
        require_population(recall, "dry run, recall")
