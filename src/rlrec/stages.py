"""Study 3 pipeline stages (docs/rlrec/PROTOCOL.md §10). Each stage is one registered run under .runs/rlrec/<stage>/;
on success it writes `latest.json` there, which the next stage reads. Model files and tables stay in .runs."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from genrec.generative import Codes
from receval import data as rdata
from receval import registry
from receval.data import file_digest

from . import config

RQ_SIZES = [64, 64, 64]


def _dir(stage: str, run_id: str) -> Path:
    out = config.RUNS / stage / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write(obj: dict, path: Path) -> Path:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8", newline="\n")
    return path


def _publish(stage: str, run_id: str, summary: dict) -> None:
    _write({"run_id": run_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **summary}, config.RUNS / stage / "latest.json")


def index_path(stage: str) -> Path:
    return config.RUNS / stage / "latest.json"


def inputs(*stages: str) -> dict:
    """sha256 of the upstream index files a run reads; each index pins the sha256 of every file it lists (§10)."""
    out = {}
    for s in stages:
        path = config.RUNS / s if s.endswith(".json") else index_path(s)
        out[s] = file_digest(path)
    return out


def latest(stage: str) -> dict:
    path = config.RUNS / stage / "latest.json"
    if not path.exists():
        raise FileNotFoundError(f"stage {stage} has not finished yet")
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.relative_to(config.ROOT).as_posix()


def device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("study 3 runs on the GPU")
    return torch.device("cuda")


def video_ids() -> np.ndarray:
    """Pool order: ascending video_id (receval.features.item_table)."""
    return np.sort(rdata.load("videos")["video_id"].to_numpy())


# ---- §3 text and semantic IDs -------------------------------------------------------------------------------------

def text() -> dict:
    from . import text as T
    with registry.logged("rlrec-text", {"captions": config.CAPTIONS.name, "categories": config.CATEGORIES.name},
                         runs=config.RUNS) as run:
        started = time.time()
        out = _dir("text", run["run_id"])
        ids = video_ids()
        frame = T.read_supplement(ids)
        texts = [T.describe(row) for row in frame.to_dict("records")]
        emb, truncated = T.embed(texts)
        np.save(out / "text_emb.npy", emb)
        _write({"video_ids": ids.tolist(), "texts": texts}, out / "texts.json")
        caption_ok = (~frame["malformed"]) & frame["caption"].fillna("").str.strip().ne("")
        summary = {"videos": int(len(ids)), "malformed": int(frame["malformed"].sum()),
                   "usable_caption": int(caption_ok.sum()), "no_text": int(sum(t == "无文本" for t in texts)),
                   "with_cover_text": int(sum("封面文字：" in t for t in texts)),
                   "with_categories": int(sum("类目：" in t for t in texts)), "texts_hitting_max_tokens": truncated,
                   "emb_sha256": file_digest(out / "text_emb.npy"), "texts_sha256": file_digest(out / "texts.json"),
                   "seconds": round(time.time() - started, 1)}
        _write(summary, out / "summary.json")
        run.update({"output": rel(out), **{k: summary[k] for k in ("no_text", "emb_sha256")}})
    _publish("text", run["run_id"], {"dir": rel(out), **summary})
    return summary


def load_embeddings() -> tuple[np.ndarray, dict]:
    info = latest("text")
    path = config.ROOT / info["dir"] / "text_emb.npy"
    if file_digest(path) != info["emb_sha256"]:
        raise PermissionError("text embeddings changed since the text stage")
    return np.load(path), info


def sids() -> dict:
    from . import text as T
    emb, info = load_embeddings()
    dev = device()
    with registry.logged("rlrec-sids", {"seeds": list(config.SEEDS), "emb_sha256": info["emb_sha256"],
                                        "inputs": inputs("text")}, runs=config.RUNS) as run:
        out = _dir("sids", run["run_id"])
        reports = {}
        for seed in config.SEEDS:
            codes, report = T.semantic_ids(emb, seed, dev)
            path = out / f"sid_seed{seed}.npy"
            np.save(path, codes)
            reports[seed] = {**report, "sha256": file_digest(path), "file": rel(path)}
        _write(reports, out / "sids.json")
        run["output"] = rel(out)
    _publish("sids", run["run_id"], {"dir": rel(out), "seeds": {str(s): r for s, r in reports.items()}})
    return {s: {k: r[k] for k in ("codes_used_per_level", "three_code_collisions", "fourth_code_max", "retried_with_double_steps")}
            for s, r in reports.items()}


def load_codes(seed: int) -> tuple[Codes, str]:
    info = latest("sids")["seeds"][str(seed)]
    path = config.ROOT / info["file"]
    if file_digest(path) != info["sha256"]:
        raise PermissionError(f"{info['file']} changed since the sids stage")
    codes = np.load(path)
    return Codes(codes, sizes=RQ_SIZES + [int(codes[:, -1].max()) + 1]), info["sha256"]


# ---- §4 SFT --------------------------------------------------------------------------------------------------------

def _valid_inputs(base) -> dict:
    from .data import validation_subset
    users, targets = validation_subset(base)
    return {"users": users, "hist": base.histories(users), "allowed": base.allowed(users), "targets": targets}


def sft() -> dict:
    from . import sft as S
    from .data import load_base
    dev = device()
    base = load_base()
    windows = base.windows()
    valid = _valid_inputs(base)
    with registry.logged("rlrec-sft-select", {"grid": S.GRID, "fixed": S.FIXED, "seeds": list(config.SEEDS),
                                              "inputs": inputs("sids")}, runs=config.RUNS) as run:
        out = _dir("sft", run["run_id"])
        codes0, sha0 = load_codes(0)
        trials = [S.train_run(cfg, 0, codes0, sha0, windows, valid, dev, purpose="select") for cfg in S.GRID]
        best = max(trials, key=lambda t: (t["valid_recall@100"], -S.GRID.index(t["config"])))
        entries = {0: best}
        for seed in config.SEEDS[1:]:
            codes, sha = load_codes(seed)
            entries[seed] = S.train_run(best["config"], seed, codes, sha, windows, valid, dev, purpose="seed")
        keep = ("run_id", "path", "sha256", "config", "seed", "codes_sha256", "valid_recall@100", "best_epoch",
                "epochs_run", "seconds", "batch")
        summary = {"windows": int(len(windows)), "valid_users": int(len(valid["users"])),
                   "trials": [{k: t[k] for k in keep} for t in trials], "config": best["config"],
                   "seeds": {str(s): {k: e[k] for k in keep} for s, e in entries.items()}}
        _write(summary, out / "selection.json")
        run.update({"output": rel(out), "selected_config": best["config"]})
    _publish("sft", run["run_id"], {"dir": rel(out), **summary})
    return {"config": best["config"], "valid": {s: e["valid_recall@100"] for s, e in entries.items()}}


def load_sft(seed: int, dev: torch.device):
    from . import sft as S
    entry = latest("sft")["seeds"][str(seed)]
    codes, sha = load_codes(seed)
    if sha != entry["codes_sha256"]:
        raise PermissionError("the SFT model was trained on other semantic IDs")
    return S.load(entry, codes, dev), codes


# ---- §5 reward tables and calibrations -----------------------------------------------------------------------------

REWARDS = ("R1", "R2", "R3", "R4", "R5", "R6", "R3_all_rows", "R3_actual_context")  # last two: §5 sensitivity
CELL_OF = {"R4": "B_standard_fixed", "R5": "B_random_fixed", "R6": "C_fixed"}


def _frozen_a():
    import lightgbm as lgb

    from receval import analysis as A
    manifest = json.loads((config.ROOT / rdata.ZOO_MANIFEST).read_text(encoding="utf-8"))
    frozen = A.artefact_index(manifest["artefacts"])[("lightgbm", "is_click")]
    out = []
    for e in frozen:
        if file_digest(config.ROOT / e["path"]) != e["sha256"]:
            raise PermissionError(f"frozen LightGBM {e['path']} does not match the manifest")
        out.append(lgb.Booster(model_file=str(config.ROOT / e["path"])))
    return out, [{"path": e["path"], "sha256": e["sha256"]} for e in frozen]


def _reward_models() -> dict:
    import lightgbm as lgb
    info = json.loads((config.RUNS / "reward_models" / "latest.json").read_text(encoding="utf-8"))
    if not info["equal"]:
        raise PermissionError("the R4–R6 retraining did not pass its check")
    out: dict = {}
    for m in info["models"]:
        path = config.ROOT / m["path"]
        if file_digest(path) != m["sha256"]:
            raise PermissionError(f"{m['path']} changed since it was saved")
        out.setdefault(m["cell"], []).append((m["seed"], lgb.Booster(model_file=str(path))))
    models = {cell: [b for _, b in sorted(v, key=lambda x: x[0])] for cell, v in out.items()}
    models["_info"] = info
    return models


def tables() -> dict:
    from receval import features as F
    from receval.train import lgb_matrix

    from . import rewards as RW
    from .data import load_base, reference_hour
    base = load_base()
    hour = reference_hour(base)
    bc = RW.Broadcast(base.p.space, base.p.users, base.p.videos, date=config.REF_DATE, hour=hour)
    if not np.array_equal(bc.video_ids, base.pool.video_ids):
        raise RuntimeError("broadcast videos are not in pool order")
    a_boosters, a_files = _frozen_a()
    rm = _reward_models()
    boosters = {"R1": a_boosters, **{r: rm[cell] for r, cell in CELL_OF.items()}}
    users = base.eligible
    cfg = {"reference_context": {"tab": 1, "date": config.REF_DATE, "hour": hour}, "users": int(len(users)),
           "reward_models_run": rm["_info"]["run_id"], "frozen_a": a_files,
           "inputs": inputs("reward_models/latest.json")}
    with registry.logged("rlrec-tables", cfg, runs=config.RUNS) as run:
        started = time.time()
        out = _dir("tables", run["run_id"])
        np.save(out / "users.npy", users)
        files = {"users": {"file": rel(out / "users.npy"), "sha256": file_digest(out / "users.npy")}}
        for name, bs in boosters.items():
            table = RW.predict_table(bs, bc, users)
            np.save(out / f"{name}.npy", table)
            files[name] = {"file": rel(out / f"{name}.npy"), "sha256": file_digest(out / f"{name}.npy")}
            del table
        item_group = RW.popularity_group_of_items(base.p.train, base.pool.video_ids)
        np.save(out / "item_group.npy", item_group)

        # calibration rows: the 61,434 random FIT rows B-random added (§5); all FIT rows for the sensitivity
        rows_info = rm["_info"]["fit_rows"]
        if file_digest(config.ROOT / rows_info["path"]) != rows_info["sha256"]:
            raise PermissionError("fit_rows.json changed")
        picked = json.loads((config.ROOT / rows_info["path"]).read_text(encoding="utf-8"))
        in_fit = base.p.calib["date"].between(*config.FIT).to_numpy()
        fit_frame = base.p.calib[in_fit].reset_index(drop=True)
        fit_built = {k: v[in_fit] for k, v in base.p.built["calib"].items()}
        pos = np.asarray(picked["random_positions"])
        if fit_frame.iloc[pos][["user_id", "video_id", "time_ms"]].to_numpy().tolist() != picked["random_keys"]:
            raise RuntimeError("the FIT rows no longer match the ones B-random added")

        def at_reference(frame) -> np.ndarray:
            ref = frame[["user_id", "video_id"]].copy()
            ref["date"], ref["hourmin"], ref["tab"], ref["time_ms"] = config.REF_DATE, hour * 100, 1, 0
            x = lgb_matrix(F.build(ref, base.p.space, base.p.users, base.p.videos, is_train=False))
            return np.mean([b.predict(x) for b in a_boosters], 0)

        def group_of_rows(frame) -> np.ndarray:
            return item_group[frame["video_id"].map(base.pool.row_of).to_numpy()]

        chosen = fit_frame.iloc[pos].reset_index(drop=True)
        y_all = fit_frame["is_click"].to_numpy(float)
        actual = np.mean([b.predict(lgb_matrix({k: v[pos] for k, v in fit_built.items()})) for b in a_boosters], 0)
        fits = {"main": RW.fit_calibrations(at_reference(chosen), y_all[pos], group_of_rows(chosen)),
                "all_rows": RW.fit_calibrations(at_reference(fit_frame), y_all, group_of_rows(fit_frame)),
                "actual_context": RW.fit_calibrations(actual, y_all[pos], group_of_rows(chosen))}
        r1 = np.load(out / "R1.npy")
        derived = {"R2": ("platt", "main"), "R3": ("grouped", "main"), "R3_all_rows": ("grouped", "all_rows"),
                   "R3_actual_context": ("grouped", "actual_context")}
        for name, (kind, which) in derived.items():
            np.save(out / f"{name}.npy", RW.calibrated(kind, r1, fits[which], item_group))
            files[name] = {"file": rel(out / f"{name}.npy"), "sha256": file_digest(out / f"{name}.npy")}
        del r1
        files["item_group"] = {"file": rel(out / "item_group.npy"), "sha256": file_digest(out / "item_group.npy")}
        calib = {k: {"platt": list(v["platt"]),
                     "grouped": {"global": list(v["grouped"]["global"]),
                                 "groups": {str(g): list(x) for g, x in v["grouped"]["groups"].items()}}}
                 for k, v in fits.items()}
        slope_zero = {k: [str(g) for g, x in v["grouped"]["groups"].items() if x[1] == 0.0]
                      + (["global"] if v["platt"][1] == 0.0 else []) for k, v in fits.items()}
        summary = {"files": files, "frozen_a": a_files, "calibrations": calib, "slope_at_zero": slope_zero,
                   "calibration_rows": {"main": int(len(pos)), "all_rows": int(len(fit_frame))},
                   "reference_context": cfg["reference_context"], "seconds": round(time.time() - started, 1)}
        _write(summary, out / "tables.json")
        run.update({"output": rel(out), "slope_at_zero": slope_zero})
    _publish("tables", run["run_id"], {"dir": rel(out), **summary})
    return {"reference_hour": hour, "slope_at_zero": slope_zero, "seconds": summary["seconds"]}


def load_table(name: str, mmap: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(eligible users, table) of a reward, hash-checked."""
    info = latest("tables")
    f = info["files"][name]
    path = config.ROOT / f["file"]
    if file_digest(path) != f["sha256"]:
        raise PermissionError(f"{f['file']} changed since the tables stage")
    users = np.load(config.ROOT / info["files"]["users"]["file"])
    return users, np.load(path, mmap_mode="r" if mmap else None)


def rows_of(users_all: np.ndarray, users: np.ndarray) -> np.ndarray:
    at = {int(u): i for i, u in enumerate(users_all)}
    return np.array([at[int(u)] for u in users], dtype=np.int64)


# ---- §6.1 π0 on the KL subset and HOLD, and β ---------------------------------------------------------------------

def o3() -> dict:
    from . import analytic
    from .data import hold, load_base, subset
    from .policy import Tree, exact_logprobs
    dev = device()
    base = load_base()
    pop = hold(base)
    kl_users = subset(pop.users, config.KL_SUBSET, config.KL_SUBSET_SEED)
    cfg = {"levels": list(config.KL_LEVELS), "kl_subset": config.KL_SUBSET, "kl_subset_seed": config.KL_SUBSET_SEED,
           "rewards": list(REWARDS), "sft_run": latest("sft")["run_id"], "tables_run": latest("tables")["run_id"],
           "inputs": inputs("sft", "sids", "tables")}
    with registry.logged("rlrec-o3", cfg, runs=config.RUNS) as run:
        started = time.time()
        out = _dir("o3", run["run_id"])
        files, betas = {}, {}
        for name, arr in (("kl_users", kl_users), ("hold_users", pop.users)):
            np.save(out / f"{name}.npy", arr)
            files[name] = {"file": rel(out / f"{name}.npy"), "sha256": file_digest(out / f"{name}.npy")}
        for seed in config.SEEDS:
            model, codes = load_sft(seed, dev)
            tree = Tree(codes)
            for name, users in (("kl", kl_users), ("hold", pop.users)):
                logp = exact_logprobs(model, codes, tree, base.histories(users), base.allowed(users), dev)
                path = out / f"pi0_{name}_seed{seed}.npy"
                np.save(path, logp)
                files[f"pi0_{name}_seed{seed}"] = {"file": rel(path), "sha256": file_digest(path)}
            logp0 = np.load(out / f"pi0_kl_seed{seed}.npy")
            for reward in REWARDS:
                all_users, table = load_table(reward)
                r = np.asarray(table[rows_of(all_users, kl_users)], dtype=np.float64)
                limit = analytic.kl_limit(logp0, r)
                for level in config.KL_LEVELS:
                    key = f"{reward}/{seed}/{level}"
                    try:
                        b, kl = analytic.bisect_beta(logp0, r, level)
                        betas[key] = {"beta": b, "kl": kl, "reachable": True, "kl_limit": limit}
                    except analytic.Unreachable as exc:
                        betas[key] = {"beta": None, "kl": None, "reachable": False, **exc.info}
            del model
            torch.cuda.empty_cache()
        summary = {"files": files, "betas": betas, "kl_users": int(len(kl_users)), "hold_users": int(len(pop.users)),
                   "hold_rows": int(pop.rows), "hold_dropped": pop.dropped, "seconds": round(time.time() - started, 1)}
        _write(summary, out / "o3.json")
        run.update({"output": rel(out), "unreachable": [k for k, v in betas.items() if not v["reachable"]]})
    _publish("o3", run["run_id"], {"dir": rel(out), **summary})
    return {"unreachable": [k for k, v in betas.items() if not v["reachable"]], "seconds": summary["seconds"]}


def beta(reward: str, seed: int, level: float) -> float:
    b = latest("o3")["betas"][f"{reward}/{seed}/{level}"]
    if not b["reachable"]:
        raise ValueError(f"{reward} cannot reach {level} nat with SFT seed {seed}")
    return float(b["beta"])


def load_o3_array(name: str) -> np.ndarray:
    f = latest("o3")["files"][name]
    path = config.ROOT / f["file"]
    if file_digest(path) != f["sha256"]:
        raise PermissionError(f"{f['file']} changed since the o3 stage")
    return np.load(path)


# ---- §6.2 RLOO -----------------------------------------------------------------------------------------------------

LR_GRID, PILOT_STEPS, LAMBDA_LEVEL = (1e-4, 3e-4, 1e-3), 1_500, 2.0
MAIN_RL = [(r, s, k, False) for r in ("R1", "R3", "R4", "R5") for s in config.SEEDS for k in config.SEEDS]  # §12.1
SECONDARY_RL = [(r, s, 0, True) for r in ("R1", "R3") for s in config.SEEDS]
RL_INPUTS = ("sft", "sids", "tables", "o3", "rl_pilot")


def _check_rl_inputs(name: str, cfg: dict) -> None:
    """A finished run is only reused or loaded on the upstream stages it was trained from (§1: one SFT starting point)."""
    if cfg["inputs"] != inputs(*RL_INPUTS):
        raise PermissionError(f"RL run {name} was trained on other upstream stages than the current ones")


def rl_lambda(reward: str, seed: int) -> float:
    """§6.2: β of the 2 nat level, else of the 1 nat level; ValueError if neither is reachable."""
    try:
        return beta(reward, seed, LAMBDA_LEVEL)
    except ValueError:
        return beta(reward, seed, config.KL_MAIN)


def rl_name(reward: str, sft_seed: int, rl_seed: int, grpo: bool) -> str:
    return f"{reward}{'_grpo' if grpo else ''}/sft{sft_seed}/rl{rl_seed}"


def _rl_data(base, reward: str, seed: int):
    from .rl import RLData
    users_all, table = load_table(reward, mmap=False)
    if not np.array_equal(users_all, base.eligible):
        raise RuntimeError("the reward table's users are not the eligible users")
    kl_users = load_o3_array("kl_users")
    return RLData(hist=base.histories(users_all), allowed=base.allowed(users_all), rewards=np.asarray(table),
                  kl_hist=base.histories(kl_users), kl_allowed=base.allowed(kl_users),
                  kl_logp0=load_o3_array(f"pi0_kl_seed{seed}"))


def rl_pilot() -> dict:
    from . import rl as RL
    from .data import load_base
    from .policy import Tree
    dev = device()
    base = load_base()
    pick = None
    for reward, seed in [("R1", s) for s in config.SEEDS] + [("R3", s) for s in config.SEEDS]:
        try:
            pick = (reward, seed, rl_lambda(reward, seed))
            break
        except ValueError:
            continue
    cfg = {"lr_grid": list(LR_GRID), "steps": PILOT_STEPS, "pick": pick, "o3_run": latest("o3")["run_id"],
           "inputs": inputs("sft", "sids", "tables", "o3")}
    with registry.logged("rlrec-rl-pilot", cfg, runs=config.RUNS) as run:
        out = _dir("rl_pilot", run["run_id"])
        results = {}
        if pick is None:  # no reward has a λ on any seed: no pilot, default learning rate (§6.2)
            summary = {"lr": max(LR_GRID), "lambda": None, "pick": None, "results": {}}
            _write(summary, out / "pilot.json")
            run.update({"output": rel(out), "lr": summary["lr"]})
            _publish("rl_pilot", run["run_id"], {"dir": rel(out), **summary})
            return {"lr": summary["lr"], "pilot": "skipped: no λ"}
        reward, seed, lam = pick
        data = _rl_data(base, reward, seed)
        for lr in LR_GRID:
            model, codes = load_sft(seed, dev)
            ref, _ = load_sft(seed, dev)
            res = RL.train(model, ref, codes, Tree(codes), data, lam, lr, seed=100 * seed, dev=dev, levels=(1.0,),
                           max_steps=PILOT_STEPS)
            sel = res["selected"][1.0]
            results[str(lr)] = {"reached": sel["reached"], "step": sel["step"], "kl": sel["kl"],
                                "max_rise_per_100": RL.max_rise(res["log"]), "max_slope": RL.max_slope(res["log"]),
                                "log": res["log"]}
            del model, ref
            torch.cuda.empty_cache()
        reached = [lr for lr in LR_GRID if results[str(lr)]["reached"]]
        chosen = min(reached) if reached else max(LR_GRID)
        summary = {"lr": chosen, "lambda": lam, "pick": list(pick[:2]), "results": results,
                   "max_rise_per_100": results[str(chosen)]["max_rise_per_100"],
                   "max_slope": results[str(chosen)]["max_slope"]}
        _write(summary, out / "pilot.json")
        run.update({"output": rel(out), "lr": chosen})
    _publish("rl_pilot", run["run_id"], {"dir": rel(out), **summary})
    return {"lr": chosen, "lambda": lam, "max_rise_per_100": summary["max_rise_per_100"], "max_slope": summary["max_slope"],
            "pilot": {k: {x: v[x] for x in ("reached", "step", "kl", "max_rise_per_100", "max_slope")}
                      for k, v in results.items()}}


def rl_runs(which: str = "main") -> dict:
    """Formal RLOO runs; resumable: runs already in rl/<which>.json are skipped."""
    from . import rl as RL
    from .data import load_base
    from .policy import Tree
    dev = device()
    base = load_base()
    lr = latest("rl_pilot")["lr"]
    specs = MAIN_RL if which == "main" else SECONDARY_RL
    index_path = config.RUNS / "rl" / f"{which}.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"runs": {}}
    cache: dict = {}
    for reward, sft_seed, rl_seed, grpo in specs:
        name = rl_name(reward, sft_seed, rl_seed, grpo)
        if name in index["runs"]:
            _check_rl_inputs(name, index["runs"][name]["config"])
            continue
        key = (reward, sft_seed)
        if key not in cache:
            cache.clear()
            cache[key] = _rl_data(base, reward, sft_seed)
        data = cache[key]
        try:
            lam = rl_lambda(reward, sft_seed)
        except ValueError:  # neither level reachable: no RL for this reward (§6.2)
            index.setdefault("skipped", []).append(name)
            _write(index, index_path)
            continue
        torch_seed = 100 * sft_seed + rl_seed
        cfg = {"reward": reward, "sft_seed": sft_seed, "rl_seed": rl_seed, "torch_seed": torch_seed, "grpo": grpo,
               "lambda": lam, "lr": lr, "levels": list(config.KL_LEVELS), "max_steps": RL.MAX_STEPS,
               "users_per_step": RL.USERS_PER_STEP, "samples": RL.SAMPLES, "check_every": RL.CHECK_EVERY,
               "fine_every": RL.FINE_EVERY, "every_step_until": RL.EVERY_STEP_UNTIL,
               "sft_run": latest("sft")["run_id"], "tables_run": latest("tables")["run_id"],
               "o3_run": latest("o3")["run_id"], "pilot_run": latest("rl_pilot")["run_id"],
               "inputs": inputs(*RL_INPUTS)}
        with registry.logged("rlrec-rl", cfg, runs=config.RUNS) as run:
            out = _dir("rl", run["run_id"])
            model, codes = load_sft(sft_seed, dev)
            ref, _ = load_sft(sft_seed, dev)
            res = RL.train(model, ref, codes, Tree(codes), data, lam, lr, seed=torch_seed, dev=dev,
                           levels=config.KL_LEVELS, normalise=grpo, every_step_until=RL.EVERY_STEP_UNTIL)
            levels = {}
            for level, sel in res["selected"].items():
                path = out / f"kl{level}.pt"
                torch.save({"state_dict": sel["state"], "level": level, "step": sel["step"], "kl": sel["kl"],
                            "reached": sel["reached"], "aligned": sel["aligned"], **cfg}, path)
                levels[str(level)] = {"file": rel(path), "sha256": file_digest(path), "step": sel["step"],
                                      "kl": sel["kl"], "reached": sel["reached"], "aligned": sel["aligned"]}
            _write({"config": cfg, "levels": levels, "log": res["log"]}, out / "rl.json")
            run.update({"output": rel(out), "levels": {k: {x: v[x] for x in ("step", "kl", "aligned")} for k, v in levels.items()}})
            del model, ref
            torch.cuda.empty_cache()
        index["runs"][name] = {"run_id": run["run_id"], "dir": rel(out), "config": cfg, "levels": levels}
        _write(index, index_path)
    return {n: {lv: (round(v["kl"], 4), v["aligned"]) for lv, v in r["levels"].items()} for n, r in index["runs"].items()}


def load_rl(name: str, level: float, which: str, dev: torch.device):
    from .sft import build
    entry = json.loads((config.RUNS / "rl" / f"{which}.json").read_text(encoding="utf-8"))["runs"][name]
    _check_rl_inputs(name, entry["config"])
    f = entry["levels"][str(level)]
    path = config.ROOT / f["file"]
    if file_digest(path) != f["sha256"]:
        raise PermissionError(f"{f['file']} changed since it was saved")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if any(saved[k] != entry["config"][k] for k in ("reward", "sft_seed", "rl_seed", "grpo", "lr", "inputs")):
        raise PermissionError(f"{f['file']} does not carry the recorded configuration")
    _, codes = load_sft(entry["config"]["sft_seed"], torch.device("cpu"))
    sft_cfg = latest("sft")["config"]
    model = build(codes, sft_cfg, entry["config"]["sft_seed"])
    model.load_state_dict(saved["state_dict"])
    return model.to(dev).eval(), codes, f


# ---- §9 precision check before the model freeze -------------------------------------------------------------------

HYPOTHESES = {"H1": (("O3", "R1"), ("pi0", None)), "H2": (("O3", "R3"), ("O3", "R1")), "H3": (("O3", "R5"), ("O3", "R4")),
              "H4": (("O4", "R1"), ("pi0", None)), "H5": (("O4", "R3"), ("O4", "R1")),
              "H6": (("O4", "R5"), ("O4", "R4"))}


def policy_scores(kind: str, reward: str | None, seed: int, users: np.ndarray, logp0: np.ndarray, base, dev,
                  rl_seed: int | None = None, level: float = config.KL_MAIN, which: str = "main") -> np.ndarray:
    """Ranking scores of one model of a policy on `users` (rows aligned with logp0)."""
    if kind == "pi0":
        return logp0
    if kind == "O3":
        all_users, table = load_table(reward)
        r = np.asarray(table[rows_of(all_users, users)], dtype=np.float64)
        return np.where(np.isfinite(logp0), logp0 + r / beta(reward, seed, level), -np.inf)
    if kind == "O4":
        from .policy import Tree, exact_logprobs
        model, codes, _ = load_rl(rl_name(reward, seed, rl_seed, False), level, which, dev)
        out = exact_logprobs(model, codes, Tree(codes), base.histories(users), base.allowed(users), dev)
        del model
        torch.cuda.empty_cache()
        return out
    raise ValueError(kind)


def seed_counts(kind: str, reward: str | None, pop, logp0s: dict, base, dev, ks) -> dict:
    """k -> per-user rows and clicks averaged over the policy's models (3 SFT seeds, x 3 RL seeds for O4), and
    per SFT seed (averaged over its RL seeds)."""
    from .evaluate import counts, row_ranks
    per_seed = {}
    for seed in config.SEEDS:
        rl_seeds = config.SEEDS if kind == "O4" else (None,)
        acc = None
        for k_rl in rl_seeds:
            scores = policy_scores(kind, reward, seed, pop.users, logp0s[seed], base, dev, rl_seed=k_rl)
            c = counts(row_ranks(scores, pop), pop, ks)
            acc = c if acc is None else {k: {m: acc[k][m] + c[k][m] for m in c[k]} for k in ks}
        per_seed[seed] = {k: {m: v / len(rl_seeds) for m, v in acc[k].items()} for k in ks}
    mean = {k: {m: np.mean([per_seed[s][k][m] for s in config.SEEDS], 0) for m in ("rows", "clicks")} for k in ks}
    return {"mean": mean, "per_seed": per_seed}


def rl_index(which: str) -> dict:
    path = config.RUNS / "rl" / f"{which}.json"
    return json.loads(path.read_text(encoding="utf-8"))["runs"] if path.exists() else {}


def hypothesis_status() -> dict:
    """§1: each main hypothesis is testable, or not for the listed reasons."""
    betas = latest("o3")["betas"]
    runs = rl_index("main")
    out = {}
    for h, (a, b) in HYPOTHESES.items():
        reasons = []
        for kind, reward in (a, b):
            if kind == "O3":
                reasons += [f"{reward}/SFT {s}: 1 nat unreachable" for s in config.SEEDS
                            if not betas[f"{reward}/{s}/{config.KL_MAIN}"]["reachable"]]
            elif kind == "O4":
                for s in config.SEEDS:
                    for k in config.SEEDS:
                        name = rl_name(reward, s, k, False)
                        if name not in runs:
                            reasons.append(f"{name}: not trained")
                        elif not runs[name]["levels"][str(config.KL_MAIN)]["aligned"]:
                            reasons.append(f"{name}: 1 nat checkpoint not aligned (KL {runs[name]['levels'][str(config.KL_MAIN)]['kl']:.4f})")
        out[h] = {"testable": not reasons, "reasons": reasons}
    return out


def power() -> dict:
    from receval import stats

    from .data import hold, load_base
    dev = device()
    base = load_base()
    pop = hold(base)
    if not np.array_equal(pop.users, load_o3_array("hold_users")):
        raise RuntimeError("HOLD users differ from the o3 stage")
    logp0s = {s: load_o3_array(f"pi0_hold_seed{s}") for s in config.SEEDS}
    ks = (500, 1000)
    status = hypothesis_status()
    testable = [h for h in HYPOTHESES if status[h]["testable"]]
    policies = sorted({("pi0", None)} | {p_ for h in testable for p_ in HYPOTHESES[h]}, key=str)
    cfg = {"ks": list(ks), "hypotheses": list(HYPOTHESES), "status": status, "o3_run": latest("o3")["run_id"],
           "inputs": inputs("sft", "sids", "tables", "o3", "rl/main.json")}
    with registry.logged("rlrec-power", cfg, runs=config.RUNS) as run:
        started = time.time()
        out = _dir("power", run["run_id"])
        got = {p_: seed_counts(p_[0], p_[1], pop, logp0s, base, dev, ks)["mean"] for p_ in policies}
        retention = pop.rows / pop.dropped["rows"]
        rows_den, users_den = config.TEST_ROWS * retention, len(base.eligible)
        rows_factor, users_factor = np.sqrt(pop.rows / rows_den), np.sqrt(len(pop.users) / users_den)
        mult = stats.mde_multipliers(len(HYPOTHESES))[0]
        result = {"hold_users": int(len(pop.users)), "hold_rows": int(pop.rows), "status": status,
                  "extrapolation": {"rows_denominator": rows_den, "hold_row_retention": retention,
                                    "users_denominator": users_den, "multiplier": mult}, "ks": {}}
        at = {p_: i for i, p_ in enumerate(policies)}
        for k in ks:
            cols = [got[p_][k][m] for p_ in policies for m in ("rows", "clicks")]
            rep = stats.joint_user_bootstrap(np.stack(cols, 1), samples=10_000, seed=0)
            value = {p_: rep[:, 2 * i + 1] / rep[:, 2 * i] for p_, i in at.items()}
            base_value = float(got[("pi0", None)][k]["clicks"].sum() / got[("pi0", None)][k]["rows"].sum())
            block = {}
            for h in testable:
                a, b = HYPOTHESES[h]
                se = float(np.std(value[a] - value[b], ddof=1))
                degenerate = not (np.isfinite(se) and se > 0)
                ext = max(se * rows_factor, se * users_factor) if not degenerate else float("nan")
                block[h] = {"se_hold": se, "se_test_extrapolated": float(ext), "mde": float(mult * ext),
                            "degenerate": degenerate, "passes": (not degenerate) and mult * ext < 0.05 * base_value}
            result["ks"][str(k)] = {"pi0_value_hold": base_value, "threshold": 0.05 * base_value, "hypotheses": block}
        p500 = {h: result["ks"]["500"]["hypotheses"][h]["passes"] for h in testable}
        p1000 = {h: result["ks"]["1000"]["hypotheses"][h]["passes"] for h in testable}
        untestable = [h for h in HYPOTHESES if h not in testable]
        if not testable:
            decision = {"branch": 0, "k": 500, "family": [], "note": "no testable main hypothesis"}
        elif all(p500.values()):
            decision = {"branch": 1, "k": 500, "family": list(HYPOTHESES)}
        elif all(p1000.values()):
            decision = {"branch": 2, "k": 1000, "family": list(HYPOTHESES)}
        elif any(p1000.values()):
            kept = [h for h in HYPOTHESES if h in untestable or p1000.get(h)]
            decision = {"branch": 3, "k": 1000, "family": kept, "removed": [h for h in testable if not p1000[h]]}
        else:
            decision = {"branch": 4, "k": 1000, "family": [], "note": "precision insufficient"}
        decision["untestable"] = untestable
        result["decision"] = decision
        result["seconds"] = round(time.time() - started, 1)
        _write(result, out / "power.json")  # standard errors and decisions only: no differences, no HOLD values of them
        run.update({"output": rel(out), "decision": decision})
    _publish("power", run["run_id"], {"dir": rel(out), **result})
    return {"decision": decision, "status": status, "pi0_value_hold": {k: v["pi0_value_hold"] for k, v in result["ks"].items()},
            "mde": {k: {h: round(x["mde"], 4) for h, x in v["hypotheses"].items()} for k, v in result["ks"].items()}}


# ---- §9-§10 model freeze, HOLD report (after the freeze) and the single test evaluation -------------------------

MANIFEST = config.ROOT / "configs" / "rlrec" / "zoo_manifest.json"
TEST_LOG = config.RUNS / "test_evaluations.jsonl"


def _git(*args: str) -> str:
    import subprocess
    return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, encoding="utf-8",
                          check=True).stdout.strip()


def _artefacts() -> list[dict]:
    out = []

    def add(role: str, path: str, sha: str) -> None:
        out.append({"role": role, "path": path, "sha256": sha})

    t = latest("text")
    add("text_emb", f"{t['dir']}/text_emb.npy", t["emb_sha256"])
    add("texts", f"{t['dir']}/texts.json", t["texts_sha256"])
    for s, r in latest("sids")["seeds"].items():
        add(f"sid/{s}", r["file"], r["sha256"])
    sft_info = latest("sft")
    for s, e in sft_info["seeds"].items():
        add(f"sft/{s}", e["path"], e["sha256"])
    rm = json.loads((config.RUNS / "reward_models" / "latest.json").read_text(encoding="utf-8"))
    for m in rm["models"]:
        add(f"reward_model/{m['cell']}/{m['seed']}", m["path"], m["sha256"])
    add("fit_rows", rm["fit_rows"]["path"], rm["fit_rows"]["sha256"])
    for name, f in latest("tables")["files"].items():
        add(f"table/{name}", f["file"], f["sha256"])
    for i, e in enumerate(latest("tables")["frozen_a"]):
        add(f"frozen_a/{i}", e["path"], e["sha256"])
    for name, f in latest("o3")["files"].items():
        add(f"o3/{name}", f["file"], f["sha256"])
    for which in ("main", "secondary"):
        index = json.loads((config.RUNS / "rl" / f"{which}.json").read_text(encoding="utf-8"))
        for name, r in index["runs"].items():
            for level, f in r["levels"].items():
                add(f"rl/{which}/{name}/{level}", f["file"], f["sha256"])
    return out


CHAIN = {"sids": ("text",), "sft": ("sids",), "tables": ("reward_models/latest.json",), "o3": ("sft", "sids", "tables"),
         "rl_pilot": ("sft", "sids", "tables", "o3"), "power": ("sft", "sids", "tables", "o3", "rl/main.json")}


def _check_chain() -> None:
    """§10: every stage's and RL run's registered inputs are still the current index files, and every run that made an
    artefact used the current data. A run's start record holds its config; its final record may add to it."""
    entries = registry.read(config.RUNS)
    finished = {e["run_id"] for e in entries if e.get("status") == "ok"}
    registered = {e["run_id"]: e for e in entries if e.get("status") == "started" and e["run_id"] in finished}
    for stage, upstream in CHAIN.items():
        if registered[latest(stage)["run_id"]]["config"]["inputs"] != inputs(*upstream):
            raise PermissionError(f"stage {stage} was run on other upstream stages than the current ones")
    runs = [latest(s)["run_id"] for s in ("text", "reward_models", *CHAIN)]
    runs += [e["run_id"] for e in latest("sft")["seeds"].values()]
    for which in ("main", "secondary"):
        for name, run in rl_index(which).items():
            _check_rl_inputs(name, run["config"])
            runs.append(run["run_id"])
    data = registry.data_hash()
    for run_id in runs:
        if registered[run_id]["data_hash"] != data:
            raise PermissionError(f"run {run_id} used other data than the current cache")


def freeze() -> dict:
    """Write configs/rlrec/zoo_manifest.json; the caller commits it (the model freeze point)."""
    from genrec import encode as genc
    rdata.verify_data()  # registry.data_hash covers the data manifest only, not the cached files
    _check_chain()
    encoder = json.loads(genc.REVISION_FILE.read_text(encoding="utf-8"))
    if encoder["revision"] != genc.REVISION:  # the text stage ran on this pinned revision (rlrec.text.embed)
        raise PermissionError(f"encoder revision {encoder['revision']} is not the pinned {genc.REVISION}")
    arts = _artefacts()
    for a in arts:
        if file_digest(config.ROOT / a["path"]) != a["sha256"]:
            raise PermissionError(f"{a['path']} changed since it was recorded")
    tables = latest("tables")
    indexes = [f"{s}/latest.json" for s in ("text", "sids", "sft", "reward_models", "tables", "o3", "rl_pilot", "power")]
    indexes += ["rl/main.json", "rl/secondary.json"]
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "head": _git("rev-parse", "HEAD"),
                "protocol_commit": config.PROTOCOL_FREEZE, "code_hash": registry.code_hash(),
                "data_hash": registry.data_hash(), "indexes": {i: file_digest(config.RUNS / i) for i in indexes},
                "stages": {s: latest(s)["run_id"] for s in ("text", "sids", "sft", "tables", "o3", "rl_pilot", "power")},
                "sft_config": latest("sft")["config"], "rl_lr": latest("rl_pilot")["lr"],
                "reference_context": tables["reference_context"], "calibrations": tables["calibrations"],
                "encoder": encoder, "betas": latest("o3")["betas"], "decision": latest("power")["decision"], "artefacts": arts}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    _write(manifest, MANIFEST)
    return {"artefacts": len(arts), "decision": manifest["decision"], "manifest": rel(MANIFEST)}


def _check_freeze() -> dict:
    """§10: the committed manifest, the append-only protocol, the code, the data, the indexes and every artefact."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    _git("ls-files", "--error-unmatch", rel(MANIFEST))
    if _git("status", "--porcelain", "--", rel(MANIFEST)):
        raise PermissionError("the model manifest has uncommitted changes")
    frozen = _git("show", f"{manifest['protocol_commit']}:docs/rlrec/PROTOCOL.md").replace("\r\n", "\n")
    now = (config.ROOT / "docs/rlrec/PROTOCOL.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    if not now.startswith(frozen.rstrip("\n")):
        raise PermissionError("the protocol changed other than by appending since its freeze")
    if registry.code_hash() != manifest["code_hash"]:
        raise PermissionError("the code differs from the model freeze")
    if registry.data_hash() != manifest["data_hash"]:
        raise PermissionError("study 1's data cache differs from the model freeze")
    rdata.verify_data()
    _check_chain()
    for i, sha in manifest["indexes"].items():
        if file_digest(config.RUNS / i) != sha:
            raise PermissionError(f"index {i} differs from the model freeze")
    for a in manifest["artefacts"]:
        if file_digest(config.ROOT / a["path"]) != a["sha256"]:
            raise PermissionError(f"{a['path']} differs from the frozen manifest")
    return manifest


def _strip(ev: dict) -> dict:
    """Drop per-user arrays before writing."""
    keep = ("models", "hot_share", "coverage", "regular_recall500", "kl", "proxy", "support_median", "ess_mix",
            "per_model", "seed_values", "checkpoints")
    return {**{k: v for k, v in ev.items() if k != "policies"},
            "policies": {n: {k: v for k, v in p.items() if k in keep} for n, p in ev["policies"].items()}}


def _claim_test(started: str, reopen_reason: str | None) -> None:
    """§0: one evaluation, after the HOLD dry run on the same manifest. The first run creates the log atomically; only a
    run that started and never finished can be resumed, on the same manifest and with a recorded reason; a completed
    one cannot be reopened here. One operator runs this, so there is no lock against a run still in progress."""
    manifest_sha = file_digest(MANIFEST)
    entries = registry.read(config.RUNS)
    if any(e.get("kind") == "rlrec-report-test" and e.get("status") == "ok" for e in entries):
        raise PermissionError("study 3 already evaluated the test once")
    if not any(e.get("kind") == "rlrec-report-hold" and e.get("status") == "ok"
               and e["config"].get("manifest_sha256") == manifest_sha for e in entries):
        raise PermissionError("run hold-report on this manifest before the test")
    entry = json.dumps({"started": started, "head": _git("rev-parse", "HEAD"), "status": "started",
                        "manifest_sha256": manifest_sha, "reopen_reason": reopen_reason}) + "\n"
    try:
        with open(TEST_LOG, "x", encoding="utf-8") as f:
            f.write(entry)
    except FileExistsError:
        previous = [json.loads(x) for x in TEST_LOG.read_text(encoding="utf-8").splitlines()]
        if any(r.get("status") == "completed" for r in previous):
            raise PermissionError("study 3 already evaluated the test once") from None
        if not reopen_reason:
            raise PermissionError("an earlier test run did not finish; resuming needs --reopen-reason") from None
        if any(r.get("manifest_sha256") not in (None, manifest_sha) for r in previous):
            raise PermissionError("an earlier test run used another manifest") from None
        with open(TEST_LOG, "a", encoding="utf-8") as f:
            f.write(entry)


def _report(population: str, reopen_reason: str | None = None) -> dict:  # noqa: C901
    from receval import features as F

    from . import report as R
    from .data import hold, load_base, population as make_population
    manifest = _check_freeze()
    dev = device()
    decision = manifest["decision"]
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    if population == "test":
        _claim_test(started, reopen_reason)
    base = load_base()
    if population == "test":
        frame = F.dedup(rdata.load("random_test"))
        pop = make_population(base, frame[frame["date"].between(*config.TEST)])
        std = F.dedup(rdata.load("standard_test"))
        std_targets = base.pool.targets(std[std["tab"] == 1])
    else:
        pop = hold(base)
        std_targets = base.pool.targets(base.p.valid)
    cfg = {"population": population, "manifest_sha256": file_digest(MANIFEST), "decision": decision}
    with registry.logged(f"rlrec-report-{population}", cfg, runs=config.RUNS) as run:
        t0 = time.time()
        out = _dir(f"report_{population}", run["run_id"])
        ev = R.evaluate(pop, base, std_targets, dev, diagnostics=(population == "hold"))
        st = R.statistics(ev, decision["k"], decision["family"], hypothesis_status(), descriptive=(population == "hold"))
        result = {"note": f"docs/rlrec/PROTOCOL.md §7–§8, population {population}", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "decision": decision, "evaluation": _strip(ev), "statistics": st, "seconds": round(time.time() - t0, 1)}
        path = _write(result, out / f"{population}.json")
        run.update({"output": rel(path), "sha256": file_digest(path)})
    if population == "test":
        with open(TEST_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"started": started, "completed": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "completed",
                                "output": rel(path), "sha256": file_digest(path)}) + "\n")
    return {h: {k: r.get(k) for k in ("point", "ci95", "p", "holm_reject", "seed_support", "holds")}
            for h, r in st["hypotheses"].items()}


def hold_report() -> dict:
    return _report("hold")


def test(reopen_reason: str | None = None) -> dict:
    return _report("test", reopen_reason)
