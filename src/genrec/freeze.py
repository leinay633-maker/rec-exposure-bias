"""Model-zoo freeze and the one-time opening of study 2's test data (protocol §6).

`inputs()` re-hashes every file the models and the analysis read (cache files, the three code tables, text
vectors and the encoder) against their recorded hashes; its fingerprint binds the proxy run, the power check,
the freeze and the opening to one and the same set of inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

from receval.data import file_digest
from receval.registry import code_hash

from . import config, data, encode, runlog, sids
from .train import LEARNED
from .tune import SENSITIVITY

CONFIGS = config.ROOT / "configs" / "kuairec"
MANIFEST, FREEZE = CONFIGS / "zoo_manifest.json", CONFIGS / "freeze.json"
PROTOCOL = "docs/kuairec/PROTOCOL.md"
FROZEN = ["src", "pyproject.toml", "uv.lock", "configs/kuairec/zoo_manifest.json"]
SEALED_DIR = config.CACHE / "sealed"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, encoding="utf-8")


def inputs() -> dict:
    """Every input file with its hash, each re-computed and checked against the recorded value."""
    cache = data.manifest()
    for name, expected in cache["files"].items():
        path = config.CACHE / ("users.json" if name == "users" else f"{name}.parquet")
        if not path.exists() or file_digest(path) != expected:
            raise PermissionError(f"cache file {path.name} is missing or does not match the cache manifest")
    codes = {}
    for seed in config.SEEDS:
        npy, meta = sids.path(seed)
        if file_digest(npy) != json.loads(meta.read_text(encoding="utf-8"))["sha256"]:
            raise PermissionError(f"{npy.name} does not match its metadata")
        codes[str(seed)] = {"npy": file_digest(npy), "meta": file_digest(meta)}
    text_meta = json.loads(encode.META_FILE.read_text(encoding="utf-8"))
    if file_digest(encode.EMB_FILE) != text_meta["sha256"]:
        raise PermissionError("text_emb.npy does not match its metadata")
    rev = json.loads(encode.REVISION_FILE.read_text(encoding="utf-8"))
    for name, expected in rev["files"].items():
        if file_digest(encode.MODEL_DIR / name) != expected:
            raise PermissionError(f"encoder file {name} does not match revision.json")
    return {"cache_manifest_sha256": file_digest(config.CACHE / "manifest.json"), "cache_manifest": cache,
            "sids": codes, "text_emb": {"npy": file_digest(encode.EMB_FILE), "meta": file_digest(encode.META_FILE)},
            "encoder": {"revision": rev["revision"], "files": rev["files"]}}


def fingerprint(snapshot: dict | None = None) -> str:
    return hashlib.sha256(json.dumps(snapshot or inputs(), sort_keys=True).encode()).hexdigest()


def _check_entries(block: dict, train_ok: dict, seen: set, label: str) -> None:
    for seed, e in block["seeds"].items():
        run = train_ok.get(e["run_id"])
        if run is None or e["run_id"] in seen:
            raise PermissionError(f"{label} seed {seed}: run {e['run_id']} is missing, failed or used twice")
        seen.add(e["run_id"])
        spec = run["config"]
        if spec["config"] != block["config"] or spec["seed"] != int(seed):
            raise PermissionError(f"{label} seed {seed}: the registered run trained something else")
        for key in ("path", "sha256", "valid_recall@100", "best_epoch", "epochs_run"):
            if run.get(key) != e.get(key):
                raise PermissionError(f"{label} seed {seed}: {key} differs from the registry")
        if file_digest(config.ROOT / e["path"]) != e["sha256"]:
            raise PermissionError(f"{label} seed {seed}: model file does not match its sha256")


def freeze_zoo() -> dict:
    sel_path = config.RUNS / "zoo" / "selection.json"
    selection, sel_sha = json.loads(sel_path.read_text(encoding="utf-8")), file_digest(sel_path)
    entries = runlog.entries()
    train_ok = {e["run_id"]: e for e in entries if e.get("kind") == "train" and e.get("status") == "ok"}
    seen: set = set()
    for kind in LEARNED:
        block = selection["models"].get(kind)
        if block is None or sorted(block["seeds"]) != sorted(str(s) for s in config.SEEDS):
            raise PermissionError(f"{kind}: the selection must hold seeds {config.SEEDS}")
        if block["seeds"]["0"]["run_id"] != block["winner_run_id"]:
            raise PermissionError(f"{kind}: seed 0 is not the winning tuning run")
        if any(train_ok[block["seeds"][s]["run_id"]]["config"]["kind"] != kind for s in block["seeds"]):
            raise PermissionError(f"{kind}: a seed entry points at another model kind")
        _check_entries(block, train_ok, seen, kind)
    for kind in SENSITIVITY["kinds"]:
        block = selection.get("sensitivity", {}).get(kind)
        if block is None or sorted(block["seeds"]) != sorted(str(s) for s in config.SEEDS):
            raise PermissionError(f"sensitivity {kind}: the selection must hold seeds {config.SEEDS} (§8)")
        if block["config"] != selection["models"][kind]["config"]:
            raise PermissionError(f"sensitivity {kind}: configuration differs from the selected one")
        _check_entries(block, train_ok, seen, f"sensitivity {kind}")
        if any(train_ok[e["run_id"]]["config"]["purpose"] != "sensitivity" for e in block["seeds"].values()):
            raise PermissionError(f"sensitivity {kind}: a seed entry is not a sensitivity run")
    snapshot = inputs()
    fp, current = fingerprint(snapshot), code_hash()
    for kind in ("analysis-proxy", "power-check"):
        if not any(e.get("kind") == kind and e.get("status") == "ok" and e.get("selection_sha256") == sel_sha
                   and e.get("inputs_sha256") == fp and e.get("code_hash") == current for e in entries):
            raise PermissionError(f"no successful {kind} run for this selection, these inputs and this code")
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "selection_sha256": sel_sha, "selection": selection,
                "inputs": snapshot, "inputs_sha256": fp, "code_hash_at_freeze": current}
    CONFIGS.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"manifest": str(MANIFEST), "sha256": file_digest(MANIFEST)}


def load_manifest() -> tuple[dict, str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if fingerprint() != manifest["inputs_sha256"]:
        raise PermissionError("the inputs no longer match the frozen manifest")
    blocks = [manifest["selection"]["models"][k] for k in LEARNED]
    blocks += [manifest["selection"]["sensitivity"][k] for k in SENSITIVITY["kinds"]]
    for block in blocks:
        for e in block["seeds"].values():
            if file_digest(config.ROOT / e["path"]) != e["sha256"]:
                raise PermissionError(f"{e['path']} changed since the freeze")
    return manifest, file_digest(MANIFEST)


def _full(rev: str) -> str:
    out = _git("rev-parse", "--verify", f"{rev}^{{commit}}")
    if out.returncode:
        raise PermissionError(f"cannot resolve commit {rev}")
    return out.stdout.strip()


def _normalised(text: str) -> str:
    return text.replace("\r\n", "\n")


def check_freeze() -> dict:
    spec = json.loads(FREEZE.read_text(encoding="utf-8"))
    protocol, zoo, head = _full(spec["protocol_commit"]), _full(spec["zoo_commit"]), _full("HEAD")
    if protocol == zoo or _git("merge-base", "--is-ancestor", protocol, zoo).returncode:
        raise PermissionError("the protocol freeze must come strictly before the zoo freeze")
    if _git("merge-base", "--is-ancestor", zoo, head).returncode:
        raise PermissionError("the zoo freeze is not an ancestor of HEAD")
    frozen_text = _git("show", f"{protocol}:{PROTOCOL}").stdout
    current = (config.ROOT / PROTOCOL).read_text(encoding="utf-8")
    if not frozen_text or not _normalised(current).startswith(_normalised(frozen_text)):
        raise PermissionError("the protocol was edited after its freeze, not only appended to")
    if _git("status", "--porcelain").stdout.strip():
        raise PermissionError("the working tree is not clean")
    if _git("diff", "--quiet", zoo, head, "--", *FROZEN).returncode:
        raise PermissionError("src/, pyproject.toml, uv.lock or the manifest changed since the zoo freeze")
    manifest, man_sha = load_manifest()
    raw = {n: file_digest(config.RAW / n) for n in (config.BIG, config.SMALL, config.CAPTION)}
    if raw != manifest["inputs"]["cache_manifest"]["raw_sha256"]:
        raise PermissionError("raw files changed")
    return {"protocol_commit": protocol, "zoo_commit": zoo, "head": head, "manifest_sha256": man_sha}


def _record(entry: dict) -> None:
    config.RUNS.mkdir(parents=True, exist_ok=True)
    with open(config.RUNS / data.OPENINGS, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _atomic_parquet(frame, path) -> str:
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return file_digest(path)


def open_test(reopen_reason: str | None = None) -> dict:
    """Open once. Any earlier opening record (even an interrupted one) or a leftover file needs a reason."""
    earlier = data.openings()
    leftovers = sorted(p.name for p in SEALED_DIR.glob("*")) if SEALED_DIR.exists() else []
    if (earlier or leftovers) and not reopen_reason:
        raise PermissionError("an opening was started before (or sealed files exist); reopening needs --reopen-reason")
    checks = check_freeze()
    with runlog.logged("open-test", {"reopen_reason": reopen_reason, **checks}) as state:
        base = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run_id": state["run_id"], **checks,
                "reopen_reason": reopen_reason}
        _record({**base, "status": "started"})
        users = data.load("users")
        SEALED_DIR.mkdir(parents=True, exist_ok=True)
        dense = data.read_small(config.RAW, set(users["test"]))[["user_id", "video_id", "watch_ratio"]]
        big = data.dedup(data.read_big(config.RAW, config.NATURAL_TEST, only_window=True))
        natural = big[big["user_id"].isin(set(users["others"]))][["user_id", "video_id", "date", "timestamp",
                                                                 "watch_ratio"]]
        files = {"dense_test": _atomic_parquet(dense.reset_index(drop=True), SEALED_DIR / "dense_test.parquet"),
                 "natural_test": _atomic_parquet(natural.reset_index(drop=True), SEALED_DIR / "natural_test.parquet")}
        rows = {"dense_test": int(len(dense)), "natural_test": int(len(natural))}
        entry = {**base, "status": "completed", "files": files, "rows": rows, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        _record(entry)
        state.update({"files": files, "rows": rows})
    return entry
