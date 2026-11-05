"""Download checks and the parquet cache for KuaiRand-Pure, with the test sets sealed.

Before opening, test-period feedback is never parsed into any DataFrame: `prepare` reads each log
twice, first only the META columns to place rows in windows, then every column while skipping the
test-period rows by line number. `open_test` runs once, only when `check_freeze` passes; it writes the
labelled files atomically and records them, and `load` serves a sealed file only if its hash matches
that record.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

FEEDBACK = ("is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate", "long_view",
            "play_time_ms", "profile_stay_time", "comment_stay_time", "is_profile_enter")
META = ("user_id", "video_id", "date", "hourmin", "time_ms", "duration_ms", "is_rand", "tab")
SEALED = ("standard_test", "random_test")
KNOWN = ("standard_train", "standard_valid", "random_calib", "standard_test_meta", "random_test_meta",
         "users", "videos") + SEALED
FREEZE_FILE = config.ROOT / "configs" / "freeze.json"
ZOO_MANIFEST = "configs/zoo_manifest.json"
OPENINGS = "test_openings.jsonl"
# Paths whose content must not change between the zoo freeze and opening the test sets.
FROZEN_PATHS = ("src", "pyproject.toml", "uv.lock", ZOO_MANIFEST)
# Artefacts the zoo must contain: model -> label -> seeds (protocol §5.2, §9.3).
REQUIRED_ARTEFACTS = {
    "logreg": {"is_click": (0,), "long_view": (0,)},
    **{m: {"is_click": (0, 1, 2), "long_view": (0, 1, 2)} for m in ("lightgbm", "two_tower", "dcn", "din")},
    "mmoe": {"is_click": (0, 1, 2)},
    "recall_two_tower": {"is_click": (0, 1, 2)},
    "recall_two_tower_logq": {"is_click": (0, 1, 2)},
}


def file_digest(path: Path, algorithm: str = "sha256", chunk: int = 1 << 20) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_and_extract(archive: Path = config.PURE_ARCHIVE, expected_md5: str = config.PURE_MD5,
                       dest: Path = config.RAW) -> Path:
    digest = file_digest(archive, "md5")
    if digest != expected_md5:
        raise ValueError(f"md5 mismatch for {archive}: got {digest}, expected {expected_md5}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    return dest


def in_window(frame: pd.DataFrame, window: tuple[int, int]) -> pd.Series:
    return frame["date"].between(*window)


def _check_columns(frame: pd.DataFrame, name: str) -> None:
    missing = set(META + FEEDBACK) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} lacks columns {sorted(missing)}")


def _split(frame: pd.DataFrame, name: str, windows: dict[str, tuple[int, int]]) -> dict[str, pd.DataFrame]:
    masks = {key: in_window(frame, window) for key, window in windows.items()}
    covered = sum(mask.astype(int) for mask in masks.values())
    if (covered != 1).any():
        bad = frame.loc[covered != 1, "date"].value_counts().sort_index()
        raise ValueError(f"{name}: rows outside or in several windows by date:\n{bad}")
    return {key: frame.loc[mask].reset_index(drop=True) for key, mask in masks.items()}


def _read_log(name: str, raw_dir: Path, windows: dict[str, tuple[int, int]],
              sealed: tuple[str, ...] = ()) -> dict[str, pd.DataFrame]:
    """Read one log; rows of windows listed in `sealed` are parsed only for their META columns.

    Pass 1 parses the META columns to place every row in its window. Pass 2 parses all columns but
    skips the sealed rows by line number, so their feedback never becomes part of a DataFrame.
    """
    path = raw_dir / config.LOG_FILES[name]
    meta_parts = _split(pd.read_csv(path, usecols=list(META)), name, windows)
    if not sealed:
        full = pd.read_csv(path)
        _check_columns(full, name)
        return _split(full, name, windows)
    dates = pd.read_csv(path, usecols=["date"])
    sealed_mask = np.zeros(len(dates), dtype=bool)
    for key in sealed:
        sealed_mask |= in_window(dates, windows[key]).to_numpy()
    skip = set((np.flatnonzero(sealed_mask) + 1).tolist())  # +1: line 0 is the header
    full = pd.read_csv(path, skiprows=lambda line: line in skip)
    _check_columns(full, name)
    parts = _split(full, name, {k: w for k, w in windows.items() if k not in sealed})
    for key in sealed:
        parts[key] = meta_parts[key][list(META)]
    return parts


def prepare(raw_dir: Path = config.PURE_DIR, cache: Path = config.CACHE) -> dict:
    """Cache every split except test-period feedback; returns and writes the manifest."""
    cache.mkdir(parents=True, exist_ok=True)
    early = _read_log("standard_early", raw_dir, {"train": config.TRAIN})
    late = _read_log("standard_late", raw_dir, {"valid": config.VALID, "test": config.TEST}, sealed=("test",))
    rand = _read_log("random", raw_dir, {"calib": config.CALIB, "test": config.TEST}, sealed=("test",))
    parts = {
        "standard_train": early["train"],
        "standard_valid": late["valid"],
        "random_calib": rand["calib"],
        "standard_test_meta": late["test"],
        "random_test_meta": rand["test"],
        "users": pd.read_csv(raw_dir / config.SIDE_FILES["users"]),
        "videos": pd.read_csv(raw_dir / config.SIDE_FILES["videos"]),
    }
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {},
                "raw": {key: file_digest(raw_dir / fname) for key, fname in config.LOG_FILES.items()}}
    for key, frame in parts.items():
        path = cache / f"{key}.parquet"
        frame.to_parquet(path, index=False)
        manifest["files"][key] = {"rows": int(len(frame)), "sha256": file_digest(path)}
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _openings(runs: Path) -> list[dict]:
    path = runs / OPENINGS
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load(name: str, cache: Path = config.CACHE, columns: list[str] | None = None,
         runs: Path = config.RUNS) -> pd.DataFrame:
    if name not in KNOWN:
        raise ValueError(f"unknown cache name {name!r}; expected one of {KNOWN}")
    path = cache / f"{name}.parquet"
    if name in SEALED:
        completed = [r for r in _openings(runs) if r.get("status") == "completed"]
        if not completed or not path.exists():
            raise PermissionError(f"{name} is sealed until `receval open-test` completes")
        if file_digest(path) != completed[-1]["files"][name]["sha256"]:
            raise PermissionError(f"{name} does not match the recorded opening; it was modified or replaced")
    return pd.read_parquet(path, columns=columns)


def _git(*args: str, cwd: Path = config.ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8")


def _ancestor(older: str, newer: str, repo: Path) -> bool:
    return _git("merge-base", "--is-ancestor", older, newer, cwd=repo).returncode == 0


def verify_data(cache: Path = config.CACHE, raw_dir: Path = config.PURE_DIR) -> dict:
    """Every cached file and raw log must still match the data manifest."""
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest.get("raw", {})) != set(config.LOG_FILES):
        raise PermissionError("data manifest lacks raw log fingerprints; rerun `receval prepare`")
    for key, info in manifest["files"].items():
        path = cache / f"{key}.parquet"
        if not path.exists() or file_digest(path) != info["sha256"]:
            raise PermissionError(f"cache file {key} is missing or differs from the data manifest")
    for key, digest in manifest["raw"].items():
        if file_digest(raw_dir / config.LOG_FILES[key]) != digest:
            raise PermissionError(f"raw log {key} differs from the data manifest")
    return manifest


def _normalised(text: str) -> str:
    return text.replace("\r\n", "\n")


def check_freeze(freeze_file: Path = FREEZE_FILE, repo: Path = config.ROOT, cache: Path = config.CACHE,
                 raw_dir: Path = config.PURE_DIR, protocol_path: str = "docs/PROTOCOL.md") -> dict:
    """Verify protocol commit -> zoo commit -> HEAD, the frozen protocol text, frozen paths, data, artefacts."""
    if not freeze_file.exists():
        raise PermissionError(f"{freeze_file} missing: freeze the protocol and the model zoo first")
    freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    protocol, zoo = freeze.get("protocol_commit"), freeze.get("zoo_commit")
    if not protocol or not zoo:
        raise PermissionError("freeze.json needs protocol_commit and zoo_commit")
    # Resolve to full hashes so a short and a long name of one commit cannot pass as two commits.
    resolved = [_git("rev-parse", "--verify", "--quiet", f"{c}^{{commit}}", cwd=repo).stdout.strip() for c in (protocol, zoo)]
    if not all(resolved):
        raise PermissionError("protocol_commit and zoo_commit must name existing commits")
    protocol, zoo = resolved
    if protocol == zoo or not _ancestor(protocol, zoo, repo):
        raise PermissionError("protocol_commit must be a strict ancestor of zoo_commit")
    if not _ancestor(zoo, "HEAD", repo):
        raise PermissionError(f"zoo_commit={zoo!r} is not a committed ancestor of HEAD")
    if _git("status", "--porcelain", cwd=repo).stdout.strip():
        raise PermissionError("working tree is dirty; commit before opening the test sets")
    frozen_text = _normalised(_git("show", f"{protocol}:{protocol_path}", cwd=repo).stdout)
    current_text = _normalised((repo / protocol_path).read_text(encoding="utf-8"))
    if not frozen_text or not current_text.startswith(frozen_text):
        raise PermissionError("the frozen protocol text was edited; only appending is allowed after the freeze")
    changed = _git("diff", "--name-only", zoo, "HEAD", "--", *FROZEN_PATHS, cwd=repo).stdout.split()
    if changed:
        raise PermissionError(f"frozen paths changed after the zoo freeze: {changed}")
    manifest_path = repo / ZOO_MANIFEST
    if not manifest_path.exists():
        raise PermissionError(f"{manifest_path} missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_data(cache, raw_dir)
    if manifest.get("data_manifest_sha256") != file_digest(cache / "manifest.json"):
        raise PermissionError("data cache differs from the one the zoo was trained on")
    verify_manifest(manifest, repo)
    return freeze


def verify_manifest(manifest: dict, repo: Path = config.ROOT) -> None:
    """Artefacts are complete and unchanged, and match the selection's run ids one to one."""
    artefacts = manifest.get("artefacts", [])
    verify_artefacts(artefacts, repo)
    listed = {(a["model"], a["label"], int(a["seed"])): a.get("run_id") for a in artefacts}
    if len(listed) != len(artefacts):
        raise PermissionError("zoo manifest lists one (model, label, seed) more than once")
    # One run or one file standing in for several seeds would inflate seed support; equal content is fine.
    for field in ("run_id", "path"):
        values = [a.get(field) for a in artefacts]
        if len(set(values)) != len(values):
            raise PermissionError(f"zoo manifest uses one {field} for more than one (model, label, seed)")
    chosen = {(model, label, int(seed)): run_id for model, entry in manifest.get("selection", {}).items()
              for label in ("is_click", "long_view") for seed, run_id in entry.get(label, {}).items()}
    if listed != chosen:
        raise PermissionError("zoo manifest artefacts do not match the selection's run ids")


def verify_artefacts(artefacts: list[dict], repo: Path = config.ROOT) -> None:
    """Every required (model, label, seed) is listed, and each listed file still has its recorded hash."""
    present = {(a["model"], a["label"], int(a["seed"])) for a in artefacts}
    missing = [(m, label, seed) for m, labels in REQUIRED_ARTEFACTS.items() for label, seeds in labels.items()
               for seed in seeds if (m, label, seed) not in present]
    if missing:
        raise PermissionError(f"zoo manifest lacks required artefacts: {missing}")
    for artefact in artefacts:
        path = repo / artefact["path"]
        if not path.exists() or file_digest(path) != artefact["sha256"]:
            raise PermissionError(f"artefact {artefact['path']} is missing or changed since the freeze")


def open_test(raw_dir: Path = config.PURE_DIR, cache: Path = config.CACHE, runs: Path = config.RUNS,
              freeze_file: Path = FREEZE_FILE, repo: Path = config.ROOT, reopen_reason: str | None = None) -> dict:
    previous = _openings(runs)
    if previous and not reopen_reason:
        raise PermissionError("the test sets were already opened; reopening needs a reason (protocol deviation)")
    if not previous and any((cache / f"{key}.parquet").exists() for key in SEALED):
        raise PermissionError("stray labelled test files exist without an opening record; investigate first")
    freeze = check_freeze(freeze_file, repo, cache, raw_dir)
    record = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "head": _git("rev-parse", "HEAD", cwd=repo).stdout.strip(),
              "freeze": freeze, "reopen_reason": reopen_reason, "status": "started"}
    runs.mkdir(parents=True, exist_ok=True)
    with open(runs / OPENINGS, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    late = _read_log("standard_late", raw_dir, {"valid": config.VALID, "test": config.TEST})["test"]
    rand = _read_log("random", raw_dir, {"calib": config.CALIB, "test": config.TEST})["test"]
    record["files"] = {}
    for key, frame in {"standard_test": late, "random_test": rand}.items():
        final, tmp = cache / f"{key}.parquet", cache / f"{key}.parquet.tmp"
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, final)
        record["files"][key] = {"rows": int(len(frame)), "sha256": file_digest(final)}
    record.update(status="completed", completed=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(runs / OPENINGS, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record
