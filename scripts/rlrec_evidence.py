"""Copy study 3's evidence into evidence/rlrec and rewrite evidence/SHA256SUMS (AGENTS.md).

Summaries and hashes only: models, reward tables and arrays stay in .runs. Paths follow .runs/. Run after the test.

Usage: python scripts/rlrec_evidence.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from rlrec import config

EVIDENCE = config.ROOT / "evidence"
STAGES = ("text", "sids", "sft", "reward_models", "tables", "o3", "rl_pilot", "power")
SUMMARIES = ("sft/*/selection.json", "tables/*/tables.json", "o3/*/o3.json", "rl_pilot/*/pilot.json", "kl_rate/*/kl_rate.json",
             "power/*/power.json", "report_hold/*/hold.json", "report_test/*/test.json", "posthoc/*/sharpen.json")
LIMIT = 20 * 2**20  # a file this large would hold data rows, not a summary


def sources() -> list[Path]:
    runs = config.RUNS
    files = [runs / "registry.jsonl", runs / "test_evaluations.jsonl", *(runs / s / "latest.json" for s in STAGES)]
    for pattern in SUMMARIES:
        found = sorted(runs.glob(pattern))
        if not found:
            raise FileNotFoundError(pattern)
        files += found
    for which in ("main", "secondary"):
        index = runs / "rl" / f"{which}.json"
        files.append(index)
        files += [config.ROOT / r["dir"] / "rl.json" for r in json.loads(index.read_text(encoding="utf-8"))["runs"].values()]
    return files


def rewrite_sums(evidence: Path = EVIDENCE) -> int:
    paths = sorted((p for p in evidence.rglob("*") if p.is_file() and p.name != "SHA256SUMS"), key=lambda p: p.as_posix())
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(evidence).as_posix()}\n" for p in paths]
    (evidence / "SHA256SUMS").write_bytes("".join(lines).encode("utf-8"))
    return len(lines)


def main() -> None:
    files = sources()
    too_large = [str(f) for f in files if f.stat().st_size > LIMIT]
    if too_large:
        raise ValueError(f"not summaries: {too_large}")
    for src in files:
        dst = EVIDENCE / src.relative_to(config.ROOT / ".runs")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    shutil.copyfile(config.ROOT / "configs" / "rlrec" / "zoo_manifest.json", EVIDENCE / "rlrec" / "zoo_manifest.json")
    print(f"copied {len(files) + 1} files, SHA256SUMS lists {rewrite_sums()}")


if __name__ == "__main__":
    main()
