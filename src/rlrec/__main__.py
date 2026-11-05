"""python -m rlrec <stage> — study 3 pipeline (docs/rlrec/PROTOCOL.md §10).

Order: text → sids → sft → (scripts/rlrec_reward_models.py) → tables → o3 → rl-pilot → rl → rl-secondary → power
→ freeze → hold-report → test.
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(prog="rlrec")
    parser.add_argument("stage", choices=["text", "sids", "sft", "tables", "o3", "rl-pilot", "rl", "rl-secondary", "hold-report",
                                          "power", "freeze", "test"])
    parser.add_argument("--reopen-reason", help="test only: why an unfinished evaluation is resumed (a protocol deviation)")
    args = parser.parse_args()
    from . import stages
    run = {"text": stages.text, "sids": stages.sids, "sft": stages.sft, "tables": stages.tables, "o3": stages.o3,
           "rl-pilot": stages.rl_pilot, "rl": lambda: stages.rl_runs("main"),
           "rl-secondary": lambda: stages.rl_runs("secondary"), "power": stages.power,
           "freeze": stages.freeze, "hold-report": stages.hold_report, "test": lambda: stages.test(args.reopen_reason)}
    print(json.dumps(run[args.stage](), indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
