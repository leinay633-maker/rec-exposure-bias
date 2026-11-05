"""Command-line entry point: `receval <command>`."""
from __future__ import annotations

import argparse
import json
import time

from . import __version__, config


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def _info(_: argparse.Namespace) -> int:
    print(_dump({"version": __version__, "root": config.ROOT, "data": config.DATA, "runs": config.RUNS,
                 "train": config.TRAIN, "valid": config.VALID, "calib": config.CALIB, "test": config.TEST}))
    return 0


def _prepare(args: argparse.Namespace) -> int:
    from .data import prepare, verify_and_extract
    if not args.skip_extract:
        verify_and_extract()
    print(_dump(prepare()))
    return 0


def _audit(_: argparse.Namespace) -> int:
    from .audit import audit
    out = config.RUNS / "audit" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    report = audit()
    (out / "audit.json").write_text(_dump(report), encoding="utf-8")
    print(out / "audit.json")
    return 0


def _smoke(args: argparse.Namespace) -> int:
    from .smoke import run
    run(args.fraction, args.epochs)
    return 0


def _tune(_: argparse.Namespace) -> int:
    from .orchestrate import run_tune
    print(run_tune())
    return 0


def _freeze_zoo(_: argparse.Namespace) -> int:
    from .tune import write_zoo_manifest
    print(write_zoo_manifest())
    return 0


def _power_check(_: argparse.Namespace) -> int:
    from .orchestrate import run_power_check
    print(run_power_check())
    return 0


def _pilot_power(args: argparse.Namespace) -> int:
    from .orchestrate import run_pilot_power
    print(run_pilot_power(args.epochs))
    return 0


def _analyze(args: argparse.Namespace) -> int:
    from .orchestrate import run_analysis
    print(run_analysis(proxy=args.proxy))
    return 0


def _open_test(args: argparse.Namespace) -> int:
    from .data import open_test
    print(_dump(open_test(reopen_reason=args.reopen_reason)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="receval", description=__doc__)
    parser.add_argument("--version", action="version", version=f"receval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="print paths and the fixed time windows").set_defaults(func=_info)
    prep = sub.add_parser("prepare", help="verify md5, extract, and cache every split except test feedback")
    prep.add_argument("--skip-extract", action="store_true", help="reuse the already extracted CSVs")
    prep.set_defaults(func=_prepare)
    sub.add_parser("audit", help="write the data audit to .runs/audit/<stamp>/").set_defaults(func=_audit)
    smoke = sub.add_parser("smoke", help="run every model end to end on validation data, with timings")
    smoke.add_argument("--fraction", type=float, default=0.01, help="share of training users to keep")
    smoke.add_argument("--epochs", type=int, default=1)
    smoke.set_defaults(func=_smoke)
    sub.add_parser("tune", help="grid search, seeds and second-label retraining; writes .runs/zoo/selection.json"
                   ).set_defaults(func=_tune)
    sub.add_parser("power-check", help="proxy-window SEs and MDEs for the selected zoo (seed 0)"
                   ).set_defaults(func=_power_check)
    sub.add_parser("freeze-zoo", help="copy the complete selection to configs/zoo_manifest.json for the zoo freeze"
                   ).set_defaults(func=_freeze_zoo)
    pilot = sub.add_parser("pilot-power", help="pre-freeze pilot: standard errors only, no effects")
    pilot.add_argument("--epochs", type=int, default=3)
    pilot.set_defaults(func=_pilot_power)
    analyze = sub.add_parser("analyze", help="protocol §7-§10 on the opened test sets")
    analyze.add_argument("--proxy", action="store_true", help="dry run on standard valid vs random calibration")
    analyze.set_defaults(func=_analyze)
    opener = sub.add_parser("open-test", help="write labelled test sets once; needs configs/freeze.json")
    opener.add_argument("--reopen-reason", help="only for a logged protocol deviation")
    opener.set_defaults(func=_open_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
