"""python -m genrec <command> — study 2 entry points (protocol §6)."""
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(prog="genrec")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "fetch-encoder", "encode", "prepare", "sids", "smoke", "tune", "sensitivity", "proxy",
                 "power-check", "freeze-zoo", "check-freeze", "analyze"):
        sub.add_parser(name)
    opener = sub.add_parser("open-test")
    opener.add_argument("--reopen-reason")
    args = parser.parse_args()

    if args.command == "audit":
        from . import audit
        audit.main()
        return
    if args.command == "fetch-encoder":
        from . import encode
        out = encode.fetch()
    elif args.command == "encode":
        from . import encode
        out = {k: v for k, v in encode.encode().items() if k != "video_ids"}
    elif args.command == "prepare":
        from . import data
        out = data.prepare()["counts"]
    elif args.command == "sids":
        from . import sids
        out = {s: {k: r[k] for k in ("codes_used_per_level", "three_code_collisions", "fourth_code_max")}
               for s, r in sids.build().items()}
    elif args.command == "smoke":
        from . import tune
        out = tune.smoke()
    elif args.command == "tune":
        from . import tune
        sel = tune.tune()
        out = {k: {"config": v["config"], "seeds": {s: e["valid_recall@100"] for s, e in v["seeds"].items()}}
               for k, v in sel["models"].items()}
    elif args.command == "sensitivity":
        from . import tune
        out = {k: {"config": v["config"], "seeds": {s: {"valid_recall@100": e["valid_recall@100"],
                                                         "best_epoch": e["best_epoch"], "epochs_run": e["epochs_run"]}
                                                     for s, e in v["seeds"].items()}}
               for k, v in tune.sensitivity().items()}
    elif args.command == "proxy":
        from . import analysis
        out = {"primary": analysis.run(proxy=True)["primary"]}
    elif args.command == "power-check":
        from . import analysis
        out = analysis.power_check()
    elif args.command == "freeze-zoo":
        from . import freeze
        out = freeze.freeze_zoo()
    elif args.command == "check-freeze":
        from . import freeze
        out = freeze.check_freeze()
    elif args.command == "open-test":
        from . import freeze
        out = freeze.open_test(args.reopen_reason)
    else:
        from . import analysis
        out = {"primary": analysis.run(proxy=False)["primary"]}
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
