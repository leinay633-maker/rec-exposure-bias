"""Full contrast tables of protocol §17 and §18 (docs/RESULTS.md §9) straight from the two ablation JSONs.

Usage: python scripts/make_random_mix_tables.py > tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"§17（按随机早停集早停）": ROOT / "evidence/ablation/20261101-160809-ablation-random-mix-8ea380/random_mix.json",
        "§18（固定轮数）": ROOT / "evidence/ablation/20261101-165659-ablation-random-mix-fixed-29339e/random_mix_fixed.json"}
CELL = {"A": "A", "A_prime": "A′", "B_standard": "B 常规", "B_random": "B 随机", "C": "C", "A_refit": "A 重训",
        "B_standard_fixed": "B 常规（固定）", "B_random_fixed": "B 随机（固定）", "C_fixed": "C（固定）"}
SETS = {"standard_test": "常规", "random_test": "随机"}
METRICS = {"uauc": "UAUC", "logloss": "LogLoss"}


def num(x: float) -> str:
    return f"{x:+.4f}".replace("-", "−")


def plain(x: float) -> str:
    return f"{x:.4f}".replace("-", "−")


def label(contrast: str) -> str:
    a, b = contrast.split(" − ")
    return f"{CELL[a]} − {CELL[b]}"


def p_text(p: float) -> str:
    return "< 0.0002" if p == 0 else f"{p:.4f}"


def main() -> None:
    lines = []
    for title, path in RUNS.items():
        r = json.loads(path.read_text(encoding="utf-8"))
        lines += [f"#### {title}：全部对比（每份测试、每个指标上四个对比一起做 Holm）", "",
                  "| 对比 | 指标 | 测试 | 点估计 | 95% 区间 | 标准误 | p | Holm | 种子支持 | 逐种子 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for metric, sets in r["contrasts"].items():
            for s, block in sets.items():
                for name, v in block.items():
                    lines.append(f"| {label(name)} | {METRICS[metric]} | {SETS[s]} | {num(v['point'])} | "
                                 f"[{plain(v['ci95'][0])}, {plain(v['ci95'][1])}] | {plain(v['se'])} | {p_text(v['p'])} | "
                                 f"{'拒绝' if v['holm_reject'] else '不拒绝'} | {v['seed_support']}/3 | "
                                 f"{'、'.join(num(x) for x in v['per_seed'])} |")
        if "check" in r:
            lines += ["", "A 重训 − A（核对用，不进 Holm）：" + "；".join(
                f"{METRICS[m]} {SETS[s]} {num(v['point'])}" for m, sets in r["check"].items() for s, v in sets.items())
                + f"。两份测试上预测的最大绝对差：" + "、".join(
                f"{SETS[k]} {v}" for k, v in r["refit_max_abs_prediction_diff"].items()) + "。"]
        lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
