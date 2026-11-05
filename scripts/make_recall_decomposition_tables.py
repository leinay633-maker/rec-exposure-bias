"""Tables of protocol §20 (docs/RESULTS.md §10) straight from the evidence copy of recall_decomposition.json.

Usage: python scripts/make_recall_decomposition_tables.py <evidence/posthoc/<run>/recall_decomposition.json> > tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NAME = {"recall_popularity": "热度", "recall_itemcf": "ItemCF", "recall_two_tower": "双塔",
        "recall_two_tower_logq": "双塔（logQ）"}
SETS = {"standard": "常规", "random": "随机"}
GROUP = ["最冷", "较冷", "中间", "较热", "最热"]


def f4(x: float) -> str:
    return f"{x:.4f}".replace("-", "−")


def s4(x: float) -> str:
    return f"{x:+.4f}".replace("-", "−")


def f3(x: float) -> str:
    return f"{x:.3f}".replace("-", "−")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci(v: dict, fmt=f4) -> str:
    return f"[{fmt(v['ci'][0])}, {fmt(v['ci'][1])}]"


def p_text(p: float) -> str:
    return "< 0.0002" if p == 0 else f"{p:.4f}"


def pair(names: list[str], op: str) -> str:
    return f"{NAME[names[0]]} {op} {NAME[names[1]]}"


def main(path: Path) -> None:
    r = json.loads(path.read_text(encoding="utf-8"))
    d = r["denominators"]
    out = [f"R\\* {r['users']:,} 人；复现正式分析 Recall@100 的最大差 {r['formal_check']['max_abs_difference']:.1e}。", "",
           "#### 10.A 分母和随机基线", "",
           "| 量 | 常规测试 | 随机测试 |", "| --- | --- | --- |"]
    c = d["candidates"]
    out.append(f"| 每人候选数 N_u（均值；P10 / P50 / P90） | {c['mean']:.0f}；{c['p10']:.0f} / {c['p50']:.0f} / {c['p90']:.0f} | 同左 |")
    t = d["targets"]
    out.append("| 每人目标数 \\|T_u\\|（均值；P10 / P50 / P90） | "
               + " | ".join(f"{t[s]['mean']:.2f}；{t[s]['p10']:.0f} / {t[s]['p50']:.0f} / {t[s]['p90']:.0f}" for s in SETS) + " |")
    for k in ("k50", "k100", "k200"):
        out.append(f"| 目标数 ≤ {k[1:]} 的人占比 | " + " | ".join(pct(d["share_targets_at_most_k"][s][k]) for s in SETS) + " |")
    out.append(f"| 随机排序的期望 b_u = 100 / N_u（均值） | {f4(d['random_baseline']['k100']['mean'])} | 同左 |")
    out.append("| 观测标签下的理想上限（K = 100，均值） | "
               + " | ".join(f4(d["ideal_upper_bound"][s]["k100"]) for s in SETS) + " |")
    out.append("| 参与策略价值的测试行（去掉训练期看过的） | "
               + " | ".join(f"{d['test_rows'][s]['eligible']:,}（去掉 {d['test_rows'][s]['dropped_seen']:,}）" for s in SETS) + " |")

    out += ["", "#### 10.B Recall@100 与按随机测试目标组成重新加权的常规 Recall@100", "",
            "| 模型 | 常规 | 随机 | 常规（按随机测试的热度组成重新加权） |", "| --- | --- | --- | --- |"]
    for m, name in NAME.items():
        rec = r["recall"][m]["recall@100"]
        rw = r["decomposition"]["reweighted_standard_recall"][m]
        out.append(f"| {name} | {f4(rec[0])} | {f4(rec[1])} | {f4(rw['point'])} {ci(rw)} |")

    out += ["", "#### 10.C 相对提升（K = 100）：原始 Recall 比和对随机基线的提升比", "",
            "对数比变化 = ln(随机测试上的比) − ln(常规测试上的比)，6 对一起做 Holm。",
            "", "| 对 | 原始比·常规 | 原始比·随机 | 对数比变化 | 95% 区间 | p | Holm | 提升比·常规 | 提升比·随机 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in r["relative"]["k100"]:
        raw, lift = e["raw"], e["lift"]
        ch = raw["log_ratio_change"]
        out.append(f"| {pair(e['pair'], '÷')} | {f3(raw['standard']['point'])} | {f3(raw['random']['point'])} | "
                   f"{s4(ch['point'])} | {ci(ch)} | {p_text(ch['p'])} | {'显著' if ch['holm_significant'] else '不显著'} | "
                   f"{f3(lift['standard']['point'])} | {f3(lift['random']['point'])} |")

    for s in SETS:
        pol = r["policy"][s]
        biased = "（只作描述：常规测试的行是旧策略挑的，这不是无偏估计）" if s == "standard" else ""
        out += ["", f"#### 10.{'D' if s == 'random' else 'F'} {SETS[s]}测试上“前 K 个里均匀推一个”的点击率{biased}", "",
                f"可推荐行的平均点击率 {f4(pol['base_click_rate']['point'])} {ci(pol['base_click_rate'])}，"
                f"共 {pol['eligible_rows']:,.0f} 行。", "",
                "| 模型 | K | 落入行数 | 点击率 | 95% 区间 | 对平均点击率的倍数 |", "| --- | --- | --- | --- | --- | --- |"]
        for k in ("k50", "k100", "k200"):
            for m, name in NAME.items():
                v = pol["by_k"][k]["models"][m]
                out.append(f"| {name} | {k[1:]} | {v['rows_in_top_k']:,.0f} | {f4(v['click_rate']['point'])} | "
                           f"{ci(v['click_rate'])} | {f3(v['lift_over_base']['point'])} |")
        if s == "random":
            out += ["", "#### 10.E 随机测试上 K = 100 的两两比较（差一列 6 对一起做 Holm）", "",
                    "| 对 | 点击率之差 | 95% 区间 | p | Holm | 点击率之比 | 95% 区间 |",
                    "| --- | --- | --- | --- | --- | --- | --- |"]
            for e in pol["by_k"]["k100"]["pairs"]:
                dv, rv = e["difference"], e["ratio"]
                out.append(f"| {pair(e['pair'], '−')} | {s4(dv['point'])} | {ci(dv)} | {p_text(dv['p'])} | "
                           f"{'显著' if dv['holm_significant'] else '不显著'} | {f3(rv['point'])} | {ci(rv, f3)} |")

    dec = r["decomposition"]
    out += ["", "#### 10.G 热度分解（K = 100；视频按训练期曝光数等分 5 组）", "",
            "| 量 | " + " | ".join(GROUP) + " |", "| --- | " + " | ".join("---" for _ in GROUP) + " |"]
    for s in SETS:
        out.append(f"| 目标占比·{SETS[s]} | " + " | ".join(pct(x) for x in dec["target_share"][s]) + " |")
    for m, name in NAME.items():
        for s in SETS:
            out.append(f"| {name} 组内 Recall·{SETS[s]} | "
                       + " | ".join("—" if x is None else f4(x) for x in dec["group_recall"][m][s]) + " |")
    for m, name in NAME.items():
        out.append(f"| {name} 前 100 的占位 | " + " | ".join(pct(x) for x in dec["slot_share"][m]) + " |")
    out += ["", f"共同支持的组：{'、'.join(GROUP[g] for g in dec['common_support_groups'])}。", "",
            "#### 10.H 每对差距的分解（K = 100，只作描述）", "",
            "目标组成部分 = 重新加权后的常规差距 − 常规差距；组内表现部分 = 随机差距 − 重新加权后的常规差距。", "",
            "| 对 | 常规差距 | 重新加权后的常规差距 | 随机差距 | 目标组成部分 | 组内表现部分 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for e in dec["pairs"]:
        out.append(f"| {pair(e['pair'], '−')} | {s4(e['standard_gap']['point'])} | "
                   f"{s4(e['reweighted_standard_gap']['point'])} {ci(e['reweighted_standard_gap'], s4)} | "
                   f"{s4(e['random_gap']['point'])} | {s4(e['composition_part']['point'])} {ci(e['composition_part'], s4)} | "
                   f"{s4(e['within_group_part']['point'])} {ci(e['within_group_part'], s4)} |")

    out += ["", "#### 10.J 两种分解顺序（K = 100，只用 10.G 的点估计算，协议 §21 补充的描述，没有区间）", "",
            "R(W, R_g) = Σ_g W_g R_g。“先换组成”是 §20 的顺序：常规 → 常规组内 × 随机组成 → 随机；"
            "“先换组内”是反向：常规 → 随机组内 × 常规组成 → 随机。", "",
            "| 对 | 口径 | 常规 | 先换组成的中间点 | 先换组内的中间点 | 随机 |", "| --- | --- | --- | --- | --- | --- |"]
    share = dec["target_share"]

    def mix(w: list, rg: list) -> float:
        return sum(a * b for a, b in zip(w, rg))

    for e in dec["pairs"]:
        a, b = (dec["group_recall"][m] for m in e["pair"])
        cells = [(share["standard"], "standard"), (share["random"], "standard"), (share["standard"], "random"),
                 (share["random"], "random")]
        gap = [mix(w, a[s]) - mix(w, b[s]) for w, s in cells]
        ratio = [mix(w, a[s]) / mix(w, b[s]) for w, s in cells]
        out.append(f"| {pair(e['pair'], '−')} | 差 | " + " | ".join(s4(x) for x in gap) + " |")
        out.append(f"| {pair(e['pair'], '÷')} | 比 | " + " | ".join(f3(x) for x in ratio) + " |")

    out += ["", "#### 10.I 逐种子（两个双塔）", "",
            "| 模型 | 种子 | Recall@100·常规 | Recall@100·随机 | 随机测试点击率（K = 100） |", "| --- | --- | --- | --- | --- |"]
    for m, seeds in r["per_seed"].items():
        for i, v in enumerate(seeds):
            out.append(f"| {NAME[m]} | {i} | {f4(v['recall@100'][0])} | {f4(v['recall@100'][1])} | "
                       f"{f4(v['random_click_rate@100'])} |")
    print("\n".join(out))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
