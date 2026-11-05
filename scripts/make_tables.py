"""Render the JSON outputs of one `receval analyze` run as Markdown tables for docs/RESULTS.md.

Reads only .runs/analysis/<run_id>/*.json (plus optional power_check.json or user_rates.json files) and formats
what is there. The only numbers it computes itself are differences, counts and medians of values in those files.
Usage: python scripts/make_tables.py .runs/analysis/<run_id> [power_check.json | user_rates.json ...]
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

NAMES = {"popularity": "热度", "logreg": "逻辑回归", "lightgbm": "LightGBM", "two_tower": "精排双塔", "dcn": "DCN-v2",
         "din": "DIN", "mmoe": "MMoE", "recall_popularity": "热度", "recall_itemcf": "ItemCF",
         "recall_two_tower": "双塔", "recall_two_tower_logq": "双塔（logQ）"}
DIRECTION = {"reversed": "反向", "widened": "扩大", "narrowed": "缩小", "unchanged": "不变"}
# Models trained with three seeds; a pair without any of them has no seed rule (protocol §7.3).
SEEDED = {"lightgbm", "two_tower", "dcn", "din", "mmoe", "recall_two_tower", "recall_two_tower_logq"}
HALVES = {"first": "4/29–5/03", "second": "5/04–5/08"}
METHODS = {"uncalibrated": "不校准", "intercept": "只调截距", "platt": "Platt", "isotonic": "保序",
           "grouped": "分组 Platt", "constant": "常数"}


def num(x, digits: int = 4) -> str:
    return "—" if x is None or x != x else f"{x:.{digits}f}"


def pval(x) -> str:
    return x if isinstance(x, str) else ("< 0.0002" if x == 0 else f"{x:.4f}")


def ci(pair, digits: int = 4) -> str:
    return f"[{num(pair[0], digits)}, {num(pair[1], digits)}]"


def yes(flag) -> str:
    return "是" if flag else "否"


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pair_name(pair) -> str:
    return f"{NAMES[pair[0]]} − {NAMES[pair[1]]}"


def comparison(m: dict, metric: str, key: str) -> list[str]:
    """The §7.3 block shared by ranking and recall: levels, gap changes, flips, tau-b, scaling."""
    out = [table(["模型", f"常规 {metric}", f"随机 {metric}", "随机 − 常规"],
                 [[NAMES[k], num(v[0]), num(v[1]), num(v[1] - v[0])] for k, v in m[key].items()])]
    gc = m["gap_changes"]
    out.append(f"\n显著差距变化 {gc['changes']} 对（Holm 后显著且过种子规则）；显著翻转 {m['flips']} 对。\n")
    out.append(table(["模型对", "Δ 常规", "Δ 随机", "D", "D 的 95% 区间", "p", "Holm 显著", "种子支持", "方向", "计入", "残差"],
                     [[pair_name(r["pair"]), num(r["delta"][0]), num(r["delta"][1]), num(r["did"]), ci(r["ci"]),
                       pval(r["p"]), yes(r["significant"]),
                       r["seed_support"] if set(r["pair"]) & SEEDED else "不适用", DIRECTION[r["direction"]],
                       yes(r["change_final"]), num(r["residual"])] for r in gc["pairs"]]))
    assert [r["pair"] for r in m["pairs"]] == [g["pair"] for g in gc["pairs"]]  # both list pairs in the same order
    out.append("\n每对的标准误和最小可检出差（功效 0.8；Holm 第一步和未校正两种）：\n")
    out.append(table(["模型对", "Δ 常规 标准误", "Δ 随机 标准误", "D 标准误", "Holm 门槛 常规", "Holm 门槛 随机", "Holm 门槛 D",
                      "未校正门槛 常规", "未校正门槛 随机", "未校正门槛 D", "Holm 显著 常规", "Holm 显著 随机"],
                     [[pair_name(r["pair"]), num(r["se"][0]), num(r["se"][1]), num(g["se"]),
                       num(r["mde_holm_first_step"][0]), num(r["mde_holm_first_step"][1]), num(g["mde_holm_first_step"]),
                       num(r["mde_unadjusted"][0]), num(r["mde_unadjusted"][1]), num(g["mde_unadjusted"]),
                       yes(r["significant"][0]), yes(r["significant"][1])]
                      for r, g in zip(m["pairs"], gc["pairs"])]))
    both = [r for r in m["pairs"] if all(r["significant"])]
    out.append(f"\n两份测试集上都显著的对 {len(both)} 个，其中方向相反（翻转）{sum(r['flip'] for r in both)} 个，"
               f"过种子规则的翻转 {m['flips']} 个。")
    flips = [pair_name(r["pair"]) for r in m["pairs"] if r["flip_final"]]
    if flips:
        out.append("翻转的对：" + "、".join(flips) + "。")
    out.append(f"\nKendall τ-b {num(m['kendall_tau_b'], 3)}，95% 区间 {ci(m['kendall_ci'], 3)}，"
               f"全部并列的重抽样 {m['kendall_undefined_replicates']} 次。")
    s = m["gap_scaling"]
    out.append(f"整体缩放：斜率 {num(s['slope'], 3)}，95% 区间 {ci(s['ci'], 3)}，R² {num(s['r2'], 3)}，"
               f"无定义的重抽样 {s['undefined_replicates']} 次。")
    mid = statistics.median
    out.append(f"Holm 第一步最小可检出差的中位数：常规 {num(mid(r['mde_holm_first_step'][0] for r in m['pairs']))}，"
               f"随机 {num(mid(r['mde_holm_first_step'][1] for r in m['pairs']))}，"
               f"D {num(mid(r['mde_holm_first_step'] for r in gc['pairs']))}。")
    return out


def coverage(cov: dict, population: str = "U\\*") -> str:
    return (f"{population} {cov['common_users']:,} 人，占两份测试集都出现过的 {cov['users_in_both_sets']:,} 人的 "
            f"{cov['share']:.1%}；因此排除的行数：常规 {cov['rows_excluded'][0]:,}，随机 {cov['rows_excluded'][1]:,}。\n")


def ranking(res: dict) -> str:
    m = res["main"]
    out = [f"### 精排 · `{res['label']}`\n"]
    if not m.get("users"):
        return "\n".join(out + [m.get("status", "无共同用户")])
    out.append(coverage(m["coverage"]))
    out += comparison(m, "UAUC", "uauc")
    if "time_halves" in res:
        rows = []
        for name, half in res["time_halves"].items():
            if not half.get("users"):
                rows.append([HALVES[name], 0, "—", "—", "—"])
                continue
            rows.append([HALVES[name], f"{half['users']:,}", half["gap_changes"]["changes"], half["flips"],
                         f"{num(half['kendall_tau_b'], 3)} {ci(half['kendall_ci'], 3)}"])
        out.append("\n按时段（§7.4，只报告；每段各自重定 U\\*，完整的表在本节末尾）：\n")
        out.append(table(["时段", "U*", "显著差距变化", "显著翻转", "τ-b 及区间"], rows))
    for s, name in enumerate(("常规", "随机")):
        own = res["own_set"][s]
        out.append(f"\n{name}测试集的次要指标（只报告；GAUC 和等权 UAUC 用这份测试集里有正有负的用户，"
                   "全局 AUC、LogLoss、Brier、ECE 用这份测试集的全部行）：\n")
        out.append(table(["模型", "行加权 GAUC", "等权 UAUC", "全局 AUC", "LogLoss", "Brier", "ECE"],
                         [[NAMES[k], num(v["gauc"]), num(v["uauc_own"]), num(v["auc"]), num(v["logloss"]),
                           num(v["brier"]), num(v["ece"])] for k, v in own.items()]))
    out.append("\n§10 的并列：全局 AUC、行加权 GAUC、各自合格用户的等权 UAUC、共同用户 U\\* 的等权 UAUC（只报告）：\n")
    out.append(table(["模型"] + [f"{name} {metric}" for name in ("常规", "随机")
                                 for metric in ("全局 AUC", "GAUC", "自身 UAUC", "U* UAUC")],
                     [[NAMES[k]] + [num(x) for s in (0, 1)
                                    for x in (res["own_set"][s][k]["auc"], res["own_set"][s][k]["gauc"],
                                              res["own_set"][s][k]["uauc_own"], m["uauc"][k][s])]
                      for k in m["uauc"]]))
    for s, name in enumerate(("常规", "随机")):
        mech = res["mechanism"][s]
        first = next(iter(mech.values()))
        out.append(f"\n{name}测试集按训练期热度分档（§10，组 0 最冷，只报告）：\n")
        header = ["模型"] + [f"组 {g['group']} AUC" for g in first] + [f"组 {g['group']} 平均预测" for g in first]
        rows = [[NAMES[k]] + [num(g["auc"]) for g in v] + [num(g["mean_pred"]) for g in v] for k, v in mech.items()]
        rows.append(["正样本率"] + [num(g["rate"]) for g in first] + ["" for _ in first])
        rows.append(["行数"] + [f"{g['rows']:,}" for g in first] + ["" for _ in first])
        out.append(table(header, rows))
    for name, half in res.get("time_halves", {}).items():
        out.append(f"\n#### 精排 · `{res['label']}` · {HALVES[name]}（§7.4，只报告）\n")
        if not half.get("users"):
            out.append(half.get("status", "无共同用户"))
            continue
        out.append(coverage(half["coverage"]))
        out += comparison(half, "UAUC", "uauc")
    return "\n".join(out)


def calibration(res: dict) -> str:
    out = ["### 校准（`is_click`）\n"]
    rows = []
    for key, cell in res["by_model_budget"].items():
        model, budget = key.split("@")
        loss = cell["logloss_user_mean"]
        rows.append([NAMES[model], budget] + [num(loss[m]) for m in METHODS]
                    + [f"{num(cell[c]['point'])} {ci(cell[c]['ci'])}"
                       for c in ("intercept_minus_platt", "platt_minus_grouped", "uncalibrated_minus_intercept")])
    out.append(table(["模型", "N"] + [f"LogLoss {v}" for v in METHODS.values()]
                     + ["d 截距 − Platt 及区间", "d Platt − 分组 及区间", "不校准 − 截距 及区间（只报告）"], rows))
    out.append(f"\nH2a（每个模型在 16,000、64,000、全部三档 d 截距 − Platt 的区间都在 0 以上）：{yes(res['H2a'])}。")
    held = [NAMES[m] for m, v in res["H2b_per_model"].items() if v]
    out.append(f"H2b（N 取全部时 d Platt − 分组的区间在 0 以上）：{len(held)}/{len(res['H2b_per_model'])} 个模型成立"
               + (f"（{'、'.join(held)}）" if held else "") + f"；整体{'成立' if res['H2b_overall'] else '不成立'}。")
    out.append(f"拟合失败 {len(res['fit_failures'])} 次。")
    rows = [[NAMES[key.split("@")[0]], key.split("@")[1]] + [num(cell["brier"][m]) for m in METHODS]
            + [num(cell["ece"][m]) for m in METHODS] for key, cell in res["by_model_budget"].items()]
    out.append("\n各档的 Brier 和 ECE（只报告；对实际的抽样和种子取平均：除“全部”档外每档抽样 5 次，“全部”档 1 次；"
               "热度、逻辑回归各 1 个种子，其余模型各 3 个）：\n")
    out.append(table(["模型", "N"] + [f"Brier {v}" for v in METHODS.values()] + [f"ECE {v}" for v in METHODS.values()],
                     rows))
    rows = [[NAMES[key.split("@")[0]], num(cell["uauc_random_test"]["uncalibrated"]),
             num(cell["uauc_random_test"]["grouped"])]
            for key, cell in res["by_model_budget"].items() if key.endswith("@all")]
    out.append("\nN 取全部时随机测试上的 UAUC（只报告；用随机测试集里有正有负的用户）：\n")
    out.append(table(["模型", "不校准", "分组 Platt"], rows))
    rows = [[NAMES[key.split("@")[0]], key.split("@")[1],
             ci(cell["spread_across_repeats"]["intercept_minus_platt"]),
             ci(cell["spread_across_repeats"]["platt_minus_grouped"])]
            for key, cell in res["by_model_budget"].items() if key.split("@")[1] != "all"]
    out.append("\n点估计在 5 次抽样之间的范围（最小、最大）：\n")
    out.append(table(["模型", "N", "d 截距 − Platt", "d Platt − 分组"], rows))
    return "\n".join(out)


def recall(res: dict) -> str:
    out = ["### 召回（Recall@100）\n"]
    if not res.get("users"):
        return "\n".join(out + [res.get("status", "无共同用户")])
    cov = res["coverage"]
    out.append(f"R\\* {cov['common_users']:,} 人，占两份测试集都出现过的 {cov['users_in_both_sets']:,} 人的 "
               f"{cov['share']:.1%}；有目标视频的用户：常规 {cov['users_with_targets'][0]:,}，随机 {cov['users_with_targets'][1]:,}。\n")
    rows = []
    for key, h in res["did"].items():
        rows.append([key, pair_name(h["pair"]), num(h["point"]), ci(h["ci"]), pval(h["p"]), num(h["se"]),
                     num(h["mde_holm_first_step"]), yes(h["holds"])])
    out.append(table(["假设", "模型对", "DiD", "95% 区间", "p", "标准误", "Holm 门槛", "成立"], rows))
    out.append("")
    out += comparison(res, "Recall@100", "recall@100")
    rows = [[NAMES[k]] + [num(res["secondary"][metric][k][s]) for metric in ("recall@50", "recall@200", "ndcg@100")
                          for s in (0, 1)] for k in res["models"]]
    out.append("\n次要指标（点估计，只报告）：\n")
    out.append(table(["模型", "Recall@50 常规", "Recall@50 随机", "Recall@200 常规", "Recall@200 随机",
                      "NDCG@100 常规", "NDCG@100 随机"], rows))
    return "\n".join(out)


def power(res: dict) -> str:
    out = ["### 功效检查（代理对，种子 0，只有标准误和门槛）\n"]
    for key, name in (("ranking_is_click", "精排"), ("recall@100", "召回")):
        block = res[key]
        if "status" in block:
            out.append(f"{name}：{block['status']}")
            continue
        rows = [[label] + [num(block[k][stat]) for stat in ("median", "min", "max")]
                for label, k in (("标准误 常规", "se_standard"), ("标准误 随机", "se_random"), ("标准误 D", "se_did"),
                                 ("Holm 门槛 常规", "mde_holm_standard"), ("Holm 门槛 随机", "mde_holm_random"),
                                 ("Holm 门槛 D", "mde_holm_did"))]
        out.append(f"{name}：{block['users']:,} 个两边都合格的用户，{block['pairs']} 对，"
                   f"Holm 第一步乘数 {num(block['holm_first_multiplier'], 2)}。\n")
        out.append(table(["量", "中位数", "最小", "最大"], rows))
        if "h3_mde_did" in block:
            out.append("\nH3 两条（Holm 两条一族）的门槛：" + "，".join(f"{k} {num(v)}" for k, v in block["h3_mde_did"].items())
                       + "。")
        out.append("")
    return "\n".join(out)


def user_rates(res: dict) -> str:
    out = ["### 用户之间正样本率的差异（§10 事后补算，只作描述）\n"]
    rows = []
    for key, v in res.items():
        name, label = key.split("/")
        q = v["user_rate_quantiles_10_25_50_75_90"]
        rows.append([{"standard_test": "常规测试", "random_test": "随机测试"}[name], f"`{label}`", f"{v['rows']:,}",
                     num(v["positive_rate"], 3), f"{v['users']:,}", f"{v['users_with_10_rows']:,}"]
                    + [num(x, 3) for x in q] + [num(v["user_rate_std"], 3)])
    out.append(table(["测试集", "标签", "行数", "正样本率", "用户数", "至少 10 行的用户", "P10", "P25", "中位数", "P75", "P90",
                      "标准差"], rows))
    return "\n".join(out)


def main() -> None:
    run = Path(sys.argv[1])
    parts = []
    reload = run / "reload_check.json"
    if reload.exists():
        check = json.loads(reload.read_text(encoding="utf-8"))
        parts.append(f"重载核对：{len(check['rows'])} 个模型文件，最大误差 {check['max_abs_difference']:.2e}，"
                     f"{'通过' if check['passed'] else '不通过'}。")
    for label in ("is_click", "long_view"):
        path = run / f"ranking_{label}.json"
        if path.exists():
            parts.append(ranking(json.loads(path.read_text(encoding="utf-8"))))
    if (run / "calibration.json").exists():
        parts.append(calibration(json.loads((run / "calibration.json").read_text(encoding="utf-8"))))
    if (run / "recall.json").exists():
        parts.append(recall(json.loads((run / "recall.json").read_text(encoding="utf-8"))))
    for extra in sys.argv[2:]:
        payload = json.loads(Path(extra).read_text(encoding="utf-8"))
        parts.append(power(payload) if "ranking_is_click" in payload else user_rates(payload))
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
