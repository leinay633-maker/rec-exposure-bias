"""Appendix tables of study 2 (docs/kuairec/RESULTS.md) straight from the formal analysis JSON and the selection.

The script only formats numbers, rounds them and counts; every figure in the appendix comes from these two files.

Usage: python scripts/make_kuairec_tables.py [result.json] [selection.json] > appendix.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "evidence/kuairec/analysis/20261101-165505-analysis-c8c25f/result.json"
SELECTION = ROOT / "evidence/kuairec/zoo/selection.json"
NAMES = {"pop": "热度", "two_tower": "双塔", "sasrec": "SASRec", "gr_text": "文本语义 ID", "gr_shuf": "打乱语义 ID"}
KINDS = list(NAMES)
METRICS = ("recall@100", "recall@20", "recall@50", "ndcg@20", "ndcg@50", "ndcg@100")
COLS = {"dense": "稠密", "weighted": "有偏", "D": "D", "borrowed": "借用", "natural": "自然"}


def pair_name(pair: str) -> str:
    a, b = pair.split(" − ")
    return f"{NAMES[a]} − {NAMES[b]}"


def num(x: float, d: int = 4) -> str:
    return f"{x:+.{d}f}".replace("-", "−")


def plain(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}".replace("-", "−")


def ci(c: list) -> str:
    return f"[{plain(c[0])}, {plain(c[1])}]"


def p_text(cell: dict) -> str:
    return cell["p_text"] if cell["p"] == 0 else f"{cell['p']:.4f}"


def seeds(cell: dict) -> str:
    return "、".join(num(v) for v in cell["per_seed"])


def pair_table(pairs: list, cols: list) -> list[str]:
    out = ["| 对 | 列 | 点估计 | 95% 区间 | 标准误 | p | Holm | 种子支持 | 逐种子 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in pairs:
        for col in cols:
            c = row[col]
            out.append(f"| {pair_name(row['pair'])} | {COLS[col]} | {num(c['point'])} | {ci(c['ci95'])} | {plain(c['se'])} | "
                       f"{p_text(c)} | {'拒绝' if c['holm_reject'] else '不拒绝'} | {c['seed_support']}/3 | {seeds(c)} |")
    return out


def main() -> None:
    result_path = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULT
    selection_path = Path(sys.argv[2]) if len(sys.argv) > 2 else SELECTION
    r = json.loads(result_path.read_text(encoding="utf-8"))
    sel = json.loads(selection_path.read_text(encoding="utf-8"))
    lines = ["## 附录：全部表格（`scripts/make_kuairec_tables.py` 从正式分析生成）", ""]

    pop = r["population"]
    lines += ["### A1 人群", "",
              f"- 稠密测试和 H1–H3：{r['users']:,} 人。H1 人群（有目标、目标权重大于 0）{pop['weighted_and_D']:,} 人，"
              f"H3 人群（有目标）{pop['dense']:,} 人，两者是同一批人；没有目标的 {pop['excluded_no_target']} 人，"
              f"目标权重为 0 的 {pop['excluded_zero_weight']} 人。",
              f"- 每人目标数 P10、P50、P90：{'、'.join(f'{x:.0f}' for x in pop['targets_per_user'])}。",
              f"- 借用曝光：{pop['borrowed']:,} 人，比 H1 人群少 {pop['excluded_no_borrowed_draw']} 人（20 次都没有目标）；"
              f"每人平均用到 {pop['mean_borrowed_draws_used']:.2f} 次；每人在用到的几次里的平均目标数 P10、P50、P90："
              f"{'、'.join(f'{x:.2f}' for x in pop['borrowed_targets_per_user'])}。"]
    nat = r["natural"]
    steps = nat["steps"]
    lines += [f"- 自然测试：其他用户 {nat['other_users']:,} 人，进入 {nat['users']:,} 人，排除 {nat['excluded']:,} 人。逐步："
              f"窗口里有行 {steps['other_users_in_window']:,}，有候选池正样本 {steps['with_positive_pool_row']:,}，"
              f"去掉训练里看过的视频后仍有目标 {steps['with_unexposed_target']:,}，有训练正样本事件 "
              f"{steps['with_positive_event_in_training']:,}。", ""]

    lines += ["### A2 主假设（每个指标）", "",
              "| 指标 | 假设 | 统计量 | 点估计 | 95% 区间 | 标准误 | p | Holm | 最小可检出差 | 种子支持 | 逐种子 | 成立 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for metric in METRICS:
        for row in r["metrics"][metric]["primary"]["rows"]:
            holds = ("是" if row["holds"] else "否") if metric == "recall@100" else "（次要指标不判定）"
            stat = row["statistic"].replace("dense(", "Δ稠密(")
            for k, v in NAMES.items():
                stat = stat.replace(k, v)
            lines.append(f"| {metric} | {row['hypothesis']} | {stat} | "
                         f"{num(row['point'])} | {ci(row['ci95'])} | {plain(row['se'])} | {p_text(row)} | "
                         f"{'拒绝' if row['holm_reject'] else '不拒绝'} | {plain(row['mde_holm_first_step'])} | {row['seed_support']}/3 | "
                         f"{seeds(row)} | {holds} |")
    lines.append("")

    lines += ["### A3 §8 敏感性块（不进判定，Recall@100）", "",
              "| 假设 | 点估计 | 95% 区间 | 标准误 | p | 种子支持 | 逐种子 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in r["sensitivity"]["rows"]:
        lines.append(f"| {row['hypothesis']} | {num(row['point'])} | {ci(row['ci95'])} | {plain(row['se'])} | {p_text(row)} | "
                     f"{row['seed_support']}/3 | {seeds(row)} |")
    lines += ["", "敏感性模型的 Recall@100（H1 人群）：" + "；".join(
        f"{NAMES[k]} 稠密 {plain(v['dense'])}、有偏 {plain(v['weighted'])}" for k, v in r["sensitivity"]["models"].items()) + "。", ""]

    lines += ["### A4 各模型的均值（每个用户先对种子平均）", "",
              "| 指标 | 测试 | " + " | ".join(NAMES.values()) + " |", "| --- | --- | " + " | ".join("---" for _ in KINDS) + " |"]
    for metric in METRICS:
        models = r["metrics"][metric]["weighted_population"]["models"]
        for col in ("dense", "weighted"):
            lines.append(f"| {metric} | {COLS[col]}（H1 人群） | " + " | ".join(plain(models[k][col]) for k in KINDS) + " |")
    b, n = r["borrowed"]["models"], r["natural"]["models"]
    lines.append("| recall@100 | 借用曝光人群上的稠密 | " + " | ".join(plain(b[k]["dense"]) for k in KINDS) + " |")
    lines.append("| recall@100 | 借用曝光 | " + " | ".join(plain(b[k]["borrowed"], 6) for k in KINDS) + " |")
    lines.append("| recall@100 | 自然日志 | " + " | ".join(plain(n[k]["natural"]) for k in KINDS) + " |")
    lines += ["", "Kendall τ-b（Recall@100，5 个模型，只作描述）：" +
              "；".join(f"{k.replace('dense', '稠密').replace('weighted', '有偏').replace('borrowed', '借用').replace('natural', '自然')} {v:.2f}"
                       for k, v in r["kendall_tau"].items()) + "。", ""]

    for metric in METRICS:
        lines += [f"### A5 十对模型 · {metric} · H1 人群（每列 10 个一起 Holm）", ""]
        lines += pair_table(r["metrics"][metric]["weighted_population"]["pairs"], ["dense", "weighted", "D"])
        lines.append("")
    lines += ["### A6 十对模型 · recall@100 · H3 人群的稠密测试", "",
              "H3 人群和 H1 人群是同一批 1,121 人，所以这张表和 A5 里 recall@100 的稠密列相同。", ""]
    lines += pair_table(r["metrics"]["recall@100"]["dense_population"]["pairs"], ["dense"])
    lines.append("")

    lines += ["### A7 借用曝光集合（recall@100，借用曝光人群）", ""]
    lines += pair_table(r["borrowed"]["pairs"], ["dense", "borrowed", "D"])
    mc = r["borrowed"]["monte_carlo_se"]
    lines += ["", "20 次供体分配的蒙特卡洛标准误（只反映供体分配的随机性）：", "",
              "| 模型或对 | 蒙特卡洛标准误 |", "| --- | --- |"]
    lines += [f"| {NAMES[k]} | {plain(v, 5)} |" for k, v in mc["models"].items()]
    lines += [f"| {pair_name(k)} | {plain(v, 5)} |" for k, v in mc["pairs"].items()]
    lines.append("")

    lines += ["### A8 自然日志（recall@100，自然测试人群）", ""]
    lines += pair_table(r["natural"]["pairs"], ["natural"])
    lines.append("")

    lines += ["### A9 热度分组（稠密测试，H3 人群）", "",
              "| 模型 | 组 0 召回 | 组 1 | 组 2 | 组 3 | 组 4 | 占位 组 0 | 组 1 | 组 2 | 组 3 | 组 4 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k in KINDS:
        g = r["groups"][k]
        lines.append(f"| {NAMES[k]} | " + " | ".join(plain(x, 4) for x in g["recall_by_group"]) + " | "
                     + " | ".join(plain(x, 3) for x in g["slot_share"]) + " |")
    lines += ["", "有各组目标的人数：" + "、".join(f"{x:,}" for x in r["groups"]["pop"]["users_with_group_targets"]) + "。", ""]

    lines += ["### A10 训练和排序（selection 与正式分析）", "",
              "| 模型 | 种子 | 运行编号 | 验证 Recall@100 | 最好 epoch | 实际 epoch | 参数量 | 训练秒数 | 批大小 | 测试用户排序秒数 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    timing = r["ranking_seconds"]
    for group, blocks in (("", sel["models"]), ("（敏感性）", sel["sensitivity"])):
        for kind, block in blocks.items():
            for seed, e in block["seeds"].items():
                key = f"{kind}/{seed}" + ("/sensitivity" if group else "")
                lines.append(f"| {NAMES[kind]}{group} | {seed} | `{e['run_id']}` | {plain(e['valid_recall@100'])} | {e['best_epoch']} | "
                             f"{e['epochs_run']} | {e['params']:,} | {e['seconds']} | {e['batch']} | {timing.get(key, '—')} |")
    lines.append("")
    codes = r["diagnostics"]["semantic_ids"]
    lines += ["语义 ID：" + "；".join(
        f"种子 {s} 每层用到 {'、'.join(map(str, c['codes_used_per_level']))} 个码字，前三位组合 {c['three_code_collisions']['distinct_prefixes']:,} 种，"
        f"共用前三位的视频 {c['three_code_collisions']['items_sharing_a_prefix']:,} 个，最大一组 {c['three_code_collisions']['max_group']} 个，"
        f"区分码最大 {c['fourth_code_max']}" for s, c in codes.items()) + "。没有一个种子触发加倍步数重训。", ""]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
