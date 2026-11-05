"""Tables of docs/kuairec/PROTOCOL.md §10 (RESULTS §9) straight from the evidence copies of the two posthoc runs.

Usage: python scripts/make_sid_posthoc_tables.py <evidence/.../sid_search.json> [<.../rqkmeans.json> [<.../tf32_check.json>]] > tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NAMES = {"pop": "热度", "two_tower": "双塔", "sasrec": "SASRec", "gr_text": "文本语义 ID", "gr_shuf": "打乱语义 ID",
         "rqvae": "文本语义 ID（RQ-VAE）", "plain": "残差 k-means（普通）", "balanced": "残差 k-means（平衡）"}
KINDS = ("pop", "two_tower", "sasrec", "gr_text", "gr_shuf")
SID = ("gr_text", "gr_shuf")
GROUP = {"main": "主模型组", "sensitivity": "§8 敏感性"}
SEARCH = {"beam100": "束宽 100", "beam200": "束宽 200", "beam400": "束宽 400", "exact": "精确", "noise_floor": "噪声底"}
TESTS = {"dense": "稠密", "weighted": "有偏", "borrowed": "借用曝光", "natural": "自然日志"}
POP = {"dense": "稠密测试 1,121 人", "natural": "自然测试 4,872 人"}


def plain(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}".replace("-", "−")


def num(x: float, d: int = 4) -> str:
    return f"{x:+.{d}f}".replace("-", "−")


def ci(c: dict, d: int = 4) -> str:
    return f"{num(c['point'], d)} [{num(c['ci95'][0], d)}, {num(c['ci95'][1], d)}]"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def label_name(label: str) -> str:
    kind, seed, *rest = label.split("/")
    return f"{NAMES[kind]}{'（敏感性）' if rest else ''} 种子 {seed}"


def pair_name(pair: str) -> str:
    a, b = pair.split(" − ")
    return f"{NAMES[a]} − {NAMES[b]}"


def verdict(group: str, row: dict) -> str:
    """The §8 sensitivity block does not enter the H1–H3 decisions (protocol §8), so it gets no verdict."""
    if group == "sensitivity":
        return "（不判定）"
    return "成立" if row["holds"] else "不成立"


def search_tables(r: dict) -> list[str]:
    pops = r["populations"]
    out = [f"人群：稠密和有偏 {pops['dense_and_weighted']:,} 人，借用曝光 {pops['borrowed']:,} 人，自然日志 {pops['natural']:,} 人；"
           f"候选池 {pops['pool']:,} 个视频，精确打分的束宽就取这个数。束宽 100 重排后与正式分析的最大差 "
           f"{r['reproduction']['max_abs_difference']:.1e}（容差 {r['reproduction']['tolerance']:.0e}）。", "",
           "#### 9.A 精确打分对 teacher forcing 的核对（每个模型文件 8 个稠密测试用户）", "",
           "| 模型文件 | 每人核对的视频数 | 最大绝对差（关 TF32） | 最大绝对差（开 TF32） |", "| --- | --- | --- | --- |"]
    for label, c in r["exactness"].items():
        out.append(f"| {label_name(label)} | {min(c['videos_per_user'])}–{max(c['videos_per_user'])} | "
                   f"{c['max_abs_diff_tf32_off']:.1e} | {c['max_abs_diff_tf32_on']:.1e} |")

    out += ["", "#### 9.B 束搜索前 100 与精确前 100 的重合率", "",
            "每格是“人均重合率 / P10 / 完全相同的人数占比”。噪声底是束宽 100 按每批 128 人和 16 人各排一次的重合率。", ""]
    for p, block in r["overlap"].items():
        out += [f"**{POP[p]}**", "", "| 模型文件 | 束宽 100 | 束宽 200 | 束宽 400 | 噪声底 |", "| --- | --- | --- | --- | --- |"]
        for label, cells in block.items():
            out.append(f"| {label_name(label)} | " + " | ".join(
                f"{pct(cells[k]['mean'])} / {pct(cells[k]['p10'])} / {pct(cells[k]['identical_share'])}"
                for k in ("beam100", "beam200", "beam400", "noise_floor")) + " |")
        out.append("")

    out += ["#### 9.C Recall@100：精确打分的值，以及各束宽减精确的差（三个种子先平均，95% 区间）", "",
            "| 模型 | 测试 | 精确 | 束宽 100 − 精确 | 束宽 200 − 精确 | 束宽 400 − 精确 |", "| --- | --- | --- | --- | --- | --- |"]
    for g, kinds in r["recall"].items():
        for k, tests in kinds.items():
            for t, cells in tests.items():
                out.append(f"| {NAMES[k]}{'（敏感性）' if g == 'sensitivity' else ''} | {TESTS[t]} | {plain(cells['exact']['value'])} | "
                           + " | ".join(ci(cells[s]["minus_exact"]) for s in ("beam100", "beam200", "beam400")) + " |")

    out += ["", "#### 9.D 各种搜索下的 H1–H3（Recall@100，一族 Holm）", "",
            "| 模型组 | 搜索 | 假设 | 点估计 | 95% 区间 | p | 种子支持 | 逐种子 | 判定 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g in GROUP:
        for s in ("beam100", "beam200", "beam400", "exact"):
            for row in r["searches"][s][g]["primary"]:
                out.append(f"| {GROUP[g]} | {SEARCH[s]} | {row['hypothesis']} | {num(row['point'])} | "
                           f"[{num(row['ci95'][0])}, {num(row['ci95'][1])}] | {row['p_text']} | {row['seed_support']}/3 | "
                           f"{'、'.join(num(v) for v in row['per_seed'])} | {verdict(g, row)} |")

    out += ["", "#### 9.E 各模型的 Recall@100 和 Kendall τ-b（主模型组）", "",
            "| 搜索 | 测试 | " + " | ".join(NAMES[k] for k in KINDS) + " |", "| --- | --- | " + " | ".join("---" for _ in KINDS) + " |"]
    for s in ("beam100", "exact"):
        models = r["searches"][s]["main"]["models"]
        for t in TESTS:
            out.append(f"| {SEARCH[s]} | {TESTS[t]} | " + " | ".join(plain(models[k][t]) for k in KINDS) + " |")
    out += ["", "| 搜索 | " + " | ".join(r["searches"]["beam100"]["main"]["kendall_tau"]) + " |",
            "| --- | " + " | ".join("---" for _ in r["searches"]["beam100"]["main"]["kendall_tau"]) + " |"]
    for s in ("beam100", "beam200", "beam400", "exact"):
        tau = r["searches"][s]["main"]["kendall_tau"]
        out.append(f"| {SEARCH[s]} | " + " | ".join(f"{v:.2f}".replace("-", "−") for v in tau.values()) + " |")

    out += ["", "#### 9.F 含语义 ID 模型的各对差：束宽 100 和精确打分（主模型组，95% 区间）", "",
            "这里只列主模型组、束宽 100 和精确两种搜索；每种搜索、两个模型组的全部模型对和 Kendall τ-b 在 JSON 的 "
            "`searches.<搜索>.<模型组>.pairs`、`kendall_tau` 里。", "",
            "| 对 | 测试 | 束宽 100 | 精确 |", "| --- | --- | --- | --- |"]
    blocks = (("dense_weighted_D", "dense"), ("dense_weighted_D", "weighted"), ("borrowed", "borrowed"), ("natural", "natural"))
    for i, row in enumerate(r["searches"]["beam100"]["main"]["pairs"]["dense_weighted_D"]):
        if not any(k in row["pair"] for k in SID):
            continue
        for block, col in blocks:
            a = r["searches"]["beam100"]["main"]["pairs"][block][i][col]
            b = r["searches"]["exact"]["main"]["pairs"][block][i][col]
            out.append(f"| {pair_name(row['pair'])} | {TESTS[col]} | {ci(a)} | {ci(b)} |")

    gate = r["gate"]
    out += ["", "#### 9.G 闸门（协议 §10.1）", "",
            f"- Recall@100 偏移达到门槛的：{len(gate['recall_shifts'])} 处；",
            f"- 方向相反且两边都显著的对：{len(gate['pair_flips'])} 处；",
            f"- 主模型组 H1–H3 判定改变的：{len(gate['decision_changes'])} 处（§8 敏感性块不进判定，协议 §8）；",
            f"- 闸门{'触发' if gate['triggered'] else '没有触发'}。"]
    for x in gate["recall_shifts"]:
        out.append(f"  - 偏移：{NAMES[x['model']]}（{GROUP[x['group']]}）{TESTS[x['test']]} {ci(x)}")
    block_of = {"dense": "dense_weighted_D", "weighted": "dense_weighted_D", "borrowed": "borrowed", "natural": "natural"}
    for x in gate["pair_flips"]:
        names = [row["pair"] for row in r["searches"]["beam100"][x["group"]]["pairs"]["dense_weighted_D"]]
        i, blk = names.index(x["pair"]), block_of[x["test"]]
        a = r["searches"]["beam100"][x["group"]]["pairs"][blk][i][x["test"]]
        b = r["searches"]["exact"][x["group"]]["pairs"][blk][i][x["test"]]
        out.append(f"  - 换号：{pair_name(x['pair'])}（{GROUP[x['group']]}）{TESTS[x['test']]}：束宽 100 {ci(a)} → 精确 {ci(b)}")
    for x in gate["decision_changes"]:
        out.append(f"  - 判定：{x['hypothesis']} {x['beam100']} → {x['exact']}")

    out += ["", "#### 9.H 排序开销（每个模型文件；秒数含前后 CUDA 同步）", "",
            "| 人群 | 模型 | 搜索 | 每批人数 | 秒数（三个种子） | 显存峰值 GiB（三个种子的最大值） |", "| --- | --- | --- | --- | --- | --- |"]
    for p, block in r["cost"].items():
        by_kind: dict = {}
        for label, cells in block.items():
            kind, _, *rest = label.split("/")
            by_kind.setdefault((kind, bool(rest)), []).append(cells)
        for (kind, sens), cells in by_kind.items():
            for s in ("beam100", "beam200", "beam400", "exact"):
                sec = [c[s]["seconds"] for c in cells]
                out.append(f"| {POP[p]} | {NAMES[kind]}{'（敏感性）' if sens else ''} | {SEARCH[s]} | {cells[0][s]['chunk']} | "
                           f"{'、'.join(f'{x:.1f}' for x in sec)} | {max(c[s]['peak_gib'] for c in cells):.2f} |")
    out.append(f"\n整次运行 {r['seconds'] / 60:.0f} 分钟。")
    return out


def rqkmeans_tables(r: dict) -> list[str]:
    out = ["", "#### 9.I 码表（全部 10,728 个视频；括号里是候选池 3,327 个视频）", "",
           "| 码表 | 种子 | 每层用到的码字 | 前三位组合数 | 共用前三位的视频 | 最大一组 | 区分码最大值 | 三层后的残差范数 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    rows = [("rqvae", s, c) for s, c in r["rqvae_tables"].items()]
    rows += [(v, s, c) for v in ("plain", "balanced") for s, c in r["tables"][v].items()]
    for name, seed, c in rows:
        a, b = c["three_code_collisions"], c["pool_three_code_collisions"]
        resid = f"{c['mean_residual_norm']:.3f}" if "mean_residual_norm" in c else "—"
        out.append(f"| {NAMES[name]} | {seed} | {'、'.join(map(str, c['codes_used_per_level']))} | "
                   f"{a['distinct_prefixes']:,}（{b['distinct_prefixes']:,}） | {a['items_sharing_a_prefix']:,}（{b['items_sharing_a_prefix']:,}） | "
                   f"{a['max_group']}（{b['max_group']}） | {c['fourth_code_max']} | {resid} |")

    out += ["", "#### 9.J 训练和验证子集 Recall@100（选模型用的量）", "",
            "| 码表 | 种子 | 运行编号 | 验证 Recall@100 | 最好 epoch / 实际 epoch | 参数量 | 训练秒数 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for seed, e in r["rqvae_training"].items():
        out.append(f"| {NAMES['rqvae']} | {seed} | `{e['run_id']}` | {plain(e['valid_recall@100'])} | {e['best_epoch']} / {e['epochs_run']} | "
                   f"{e['params']:,} | — |")
    for v in ("plain", "balanced"):
        for seed, e in enumerate(r["training"][v]):
            out.append(f"| {NAMES[v]} | {seed} | `{e['run_id']}` | {plain(e['valid_recall@100'])} | {e['best_epoch']} / {e['epochs_run']} | "
                       f"{e['params']:,} | {e['seconds']:.0f} |")
    val = r["validation"]
    out += ["", f"验证子集上有目标的 {val['users']:,} 人，三个种子先平均：" + "；".join(
        f"{NAMES[n]} {plain(m['mean'])}" for n, m in val["models"].items()) + "。差（95% 区间）："
        + "；".join(f"{pair_name(k)} {ci(c)}" for k, c in val["differences"].items()) + "。"]

    out += ["", "#### 9.K 测试集上的 Recall@100（只作探索；三个种子先平均，括号里是种子 0、1、2，差带 95% 区间）", "",
            "| 搜索 | 测试 | " + " | ".join(NAMES[n] for n in ("rqvae", "plain", "balanced", "sasrec", "two_tower")) + " |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for s in ("beam100", "exact"):
        for t in TESTS:
            m = r["test"][t][s]["models"]
            out.append(f"| {SEARCH[s]} | {TESTS[t]} | " + " | ".join(
                f"{plain(m[n]['mean'])}（{'、'.join(plain(x) for x in m[n]['per_seed'])}）"
                for n in ("rqvae", "plain", "balanced", "sasrec", "two_tower")) + " |")
    out += ["", "| 搜索 | 测试 | 差 | 点估计和 95% 区间 | 逐种子的差（种子 0、1、2） |", "| --- | --- | --- | --- | --- |"]
    for s in ("beam100", "exact"):
        for t in TESTS:
            block = r["test"][t][s]
            for k, c in block["differences"].items():
                v, o = k.split(" − ")
                diffs = [a - b for a, b in zip(block["models"][v]["per_seed"], block["models"][o]["per_seed"])]
                out.append(f"| {SEARCH[s]} | {TESTS[t]} | {pair_name(k)} | {ci(c)} | {'、'.join(num(x) for x in diffs)} |")
    return out


def tf32_tables(r: dict) -> list[str]:
    """§10.3: the exact search with TF32 on and off, same batch size; the §10.1 gate recomputed against TF32 off."""
    rep = r["reproduction_of_reference"]["max_abs_difference"]
    out = ["", "#### 9.L TF32 开关对精确打分的影响（协议 §10.3）", "",
           f"TF32 开的两遍和 §10.1 运行的最大差：束宽 100 {rep['beam100']:.1e}，精确 {rep['exact_on']:.1e}（容差 1e-12，只报告）。", "",
           "前 100 的重合（TF32 开对关，同为精确打分、每批 16 人；人均 / P10 / 完全相同的人数占比）：", "",
           "| 模型文件 | " + " | ".join(POP[p] for p in r["overlap_on_off"]) + " |",
           "| --- | " + " | ".join("---" for _ in r["overlap_on_off"]) + " |"]
    labels = list(next(iter(r["overlap_on_off"].values())))
    for label in labels:
        out.append(f"| {label_name(label)} | " + " | ".join(
            f"{pct(o[label]['mean'])} / {pct(o[label]['p10'])} / {pct(o[label]['identical_share'])}"
            for o in r["overlap_on_off"].values()) + " |")
    out += ["", "Recall@100（三个种子先平均，95% 区间）：", "",
            "| 模型 | 测试 | 精确·TF32 开 | 精确·TF32 关 | 开 − 关 | 束宽 100 − 精确·TF32 关 |", "| --- | --- | --- | --- | --- | --- |"]
    for g, kinds in r["recall"].items():
        for k, tests in kinds.items():
            for t_name in TESTS:
                c = tests[t_name]
                out.append(f"| {NAMES[k]}{'（敏感性）' if g == 'sensitivity' else ''} | {TESTS[t_name]} | {plain(c['exact_on'])} | "
                           f"{plain(c['exact_off'])} | {ci(c['on_minus_off'], 6)} | {ci(c['beam100_minus_off'])} |")
    gate = r["gate_against_tf32_off"]
    out += ["", "按 TF32 关的精确打分重算 §10.1 的闸门：", "",
            f"- Recall@100 偏移达到门槛的：{len(gate['recall_shifts'])} 处；",
            f"- 方向相反且两边都显著的对：{len(gate['pair_flips'])} 处；",
            f"- 主模型组 H1–H3 判定改变的：{len(gate['decision_changes'])} 处。", "",
            "§10.1 的 5 处换号在 TF32 关时的精确端：", ""]
    for x in r["reference_flips"]:
        out.append(f"- {pair_name(x['pair'])}（{GROUP[x['group']]}）{TESTS[x['test']]}：TF32 开 {ci(x['exact_on_now'])}，"
                   f"TF32 关 {ci(x['exact_off'])}")
    out += ["", "| 模型组 | TF32 | 假设 | 点估计 | 95% 区间 | 种子支持 | 判定 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for g in GROUP:
        for m, name in (("exact_on", "开"), ("exact_off", "关")):
            for row in r["modes"][m][g]["primary"]:
                out.append(f"| {GROUP[g]} | {name} | {row['hypothesis']} | {num(row['point'])} | "
                           f"[{num(row['ci95'][0])}, {num(row['ci95'][1])}] | {row['seed_support']}/3 | {verdict(g, row)} |")
    out += ["", "| TF32 | " + " | ".join(r["modes"]["exact_off"]["main"]["kendall_tau"]) + " |",
            "| --- | " + " | ".join("---" for _ in r["modes"]["exact_off"]["main"]["kendall_tau"]) + " |"]
    for m, name in (("exact_on", "开"), ("exact_off", "关")):
        out.append(f"| {name} | " + " | ".join(f"{v:.2f}".replace("-", "−") for v in r["modes"][m]["main"]["kendall_tau"].values()) + " |")
    return out


def main() -> None:
    out = search_tables(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    if len(sys.argv) > 2:
        out += rqkmeans_tables(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
    if len(sys.argv) > 3:
        out += tf32_tables(json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")))
    print("\n".join(out))


if __name__ == "__main__":
    main()
