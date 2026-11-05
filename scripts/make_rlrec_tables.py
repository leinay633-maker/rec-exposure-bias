"""Markdown tables for docs/rlrec/RESULTS.md from the study 3 report (python -m rlrec test / hold-report).

Usage: python scripts/make_rlrec_tables.py <test.json> [<hold.json> [<registry.jsonl> [<sharpen.json>]]]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

LEVELS = ("0.5", "1.0", "2.0")
NAMES = {"pi0": "π0（SFT）"}
H_TEXT = {"H1": "H1 迁移（解析最优）：π_β^R1 − π0", "H2": "H2 校准（解析最优）：π_β^R3 − π_β^R1",
          "H3": "H3 随机数据（解析最优）：π_β^R5 − π_β^R4", "H4": "H4 迁移（RLOO）：π_RL^R1 − π0",
          "H5": "H5 校准（RLOO）：π_RL^R3 − π_RL^R1", "H6": "H6 随机数据（RLOO）：π_RL^R5 − π_RL^R4"}


def f(x: float, d: int = 4) -> str:
    return "—" if x is None else f"{x:+.{d}f}".replace("-", "−")


def v(x: float, d: int = 4) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def ci(c: list) -> str:
    return f"[{f(c[0])}, {f(c[1])}]"


def p_text(p: float) -> str:
    return "< 0.0002" if p == 0 else f"{p:.4f}"


def hypotheses(st: dict, overlaps: dict, descriptive: bool = False) -> list[str]:
    k = st["k_main"]
    note = "只描述，不做 Holm、不下判定。" if descriptive else f"p 是双侧；Holm 在主假设族 {', '.join(st['family']) or '（空）'} 里做。"
    out = [f"主指标 V_{k}，相同平均 KL 1 nat。区间是联合用户 bootstrap 的 95% 区间；{note}",
           "", "| 假设 | 差 | 95% 区间 | p | Holm | 逐种子（SFT 种子 0、1、2） | 种子支持 | 判定 | 最小可检出差 | 前 500 人均重合 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for h, r in st["hypotheses"].items():
        if "point" not in r:
            out.append(f"| {H_TEXT[h]} | 策略缺失 | | | | | | {'（只描述）' if descriptive else '不成立'} | |")
            continue
        seeds = "、".join(f(x) for x in r["per_seed"])
        if descriptive:
            holm = verdict = "（只描述）"
        elif not r.get("testable", True):
            holm, verdict = "不可检验", "不成立（只描述）"
        else:
            holm = ("拒绝" if r.get("holm_reject") else "不拒绝") if r.get("in_family") else "（不在族内）"
            verdict = ("成立" if r.get("holds") else "不成立") if r.get("in_family") else "（只报告）"
        ov = overlaps.get(h)
        out.append(f"| {H_TEXT[h]} | {f(r['point'])} | {ci(r['ci95'])} | {p_text(r['p']) if 'p' in r else '—'} | {holm} | {seeds} | "
                   f"{r['seed_support']}/3 | {verdict} | {v(r.get('mde'))} | {v(ov, 3) if ov is not None else '—'} |")
    return out


def value_table(st: dict, ev: dict, kinds: tuple, title: str) -> list[str]:
    k = str(st["k_main"])
    out = [title, "", f"| 策略 | 奖励 | KL 档 | V_{k} | 95% 区间 | 测试人群 KL 均值（P10 / P50 / P90） | E[R1] | E[R3] | 最热组占比 | 覆盖视频数 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, row in sorted(st["values"].items()):
        parts = name.split("/")
        if parts[0] not in kinds:
            continue
        pol = ev["policies"][name]
        kl = pol.get("kl")
        kl_text = f"{kl['mean']:.3f}（{kl['p10']:.3f} / {kl['p50']:.3f} / {kl['p90']:.3f}）" if kl else "—"
        proxy = pol.get("proxy", {})
        reward = parts[1] if len(parts) > 1 else "—"
        level = parts[2] if len(parts) > 2 else "—"
        cp = pol.get("checkpoints")
        if cp:
            level += f"（{cp['aligned']}/{cp['models']} 对齐）"
        out.append(f"| {NAMES.get(parts[0], parts[0]) if parts[0] != 'pi0' else 'π0'} | {reward} | {level} | {v(row[k]['point'])} | "
                   f"[{row[k]['ci95'][0]:.4f}, {row[k]['ci95'][1]:.4f}] | {kl_text} | {v(proxy.get('R1'))} | {v(proxy.get('R3'))} | "
                   f"{pol['hot_share']:.3f} | {pol['coverage']:,.0f} |")
    return out


def o4_table(st: dict) -> list[str]:
    out = ["| 奖励 / 档 | O4 − O3 | O4 − π0 | O3 − π0 | 追上比例 |", "| --- | --- | --- | --- | --- |"]
    for key, e in st["o4_vs_o3"].items():
        ratio = e.get("catch_up_ratio")
        out.append(f"| {key} | {f(e['o4_minus_o3']['point'])} {ci(e['o4_minus_o3']['ci95'])} | "
                   f"{f(e['o4_minus_pi0']['point'])} {ci(e['o4_minus_pi0']['ci95'])} | "
                   f"{f(e['o3_minus_pi0']['point'])} {ci(e['o3_minus_pi0']['ci95'])} | {f(ratio, 2) if ratio is not None else '不报'} |")
    return out


def mechanism(st: dict) -> list[str]:
    k = str(st["k_main"])
    out = [f"| 对比（解析最优，1 nat） | 差（V_{k}） | 95% 区间 |", "| --- | --- | --- |"]
    for label, rows in st["mechanism"].items():
        out.append(f"| {label} | {f(rows[k]['point'])} | {ci(rows[k]['ci95'])} |")
    return out


def other_k(st: dict) -> list[str]:
    ks = sorted({kk for r in st["hypotheses"].values() for kk in r.get("other_k", {})}, key=int)
    out = ["| 假设 | " + " | ".join(f"K = {kk}" for kk in ks) + " |", "| --- |" + " --- |" * len(ks)]
    for h, r in st["hypotheses"].items():
        if "other_k" in r:
            out.append(f"| {h} | " + " | ".join(f"{f(r['other_k'][kk]['point'])} {ci(r['other_k'][kk]['ci95'])}" for kk in ks) + " |")
    return out


def sampled(st: dict) -> list[str]:
    out = ["| 策略 | 按概率抽样的价值 | 95% 区间 | 有效样本量 | 每人有效支撑（中位数） |", "| --- | --- | --- | --- | --- |"]
    for name, row in sorted(st["values"].items()):
        s = row.get("sampled")
        if s:
            out.append(f"| {name} | {v(s['point'])} | [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}] | {s['ess']:,.0f} | {s['support_median']:,.1f} |")
    return out


def sampled_main(st: dict) -> list[str]:
    """§5.6: the policies the text talks about, at the main KL level."""
    out = ["| 策略 | 抽样价值 | 95% 区间 | 有效样本量 | 每人有效支撑（中位数） |", "| --- | --- | --- | --- | --- |"]
    for label, name in (("O3 R1", "O3/R1/1.0"), ("O4 R1", "O4/R1/1.0/main"), ("O3 R5", "O3/R5/1.0"), ("O4 R5", "O4/R5/1.0/main"), ("O3 R6", "O3/R6/1.0")):
        s = st["values"][name]["sampled"]
        out.append(f"| {label} | {v(s['point'])} | [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}] | {s['ess']:,.0f} | {s['support_median']:,.1f} |")
    return out


def delta_table(st: dict) -> list[str]:
    k = str(st["k_main"])
    base = st["values"]["pi0"][k]["point"]
    out = ["| 奖励 | 方法 | 0.5 nat | 1 nat | 2 nat（O4 都没越过） | 无 KL：O2 | 无 KL：O2′ |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for reward in ("R1", "R3", "R4", "R5"):
        for name, key in (("O3", "o3_minus_pi0"), ("O4", "o4_minus_pi0")):
            cells = [f"{f(e[key]['point'])} {ci(e[key]['ci95'])}" for e in (st["o4_vs_o3"][f"{reward}/{lv}"] for lv in LEVELS)]
            tail = [f(st["values"][f"{o}/{reward}"][k]["point"] - base) for o in ("O2", "O2p")] if name == "O3" else ["", ""]
            out.append(f"| {reward} | {name} | " + " | ".join(cells + tail) + " |")
    for reward in ("R2", "R6"):  # the report has no paired interval against π0 for these two
        cells = [f(st["values"][f"O3/{reward}/{lv}"][k]["point"] - base) for lv in LEVELS]
        tail = [f(st["values"][f"{o}/{reward}"][k]["point"] - base) for o in ("O2", "O2p")]
        out.append(f"| {reward} | O3 | " + " | ".join(cells + tail) + " |")
    return out


def secondary(st: dict, ev: dict) -> list[str]:
    k = str(st["k_main"])
    out = [f"| 策略 | V_100 | V_200 | V_1000 | long_view 列表价值 V_{k} | Recall@500（有偏的常规日志） | KL（按每人随机行数加权） |",
           "| --- | --- | --- | --- | --- | --- | --- |"]

    def cell(x: dict) -> str:
        return f"{v(x['point'])} [{x['ci95'][0]:.4f}, {x['ci95'][1]:.4f}]"

    for name, row in sorted(st["values"].items()):
        pol = ev["policies"][name]
        kl = pol.get("kl")
        out.append(f"| {name} | {cell(row['100'])} | {cell(row['200'])} | {cell(row['1000'])} | {cell(row[f'{k}_long_view'])} | "
                   f"{v(pol.get('regular_recall500'))} | {v(kl['row_weighted_mean'], 3) if kl else '—'} |")
    return out


def per_model(ev: dict, k: str) -> list[str]:
    out = [f"V_{k}：", "", f"| 策略 | 模型自己的 V_{k}（SFT 种子 0 ／ 1 ／ 2；O4 每格是 RL 种子 0、1、2） | 检查点 KL 最小–最大 | 检查点的步 |", "| --- | --- | --- | --- |"]
    for name, pol in ev["policies"].items():
        models = pol.get("per_model")
        if not models:
            continue
        by: dict = {}
        for m in models:
            by.setdefault(m["sft_seed"], []).append(m)
        cells = " ／ ".join("、".join(f"{m['V'][k]:.4f}" for m in sorted(by[s], key=lambda x: x["rl_seed"] or 0)) for s in sorted(by))
        cps = [m["checkpoint"] for m in models if m.get("checkpoint")]
        kl = f"{min(c['kl'] for c in cps):.3f}–{max(c['kl'] for c in cps):.3f}" if cps else "—"
        step = f"{min(c['step'] for c in cps)}–{max(c['step'] for c in cps)}" if cps else "—"
        out.append(f"| {name} | {cells} | {kl} | {step} |")
    return out


def calibration(ev: dict) -> list[str]:
    out = ["| 奖励 | HOLD 行上的 LogLoss | 预测均值 | 点击率 |", "| --- | --- | --- | --- |"]
    for name, row in ev["reward_calibration"].items():
        out.append(f"| {name} | {row['logloss']:.4f} | {row['mean_prediction']:.4f} | {row['click_rate']:.4f} |")
    return out


def run_hours(registry: Path) -> dict:
    """Hours of each finished run, keyed by run id, from its start and end records in the registry."""
    first: dict = {}
    out: dict = {}
    for line in registry.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        first.setdefault(r["run_id"], r)
        if r["status"] == "ok":
            t0, t1 = datetime.fromisoformat(first[r["run_id"]]["time"]), datetime.fromisoformat(r["time"])
            out[r["run_id"]] = {"record": r, "hours": (t1 - t0).total_seconds() / 3600}
    return out


def cost_table(st: dict, ev: dict, registry: Path) -> list[str]:
    k = str(st["k_main"])
    base = st["values"]["pi0"][k]["point"]
    runs = run_hours(registry)
    o3_hours = sum(x["hours"] for x in runs.values() if x["record"]["kind"] == "rlrec-o3")
    minutes: dict = {}
    for x in runs.values():
        if x["record"]["kind"] == "rlrec-rl":
            c = x["record"]["config"]
            minutes.setdefault((c["reward"], c["grpo"]), []).append(x["hours"] * 60)
    out = [f"| 策略 | V_{k} − π0 | 相对 π0 | 覆盖视频数 | 最热组占比 | 额外的成本（本机 RTX 3080） |", "| --- | --- | --- | --- | --- | --- |"]

    def line(name: str, label: str, cost: str) -> str:
        pol = ev["policies"][name]
        d = st["values"][name][k]["point"] - base
        return f"| {label} | {f(d)} | {f(d / base * 100, 1)}% | {pol['coverage']:,.0f} | {pol['hot_share']:.3f} | {cost} |"

    def rl_cost(reward: str, grpo: bool) -> str:
        m = minutes[(reward, grpo)]
        return f"{len(m)} 次运行，平均 {sum(m) / len(m):.0f} 分钟，最长 {max(m):.0f} 分钟"

    out.append(line("pi0", "π0", "SFT 本身"))
    out += [line(f"O2p/{r}", f"O2′ {r}", "只查奖励表") for r in ("R1", "R3", "R5", "R6")]
    out += [line(f"O2/{r}", f"O2 {r}", "只查奖励表") for r in ("R1", "R3", "R5", "R6")]
    out += [line(f"O3/{r}/1.0", f"O3 {r} 1 nat", f"不训练；全部组合的 β 一共 {o3_hours:.2f} 小时") for r in ("R1", "R3", "R4", "R5", "R6")]
    out += [line(f"O4/{r}/1.0/main", f"O4 {r} 1 nat", rl_cost(r, False)) for r in ("R1", "R3", "R4", "R5")]
    out += [line(f"O4g/{r}/1.0/secondary", f"O4g {r} 1 nat", rl_cost(r, True)) for r in ("R1", "R3")]
    return out


def sharpen(block: dict, levels: dict) -> list[str]:
    """§12.6: the sampled value of π0 sharpened to each KL level, then each policy against it."""
    vals, diffs = block["values"], block["differences"]
    def seeds(x: dict) -> str:
        return "、".join(v(x["seed_values"][str(s)]) for s in range(3))

    out = ["| π0 调尖到 | 抽样价值 | 95% 区间 | 减 π0 | 逐种子（SFT 种子 0、1、2） | 有效样本量 | 每人有效支撑（中位数） | 这批人的平均 KL | τ（3 个 SFT 种子） |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for lv in LEVELS:
        x = vals[f"T/{lv}"]
        d = diffs[f"T/{lv} − pi0"]
        out.append(f"| {lv} nat | {v(x['point'])} | [{x['ci95'][0]:.4f}, {x['ci95'][1]:.4f}] | {f(d['point'])} {ci(d['ci95'])} | {seeds(x)} | {x['ess']:,.0f} | "
                   f"{x['support_median']:,.1f} | {x['kl_mean']:.3f} | {'、'.join(f'{levels[f"{s}/{lv}"]["tau"]:.3f}' for s in range(3))} |")
    out += ["", "| 策略 | 抽样价值 | 减同档调尖的 π0 | 逐种子（SFT 种子 0、1、2） | 有效样本量 |", "| --- | --- | --- | --- | --- |"]
    for label, x in diffs.items():
        a, b = label.split(" − ")
        if b != "pi0":
            out.append(f"| {a} | {v(vals[a]['point'])} | {f(x['point'])} {ci(x['ci95'])} | {seeds(vals[a])} | {vals[a]['ess']:,.0f} |")
    return out


def sharpen_main(sh: dict, st: dict) -> list[str]:
    """§12.6 at 1 nat, test and HOLD side by side; sharpening keeps π0's order, so V_K minus π0 is also V_K minus π_τ."""
    k = str(st["k_main"])
    base = st["values"]["pi0"][k]["point"]
    out = [f"| 策略 | 抽样价值（测试） | 减调尖的 π0（测试） | 减调尖的 π0（HOLD） | 每人有效支撑（测试） | V_{k} − π0（测试） |", "| --- | --- | --- | --- | --- | --- |"]
    for name, rewards, key, tail in (("O3", ("R1", "R3", "R4", "R5", "R6"), "o3_minus_pi0", ""), ("O4", ("R1", "R3", "R4", "R5"), "o4_minus_pi0", "/main")):
        for r in rewards:
            policy = f"{name}/{r}/1.0{tail}"
            test, held = (sh[w]["differences"][f"{policy} − T/1.0"] for w in ("test", "hold"))
            s = st["values"][policy]["sampled"]
            pair = st["o4_vs_o3"].get(f"{r}/1.0")  # R6 has no paired interval against π0
            value = f"{f(pair[key]['point'])} {ci(pair[key]['ci95'])}" if pair else f(st["values"][policy][k]["point"] - base)
            out.append(f"| {name} {r} | {v(s['point'])} | {f(test['point'])} {ci(test['ci95'])} | {f(held['point'])} {ci(held['ci95'])} | {s['support_median']:,.1f} | {value} |")
    return out


def main(test_path: str, hold_path: str | None = None, registry_path: str | None = None, sharpen_path: str | None = None) -> None:
    rep = json.loads(Path(test_path).read_text(encoding="utf-8"))
    st, ev = rep["statistics"], rep["evaluation"]
    lines = [f"#### 主假设（{ev['users']:,} 人、{ev['rows']:,} 行）", ""] + hypotheses(st, ev["overlaps"])
    lines += ["", "#### 其他 K 下的主假设差（只报告）", ""] + other_k(st)
    lines += ["", "#### 各档相对 π0 的差（只报告）", ""] + delta_table(st)
    lines += ["", "#### 机制拆分（只报告）", ""] + mechanism(st)
    lines += ["", "#### RLOO 对解析最优（只报告）", ""] + o4_table(st)
    lines += ["", ""] + value_table(st, ev, ("pi0", "O3"), "#### π0 和解析最优")
    lines += ["", ""] + value_table(st, ev, ("O4", "O4g"), "#### RLOO（主运行、GRPO 式变体）")
    lines += ["", ""] + value_table(st, ev, ("O2", "O2p"), "#### 重排和纯奖励排序（没有 KL）")
    lines += ["", "#### 策略按概率抽样的价值（只描述）", ""] + sampled(st)
    lines += ["", "#### 1 nat 的抽样价值（只描述）", ""] + sampled_main(st)
    lines += ["", "#### 其他 K、long_view、常规测试和加权 KL（只报告）", ""] + secondary(st, ev)
    lines += ["", "#### 每个模型自己的值（只报告）", ""]
    for k in ("500", "100", "200", "1000"):
        lines += per_model(ev, k) + [""]
    if registry_path:
        lines += ["", "#### 效果、覆盖和成本（只报告）", ""] + cost_table(st, ev, Path(registry_path))
    if ev.get("unreachable"):
        lines += ["", "不可达、没算的策略：" + "、".join(ev["unreachable"])]
    if hold_path:
        hold = json.loads(Path(hold_path).read_text(encoding="utf-8"))
        lines += ["", f"#### HOLD 上的主假设差（模型组冻结后算，只描述；{hold['evaluation']['users']:,} 人）", ""]
        hs, hv = hold["statistics"], hold["evaluation"]
        lines += hypotheses(hs, hv["overlaps"], descriptive=True)
        lines += ["", "#### HOLD 上的校准诊断（只描述）", ""] + calibration(hv)
        lines += ["", ""] + value_table(hs, hv, ("pi0", "O3"), "#### HOLD 上的 π0 和解析最优（只描述）")
        lines += ["", ""] + value_table(hs, hv, ("O4", "O4g"), "#### HOLD 上的 RLOO（只描述）")
        lines += ["", ""] + value_table(hs, hv, ("O2", "O2p"), "#### HOLD 上的重排和纯奖励排序（只描述）")
        lines += ["", "#### HOLD 上按概率抽样的价值（只描述）", ""] + sampled(hs)
        lines += ["", "#### HOLD 上的其他 K、long_view、常规验证 Recall@500 和加权 KL（只描述）", ""] + secondary(hs, hv)
        lines += ["", "#### HOLD 上每个模型自己的值（只描述；其他 K 见 `hold.json`）", ""] + per_model(hv, "500")
    if sharpen_path:
        sh = json.loads(Path(sharpen_path).read_text(encoding="utf-8"))
        lines += ["", f"#### 开封后的探索：π0 调尖到相同 KL，随机测试（{sh['test']['users']:,} 人）", ""] + sharpen(sh["test"], sh["temperatures"])
        lines += ["", f"#### 开封后的探索：π0 调尖到相同 KL，HOLD（{sh['hold']['users']:,} 人）", ""] + sharpen(sh["hold"], sh["temperatures"])
        lines += ["", "#### 开封后的探索：1 nat 下各策略减调尖的 π0", ""] + sharpen_main(sh, st)
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
