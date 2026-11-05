"""SVG figures for docs/rlrec/RESULTS.md from the study 3 report (PROTOCOL.md §7).

Standard library only: the figures are a few lines and dots, and the repo has no plotting dependency.

Usage: python scripts/make_rlrec_figures.py <test.json> <output directory>
"""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

LEVELS = ("0.5", "1.0", "2.0")
K = "500"
REWARDS = {"R1": ("R1 常规日志模型", "circle"), "R3": ("R3 分组校准", "square"), "R4": ("R4 多一段常规数据", "triangle"),
           "R5": ("R5 混入随机数据", "diamond"), "R6": ("R6 只用随机数据", "cross")}
# Validated with the dataviz palette script in both modes; slot order is the colour-blind safety mechanism.
LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
         "R1": "#2a78d6", "R3": "#eb6834", "R4": "#1baf7a", "R5": "#eda100", "R6": "#e87ba4"}
DARK = {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "R1": "#3987e5", "R3": "#d95926", "R4": "#199e70", "R5": "#c98500", "R6": "#d55181"}


def style() -> str:
    def block(p: dict) -> str:
        return " ".join(f"--{k}:{v};" for k, v in p.items())
    series = " ".join(f".l{r}{{stroke:var(--{r})}} .f{r}{{fill:var(--{r})}} .h{r}{{stroke:var(--{r});fill:var(--surface)}}" for r in REWARDS)
    return f"""<style>
svg{{{block(LIGHT)}font-family:system-ui,"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",sans-serif}}
@media (prefers-color-scheme:dark){{svg{{{block(DARK)}}}}}
.bg{{fill:var(--surface)}} .title{{fill:var(--ink);font-size:21px;font-weight:600}} .sub{{fill:var(--ink2);font-size:15px}}
.panel{{fill:var(--ink);font-size:16px;font-weight:600}} .tick{{fill:var(--muted);font-size:13px;font-variant-numeric:tabular-nums}}
.axis-label{{fill:var(--ink2);font-size:14px}} .legend{{fill:var(--ink2);font-size:14px}} .label{{fill:var(--ink2);font-size:13px}}
.grid{{stroke:var(--grid);stroke-width:1}} .base{{stroke:var(--axis);stroke-width:1.5}}
.line{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}} .ci{{stroke-width:1.25;opacity:.7}}
.ring{{stroke:var(--surface);stroke-width:2}} .hollow{{stroke-width:2}} .stem{{stroke-width:2;stroke-linecap:round}}
{series}
</style>"""


class Panel:
    def __init__(self, left: float, top: float, width: float, height: float, xlim: tuple, ylim: tuple):
        self.left, self.top, self.width, self.height, self.xlim, self.ylim = left, top, width, height, xlim, ylim

    def x(self, value: float) -> float:
        return self.left + (value - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.width

    def y(self, value: float) -> float:
        return self.top + (self.ylim[1] - value) / (self.ylim[1] - self.ylim[0]) * self.height

    def frame(self, title: str, xlabel: str, xticks: list, yticks: list, xfmt, yfmt) -> list[str]:
        out = [f'<text class="panel" x="{self.left}" y="{self.top - 14}">{escape(title)}</text>']
        bottom = self.top + self.height
        for t in yticks:
            out.append(f'<line class="grid" x1="{self.left}" x2="{self.left + self.width}" y1="{self.y(t):.1f}" y2="{self.y(t):.1f}"/>')
            out.append(f'<text class="tick" text-anchor="end" x="{self.left - 8}" y="{self.y(t) + 4:.1f}">{yfmt(t)}</text>')
        out.append(f'<line class="base" x1="{self.left}" x2="{self.left + self.width}" y1="{bottom}" y2="{bottom}"/>')
        for t in xticks:
            out.append(f'<text class="tick" text-anchor="middle" x="{self.x(t):.1f}" y="{bottom + 20}">{xfmt(t)}</text>')
        if xlabel:
            out.append(f'<text class="axis-label" text-anchor="middle" x="{self.left + self.width / 2}" y="{bottom + 44}">{escape(xlabel)}</text>')
        return out

    def reference(self, value: float, text: str) -> list[str]:
        return [f'<line class="base" x1="{self.left}" x2="{self.left + self.width}" y1="{self.y(value):.1f}" y2="{self.y(value):.1f}"/>',
                f'<text class="label" text-anchor="end" x="{self.left + self.width}" y="{self.y(value) - 6:.1f}">{escape(text)}</text>']


def marker(reward: str, cx: float, cy: float, tip: str, hollow: bool = False, r: float = 5.0) -> str:
    shape = REWARDS[reward][1]
    cls = f'class="hollow h{reward}"' if hollow else f'class="ring f{reward}"'
    title = f"<title>{escape(tip)}</title>"
    if shape == "circle":
        return f'<circle {cls} cx="{cx:.1f}" cy="{cy:.1f}" r="{r}">{title}</circle>'
    if shape == "square":
        s = r * 0.9
        return f'<rect {cls} x="{cx - s:.1f}" y="{cy - s:.1f}" width="{2 * s:.1f}" height="{2 * s:.1f}">{title}</rect>'
    points = {"triangle": [(0, -1.2), (1.1, 0.8), (-1.1, 0.8)], "diamond": [(0, -1.3), (1.3, 0), (0, 1.3), (-1.3, 0)],
              "cross": [(-.4, -1.2), (.4, -1.2), (.4, -.4), (1.2, -.4), (1.2, .4), (.4, .4), (.4, 1.2), (-.4, 1.2), (-.4, .4), (-1.2, .4), (-1.2, -.4), (-.4, -.4)]}[shape]
    text = " ".join(f"{cx + a * r:.1f},{cy + b * r:.1f}" for a, b in points)
    return f'<polygon {cls} points="{text}">{title}</polygon>'


def legend(rewards: tuple, left: float, y: float, extra: str = "") -> list[str]:
    out, x = [], left
    for r in rewards:
        label = REWARDS[r][0]
        out.append(f'<line class="line l{r}" x1="{x}" x2="{x + 26}" y1="{y - 5}" y2="{y - 5}"/>')
        out.append(marker(r, x + 13, y - 5, label))
        out.append(f'<text class="legend" x="{x + 33}" y="{y}">{escape(label)}</text>')
        x += 33 + 14.5 * len(label) * 0.82 + 22
    if extra:
        out.append(f'<text class="legend" x="{x}" y="{y}">{escape(extra)}</text>')
    return out


def end_labels(panel: Panel, ends: dict) -> list[str]:
    """Reward codes at the line ends; ends too close to label apart share one label instead of being nudged off their lines."""
    groups: list[list] = []
    for reward, (px, py) in sorted(ends.items(), key=lambda e: e[1][1]):
        if groups and py - groups[-1][-1][2] < 14:
            groups[-1].append((reward, px, py))
        else:
            groups.append([(reward, px, py)])
    out = []
    for g in groups:
        y = sum(p[2] for p in g) / len(g)
        x = max(p[1] for p in g) + 11
        out.append(f'<text class="label" x="{x:.1f}" y="{y + 4:.1f}">{"、".join(sorted(p[0] for p in g))}</text>')
    return out


def draw(panel: Panel, series: dict, cis: dict | None = None) -> list[str]:
    """series: reward -> list of (x, y, hollow, tip); cis: reward -> list of (x, low, high), drawn under the lines."""
    out = []
    rewards = list(series)
    for i, r in enumerate(rewards):
        for x, low, high in (cis or {}).get(r, []):
            dodge = (i - (len(rewards) - 1) / 2) * 4
            out.append(f'<line class="ci l{r}" x1="{panel.x(x) + dodge:.1f}" x2="{panel.x(x) + dodge:.1f}" y1="{panel.y(low):.1f}" y2="{panel.y(high):.1f}"/>')
    for r, pts in series.items():
        path = " ".join(f"{panel.x(x):.1f},{panel.y(y):.1f}" for x, y, _, _ in pts)
        out.append(f'<polyline class="line l{r}" points="{path}"/>')
    for r, pts in series.items():
        out += [marker(r, panel.x(x), panel.y(y), tip, hollow) for x, y, hollow, tip in pts[1:]]
    out += end_labels(panel, {r: (panel.x(p[-1][0]), panel.y(p[-1][1])) for r, p in series.items()})
    return out


def signed(v: float) -> str:
    return "0" if abs(v) < 1e-12 else f"{v:+.2f}"


def document(width: int, height: int, title: str, subtitle: str, body: list[str]) -> str:
    return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
                      style(), f'<rect class="bg" width="{width}" height="{height}"/>', f'<text class="title" x="24" y="34">{escape(title)}</text>',
                      f'<text class="sub" x="24" y="60">{escape(subtitle)}</text>', *body, "</svg>", ""])


class Report:
    def __init__(self, path: str):
        rep = json.loads(Path(path).read_text(encoding="utf-8"))
        self.st, self.ev = rep["statistics"], rep["evaluation"]
        self.base = self.value("pi0")

    def value(self, name: str) -> float:
        return self.st["values"][name][K]["point"]

    def policy(self, method: str, reward: str, level: str) -> str:
        return f"O3/{reward}/{level}" if method == "O3" else f"O4/{reward}/{level}/main"

    def kl(self, method: str, reward: str, level: str) -> float:
        if method == "O3":
            return float(level)  # O3 lands on the tier within 0.1% on the KL subset
        cps = [m["checkpoint"]["kl"] for m in self.ev["policies"][self.policy(method, reward, level)]["per_model"]]
        return sum(cps) / len(cps)

    def hollow(self, method: str, reward: str, level: str) -> bool:
        cp = self.ev["policies"][self.policy(method, reward, level)].get("checkpoints")
        return bool(cp) and cp["aligned"] < cp["models"]

    def delta(self, method: str, reward: str, level: str) -> float:
        return self.value(self.policy(method, reward, level)) - self.base

    def ci(self, method: str, reward: str, level: str) -> list | None:
        e = self.st["o4_vs_o3"].get(f"{reward}/{level}")
        return e["o3_minus_pi0" if method == "O3" else "o4_minus_pi0"]["ci95"] if e else None

    def proxy(self, method: str, reward: str, level: str) -> float:
        return self.ev["policies"][self.policy(method, reward, level)]["proxy"]["R1"]

    def hot(self, method: str, reward: str, level: str) -> float:
        return self.ev["policies"][self.policy(method, reward, level)]["hot_share"]


METHODS = {"O3": ("解析最优 O3", ("R1", "R3", "R4", "R5", "R6")), "O4": ("RLOO O4（9 个模型的混合）", ("R1", "R3", "R4", "R5"))}


def value_kl(rep: Report) -> str:
    body = legend(tuple(REWARDS), 24, 92, "空心：这一档有模型没对齐")
    yticks = [-0.01, 0, 0.01, 0.02, 0.03]
    ylim = (-0.0135, 0.0315)
    top, height, width = 142, 330, 390
    for i, (method, (title, rewards)) in enumerate(METHODS.items()):
        p = Panel(74 + i * (width + 72), top, width, height, (0, 2.3), ylim)
        body += p.frame(title, "平均 KL（nat）", [0, 0.5, 1, 1.5, 2], yticks, lambda t: f"{t:g}", lambda t: "π0" if t == 0 else f"{t:+.2f}")
        series = {r: [(0, 0, False, "π0")] + [(rep.kl(method, r, lv), rep.delta(method, r, lv), rep.hollow(method, r, lv),
                                               f"{method} {r} {lv} nat：{rep.delta(method, r, lv):+.4f}") for lv in LEVELS] for r in rewards}
        cis = {r: [(rep.kl(method, r, lv), *rep.ci(method, r, lv)) for lv in LEVELS if rep.ci(method, r, lv)] for r in rewards}
        body += draw(p, series, cis)
    rewards = ("R1", "R3", "R4", "R5", "R6")
    p = Panel(74 + 2 * (width + 72) - 0, top, 190, height, (-0.5, len(rewards) - 0.5), (-0.1, 0.02))
    body += p.frame("纯奖励排序 O2′（没有 KL）", "", [], [-0.08, -0.04, 0], lambda t: "", lambda t: "π0" if t == 0 else f"{t:+.2f}")
    for i, r in enumerate(rewards):
        d = rep.value(f"O2p/{r}") - rep.base
        body.append(f'<line class="stem l{r}" x1="{p.x(i):.1f}" x2="{p.x(i):.1f}" y1="{p.y(0):.1f}" y2="{p.y(d):.1f}"/>')
        body.append(marker(r, p.x(i), p.y(d), f"O2′ {r}：{d:+.4f}"))
        anchor_y = p.y(d) + 20 if d < 0 else p.y(d) - 11
        body.append(f'<text class="label" text-anchor="middle" x="{p.x(i):.1f}" y="{anchor_y:.1f}">{r}</text>')
    body.append(f'<text class="tick" text-anchor="middle" x="{p.left + p.width / 2}" y="{top + height + 44}">这一格的纵轴和左边不同</text>')
    return document(1240, 530, "解析最优下 R1、R3、R4、R5 先涨后跌，R6 一路上涨；RLOO 各档都没有检出高于 π0", "纵轴：V₅₀₀ 减去 π0 的 V₅₀₀（随机测试）。竖线是相对 π0 的配对 95% 区间；R6 的这个区间没有算", body)


def proxy_value(rep: Report) -> str:
    body = legend(tuple(REWARDS), 24, 92, "空心：这一档有模型没对齐")
    yticks = [-0.01, 0, 0.01, 0.02, 0.03]
    ylim = (-0.0135, 0.0315)
    top, height, width = 142, 330, 480
    start = rep.ev["policies"]["pi0"]["proxy"]["R1"]
    for i, (method, (title, rewards)) in enumerate(METHODS.items()):
        p = Panel(74 + i * (width + 90), top, width, height, (0.555, 0.725), ylim)
        body += p.frame(title, "策略下 R1 的期望奖励 E[R1]（有偏模型自己的看法）", [0.56, 0.60, 0.64, 0.68, 0.72], yticks, lambda t: f"{t:.2f}", lambda t: "π0" if t == 0 else f"{t:+.2f}")
        series = {r: [(start, 0, False, "π0")] + [(rep.proxy(method, r, lv), rep.delta(method, r, lv), rep.hollow(method, r, lv),
                                                   f"{method} {r} {lv} nat：E[R1] {rep.proxy(method, r, lv):.4f}，差 {rep.delta(method, r, lv):+.4f}") for lv in LEVELS] for r in rewards}
        body += draw(p, series)
    return document(1160, 530, "R1、R4 的策略：代理奖励一路上涨，列表价值回落；R6 的策略：两个都涨", "每条线从 π0 出发，依次经过 0.5、1、2 nat 三档。纵轴：V₅₀₀ 减去 π0 的 V₅₀₀（随机测试）", body)


def hot_share(rep: Report) -> str:
    body = legend(tuple(REWARDS), 24, 92, "空心：这一档有模型没对齐")
    top, height, width = 142, 300, 480
    start = rep.ev["policies"]["pi0"]["hot_share"]
    for i, (method, (title, rewards)) in enumerate(METHODS.items()):
        p = Panel(74 + i * (width + 90), top, width, height, (0, 2.25), (0.68, 0.90))
        body += p.frame(title, "平均 KL（nat）", [0, 0.5, 1, 1.5, 2], [0.70, 0.75, 0.80, 0.85, 0.90], lambda t: f"{t:g}", lambda t: f"{t:.2f}")
        body += p.reference(start, f"π0 {start:.3f}")
        series = {r: [(0, start, False, "π0")] + [(rep.kl(method, r, lv), rep.hot(method, r, lv), rep.hollow(method, r, lv),
                                                   f"{method} {r} {lv} nat：{rep.hot(method, r, lv):.3f}") for lv in LEVELS] for r in rewards}
        body += draw(p, series)
    return document(1160, 500, "分组校准把列表推向热门，常规日志奖励把列表推离热门，混入随机数据的 R5 基本不动", "纵轴：前 500 里属于训练期曝光最热一组（5 等分）的比例", body)


def main(test_path: str, out_dir: str) -> None:
    rep = Report(test_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, make in (("value_kl", value_kl), ("proxy_value", proxy_value), ("hot_share", hot_share)):
        (out / f"{name}.svg").write_text(make(rep), encoding="utf-8", newline="\n")
        print(out / f"{name}.svg")


if __name__ == "__main__":
    main(*sys.argv[1:])
