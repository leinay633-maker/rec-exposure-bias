"""Fill docs/rlrec/RESULTS.md: the body keeps <!-- ... --> slots for tables, the appendix is the full generated output.

Usage (from the repo root): python scripts/assemble_rlrec_results.py <tables.md>
A second run keeps working: each slot sits between its marker and the next blank line, and the appendix is rebuilt.
"""
import re
import sys
from pathlib import Path

SLOTS = {"主假设表": "#### 主假设", "相对 π0 的差": "#### 各档相对 π0 的差", "O4 对 O3": "#### RLOO 对解析最优", "机制拆分": "#### 机制拆分",
         "成本表": "#### 效果、覆盖和成本", "HOLD 表": "#### HOLD 上的主假设差", "抽样价值表": "#### 1 nat 的抽样价值",
         "调尖表": "#### 开封后的探索：π0 调尖到相同 KL，随机测试", "调尖 1 nat 表": "#### 开封后的探索：1 nat 下各策略减调尖的 π0"}
APPENDIX = "## 附录：全部表格（`scripts/make_rlrec_tables.py` 从 `evidence/rlrec/` 生成）"


def table_after(text: str, heading: str) -> str:
    start = text.index(heading)
    lines = text[start:].split("\n")
    rows = []
    for line in lines[1:]:
        if line.startswith("|"):
            rows.append(line)
        elif rows:
            break
    return "\n".join(rows)


def main(tables_path: str) -> None:
    tables = Path(tables_path).read_text(encoding="utf-8").strip()
    doc = Path("docs/rlrec/RESULTS.md")
    text = doc.read_text(encoding="utf-8")
    text = text[: text.index(APPENDIX)]
    for slot, heading in SLOTS.items():
        marker = f"<!-- {slot} -->"
        assert text.count(marker) == 1, slot
        text = re.sub(re.escape(marker) + r"\n(\|[^\n]*\n)*", lambda m: marker + "\n" + table_after(tables, heading) + "\n", text)
    doc.write_text(text + APPENDIX + "\n\n" + tables + "\n", encoding="utf-8", newline="\n")
    print("rows", text.count("\n"), "appendix lines", tables.count("\n"))


if __name__ == "__main__":
    main(*sys.argv[1:])
