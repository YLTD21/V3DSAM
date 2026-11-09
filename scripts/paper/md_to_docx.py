#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Markdown → .docx converter (SCI 论文用)。处理标题/段落/加粗/斜体/表格/列表/图片,中文宋体。"""
import os
import re
from PIL import Image
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

BODY_EA = "宋体"
HEAD_EA = "黑体"
LATIN = "Times New Roman"

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
FIG_FILES = {
    1: "fig_venn.png",
    2: "fig_framework_v2.png",
    3: "fig2_motivation.png",
    4: "fig_falseloop_method.png",
    5: "fig_semconsistency_method.png",
    6: "fig3_semantic_cloud.png",
    7: "fig4_discriminator.png",
    8: "fig5_trajectory.png",
    9: "fig6_false_loop.png",
    10: "fig7_tradeoff.png",
    11: "fig_scannet_traj.png",
    12: "fig_scannet_pcd.png",
    13: "fig_scannet_sem_traj.png",
    14: "fig8_geom_anchor.png",
    15: "fig9_reid_recall.png",
    16: "fig_instance_grid_scannet.png",
}

# 个别图需要更大的显示尺寸（默认 max 6.0 × 4.5 英寸）
FIG_SIZE_OVERRIDE = {
    9: (6.2, 6.2),    # 2×2 假回环网格，放大以读清面板与标签
    11: (6.2, 6.0),   # ScanNet 轨迹 2×2，放大 + 含放大细节小图
    16: (6.2, 5.8),   # 持久实例身份 3×5 网格，放大面板与标签
}


def set_run_font(run, ea=BODY_EA, size=None, bold=None, italic=None, color=None):
    run.font.name = LATIN
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:eastAsia"), ea)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


_LATEX_MAP = [
    (r"\tau", "τ"), (r"\to", "→"), (r"\in", "∈"), (r"\ge", "≥"), (r"\le", "≤"),
    (r"\times", "×"), (r"\rightarrow", "→"), (r"\leftarrow", "←"),
    (r"\sim", "~"), (r"\approx", "≈"), (r"\neq", "≠"), (r"\sum", "Σ"),
]


def latex2text(s):
    s = re.sub(r"\\[a-zA-Z]+\b", "", s)
    for a, b in _LATEX_MAP:
        s = s.replace(a, b)
    s = s.replace(r"\_", "_").replace(r"\{", "{").replace(r"\}", "}")
    s = re.sub(r"_{([^}]*)}", r"_\1", s)
    s = re.sub(r"\^{([^}]*)}", r"^\1", s)
    s = s.replace("{", "").replace("}", "")
    return s


TOKEN_RE = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|\$[^$]+?\$)")


def add_inline(par, text, size=None):
    for tok in TOKEN_RE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            r = par.add_run(tok[2:-2]); set_run_font(r, size=size, bold=True)
        elif tok.startswith("$") and tok.endswith("$"):
            r = par.add_run(latex2text(tok[1:-1])); set_run_font(r, size=size, italic=True)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = par.add_run(tok[1:-1]); set_run_font(r, size=size, italic=True)
        else:
            r = par.add_run(tok); set_run_font(r, size=size)


def add_figure(doc, fig_num, caption):
    fname = FIG_FILES.get(fig_num)
    path = os.path.join(FIG_DIR, fname) if fname else None
    if not path or not os.path.exists(path):
        p = doc.add_paragraph()
        add_inline(p, f"[图 {fig_num} 缺失] {caption}")
        return
    w, h = Image.open(path).size
    max_w, max_h = FIG_SIZE_OVERRIDE.get(fig_num, (6.0, 4.5))  # 英寸
    disp_w, disp_h = max_w, max_w * h / w
    if disp_h > max_h:
        disp_h = max_h
        disp_w = max_h * w / h
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(disp_w), height=Inches(disp_h))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    set_run_font(r, ea=BODY_EA, size=9, bold=True)


def add_table(doc, rows):
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = t.cell(i, j)
            cell.text = ""
            par = cell.paragraphs[0]
            txt = row[j] if j < len(row) else ""
            add_inline(par, txt, size=9)
            for run in par.runs:
                if i == 0:
                    run.font.bold = True
                run.font.size = Pt(9)
    return t


def split_table_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def is_separator_row(line):
    return bool(re.fullmatch(r"\|?[\s:|-]+\|?", line.strip())) and "-" in line


def main(src, dst):
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = LATIN
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_EA)

    with open(src, encoding="utf-8") as f:
        lines = f.read().split("\n")

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()
        stripped = line.strip()
        if stripped.startswith("## 图表占位索引"):
            break
        if not stripped or stripped == "---":
            i += 1
            continue
        # 图片占位 [图 N：...]（可含 ** 包裹）
        fig_match = re.match(r"^\*\*\[图\s*(\d+)[：:](.+)\]\*\*$", stripped)
        if fig_match:
            add_figure(doc, int(fig_match.group(1)),
                       f"图 {fig_match.group(1)}：{fig_match.group(2)}")
            i += 1
            continue
        # 英文图片占位 [Figure N: ...]
        fig_match = re.match(r"^\[Figure\s*(\d+)[：:](.+)\]$", stripped)
        if fig_match:
            add_figure(doc, int(fig_match.group(1)),
                       f"Figure {fig_match.group(1)}: {fig_match.group(2)}")
            i += 1
            continue
        if stripped.startswith("### "):
            p = doc.add_heading(level=3)
            add_inline(p, stripped[4:], size=12)
            for r in p.runs:
                set_run_font(r, ea=HEAD_EA, size=12, bold=True, color=(0, 0, 0))
            i += 1; continue
        if stripped.startswith("## "):
            p = doc.add_heading(level=2)
            add_inline(p, stripped[3:], size=14)
            for r in p.runs:
                set_run_font(r, ea=HEAD_EA, size=14, bold=True, color=(0, 0, 0))
            i += 1; continue
        if stripped.startswith("# "):
            p = doc.add_heading(level=1)
            add_inline(p, stripped[2:], size=16)
            for r in p.runs:
                set_run_font(r, ea=HEAD_EA, size=16, bold=True, color=(0, 0, 0))
            i += 1; continue
        if stripped.startswith("|"):
            block = []
            j = i
            while j < n and lines[j].strip().startswith("|"):
                block.append(lines[j].strip())
                j += 1
            rows = []
            for bl in block:
                if is_separator_row(bl):
                    continue
                rows.append(split_table_row(bl))
            if rows:
                add_table(doc, rows)
                doc.add_paragraph()
            i = j
            continue
        m = re.match(r"^[-•]\s+(.*)", stripped)
        if m:
            p = doc.add_paragraph(style="List Bullet")
            add_inline(p, m.group(1))
            i += 1; continue
        m = re.match(r"^\d+\.\s+(.*)", stripped)
        if m:
            p = doc.add_paragraph(style="List Number")
            add_inline(p, m.group(1))
            i += 1; continue
        p = doc.add_paragraph()
        add_inline(p, stripped)
        p.paragraph_format.first_line_indent = Pt(21)
        i += 1

    doc.save(dst)
    print(f"generated {dst}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "817_论文.md",
         sys.argv[2] if len(sys.argv) > 2 else "817.docx")
