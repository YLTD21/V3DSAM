#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将两份总结报告 md -> Word (.docx)，遵循国家级基金结题报告版式。

- 标题 #/##/###/#### -> Word 标题样式
- markdown 表格 -> Word 表格（表头加粗，Table Grid）
- ```代码块``` -> 等宽字体段落
- 行内 **加粗** *斜体* `代码`
- 列表（- / 数字.）-> 项目符号/编号
- 中文字体：宋体（eastAsia），西文 Times New Roman

Run: python md2docx_report.py
"""
import re
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

JOBS = [
    ("E:/0722v3dsam/829v3dsam/docs/工作总结报告.md",
     "E:/0722v3dsam/829v3dsam/docs/工作总结报告.docx"),
    ("E:/0722v3dsam/829v3dsam/docs/技术总结报告.md",
     "E:/0722v3dsam/829v3dsam/docs/技术总结报告.docx"),
]

INLINE = re.compile(r'(\*\*.+?\*\*|\*[^*\n][^*]*?\*|`[^`]+?`)', re.S)


def set_ea(run, ea="宋体"):
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn('w:eastAsia'), ea)


def add_inline(p, text, size=None, color=None):
    pos = 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            r = p.add_run(text[pos:m.start()]); _style(r, size, color)
        tok = m.group(1)
        if tok.startswith('**') and tok.endswith('**') and len(tok) > 4:
            r = p.add_run(tok[2:-2]); r.bold = True; _style(r, size, color)
        elif tok.startswith('`') and tok.endswith('`'):
            r = p.add_run(tok[1:-1]); r.font.name = 'Consolas'
            r.font.color.rgb = RGBColor(0x8a, 0x2b, 0x2b); _style(r, size, color)
        elif tok.startswith('*') and tok.endswith('*') and len(tok) > 2:
            r = p.add_run(tok[1:-1]); r.italic = True; _style(r, size, color)
        else:
            r = p.add_run(tok); _style(r, size, color)
        pos = m.end()
    if pos < len(text):
        r = p.add_run(text[pos:]); _style(r, size, color)
    return p


def _style(r, size=None, color=None):
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    set_ea(r)


def add_code_block(doc, lines):
    for ln in lines:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Pt(18)
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(ln if ln else ' ')
        r.font.name = 'Consolas'
        r.font.size = Pt(8.5)
        set_ea(r, '宋体')


def main():
    for src, out in JOBS:
        doc = Document()
        style = doc.styles['Normal']
        style.font.name = 'Times New Roman'
        style.font.size = Pt(10.5)
        style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

        with open(src, encoding='utf-8') as f:
            lines = f.read().split('\n')

        i, n = 0, len(lines)
        while i < n:
            line = lines[i].rstrip()
            stripped = line.strip()

            if not stripped:
                i += 1
                continue
            if stripped in ('---', '***', '___'):
                i += 1
                continue

            # 代码块
            if stripped.startswith('```'):
                i += 1
                buf = []
                while i < n and not lines[i].strip().startswith('```'):
                    buf.append(lines[i].rstrip())
                    i += 1
                i += 1  # 跳过结束 ```
                add_code_block(doc, buf)
                continue

            # 标题
            mh = re.match(r'^(#{1,4})\s+(.*)$', stripped)
            if mh:
                level = len(mh.group(1))
                h = doc.add_heading(level=min(level, 4))
                add_inline(h, mh.group(2))
                for r in h.runs:
                    r.font.color.rgb = RGBColor(0, 0, 0)
                    set_ea(r)
                i += 1
                continue

            # 表格
            if stripped.startswith('|'):
                rows = []
                while i < n and lines[i].strip().startswith('|'):
                    rows.append(lines[i].strip())
                    i += 1
                data = [r for r in rows if not re.match(r'^\|[\s:|-]+\|$', r)]
                if data:
                    ncols = len(data[0].strip('|').split('|'))
                    table = doc.add_table(rows=len(data), cols=ncols)
                    table.style = 'Table Grid'
                    for ri, r in enumerate(data):
                        cells = [c.strip() for c in r.strip('|').split('|')]
                        for ci in range(ncols):
                            cell = table.rows[ri].cells[ci]
                            cell.text = ''
                            txt = cells[ci] if ci < len(cells) else ''
                            add_inline(cell.paragraphs[0], txt, size=9)
                            for para in cell.paragraphs:
                                for run in para.runs:
                                    run.font.size = Pt(9)
                                    if ri == 0:
                                        run.bold = True
                doc.add_paragraph()
                continue

            # 列表
            if re.match(r'^[-*]\s+', stripped):
                p = doc.add_paragraph(style='List Bullet')
                add_inline(p, stripped[2:].strip())
                i += 1
                continue
            if re.match(r'^\d+\.\s+', stripped):
                p = doc.add_paragraph(style='List Number')
                add_inline(p, re.sub(r'^\d+\.\s+', '', stripped))
                i += 1
                continue

            # 普通段落
            p = doc.add_paragraph()
            p.paragraph_format.first_line_indent = Pt(21)
            add_inline(p, stripped)
            i += 1

        doc.save(out)
        print('saved', out)


if __name__ == '__main__':
    main()
