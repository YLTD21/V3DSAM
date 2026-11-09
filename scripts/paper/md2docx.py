#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert 824v3dsam.md -> 824v3dsam.docx (subset-aware, no pandoc needed)."""
import re
import os
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn


INLINE = re.compile(r'(\*\*.+?\*\*|\*.+?\*|`[^`]+`)')
IMG = re.compile(r'^!\[([^\]]*)\]\(([^)]+)\)$')
BASE = r'E:\0722v3dsam\paper'


def add_runs(p, text):
    """Parse **bold**, *italic*, `code` inline and add as runs."""
    for tok in INLINE.split(text):
        if not tok:
            continue
        if tok.startswith('**') and tok.endswith('**'):
            r = p.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith('*') and tok.endswith('*') and len(tok) > 1:
            r = p.add_run(tok[1:-1]); r.italic = True
        elif tok.startswith('`') and tok.endswith('`'):
            r = p.add_run(tok[1:-1]); r.font.name = 'Consolas'
            r.font.color.rgb = RGBColor(0x8a, 0x2b, 0x2b)
        else:
            p.add_run(tok)
    return p


def set_cn_font(doc):
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')


def main():
    doc = Document()
    set_cn_font(doc)

    with open(r'E:\0722v3dsam\paper\824v3dsam.md', encoding='utf-8') as f:
        lines = f.read().split('\n')

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()
        stripped = line.strip()

        # horizontal rule / blank
        if not stripped:
            i += 1
            continue
        if stripped in ('---', '***', '___'):
            i += 1
            continue

        # image embed: ![alt](path)
        m_img = IMG.match(stripped)
        if m_img:
            path = m_img.group(2)
            if not os.path.isabs(path):
                path = os.path.join(BASE, path)
            if os.path.exists(path):
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run()
                run.add_picture(path, width=Inches(6.0))
            i += 1
            continue

        # table block
        if stripped.startswith('|'):
            rows = []
            while i < n and lines[i].strip().startswith('|'):
                rows.append(lines[i].strip())
                i += 1
            # drop separator rows like |---|:---|
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
                        add_runs(cell.paragraphs[0], txt)
                        for para in cell.paragraphs:
                            for run in para.runs:
                                run.font.size = Pt(9)
            doc.add_paragraph()
            continue

        # headings
        m = re.match(r'^(#{1,4})\s+(.*)$', stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            h = doc.add_heading(level=min(level, 4))
            add_runs(h, text)
            i += 1
            continue

        # blockquote
        if stripped.startswith('>'):
            lines_buf = []
            while i < n and lines[i].strip().startswith('>'):
                lines_buf.append(lines[i].strip()[1:].strip())
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Pt(18)
            add_runs(p, ' '.join(lines_buf))
            for run in p.runs:
                run.italic = True
                run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
            continue

        # bullet list
        if re.match(r'^[-*]\s+', stripped):
            p = doc.add_paragraph(style='List Bullet')
            add_runs(p, stripped[2:].strip())
            i += 1
            continue

        # ordered list
        if re.match(r'^\d+\.\s+', stripped):
            p = doc.add_paragraph(style='List Number')
            add_runs(p, re.sub(r'^\d+\.\s+', '', stripped))
            i += 1
            continue

        # normal paragraph
        p = doc.add_paragraph()
        add_runs(p, stripped)
        i += 1

    out = r'E:\0722v3dsam\paper\824v3dsam.docx'
    doc.save(out)
    print('saved', out)


if __name__ == '__main__':
    main()
