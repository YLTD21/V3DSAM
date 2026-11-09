#!/usr/bin/env python3
"""Convert paper/818_论文.md -> Word (.docx), embedding figures + rendering tables.

- headings (#/##/###/####) -> Word heading styles
- markdown tables -> Word tables (bold header)
- [图 N: caption] -> embedded PNG (paper/figures/figN_*.png) + caption
- [表 N: caption] -> centered caption paragraph
- inline: **bold**, *italic*, `code`, $math$ (LaTeX -> unicode)
- lists / blockquotes -> indented paragraphs
"""
import re, glob, os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

MD = "E:/0722v3dsam/paper/818_论文.md"
FIGDIR = "E:/0722v3dsam/paper/figures"
OUT = "E:/0722v3dsam/V3DSAM_论文_822.docx"

LATEX = [
    (r'\tau', 'τ'), (r'\sigma', 'σ'), (r'\alpha', 'α'), (r'\beta', 'β'),
    (r'\to', '→'), (r'\rightarrow', '→'), (r'\leftarrow', '←'),
    (r'\times', '×'), (r'\approx', '≈'), (r'\leq', '≤'), (r'\geq', '≥'),
    (r'\neq', '≠'), (r'\in', '∈'), (r'\pm', '±'), (r'\cdot', '·'),
    (r'\{', '{'), (r'\}', '}'),
]


def latex2uni(s):
    for a, b in LATEX:
        s = s.replace(a, b)
    s = s.replace('_{', '_').replace('^{', '^')
    s = s.replace('{', '').replace('}', '')
    return s


INLINE = re.compile(r'(\*\*.+?\*\*|\*[^*\n][^*]*?\*|`[^`]+?`|\$[^$]+?\$)', re.S)


def set_ea(run, ea='宋体'):
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn('w:eastAsia'), ea)


def add_inline(p, text):
    """Add runs to paragraph p, honouring **bold** *italic* `code` $math$."""
    pos = 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            r = p.add_run(text[pos:m.start()]); set_ea(r)
        tok = m.group(1)
        if tok.startswith('**') and tok.endswith('**') and len(tok) > 4:
            r = p.add_run(tok[2:-2]); r.bold = True; set_ea(r)
        elif tok.startswith('`') and tok.endswith('`'):
            r = p.add_run(tok[1:-1]); r.font.name = 'Consolas'; set_ea(r)
        elif tok.startswith('$') and tok.endswith('$'):
            r = p.add_run(latex2uni(tok[1:-1])); r.italic = True; set_ea(r)
        elif tok.startswith('*') and tok.endswith('*') and len(tok) > 2:
            r = p.add_run(tok[1:-1]); r.italic = True; set_ea(r)
        else:
            r = p.add_run(tok); set_ea(r)
        pos = m.end()
    if pos < len(text):
        r = p.add_run(text[pos:]); set_ea(r)


def strip_inline(text):
    text = text.replace('**', '')
    text = re.sub(r'\$([^$]+?)\$', lambda m: latex2uni(m.group(1)), text)
    text = text.replace('`', '')
    return text


def figure_path(n):
    hits = glob.glob(f"{FIGDIR}/fig{n}_*.png")
    return hits[0] if hits else None


def parse_row(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def is_sep(line):
    return re.match(r'^\s*\|[\s:\-|]+\|\s*$', line) is not None


doc = Document()

# default body font (Latin + East Asian)
normal = doc.styles['Normal']
normal.font.name = 'Times New Roman'
normal.font.size = Pt(11)
set_ea(normal.element.get_or_add_rPr().get_or_add_rFonts() and normal.element.rPr.rFonts,
      '宋体') if False else None
# set east-asian font on Normal style properly
_rpr = normal.element.get_or_add_rPr()
_rfonts = _rpr.get_or_add_rFonts()
_rfonts.set(qn('w:eastAsia'), '宋体')

for hname in ['Heading 1', 'Heading 2', 'Heading 3', 'Heading 4', 'Title']:
    try:
        st = doc.styles[hname]
        st.font.name = 'Times New Roman'
        st.font.color.rgb = RGBColor(0, 0, 0)
        _rpr = st.element.get_or_add_rPr()
        _rfonts = _rpr.get_or_add_rFonts()
        _rfonts.set(qn('w:eastAsia'), '黑体')
    except Exception:
        pass

lines = open(MD, encoding='utf-8').read().split('\n')
n = len(lines)
i = 0
while i < n:
    line = lines[i].rstrip()
    s = line.strip()
    i += 1

    if not s:
        continue
    if s == '---':
        continue

    # table block
    if s.startswith('|') and i < n and is_sep(lines[i]):
        header = parse_row(s)
        i += 1  # skip separator
        rows = []
        while i < n and lines[i].strip().startswith('|') and not is_sep(lines[i]):
            rows.append(parse_row(lines[i].strip()))
            i += 1
        ncols = len(header)
        tbl = doc.add_table(rows=len(rows) + 1, cols=ncols)
        tbl.style = 'Light Grid Accent 1'
        for j, htxt in enumerate(header):
            cell = tbl.rows[0].cells[j]
            cell.text = ''
            p = cell.paragraphs[0]
            r = p.add_run(strip_inline(htxt)); r.bold = True; set_ea(r)
        for ri, row in enumerate(rows):
            for j in range(ncols):
                cell = tbl.rows[ri + 1].cells[j]
                cell.text = ''
                p = cell.paragraphs[0]
                txt = row[j] if j < len(row) else ''
                add_inline(p, txt)
        doc.add_paragraph('')
        continue

    # figure placeholder (optionally wrapped in **...**)
    s2 = s[2:-2] if (s.startswith('**') and s.endswith('**')) else s
    m = re.match(r'^\[图 (\d+):(.*)\]$', s2)
    if m:
        num = int(m.group(1)); cap = m.group(2)
        fp = figure_path(num)
        if fp and os.path.exists(fp):
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(); run.add_picture(fp, width=Inches(5.8))
        pcap = doc.add_paragraph(); pcap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = pcap.add_run(f"图 {num}  {cap}")
        r.bold = True; r.font.size = Pt(10); set_ea(r)
        doc.add_paragraph('')
        continue

    # table placeholder (optionally wrapped in **...**)
    s3 = s[2:-2] if (s.startswith('**') and s.endswith('**')) else s
    m = re.match(r'^\[表 (\d+):(.*)\]$', s3)
    if m:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(f"表 {m.group(1)}  {m.group(2)}")
        r.bold = True; r.font.size = Pt(10); set_ea(r)
        doc.add_paragraph('')
        continue

    # headings
    if s.startswith('#### '):
        h = doc.add_heading(strip_inline(s[5:]), level=4)
    elif s.startswith('### '):
        h = doc.add_heading(strip_inline(s[4:]), level=3)
    elif s.startswith('## '):
        h = doc.add_heading(strip_inline(s[3:]), level=2)
    elif s.startswith('# '):
        h = doc.add_heading(strip_inline(s[2:]), level=0)
    # blockquote
    elif s.startswith('>'):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.4)
        add_inline(p, s.lstrip('>').strip())
        for r in p.runs:
            r.italic = True
    # bullet list
    elif re.match(r'^[-*]\s+', s):
        p = doc.add_paragraph(style='List Bullet')
        add_inline(p, re.sub(r'^[-*]\s+', '', s))
    # numbered list
    elif re.match(r'^\d+\.\s+', s):
        p = doc.add_paragraph(style='List Number')
        add_inline(p, re.sub(r'^\d+\.\s+', '', s))
    # regular paragraph
    else:
        p = doc.add_paragraph()
        # align title-like bold english line centered
        add_inline(p, s)

doc.save(OUT)
print(f"SAVED -> {OUT}")
print(f"size = {os.path.getsize(OUT):,} bytes")
