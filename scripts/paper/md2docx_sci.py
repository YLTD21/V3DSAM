#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SCI_稿件_中文.md -> SCI_稿件_中文.docx（内嵌图片 + 图注）。"""
import re
import os
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from PIL import Image as PILImage

SRC = r'E:\0722v3dsam\paper\SCI_稿件_中文.md'
OUT = r'E:\0722v3dsam\paper\SCI_稿件_中文.docx'
BASE = r'E:\0722v3dsam\paper'

INLINE = re.compile(r'(\*\*.+?\*\*|\*.+?\*|`[^`]+`)')

# (图号, 图注, [相对路径...], 定位marker——在包含该子串的段落后插入)
FIGURES = [
    ("图 1", "方法定位——三圈 Venn：VGGT 前馈几何、SAM3 开放词汇语义、3DGS 稠密重建三者的交汇即 V3DSAM。",
     ["paper_figs/image1.png"], "远超外观匹配的 2.6%/1.5%。"),
    ("图 2", "方法框架——三阶段流水线：① VGGT 前馈几何前端 ② SAM3 开放词汇语义 ③ 3DGS 稠密重建。",
     ["paper_figs/image2.png"], "以验证后的位姿与深度初始化高斯地图并渲染。"),
    ("图 3", "动机示意——15 个回环候选（4 假 + 11 真）的类别级重叠（Jaccard）散点图：假回环（红×）与真回环（蓝○）类别重叠交叠，说明类别集合无法区分真假回环。",
     ["figs/fig2_motivation_honest.png"], "假回环则因朝向翻转而镜像错位。"),
    ("图 4", "假回环检测方法示意——帧对 (Ii, Ij) 外观相似度高于阈值时产生候选相对位姿 Tij。",
     ["paper_figs/image4.png"], "图 5 给出该度量的核心四步。"),
    ("图 5", "语义-几何三维一致性度量——核心四步：① 反投影 ② 位姿变换 ③ 重投影比对标签 ④ 双向对称平均。",
     ["paper_figs/image5.png"], "图 5 给出该度量的核心四步。"),
    ("图 6", "语义三维地图——V3DSAM 把 SAM3 类别掩码反投影到世界坐标系的 3DGS 初始化点云。",
     ["paper_figs/image6.png"], "（图 6 展示聚合到世界坐标系的语义三维地图）。"),
    ("图 7", "判别器对比——两行散点图（上 TUM-RGBD fr1 15 候选、下 ScanNet 31 候选），五列判别器：(a) αmatch (b) 外观 (c) 类别重叠 (d) 几何 (e) 语义-几何三维一致性（本文）。αmatch 与外观取值虽在更紧阈值下可分，但其原生阈值（−0.95/0.95）全数接受假回环，且外观即回环检索分数本身（循环论证）；类别重叠与几何在 TUM 上各产生 2 个假正例。仅本文语义-几何三维一致性以大幅余量干净分离（TUM 假≤0.078/真≥0.152；ScanNet 假=0.0/真≥0.392）。",
     ["figs/fig3_discriminator_honest.png"], "仅语义-几何三维一致性实现零误判。"),
    ("图 8", "阈值扫描与 PR 曲线——τ∈(0.078,0.152) 平台内 P=R=1.0。",
     ["figs/fig5_pr_curve.png", "figs/fig5_threshold_sweep.png"], "说明 τ=0.10 非刀锋取值。"),
    ("图 9", "噪声鲁棒性——精确率恒定 1.0，召回率随噪声缓降。",
     ["figs/figS2_noise_robustness.png"], "召回率在 0.2 噪声下仍为 0.909"),
    ("图 10", "前后轨迹对比——fr1/room 与 fr1/360 的对齐轨迹（真值 vs 处理前 vs 处理后）。",
     ["../figures/fig2_trajectory.png"], "fr1/360 由 0.188 m 降至 0.0488 m（图 10）。"),
    ("图 11", "几何优先 vs 类别门控重识别召回对照——room（89.5% vs 59.2%）与 360（86.6% vs 52.2%）。",
     ["paper_figs/fig7_recall_honest.png"], "这直接证明：实例身份在纯前馈 3DGS 中必须靠几何锚定。"),
    ("图 12", "几何锚定对照——本文 SL(4) 全局一致几何（自一致约 7 cm）。",
     ["figs/fig6_centroid_honest.png"], "证明 SL(4) 全局一致几何提供了紧致且稳定的身份锚。"),
    ("图 13", "持久实例身份 4×5 多帧可视化（ScanNet）——4 场景 × 5 帧（3 近连续局部帧 + 2 回环帧），同物体同色同编号跨回环一致。",
     ["../oneshot_preview/fig_instance_grid_3d2d_4scene.png"], "scene0207（浴室+卧室）3 个（椅子/窗户/瓶子）。"),
    ("图 14", "ScanNet scene0207 假回环剔除与轨迹/语义对照。",
     ["figs/scannet_loop/fig_scene0207_pcd_traj.png", "figs/scannet_loop/fig_scene0207_sem.png"], "稀有细粒度类召回低"),
]


def add_runs(p, text):
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


def resolve(path):
    if not os.path.isabs(path):
        path = os.path.join(BASE, path)
    return os.path.normpath(path)


def add_figure(doc, num, caption, paths):
    for path in paths:
        full = resolve(path)
        if not os.path.exists(full):
            continue
        try:
            w, h = PILImage.open(full).size
        except Exception:
            w, h = 6, 4
        target_w = 6.0
        max_h = 4.6
        if target_w * (h / w) > max_h:
            target_w = max_h * (w / h)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(full, width=Inches(target_w))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(2)
    r = cap.add_run(num + '　')
    r.bold = True
    r.font.size = Pt(9)
    r2 = cap.add_run(caption)
    r2.font.size = Pt(9)


def maybe_insert_figures(doc, text):
    plain = text.replace('**', '').replace('*', '')
    for num, caption, paths, marker in FIGURES:
        if marker in plain:
            add_figure(doc, num, caption, paths)


def main():
    doc = Document()
    set_cn_font(doc)

    with open(SRC, encoding='utf-8') as f:
        lines = f.read().split('\n')

    skip = False  # 跳过"图清单"节
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue
        if stripped in ('---', '***', '___'):
            i += 1
            continue

        # 图清单节整体跳过（图已内嵌）
        mh = re.match(r'^#{1,4}\s+(.*)$', stripped)
        if mh:
            if '图清单' in mh.group(1):
                skip = True
                i += 1
                continue
            if skip and '参考文献' in mh.group(1):
                skip = False
                # 落到普通标题处理
            if skip:
                i += 1
                continue
            level = len(mh.group(1))
            h = doc.add_heading(level=min(level, 4))
            add_runs(h, mh.group(1))
            i += 1
            continue

        if skip:
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
                        add_runs(cell.paragraphs[0], txt)
                        for para in cell.paragraphs:
                            for run in para.runs:
                                run.font.size = Pt(9)
            doc.add_paragraph()
            continue

        # 引用块
        if stripped.startswith('>'):
            buf = []
            while i < n and lines[i].strip().startswith('>'):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Pt(18)
            add_runs(p, ' '.join(buf))
            for run in p.runs:
                run.italic = True
                run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
            continue

        # 列表
        if re.match(r'^[-*]\s+', stripped):
            p = doc.add_paragraph(style='List Bullet')
            text = stripped[2:].strip()
            add_runs(p, text)
            maybe_insert_figures(doc, text)
            i += 1
            continue
        if re.match(r'^\d+\.\s+', stripped):
            p = doc.add_paragraph(style='List Number')
            text = re.sub(r'^\d+\.\s+', '', stripped)
            add_runs(p, text)
            maybe_insert_figures(doc, text)
            i += 1
            continue

        # 普通段落
        p = doc.add_paragraph()
        add_runs(p, stripped)
        maybe_insert_figures(doc, stripped)
        i += 1

    doc.save(OUT)
    print('saved', OUT)


if __name__ == '__main__':
    main()
