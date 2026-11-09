#!/usr/bin/env python3
"""V3DSAM paper figure style — a faithful mirror of the nature-figure skill.

This module reproduces, verbatim, the constants and helper functions from the
official nature-figure skill's `references/api.md` (plus the `pdf.fonttype=42`
line from `static/fragments/backend/python.md`), so every figure script in this
project draws from the same official Nature/CNS convention instead of a
hand-rolled approximation.

Official sources mirrored here:
  * references/api.md      — PALETTE, DEFAULT_COLORS, PALETTE_NMI_PASTEL,
                            PALETTE_NATURE_*, apply_publication_style, is_dark,
                            add_panel_label, style_dark_image_ax,
                            make_grouped_bar, make_trend, make_forest_plot,
                            make_heatmap, finalize_figure.
  * static/fragments/backend/python.md — pdf.fonttype = 42 (editable TrueType).

Only two tiny conveniences are added beyond the official API (documented in-line):
  * C_TRUE/C_FALSE/C_GT/C_OURS/C_BASELINE — semantic aliases onto PALETTE values.
  * set_agg_backend() — headless `matplotlib.use("Agg")` before pyplot import.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────
# Constants (verbatim from references/api.md)
# ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1":   "#F6CFCB",
    "red_2":   "#E9A6A1",
    "red_strong": "#B64342",
    "neutral_light": "#CFCECE",
    "neutral_mid":   "#767676",
    "neutral_dark":  "#4D4D4D",
    "neutral_black": "#272727",
    "gold":   "#FFD700",
    "teal":   "#42949E",
    "violet": "#9A4D8E",
    "magenta":"#EA84DD",
}

DEFAULT_COLORS = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral_light"],
]

PALETTE_NMI_PASTEL = {
    "baseline_dark": "#484878",
    "baseline_mid":  "#7884B4",
    "baseline_soft": "#B4C0E4",
    "ours_tiny":  "#E4E4F0",
    "ours_base":  "#E4CCD8",
    "ours_large": "#F0C0CC",
    "bg_lilac": "#E0E0F0",
    "bg_aqua":  "#E0F0F0",
    "bg_peach": "#F0E0D0",
    "neutral_light": "#D8D8D8",
    "neutral_mid":   "#A8A8A8",
    "neutral_dark":  "#606060",
    "delta_up":   "#2E9E44",
    "delta_down": "#E53935",
}

DEFAULT_COLORS_NMI_PASTEL = [
    PALETTE_NMI_PASTEL["baseline_dark"],
    PALETTE_NMI_PASTEL["baseline_mid"],
    PALETTE_NMI_PASTEL["baseline_soft"],
    PALETTE_NMI_PASTEL["ours_tiny"],
    PALETTE_NMI_PASTEL["ours_base"],
    PALETTE_NMI_PASTEL["ours_large"],
]

PALETTE_NATURE_IMAGING = {
    "bg": "#000000",
    "context": "#B8B8B8",
    "cyan": "#22D7E6",
    "magenta": "#FF2AD4",
    "white": "#FFFFFF",
}

PALETTE_NATURE_MATERIAL = {
    "aqua": "#77D7D1",
    "teal": "#33B5A5",
    "lilac": "#B9A7E8",
    "violet": "#7C6CCF",
    "callout_red": "#E53935",
    "neutral": "#D9D9D9",
}

PALETTE_NATURE_CLINICAL = {
    "baseline": "#272727",
    "week6": "#E28E2C",
    "week13": "#D24B40",
    "week26": "#5B8FD6",
    "year1": "#7BAA5B",
    "year2": "#C45AD6",
    "group_band": "#F2E6D9",
}

PALETTE_NATURE_GENOMICS = {
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#8F8F8F",
    "wave1": "#D9544D",
    "wave2": "#5B7FCA",
    "wave3": "#B89BD9",
    "outline": "#4D4D4D",
}

# Semantic role aliases (project convenience — map onto the official PALETTE).
C_TRUE = PALETTE["green_3"]      # true loop / success / gain
C_FALSE = PALETTE["red_strong"]  # false loop / failure / drop
C_GT = PALETTE["neutral_mid"]    # ground truth
C_OURS = PALETTE["blue_main"]    # our method (hero)
C_BASELINE = PALETTE["neutral_dark"]


# ─────────────────────────────────────────────────────────────────────
# Global style (verbatim from references/api.md, + pdf.fonttype from python.md)
# ─────────────────────────────────────────────────────────────────────
def apply_publication_style(font_size=16, axes_linewidth=2.5, use_tex=False):
    """Apply Nature-style rcParams. Call once before creating any figures.

    Presets (from api.md):
      * Large bar panels:          apply_publication_style(font_size=24, axes_linewidth=3)
      * Compact figures:           apply_publication_style(font_size=15, axes_linewidth=2)
      * Dense journal multi-panels: apply_publication_style(font_size=8, axes_linewidth=1)
    """
    # ── MANDATORY: editable SVG/PDF text ─────────────────────────────────────
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['svg.fonttype'] = 'none'   # keep text as <text> nodes, not paths
    plt.rcParams['pdf.fonttype'] = 42       # editable TrueType (from python.md)
    # ── Layout & style ───────────────────────────────────────────────────────
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = axes_linewidth
    plt.rcParams['legend.frameon'] = False
    # No grid lines by default (design-theory.md §2): sparse y-ticks guide the eye.
    plt.rcParams['axes.grid'] = False
    # Consistent export defaults so tight bbox + raster density are uniform.
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.03
    plt.rcParams['savefig.dpi'] = 300
    if use_tex:
        plt.rcParams['text.usetex'] = True


def is_dark(hex_color, threshold=128):
    """Return True if hex color is dark (use white text on it)."""
    c = hex_color.lstrip('#')
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (0.299*r + 0.587*g + 0.114*b) < threshold


def add_panel_label(ax, label, x=-0.06, y=1.02, fontsize=14,
                    color='black', fontweight='bold'):
    """Place a Nature-style panel label near the top-left edge."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        ha='left',
        va='bottom',
    )


def style_dark_image_ax(ax, facecolor='black'):
    """Prepare an axes for microscopy / rendering plates."""
    ax.set_facecolor(facecolor)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


# ─────────────────────────────────────────────────────────────────────
# Chart helpers (verbatim from references/api.md)
# ─────────────────────────────────────────────────────────────────────
def make_grouped_bar(ax, categories, series, labels,
                     ylabel='Value', colors=None,
                     annotate=False, bar_width=0.8,
                     series_spread=None, error_kw=None):
    """Grouped bar chart. See api.md for the full parameter contract."""
    import numpy as np
    if colors is None:
        colors = DEFAULT_COLORS
    if error_kw is None:
        error_kw = {'elinewidth': 2, 'capthick': 2, 'capsize': 10}
    n_groups = len(series)
    n_cats = len(categories)
    w = bar_width / n_groups
    x = np.arange(n_cats)
    flat_values = np.concatenate([np.asarray(values, dtype=float) for values in series])
    label_pad = 0.02 * max(float(np.ptp(flat_values)), float(np.max(np.abs(flat_values))), 1.0)
    containers = []
    for i, (vals, label, color) in enumerate(zip(series, labels, colors)):
        spread = None if series_spread is None else np.asarray(series_spread[i], dtype=float)
        offset = (i - (n_groups - 1) / 2) * w
        bars = ax.bar(x + offset, vals, width=w, label=label,
                      color=color, edgecolor='black', linewidth=1.5,
                      yerr=spread, error_kw=error_kw)
        containers.append(bars)
        if annotate:
            for j, (bar, val) in enumerate(zip(bars, vals)):
                upper = bar.get_height() + (0 if spread is None else spread[j])
                ax.text(bar.get_x() + bar.get_width() / 2,
                        upper + label_pad,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.legend()
    return containers


def make_trend(ax, x, y_series, labels,
               colors=None, ylabel=None, xlabel=None,
               show_shadow=True, shadow_alpha=0.15,
               lw=2.5, marker='o', markersize=8):
    """Multi-line trend plot. 2D y rows are summarized as mean ± std band."""
    import numpy as np
    if colors is None:
        colors = DEFAULT_COLORS
    for y, label, color in zip(y_series, labels, colors):
        y = np.asarray(y)
        if y.ndim == 2:
            mean, std = y.mean(0), y.std(0)
        else:
            mean, std = y, None
        ax.plot(x, mean, color=color, lw=lw, marker=marker,
                markersize=markersize, label=label)
        if show_shadow and std is not None:
            ax.fill_between(x, mean - std, mean + std,
                            color=color, alpha=shadow_alpha)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.legend()


def make_forest_plot(ax, labels, estimates, ci_low, ci_high,
                     colors=None, ref=0.0, xlabel=None, xlim=None,
                     marker='o', markersize=5, lw=1.5):
    """Minimal forest plot helper for Nature-style clinical/statistical panels."""
    import numpy as np
    y = np.arange(len(labels))[::-1]
    if colors is None:
        colors = ['#B64342'] * len(labels)
    for yi, est, lo, hi, color in zip(y, estimates, ci_low, ci_high, colors):
        ax.plot([lo, hi], [yi, yi], color=color, lw=lw)
        ax.plot(est, yi, marker=marker, ms=markersize, color=color)
    ax.axvline(ref, color='#767676', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    if xlabel:
        ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def make_heatmap(ax, matrix, x_labels=None, y_labels=None,
                 cmap='magma', cbar_label=None, annotate=False,
                 fmt='{:.2f}', fontsize=12):
    """2D heatmap with optional colorbar and cell annotations."""
    import numpy as np
    import matplotlib as mpl
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    if cbar_label:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
    if x_labels:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=30, ha='right', rotation_mode='anchor')
    if y_labels:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
    if annotate:
        norm = mpl.colors.Normalize(vmin=matrix.min(), vmax=matrix.max())
        cm_obj = plt.get_cmap(cmap)
        for (i, j), val in np.ndenumerate(matrix):
            r, g, b, _ = cm_obj(norm(val))
            lum = 0.299*r + 0.587*g + 0.114*b
            color = 'white' if lum < 0.5 else 'black'
            ax.text(j, i, fmt.format(val), ha='center', va='center',
                    fontsize=fontsize, color=color)
    ax.set_frame_on(False)


# ─────────────────────────────────────────────────────────────────────
# Export (verbatim from references/api.md, with the §9 SVG-primary policy)
# ─────────────────────────────────────────────────────────────────────
def finalize_figure(fig, out_path, formats=None, dpi=300,
                    pad=2, bbox_inches='tight', close=True):
    """Apply tight_layout and save figure.

    Follows the design-theory §9 export policy: SVG is the required primary
    format (editable text), PNG is the raster preview, PDF optional. When
    `formats` is omitted and `out_path` has no extension, all three are written.
    """
    fig.tight_layout(pad=pad)
    base = Path(out_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    if formats is None:
        if base.suffix:
            formats = [base.suffix.lstrip('.')]
            base = base.with_suffix('')
        else:
            # SVG primary + PNG secondary + PDF (design-theory §9).
            formats = ['svg', 'png', 'pdf']
    saved = []
    for fmt in formats:
        p = str(base) + f'.{fmt}'
        fig.savefig(p, dpi=dpi, bbox_inches=bbox_inches)
        saved.append(p)
    if close:
        plt.close(fig)
    return saved


def set_agg_backend():
    """Headless-safe non-interactive backend (call before pyplot import)."""
    matplotlib.use('Agg')
