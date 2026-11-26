"""Shared publication style for every current figure panel."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MM = 1 / 25.4
ROOT = Path(__file__).resolve().parent

COLORS = {
    "ink": "#111111", "charcoal": "#4F5660", "white": "#FFFFFF",
    "pale_grey": "#ECEFF2", "band_grey": "#F5F6F7", "grid": "#C9CED4",
    "gray": "#7B818A", "gray_dark": "#4F5660", "gray_light": "#D2D6DB",
    "blue": "#3264B0", "blue_dark": "#1F4788", "blue_light": "#7FA6D8",
    "slate": "#65758B", "teal": "#178C8C", "teal_light": "#75BDB7",
    "coral": "#D9574F", "coral_dark": "#A83C3A", "coral_light": "#EA9487",
    "orange": "#E08B2C", "purple": "#7651B5", "rose": "#B54878", "green": "#3E8B5B",
}
TASK_COLORS = [COLORS[x] for x in ("blue", "coral", "teal", "orange", "purple")]

FONT = {
    "tick": 5.8, "legend": 5.8, "data": 5.8, "annotation": 5.8,
    "category": 7.0, "emphasis": 7.0, "axis": 8.5, "compact": 5.65,
    "milestone": 6.3,
}


@lru_cache(maxsize=1)
def _register_arial() -> None:
    # 显式注册字体文件；仅写 Arial 为首选可能在导出 PDF 时悄悄回退。
    for path in fm.findSystemFonts(fontext="ttf"):
        if "arial" not in Path(path).stem.lower():
            continue
        if fm.FontProperties(fname=path).get_name() == "Arial":
            fm.fontManager.addfont(path)
    for weight in ("normal", "bold"):
        prop = fm.FontProperties(family="Arial", weight=weight)
        try:
            path = fm.findfont(prop, fallback_to_default=False)
        except ValueError as exc:
            raise RuntimeError("缺少 Arial 字体，已停止导出图件；请安装 Arial 后重试。") from exc
        if fm.FontProperties(fname=path).get_name() != "Arial":
            raise RuntimeError(f"{weight} 字体不是 Arial：{path}")


def configure() -> None:
    _register_arial()
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": FONT["tick"], "axes.labelsize": FONT["axis"],
        "axes.labelweight": "bold", "axes.titlesize": FONT["axis"],
        "xtick.labelsize": FONT["tick"], "ytick.labelsize": FONT["tick"],
        "legend.fontsize": FONT["legend"], "text.color": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"], "axes.edgecolor": COLORS["charcoal"],
        "xtick.color": COLORS["ink"], "ytick.color": COLORS["ink"],
        "axes.linewidth": 0.78, "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": COLORS["white"], "figure.facecolor": COLORS["white"],
        "savefig.bbox": "standard", "savefig.pad_inches": 0,
    })


def figure(width_mm: float, height_mm: float):
    configure()
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), facecolor=COLORS["white"])
    # Some backends quantize the initial canvas at the screen DPI. Resetting the
    # physical size preserves the requested millimetres in PDF/SVG and at export.
    fig.set_size_inches(width_mm * MM, height_mm * MM, forward=False)
    return fig


def clean_axis(ax, grid_axis="x"):
    # Publication panels use a complete rectangular plotting frame.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["charcoal"])
        spine.set_linewidth(0.78)
    ax.grid(False)
    ax.set_axisbelow(True)


def legend(ax, handles=None, labels=None, loc="lower right", **kwargs):
    options = dict(frameon=False, borderpad=0.0, handletextpad=0.45,
                   labelspacing=0.35)
    options.update(kwargs)
    return ax.legend(handles=handles, labels=labels, loc=loc, **options)


def task_handles(task_labels):
    return [Line2D([], [], marker="o", ls="", ms=4.5, color=c, label=l)
            for c, l in zip(TASK_COLORS, task_labels)]


def export(fig, outdir: Path, stem: str, dpi: int = 500):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(outdir / f"{stem}.{ext}", bbox_inches=None, pad_inches=0)
    fig.savefig(outdir / f"{stem}.png", dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def legend_figure(handles, width_mm: float, height_mm: float, ncol: int):
    """Build a standalone, frameless legend for manual figure assembly."""
    fig = figure(width_mm, height_mm)
    fig.legend(handles=handles, loc="center", ncol=ncol, frameon=False,
               borderpad=0, handletextpad=.55, columnspacing=1.25,
               fontsize=FONT["legend"])
    return fig


def flow_arrow(ax, x0, x1, y, color=None, lw=1.4):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color or COLORS["gray_dark"],
                                shrinkA=1, shrinkB=1))


def rounded_box(ax, xy, width, height, face, line, text, subtext=None):
    from matplotlib.patches import FancyBboxPatch, Rectangle
    x, y = xy
    patch = FancyBboxPatch((x, y), width, height, transform=ax.transAxes,
                           boxstyle="round,pad=0.012,rounding_size=0.025",
                           fc=COLORS["band_grey"], ec=line, lw=1.15)
    ax.add_patch(patch)
    stripe = Rectangle((x + .012, y + height - .052), width - .024, .025,
                       transform=ax.transAxes, fc=line, ec="none", zorder=3)
    ax.add_patch(stripe)
    ax.text(x + width / 2, y + height * (0.58 if subtext else 0.5), text,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=FONT["emphasis"], weight="bold", color=COLORS["ink"])
    if subtext:
        ax.text(x + width / 2, y + height * 0.27, subtext, transform=ax.transAxes,
                ha="center", va="center", fontsize=FONT["milestone"], color=COLORS["ink"])
    return patch
