# -*- coding: utf-8 -*-
"""
Tidy "Where used and PFEP" renderer driven by column A: "Structure Level".

- Processes every .xlsx in the same folder (first sheet only).
- Trims spaces; preserves leading zeros (dtype=str).
- Filters out rows where column "Number" CONTAINS any banned token (case-insensitive).
- Level-aware tree layout:
  * build parent->child from Structure Level stack
  * compute subtree sizes
  * assign vertical slots by subtree size
  * parent Y = midpoint(children Y)
- Robust right-angle connectors (never raises intersection error).
- Outputs: .png and .pdf
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path
from matplotlib.backends.backend_pdf import PdfPages


# =========================
# USER SETTINGS
# =========================
TITLE = "Where used and PFEP"

# Colors by level (cycled if needed)
LEVEL_COLORS = [
    "#b7e3a1",
    "#bfbfbf",
    "#cfe9fb",
    "#eec8e9",
    "#f6c7ac",
    "#c6ecbd",
    "#ffd9a6",
    "#d7c9ff",
    "#b0e0ff",
    "#ffcabd",
]

# Drop rows whose 'Number' contains any of these tokens (case-insensitive, substring)
BANNED_NUMBER_TOKENS = {"DONOTUSE"}

# Geometry
COL_GAP = 3.2
ROW_GAP = 1.2  # base vertical spacing between slots
BOX_PAD_X = 0.35
BOX_PAD_Y = 0.35
CONTENT_TOP_PAD = 2.0  # gap before first row to keep headers clear
BOTTOM_MARGIN = 2.0
ROOT_GAP_SLOTS = 1.0  # blank slots inserted before each root tree
FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE = 22
LEVEL_SIZE = 16
LABEL_SIZE = 10
FIG_DPI = 220
# PDF page size limit (in inches). Acrobat starts warning past ~200in.
PDF_PAGE_LIMIT_IN = 180
PDF_TILE_OVERLAP = 1.5  # re-show edges so relationships are not split across pages

# Arrow styling
ARROW_LINEWIDTH = 1.2
ARROW_STYLE = "-|>"
ARROW_MUTATION = 12


def _norm_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _read_first_sheet(path: str) -> pd.DataFrame:
    """Load the first sheet, keep strings, and trim whitespace."""

    df = pd.read_excel(
        path, sheet_name=0, dtype=str, keep_default_na=False, engine="openpyxl"
    )
    return df.applymap(lambda cell: cell.strip() if isinstance(cell, str) else cell)


def _find_col(df: pd.DataFrame, wanted: str):
    target = _norm_header(wanted)
    for column in df.columns:
        if _norm_header(column) == target:
            return column
    return None


def _build_forest(
    df: pd.DataFrame, col_level: str, col_num: str, col_desc: str | None
):
    """From ordered rows: parent(level L) = last seen node at level L-1."""

    banned_tokens = tuple(token.lower() for token in BANNED_NUMBER_TOKENS)

    nodes: list[str] = []
    level: dict[str, int] = {}
    label: dict[str, str] = {}
    parent: dict[str, str] = {}
    last_at_level: dict[int, str] = {}

    for _, row in df.iterrows():
        lv_raw = str(row[col_level])
        if lv_raw == "":
            continue

        try:
            level_value = int(float(lv_raw))
        except ValueError:
            continue

        num = str(row[col_num])
        if not num:
            continue

        num_lower = num.lower()
        if banned_tokens and any(token in num_lower for token in banned_tokens):
            continue

        if col_desc:
            desc = str(row[col_desc]).strip()
            display = f"{num}\n{desc}" if desc else num
        else:
            display = num

        if num not in level:
            nodes.append(num)
            level[num] = level_value
            label[num] = display

            if (level_value - 1) in last_at_level:
                parent[num] = last_at_level[level_value - 1]

        last_at_level[level_value] = num
        for depth in list(last_at_level.keys()):
            if depth > level_value:
                del last_at_level[depth]

    return nodes, parent, level, label


def _children_map(parent: dict[str, str]):
    children: dict[str, list[str]] = defaultdict(list)
    for child, par in parent.items():
        children[par].append(child)
    return children


def _subtree_size(node: str, children: dict[str, list[str]], memo: dict[str, int]) -> int:
    if node in memo:
        return memo[node]

    if node not in children or len(children[node]) == 0:
        memo[node] = 1
        return 1

    size = sum(_subtree_size(child, children, memo) for child in children[node])
    memo[node] = max(2, size)  # give parents at least 2 slots
    return memo[node]


def _layout_slots(nodes, parent, level):
    """Assign y-slots by subtree size so big families get more vertical room."""

    children = _children_map(parent)
    roots = [node for node in nodes if node not in parent]

    memo: dict[str, int] = {}
    for node in nodes:
        _subtree_size(node, children, memo)

    y_slot = 0
    y_of: dict[str, float] = {}

    def place(node: str):
        nonlocal y_slot
        if node not in children or len(children[node]) == 0:
            y_of[node] = y_slot
            y_slot += 1
        else:
            for child in children[node]:
                place(child)
            child_slots = [y_of[child] for child in children[node]]
            y_of[node] = (min(child_slots) + max(child_slots)) / 2.0

    for idx, root in enumerate(roots):
        if idx > 0:
            y_slot += ROOT_GAP_SLOTS
        place(root)

    return y_of


def _auto_box_size(text, base_w=2.6, base_h=1.0):
    clean_text = str(text)
    lines = clean_text.replace("\\n", "\n").split("\n")
    width = max(base_w, 0.12 * max(len(line) for line in lines) + 1.4)
    height = max(base_h, 0.52 * len(lines) + 0.6)
    return width, height


def _draw_box(ax, x, y, w, h, label, color):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.2,
        edgecolor="#333",
        facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=LABEL_SIZE,
        family=FONT_FAMILY,
    )
    return rect


def _draw_connector(ax, boxA, boxB):
    """Right-angle polyline connector (robust)."""

    x1 = boxA.get_x() + boxA.get_width()
    y1 = boxA.get_y() + boxA.get_height() / 2
    x2 = boxB.get_x()
    y2 = boxB.get_y() + boxB.get_height() / 2
    eps = 1e-6

    if abs(y1 - y2) < eps or abs(x1 - x2) < eps:
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=ARROW_STYLE,
            mutation_scale=ARROW_MUTATION,
            linewidth=ARROW_LINEWIDTH,
            color="#444",
        )
        ax.add_patch(arrow)
        return

    mid_x = (x1 + x2) / 2.0
    verts = [(x1, y1), (mid_x, y1), (mid_x, y2), (x2, y2)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
    path = Path(verts, codes)
    arrow = FancyArrowPatch(
        path=path,
        arrowstyle=ARROW_STYLE,
        mutation_scale=ARROW_MUTATION,
        linewidth=ARROW_LINEWIDTH,
        color="#444",
    )
    ax.add_patch(arrow)


def _render_one_excel(path: str):
    df = _read_first_sheet(path)
    col_level = _find_col(df, "Structure Level")
    col_num = _find_col(df, "Number")

    if col_level is None or col_num is None:
        raise ValueError("Columns 'Structure Level' and 'Number' are required.")

    col_desc = _find_col(df, "Description")
    nodes, parent, level, label = _build_forest(df, col_level, col_num, col_desc)

    if not nodes:
        raise ValueError("No drawable rows after filtering.")

    y_slot = _layout_slots(nodes, parent, level)
    max_y = max(y_slot.values()) if y_slot else 0
    y_pos = {node: (y_slot[node] * ROW_GAP + CONTENT_TOP_PAD) for node in nodes}

    max_lv = max(level.values())
    x_pos = {node: (level[node] * COL_GAP) for node in nodes}

    height_units = (max_y + 1) * ROW_GAP + CONTENT_TOP_PAD + BOTTOM_MARGIN + 1.5
    width_units = (max_lv + 1) * COL_GAP + 2

    # Guard against matplotlib's 2^16 px limit for raster outputs.
    # PDF stays vector-based, but PNG needs dimensions within the cap.
    max_px = (2**16) - 1
    width_px = width_units * FIG_DPI
    height_px = height_units * FIG_DPI
    scale = min(1.0, max_px / width_px, max_px / height_px)
    effective_dpi = FIG_DPI * scale

    layout = {}
    for node in nodes:
        label_text = label[node]
        width, height = _auto_box_size(label_text)
        cx = x_pos[node] + BOX_PAD_X
        cy = y_pos[node] + BOX_PAD_Y
        color = LEVEL_COLORS[level[node] % len(LEVEL_COLORS)]
        layout[node] = {
            "label": label_text,
            "width": width,
            "height": height,
            "x": cx,
            "y": cy,
            "color": color,
        }

    def draw_page(ax, x0: float, y0: float, page_w: float, page_h: float, page_idx: int | None):
        ax.set_xlim(x0 - 1, x0 + page_w + 0.4)
        ax.set_ylim(y0 - BOTTOM_MARGIN, y0 + page_h + 0.8)
        ax.axis("off")

        title_suffix = f" (page {page_idx})" if page_idx is not None else ""
        ax.text(
            x0,
            y0 + page_h - 0.6,
            TITLE + title_suffix,
            fontsize=TITLE_SIZE,
            family=FONT_FAMILY,
            weight="bold",
        )

        for lv in range(max_lv + 1):
            x_lv = lv * COL_GAP + 0.5
            if x0 - 0.5 <= x_lv <= x0 + page_w + 0.5:
                ax.text(
                    x_lv,
                    y0 + page_h - 1.6,
                    f"Level-{lv}",
                    fontsize=LEVEL_SIZE,
                    color="#c00",
                    family=FONT_FAMILY,
                )

        visible_nodes = {}
        for node, info in layout.items():
            if (info["x"] + info["width"] >= x0) and (info["x"] <= x0 + page_w) and (
                info["y"] + info["height"] >= y0
            ) and (info["y"] <= y0 + page_h):
                visible_nodes[node] = _draw_box(
                    ax,
                    info["x"],
                    info["y"],
                    info["width"],
                    info["height"],
                    info["label"],
                    info["color"],
                )

        for child, par in parent.items():
            if par in visible_nodes and child in visible_nodes:
                _draw_connector(ax, visible_nodes[par], visible_nodes[child])

    stem = os.path.splitext(path)[0]

    # Full-resolution PNG (single page, raster-safe DPI).
    fig_png, ax_png = plt.subplots(figsize=(width_units, height_units), dpi=effective_dpi)
    draw_page(ax_png, 0, 0, width_units + 1, height_units + 2.5, None)
    fig_png.savefig(stem + ".png", bbox_inches="tight")
    plt.close(fig_png)

    # Multi-page PDF if the page would exceed Acrobat's size limit.
    tile_stride_x = PDF_PAGE_LIMIT_IN - PDF_TILE_OVERLAP
    tile_stride_y = PDF_PAGE_LIMIT_IN - PDF_TILE_OVERLAP
    tiles_x = max(1, math.ceil((width_units - PDF_TILE_OVERLAP) / tile_stride_x))
    tiles_y = max(1, math.ceil((height_units - PDF_TILE_OVERLAP) / tile_stride_y))
    page_total = tiles_x * tiles_y

    with PdfPages(stem + ".pdf") as pdf:
        page_idx = 1
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                x0 = tx * tile_stride_x
                y0 = ty * tile_stride_y
                page_w = min(PDF_PAGE_LIMIT_IN, width_units - x0 + PDF_TILE_OVERLAP)
                page_h = min(PDF_PAGE_LIMIT_IN, height_units - y0 + PDF_TILE_OVERLAP)
                fig, ax = plt.subplots(figsize=(page_w, page_h), dpi=FIG_DPI)
                draw_page(ax, x0, y0, page_w, page_h, page_idx if page_total > 1 else None)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                page_idx += 1

    print(f" ✓ Wrote {os.path.basename(stem)}.png and .pdf")


def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    xl_files = [
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if filename.lower().endswith(".xlsx")
    ]

    if not xl_files:
        print("No .xlsx files found next to the script.")
        return

    print(f"Found {len(xl_files)} Excel file(s).")
    for path in sorted(xl_files):
        print(f"- Processing {os.path.basename(path)} …")
        try:
            _render_one_excel(path)
        except Exception as exc:  # noqa: BLE001
            print(f" [error] {exc}")


if __name__ == "__main__":
    main()
