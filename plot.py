"""
Renders the UMAP embedding as a single self-contained HTML page.

The page is plain HTML with an inline SVG: no scripts or CDN assets, so it renders offline and
inside the sandboxed iframe the platform shows app outputs in. Hovering a point shows the
patient id through the SVG <title> element.
"""
from __future__ import annotations

import html
import math
from typing import Optional

import numpy as np
import pandas as pd

from umap_pipeline import PATIENT_ID, PipelineResult, PipelineSettings

WIDTH, HEIGHT = 960, 620
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 48, 24, 16, 44

# Tableau 10 - distinguishable categorical colours
CATEGORICAL = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
               "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]
OTHER_COLOR = "#8c8c8c"
MISSING_COLOR = "#c8c8c8"
SINGLE_COLOR = "#4e79a7"
# Viridis anchor stops for continuous values
VIRIDIS = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e",
           "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"]


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def viridis(t: float) -> str:
    t = min(1.0, max(0.0, t))
    position = t * (len(VIRIDIS) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(VIRIDIS) - 1)
    fraction = position - low
    a, b = _hex_to_rgb(VIRIDIS[low]), _hex_to_rgb(VIRIDIS[high])
    r, g, bl = (round(a[i] + (b[i] - a[i]) * fraction) for i in range(3))
    return f"#{r:02x}{g:02x}{bl:02x}"


def _format_value(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        return "missing"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    return str(value)


def _point_colors(result: PipelineResult) -> tuple[list[str], str]:
    """Colour of every point plus the legend markup."""
    embedding = result.embedding
    column = result.color_column
    if column is None:
        return [SINGLE_COLOR] * len(embedding), ""

    values = embedding[column]
    if result.color_kind == "numeric":
        numeric = pd.to_numeric(values, errors="coerce")
        low, high = numeric.min(), numeric.max()
        span = (high - low) if pd.notna(high) and high != low else 1.0
        colors = [MISSING_COLOR if pd.isna(v) else viridis((v - low) / span) for v in numeric]
        stops = "".join(f'<stop offset="{i / (len(VIRIDIS) - 1):.3f}" stop-color="{c}"/>'
                        for i, c in enumerate(VIRIDIS))
        legend = (
            f'<div class="legend"><div class="legend-title">{html.escape(column)}</div>'
            f'<svg width="220" height="34" role="img" aria-label="colour scale">'
            f'<defs><linearGradient id="scale">{stops}</linearGradient></defs>'
            f'<rect x="0" y="0" width="220" height="12" rx="2" fill="url(#scale)"/>'
            f'<text x="0" y="28" class="tick">{html.escape(_format_value(low))}</text>'
            f'<text x="220" y="28" class="tick" text-anchor="end">{html.escape(_format_value(high))}</text>'
            f'</svg>'
            + (f'<div class="legend-item"><span class="swatch" style="background:{MISSING_COLOR}"></span>missing</div>'
               if numeric.isna().any() else "")
            + '</div>'
        )
        return colors, legend

    labels = values.astype("string")
    counts = labels.value_counts(dropna=True)
    top = list(counts.index[:len(CATEGORICAL)])
    palette = {label: CATEGORICAL[i] for i, label in enumerate(top)}
    colors = [MISSING_COLOR if pd.isna(v) else palette.get(v, OTHER_COLOR) for v in labels]
    items = [f'<div class="legend-item"><span class="swatch" style="background:{palette[label]}"></span>'
             f'{html.escape(str(label))} <span class="count">{int(counts[label])}</span></div>'
             for label in top]
    other = int(counts.iloc[len(CATEGORICAL):].sum()) if len(counts) > len(CATEGORICAL) else 0
    if other:
        items.append(f'<div class="legend-item"><span class="swatch" style="background:{OTHER_COLOR}"></span>'
                     f'other <span class="count">{other}</span></div>')
    missing = int(labels.isna().sum())
    if missing:
        items.append(f'<div class="legend-item"><span class="swatch" style="background:{MISSING_COLOR}"></span>'
                     f'missing <span class="count">{missing}</span></div>')
    legend = f'<div class="legend"><div class="legend-title">{html.escape(column)}</div>{"".join(items)}</div>'
    return colors, legend


def _scale(values: np.ndarray, start: float, end: float) -> np.ndarray:
    low, high = float(np.min(values)), float(np.max(values))
    padding = (high - low) * 0.04 or 1.0
    low, high = low - padding, high + padding
    return start + (values - low) / (high - low) * (end - start)


def render_html(result: PipelineResult, settings: PipelineSettings,
                title: str = "UMAP of patient data") -> str:
    embedding = result.embedding
    count = len(embedding)
    radius = 4.0 if count <= 500 else 3.0 if count <= 5000 else 2.0

    plot_left, plot_right = MARGIN_LEFT, WIDTH - MARGIN_RIGHT
    plot_top, plot_bottom = MARGIN_TOP, HEIGHT - MARGIN_BOTTOM
    xs = _scale(embedding["umap_1"].to_numpy(dtype=float), plot_left, plot_right)
    # SVG y grows downwards
    ys = _scale(embedding["umap_2"].to_numpy(dtype=float), plot_bottom, plot_top)

    colors, legend = _point_colors(result)
    color_values: Optional[pd.Series] = embedding[result.color_column] if result.color_column else None

    points = []
    for i in range(count):
        tooltip = f"Patient {embedding[PATIENT_ID].iat[i]}"
        if color_values is not None:
            tooltip += f"\n{result.color_column}: {_format_value(color_values.iat[i])}"
        points.append(
            f'<circle cx="{xs[i]:.1f}" cy="{ys[i]:.1f}" r="{radius}" fill="{colors[i]}">'
            f'<title>{html.escape(tooltip)}</title></circle>'
        )

    used = result.used_features
    numeric = sum(1 for f in used if f.kind == "numeric")
    categorical = len(used) - numeric
    dropped = len(result.features) - len(used)
    summary = (f"{count} patients &middot; {len(used)} features "
               f"({numeric} numeric, {categorical} categorical, {dropped} dropped) &middot; "
               f"n_neighbors {result.n_neighbors} &middot; min_dist {settings.min_dist:g} &middot; "
               f"{html.escape(settings.metric)}")
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in result.notes)

    rows = "".join(
        f"<tr><td>{html.escape(f.feature)}</td><td>{f.kind}</td><td>{f.coverage:.0%}</td>"
        f"<td>{f.columns}</td><td>{html.escape(f.reason)}</td></tr>"
        for f in result.features
    )

    svg = (
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{html.escape(title)}" class="plot">'
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_right - plot_left}" '
        f'height="{plot_bottom - plot_top}" class="frame"/>'
        f'<text x="{(plot_left + plot_right) / 2}" y="{HEIGHT - 12}" text-anchor="middle" class="axis">UMAP 1</text>'
        f'<text x="16" y="{(plot_top + plot_bottom) / 2}" text-anchor="middle" class="axis" '
        f'transform="rotate(-90 16 {(plot_top + plot_bottom) / 2})">UMAP 2</text>'
        f'<g class="points">{"".join(points)}</g>'
        f'</svg>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  body {{ margin: 0; padding: 16px 20px; font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
         color: #1f2328; background: #ffffff; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  .summary {{ color: #57606a; margin-bottom: 8px; }}
  .notes {{ color: #9a6700; margin: 0 0 8px; padding-left: 18px; }}
  .layout {{ display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }}
  .plot {{ flex: 1 1 640px; max-width: {WIDTH}px; width: 100%; height: auto; }}
  .frame {{ fill: none; stroke: #d0d7de; }}
  .axis {{ fill: #57606a; font-size: 12px; }}
  .tick {{ fill: #57606a; font-size: 11px; }}
  .points circle {{ fill-opacity: 0.8; stroke: #ffffff; stroke-width: 0.5; }}
  .points circle:hover {{ fill-opacity: 1; stroke: #1f2328; stroke-width: 1.5; }}
  .legend {{ min-width: 180px; }}
  .legend-title {{ font-weight: 600; margin-bottom: 6px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; margin: 2px 0; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; }}
  .count {{ color: #57606a; }}
  details {{ margin-top: 12px; }}
  table {{ border-collapse: collapse; margin-top: 8px; }}
  th, td {{ text-align: left; padding: 3px 10px 3px 0; border-bottom: 1px solid #eaeef2; }}
  th {{ color: #57606a; font-weight: 600; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="summary">{summary}</div>
{f'<ul class="notes">{notes}</ul>' if notes else ''}
<div class="layout">
{svg}
{legend}
</div>
<details>
<summary>Features ({len(result.features)})</summary>
<table>
<thead><tr><th>Feature</th><th>Kind</th><th>Coverage</th><th>Model columns</th><th>Note</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</details>
</body>
</html>
"""
