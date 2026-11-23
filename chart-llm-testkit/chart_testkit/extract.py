"""Turn a rendered matplotlib or plotly figure into a plain dict ("chart spec").

Runs inside the sandbox child, so it must not trust anything the generated code left
behind beyond the figure objects themselves.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _seq(v) -> list:
    """Plotly trace arrays: None, list, numpy array, or {'dtype', 'bdata'} (plotly 6 typed arrays)."""
    if v is None:
        return []
    if isinstance(v, dict) and "bdata" in v:
        import base64

        arr = np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v.get("dtype", "f8")))
        return arr.tolist()
    if hasattr(v, "tolist"):
        return v.tolist()
    return list(v)


def _label(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


# ------------------------------------------------------------------ matplotlib


def _tick_map(axis) -> dict[float, str]:
    out = {}
    for pos, lab in zip(axis.get_ticklocs(), axis.get_ticklabels(), strict=False):
        out[float(pos)] = lab.get_text()
    return out


def _nearest(mapping: dict[float, str], x: float) -> str | None:
    if not mapping:
        return None
    pos = min(mapping, key=lambda p: abs(p - x))
    return mapping[pos] if abs(pos - x) < 0.51 else None


def _shown_ticklabels(ax, which: str) -> list:
    """Tick labels inside the view limits; matplotlib also returns ticks it doesn't draw."""
    labels = ax.get_xticklabels() if which == "x" else ax.get_yticklabels()
    lo, hi = sorted(ax.get_xlim() if which == "x" else ax.get_ylim())
    idx = 0 if which == "x" else 1
    eps = (hi - lo) * 1e-9
    return [t for t in labels if lo - eps <= t.get_position()[idx] <= hi + eps]


def _overlap_stats(texts, renderer) -> tuple[float, int]:
    boxes = [t.get_window_extent(renderer) for t in texts if t.get_text().strip() and t.get_visible()]
    worst, pairs = 0.0, 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            w = min(a.x1, b.x1) - max(a.x0, b.x0)
            h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if w <= 0 or h <= 0:
                continue
            ratio = (w * h) / max(min(a.width * a.height, b.width * b.height), 1e-6)
            if ratio > 0.05:
                pairs += 1
            worst = max(worst, ratio)
    return worst, pairs


def _bars(ax) -> list[dict]:
    from matplotlib.container import BarContainer

    out = []
    xmap, ymap = _tick_map(ax.xaxis), _tick_map(ax.yaxis)
    for c in ax.containers:
        if not isinstance(c, BarContainer) or not len(c.patches):
            continue
        horizontal = getattr(c, "orientation", "vertical") == "horizontal"
        cats, vals, base, pos = [], [], [], []
        for p in c.patches:
            if horizontal:
                center, value, start = p.get_y() + p.get_height() / 2, p.get_width(), p.get_x()
                cats.append(_nearest(ymap, center))
            else:
                center, value, start = p.get_x() + p.get_width() / 2, p.get_height(), p.get_y()
                cats.append(_nearest(xmap, center))
            vals.append(_num(value))
            base.append(_num(start))
            pos.append(center)
        out.append({"name": c.get_label() or "", "horizontal": horizontal, "categories": cats,
                    "values": vals, "base": base, "positions": pos})
    return out


def _date_axis(axis) -> bool:
    try:
        import matplotlib.dates as mdates

        conv = axis.get_converter() if hasattr(axis, "get_converter") else axis.converter
        return isinstance(conv, (mdates.DateConverter, mdates._SwitchableDateConverter))
    except Exception:
        return False


def _x_values(xs, is_date: bool) -> list:
    """x data as plotted: datetimes, datetime64, date numbers (date axis) or plain numbers."""
    import matplotlib.dates as mdates

    out = []
    for x in xs:
        if isinstance(x, np.datetime64):
            out.append(str(np.datetime_as_string(x, unit="D")))
        elif hasattr(x, "isoformat"):
            out.append(x.isoformat()[:10])
        elif is_date and _num(x) is not None:
            out.append(mdates.num2date(float(x)).strftime("%Y-%m-%d"))
        else:
            n = _num(x)
            out.append(n if n is not None else str(x))
    return out


def extract_matplotlib(fig) -> dict[str, Any]:
    from matplotlib.collections import PathCollection
    from matplotlib.patches import Wedge

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
    axes = [ax for ax in fig.axes if ax.get_visible()]
    spec: dict[str, Any] = {"library": "matplotlib", "n_axes": len(axes), "axes": []}
    overflow = []
    for ax in axes:
        is_date = _date_axis(ax.xaxis)
        a: dict[str, Any] = {
            "title": ax.get_title() or (fig._suptitle.get_text() if fig._suptitle else ""),
            "x_label": ax.get_xlabel(),
            "y_label": ax.get_ylabel(),
            "xlim": [_num(v) for v in ax.get_xlim()],
            "ylim": [_num(v) for v in ax.get_ylim()],
            "y_inverted": bool(ax.yaxis_inverted()),
            "x_ticklabels": [t.get_text() for t in _shown_ticklabels(ax, "x")],
            "y_ticklabels": [t.get_text() for t in _shown_ticklabels(ax, "y")],
            "legend": [t.get_text() for t in ax.get_legend().get_texts()] if ax.get_legend() else [],
            "bars": _bars(ax),
            "lines": [],
            "scatter": [],
            "pie": [],
        }
        for ln in ax.get_lines():
            xs, ys = ln.get_xdata(), ln.get_ydata()
            if len(xs) < 2 or str(ln.get_label()).startswith("_child") and len(set(map(str, ys))) <= 1:
                continue  # skip axhline/axvline style helpers
            a["lines"].append({"name": ln.get_label() if not str(ln.get_label()).startswith("_") else "",
                               "x": _x_values(xs, is_date), "y": [_num(v) for v in ys]})
        for coll in ax.collections:
            if isinstance(coll, PathCollection) and len(coll.get_offsets()):
                off = np.asarray(coll.get_offsets())
                a["scatter"].append({"x": [_num(v) for v in off[:, 0]], "y": [_num(v) for v in off[:, 1]]})
        wedges = [p for p in ax.patches if isinstance(p, Wedge)]
        if wedges:
            total = sum((w.theta2 - w.theta1) for w in wedges) or 1.0
            texts = [t.get_text() for t in ax.texts]
            a["pie"] = [{"fraction": (w.theta2 - w.theta1) / 360.0, "share_of_drawn": (w.theta2 - w.theta1) / total,
                         "label": w.get_label() if not str(w.get_label()).startswith("_") else ""} for w in wedges]
            a["texts"] = texts
        xo, xp = _overlap_stats(_shown_ticklabels(ax, "x"), renderer)
        a["x_tick_overlap"] = {"max_ratio": xo, "pairs": xp}
        shown = [*_shown_ticklabels(ax, "x"), *_shown_ticklabels(ax, "y")]
        for t in [*shown, ax.title, ax.xaxis.label, ax.yaxis.label]:
            if not t.get_text().strip() or not t.get_visible():
                continue
            bb = t.get_window_extent(renderer)
            if bb.x0 < fig_box.x0 - 1 or bb.x1 > fig_box.x1 + 1 or bb.y0 < fig_box.y0 - 1 or bb.y1 > fig_box.y1 + 1:
                overflow.append(t.get_text()[:40])
        spec["axes"].append(a)
    spec["overflow"] = overflow
    return spec


# ------------------------------------------------------------------ plotly


def extract_plotly(fig) -> dict[str, Any]:
    d = fig.to_dict()
    layout = d.get("layout", {})

    def title(obj) -> str:
        t = obj.get("title") if isinstance(obj, dict) else None
        if isinstance(t, dict):
            return t.get("text") or ""
        return t or ""

    a: dict[str, Any] = {
        "title": title(layout),
        "x_label": title(layout.get("xaxis", {})),
        "y_label": title(layout.get("yaxis", {})),
        "xlim": list(layout.get("xaxis", {}).get("range") or [None, None]),
        "ylim": list(layout.get("yaxis", {}).get("range") or [None, None]),
        "y_inverted": layout.get("yaxis", {}).get("autorange") == "reversed",
        "x_ticklabels": [],
        "y_ticklabels": [],
        "legend": [],
        "bars": [],
        "lines": [],
        "scatter": [],
        "pie": [],
        "x_tick_overlap": {"max_ratio": None, "pairs": None},
    }
    for tr in d.get("data", []):
        kind = tr.get("type", "scatter")
        name = tr.get("name") or ""
        if name and tr.get("showlegend", True) is not False:
            a["legend"].append(name)
        if kind == "bar":
            horizontal = tr.get("orientation") == "h"
            cats = _seq(tr.get("y" if horizontal else "x"))
            vals = _seq(tr.get("x" if horizontal else "y"))
            a["bars"].append({"name": name, "horizontal": horizontal, "categories": [_label(c) for c in cats],
                              "values": [_num(v) for v in vals], "base": [0.0] * len(vals),
                              "positions": list(range(len(vals)))})
        elif kind == "pie":
            vals = [_num(v) or 0.0 for v in _seq(tr.get("values"))]
            total = sum(vals) or 1.0
            labels = _seq(tr.get("labels")) or [""] * len(vals)
            a["pie"] = [{"fraction": v / total, "share_of_drawn": v / total, "label": _label(lab)}
                        for v, lab in zip(vals, labels, strict=False)]
        elif kind in ("scatter", "scattergl"):
            xs = [_label(x) if isinstance(x, str) else _num(x) for x in _seq(tr.get("x"))]
            ys = [_num(y) for y in _seq(tr.get("y"))]
            mode = tr.get("mode") or "lines+markers"
            target = "lines" if "lines" in mode else "scatter"
            entry = {"name": name, "x": [str(x)[:10] if isinstance(x, str) else x for x in xs], "y": ys}
            a[target].append(entry)
    barmode = layout.get("barmode")
    return {"library": "plotly", "n_axes": 1, "axes": [a], "overflow": [], "barmode": barmode}


# ------------------------------------------------------------------ summary


def summarise(spec: dict[str, Any]) -> dict[str, Any]:
    """Add chart_type, display-ordered categories and the main value series."""
    if not spec.get("axes"):
        spec["chart_type"] = "none"
        return spec
    a = spec["axes"][0]
    bars, lines, scatter, pie = a["bars"], a["lines"], a["scatter"], a["pie"]
    stacked = len(bars) > 1 and (
        spec.get("barmode") in ("stack", "relative")
        or any(any((b or 0) != 0 for b in s["base"]) for s in bars[1:])
    )
    if pie:
        kind = "pie"
    elif bars:
        horizontal = bars[0]["horizontal"]
        named = any(c not in (None, "") for c in bars[0]["categories"])
        if not named and not horizontal and len(bars[0]["values"]) > 3:
            kind = "histogram"
        else:
            kind = ("stacked_" if stacked else "") + ("barh" if horizontal else "bar")
    elif lines:
        kind = "line"
    elif scatter:
        kind = "scatter"
    else:
        kind = "unknown"
    spec["chart_type"] = kind

    categories: list = []
    values: list = []
    if bars:
        first = bars[0]
        order = np.argsort(first["positions"])
        if first["horizontal"] and not a["y_inverted"] and spec["library"] == "matplotlib":
            order = order[::-1]  # matplotlib draws barh bottom-up; read top-down
        if first["horizontal"] and spec["library"] == "plotly" and not a["y_inverted"]:
            order = order[::-1]
        categories = [first["categories"][i] for i in order]
        totals = np.zeros(len(first["values"]))
        for s in bars if stacked else bars[:1]:
            if len(s["values"]) == len(totals):
                totals = totals + np.array([v or 0.0 for v in s["values"]])
        values = [float(totals[i]) for i in order]
        spec["value_axis"] = a["xlim"] if first["horizontal"] else a["ylim"]
    elif lines:
        ln = lines[0]
        categories, values = ln["x"], ln["y"]
        spec["value_axis"] = a["ylim"]
    elif pie:
        categories = [p["label"] for p in pie]
        values = [p["fraction"] for p in pie]
        spec["value_axis"] = [None, None]
    spec["categories"] = [None if c is None else _label(c) for c in categories]
    spec["values"] = values
    spec["series_count"] = len(bars) or len(lines) or len(scatter)
    return spec
