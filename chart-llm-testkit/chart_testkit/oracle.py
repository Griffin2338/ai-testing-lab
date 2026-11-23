"""Compare an extracted chart spec with what the data says it should show.

Each `expect:` key in a case maps to one check function. A check returns
(passed, detail). Unknown keys raise, so a typo in a case file can't pass silently.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from chart_testkit.policy import PolicyResult


@dataclass
class Outcome:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Evidence:
    """Everything a check may look at for one generation."""

    df: pd.DataFrame
    spec: dict
    policy: PolicyResult
    clarification: str | None = None


# ------------------------------------------------------------------ expected data


def _key(v) -> str:
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def expected_series(df: pd.DataFrame, spec: dict) -> pd.Series:
    d = df
    if spec.get("filter"):
        d = d.query(spec["filter"])
    for name, expr in (spec.get("derive") or {}).items():
        d = d.assign(**{name: d.eval(expr)})
    agg = spec.get("agg", "sum")
    if spec.get("freq"):
        grouper = pd.Grouper(key=spec["groupby"], freq=spec["freq"])
        g = d.groupby(grouper)
    else:
        g = d.groupby(spec["groupby"])
    s = g.size() if agg == "count" else g[spec["value"]].agg(agg)
    if spec.get("freq"):
        s = s[s.index.notna()].sort_index()
    if spec.get("sort"):
        s = s.sort_values(ascending=spec["sort"] == "asc")
    if spec.get("top_n"):
        s = s.sort_values(ascending=False).head(int(spec["top_n"]))
        if spec.get("sort") == "asc":
            s = s.sort_values()
    s.index = [_key(i) for i in s.index]
    return s.astype(float)


def _close(a: float, b: float, tol: float) -> bool:
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= tol * max(abs(b), 1e-9)


def _norm_cat(c) -> str:
    return "" if c is None else str(c).strip().lower()


# ------------------------------------------------------------------ checks


def check_chart_type(ev: Evidence, allowed) -> tuple[bool, str]:
    allowed = [allowed] if isinstance(allowed, str) else list(allowed)
    got = ev.spec.get("chart_type", "none")
    return got in allowed, f"got {got}, allowed {allowed}"


def check_data(ev: Evidence, spec: dict) -> tuple[bool, str]:
    exp = expected_series(ev.df, spec)
    tol = float(spec.get("tolerance", 0.005))
    cats = ev.spec.get("categories") or []
    vals = ev.spec.get("values") or []
    if not cats or len(cats) != len(vals):
        return False, "no data series found in the chart"
    kind = ev.spec.get("chart_type", "")

    if spec.get("freq") or kind == "line":
        # time series: compare in chronological order, ignore the exact x encoding
        def order_key(p):
            try:
                return (0, float(p[0]), "")
            except (TypeError, ValueError):
                return (1, 0.0, str(p[0]))

        pairs = sorted(zip(cats, vals, strict=True), key=order_key)
        got = [v for _, v in pairs]
        want = exp.tolist()
        if len(got) != len(want):
            return False, f"{len(got)} points, expected {len(want)}"
        bad = [i for i, (g, w) in enumerate(zip(got, want, strict=True)) if not _close(g, w, tol)]
        return not bad, f"{len(bad)} of {len(want)} points off by more than {tol:.1%}" if bad else "all points match"

    if kind == "pie":
        shares = exp / exp.sum()
        got = {_norm_cat(c): v for c, v in zip(cats, vals, strict=True)}
        missing = [k for k in shares.index if _norm_cat(k) not in got]
        if missing:
            return False, f"slices missing: {missing[:5]}"
        bad = [k for k in shares.index if abs(got[_norm_cat(k)] - shares[k]) > max(tol, 0.005)]
        return not bad, f"slice shares off: {bad[:5]}" if bad else "slice shares match"

    got = {}
    for c, v in zip(cats, vals, strict=True):
        got[_norm_cat(c)] = v
    want = {_norm_cat(k): v for k, v in exp.items()}
    extra = sorted(set(got) - set(want))
    missing = sorted(set(want) - set(got))
    if extra or missing:
        return False, f"categories differ: missing {missing[:5]}, unexpected {extra[:5]}"
    bad = [k for k in want if not _close(got[k], want[k], tol)]
    if bad:
        k = bad[0]
        return False, f"{len(bad)} value(s) off, e.g. {k}: chart {got[k]:.4g} vs data {want[k]:.4g}"
    if spec.get("order"):
        shown = [_norm_cat(c) for c in cats]
        expected_order = [_norm_cat(k) for k in exp.index]
        if shown != expected_order:
            return False, f"order {shown[:5]} differs from {expected_order[:5]}"
    return True, f"{len(want)} categories match within {tol:.1%}"


def check_zero_baseline(ev: Evidence, on: bool) -> tuple[bool, str]:
    if not on:
        return True, "not required"
    axis = ev.spec.get("value_axis") or [None, None]
    lo = axis[0]
    vals = [v for v in (ev.spec.get("values") or []) if v is not None]
    if lo is None:
        ok = ev.spec.get("library") == "plotly"  # plotly autorange starts bars at 0
        return ok, "autorange" if ok else "no axis range found"
    if vals and min(vals) >= 0:
        return abs(lo) <= 1e-9 * max(abs(max(vals)), 1), f"value axis starts at {lo:.4g}"
    return lo <= 0, f"value axis starts at {lo:.4g}"


def _axes0(ev: Evidence) -> dict:
    return (ev.spec.get("axes") or [{}])[0]


def check_labels(ev: Evidence, on: bool) -> tuple[bool, str]:
    if not on:
        return True, "not required"
    a = _axes0(ev)
    xl, yl = (a.get("x_label") or "").strip(), (a.get("y_label") or "").strip()
    raw = re.compile(r"^(col_?\d+|unnamed.*|index|none|null)$", re.I)
    problems = []
    if not xl or raw.match(xl):
        problems.append(f"x label {xl!r}")
    if not yl or raw.match(yl):
        problems.append(f"y label {yl!r}")
    if ev.spec.get("chart_type") in ("barh", "stacked_barh"):
        problems = [p for p in problems if not p.startswith("y label ''")]  # category axis on y needs no title
    return not problems, ", ".join(problems) or "axis labels present"


def _text_blob(ev: Evidence) -> str:
    a = _axes0(ev)
    return " ".join(str(a.get(k) or "") for k in ("title", "x_label", "y_label")).lower()


def check_y_label_any(ev: Evidence, words) -> tuple[bool, str]:
    a = _axes0(ev)
    blob = " ".join(str(a.get(k) or "") for k in ("title", "y_label")).lower()
    hit = [w for w in words if str(w).lower() in blob]
    return bool(hit), f"found {hit}" if hit else f"none of {words} in title / y label"


def check_axis_names_any(ev: Evidence, names) -> tuple[bool, str]:
    blob = _text_blob(ev).replace("_", " ")
    hit = [n for n in names if n.replace("_", " ").lower() in blob]
    return bool(hit), f"labels name {hit}" if hit else f"labels don't say which of {names} was used"


def check_legend_count(ev: Evidence, spec: dict) -> tuple[bool, str]:
    want = int(ev.df[spec["distinct_of"]].nunique())
    got = len(_axes0(ev).get("legend") or [])
    return got == want, f"legend has {got} entries, data has {want} groups"


def check_stacked_totals(ev: Evidence, spec: dict) -> tuple[bool, str]:
    return check_data(ev, {**spec, "agg": spec.get("agg", "sum")})


def check_max_tick_overlap(ev: Evidence, limit: float) -> tuple[bool, str]:
    ov = _axes0(ev).get("x_tick_overlap") or {}
    r = ov.get("max_ratio")
    if r is None:
        return True, "n/a for this library"
    return r <= float(limit), f"worst tick-label overlap {r:.0%} (limit {float(limit):.0%})"


def check_no_overflow(ev: Evidence, on: bool) -> tuple[bool, str]:
    of = ev.spec.get("overflow") or []
    return (not on) or not of, f"text outside the figure: {of[:3]}" if of else "no text outside the figure"


_DECIMAL_YEAR = re.compile(r"^(19|20)\d\d\.\d+$")
_FLOAT_NOISE = re.compile(r"\d\.\d{7,}")


def check_no_float_noise(ev: Evidence, on: bool) -> tuple[bool, str]:
    a = _axes0(ev)
    bad = [t for t in (a.get("x_ticklabels") or []) + (a.get("y_ticklabels") or [])
           if _DECIMAL_YEAR.match(t.replace("−", "-")) or _FLOAT_NOISE.search(t)]
    return (not on) or not bad, f"odd tick labels: {bad[:4]}" if bad else "tick labels clean"


def check_max_slices(ev: Evidence, limit: int) -> tuple[bool, str]:
    if ev.spec.get("chart_type") != "pie":
        return True, "not a pie"
    n = len(_axes0(ev).get("pie") or [])
    return n <= int(limit), f"{n} slices (limit {limit})"


def check_only_columns(ev: Evidence, allowed) -> tuple[bool, str]:
    extra = sorted(ev.policy.columns_read - set(allowed))
    return not extra, f"also reads {extra}" if extra else f"reads only {sorted(ev.policy.columns_read)}"


def check_no_fabrication(ev: Evidence, on: bool) -> tuple[bool, str]:
    issues = list(ev.policy.fabrication) + [f"unknown column {c!r}" for c in sorted(ev.policy.unknown_columns)]
    return (not on) or not issues, "; ".join(issues) if issues else "no invented data"


def check_clarification(ev: Evidence, on: bool) -> tuple[bool, str]:
    asked = bool(ev.clarification)
    return asked == bool(on), "asked a clarifying question" if asked else "returned code"


CHECKS: dict[str, Callable[[Evidence, object], tuple[bool, str]]] = {
    "chart_type": check_chart_type,
    "data": check_data,
    "zero_baseline": check_zero_baseline,
    "labels": check_labels,
    "y_label_any": check_y_label_any,
    "axis_names_any": check_axis_names_any,
    "legend_count": check_legend_count,
    "stacked_totals": check_stacked_totals,
    "max_tick_overlap": check_max_tick_overlap,
    "no_overflow": check_no_overflow,
    "no_float_noise": check_no_float_noise,
    "max_slices": check_max_slices,
    "only_columns": check_only_columns,
    "no_fabrication": check_no_fabrication,
    "clarification": check_clarification,
}


def evaluate(ev: Evidence, expect: dict) -> list[Outcome]:
    """Run every check in `expect`. `any_of` holds alternatives; one fully passing is enough."""
    out: list[Outcome] = []
    for key, arg in expect.items():
        if key == "any_of":
            tried = []
            for i, alt in enumerate(arg):
                res = evaluate(ev, alt)
                if all(o.passed for o in res):
                    out.append(Outcome("any_of", True, f"alternative {i + 1} passed"))
                    break
                tried.append("; ".join(f"{o.name}: {o.detail}" for o in res if not o.passed))
            else:
                out.append(Outcome("any_of", False, " | ".join(tried)))
            continue
        if key not in CHECKS:
            raise KeyError(f"unknown expectation {key!r}; known: {sorted(CHECKS)}")
        if key != "clarification" and ev.clarification:
            out.append(Outcome(key, False, "no chart: the app asked a question instead"))
            continue
        try:
            passed, detail = CHECKS[key](ev, arg)
        except Exception as exc:  # a broken chart should fail the check, not the run
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        out.append(Outcome(key, bool(passed), detail))
    return out
