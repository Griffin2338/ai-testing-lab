from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from chart_testkit import datasets, policy
from chart_testkit.app import parse_reply
from chart_testkit.oracle import Evidence, evaluate, expected_series
from chart_testkit.runner import _relation, wilson

COLS = ["sector", "traded_value", "traded_value_prev", "trade_date"]


# ----------------------------------------------------------------- policy


def test_clean_code_passes():
    r = policy.check("s = df.groupby('sector')['traded_value'].sum()\nplt.bar(s.index, s.values)", COLS)
    assert r.ok and not r.fabrication
    assert r.columns_read == {"sector", "traded_value"}


@pytest.mark.parametrize("code, needle", [
    ("import os", "import not allowed: os"),
    ("from subprocess import run", "import not allowed: subprocess"),
    ("import urllib.request", "import not allowed: urllib.request"),
    ("eval('1')", "eval()"),
    ("x = ().__class__", "dunder"),
    ("plt.savefig('/tmp/x.png')", ".savefig()"),
    ("d = pd.read_csv('other.csv')", ".read_csv()"),
])
def test_violations(code, needle):
    r = policy.check(code, COLS)
    assert not r.ok and any(needle in v for v in r.violations), r.violations


def test_syntax_error_reported():
    assert policy.check("def (", COLS).syntax_error


def test_fabrication_signals():
    r = policy.check("y = np.random.normal(size=12)\nd = pd.DataFrame({'a': [1, 2, 3]})", COLS)
    assert "uses np.random" in r.fabrication
    assert any("builds a DataFrame" in f for f in r.fabrication)


def test_unknown_columns_but_not_derived_ones():
    code = "d = df.copy()\nd['profit'] = d['proceeds'] - d['cost_basis']\nx = d['profit']\ny = df['traded_value']"
    r = policy.check(code, ["proceeds", "cost_basis", "traded_value"])
    assert r.unknown_columns == set()
    r2 = policy.check("x = df['profit']", ["proceeds", "cost_basis"])
    assert r2.unknown_columns == {"profit"}


# ----------------------------------------------------------------- replies


def test_parse_reply():
    assert parse_reply("```python\nx = 1\n```").code == "x = 1"
    g = parse_reply("CLARIFY: which date column?")
    assert g.code is None and g.clarification == "which date column?"


# ----------------------------------------------------------------- oracle


@pytest.fixture(scope="module")
def df():
    return datasets.build({"builtin": "trades", "seed": 3})


def _spec(categories, values, kind="bar", lo=0.0, **axes):
    a = {"title": "Traded value by sector", "x_label": "Sector", "y_label": "Traded value", "legend": [],
         "x_ticklabels": [],
         "y_ticklabels": [], "x_tick_overlap": {"max_ratio": 0.0}, **axes}
    return {"library": "matplotlib", "chart_type": kind, "categories": categories, "values": values,
            "value_axis": [lo, max(values) * 1.05], "axes": [a], "overflow": []}


def test_data_check_passes_and_catches_mean(df):
    exp = expected_series(df, {"groupby": "sector", "value": "traded_value"})
    good = _spec(list(exp.index), list(exp.values))
    ok = evaluate(Evidence(df, good, policy.PolicyResult()), {"data": {"groupby": "sector", "value": "traded_value"}})
    assert ok[0].passed, ok[0].detail
    mean = df.groupby("sector")["traded_value"].mean()
    bad = _spec(list(mean.index), list(mean.values))
    out = evaluate(Evidence(df, bad, policy.PolicyResult()), {"data": {"groupby": "sector", "value": "traded_value"}})
    assert not out[0].passed and "off" in out[0].detail


def test_order_and_top_n(df):
    spec = {"groupby": "ticker", "value": "proceeds", "top_n": 3, "order": True}
    exp = expected_series(df, spec)
    shown = _spec(list(exp.index), list(exp.values))
    assert evaluate(Evidence(df, shown, policy.PolicyResult()), {"data": spec})[0].passed
    rev = _spec(list(exp.index[::-1]), list(exp.values[::-1]))
    assert not evaluate(Evidence(df, rev, policy.PolicyResult()), {"data": spec})[0].passed


def test_zero_baseline():
    ev = Evidence(pd.DataFrame(), _spec(["a", "b"], [10.0, 12.0], lo=9.0), policy.PolicyResult())
    assert not evaluate(ev, {"zero_baseline": True})[0].passed
    ev = Evidence(pd.DataFrame(), _spec(["a", "b"], [10.0, 12.0], lo=0.0), policy.PolicyResult())
    assert evaluate(ev, {"zero_baseline": True})[0].passed


def test_any_of_and_clarification(df):
    ev = Evidence(df, {}, policy.PolicyResult(), clarification="Which date?")
    out = evaluate(ev, {"any_of": [{"clarification": True}, {"chart_type": "line"}]})
    assert out[0].passed
    out = evaluate(ev, {"chart_type": "line"})
    assert not out[0].passed and "asked" in out[0].detail


def test_unknown_expectation_key_is_an_error(df):
    with pytest.raises(KeyError):
        evaluate(Evidence(df, {}, policy.PolicyResult()), {"chart_typo": "bar"})


def test_float_noise_and_decimal_years():
    spec = _spec(["a"], [1.0], x_ticklabels=["2021.5", "2022.0"])
    out = evaluate(Evidence(pd.DataFrame(), spec, policy.PolicyResult()), {"no_float_noise": True})
    assert not out[0].passed


# ----------------------------------------------------------------- relations and stats


def test_relations():
    a = {"categories": ["Energy", "Finance"], "values": [10.0, 20.0]}
    assert _relation("shuffle_rows", {}, a, {"categories": ["Finance", "Energy"], "values": [20.0, 10.0]})[0]
    assert _relation("scale", {"factor": 1000}, a, {"categories": ["Energy", "Finance"], "values": [1e4, 2e4]})[0]
    assert not _relation("scale", {"factor": 1000}, a, {"categories": ["Energy", "Finance"], "values": [1e4, 2e3]})[0]
    renamed = {"categories": ["Power", "Finance"], "values": [10.0, 20.0]}
    assert _relation("rename_value", {"from": "Energy", "to": "Power"}, a, renamed)[0]


def test_wilson_matches_reference():
    lo, hi = wilson(873, 900)
    assert (round(lo, 4), round(hi, 4)) == (0.9567, 0.9793)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(st.sampled_from(["a", "b", "c"]), st.floats(0, 1e6, allow_nan=False)), min_size=1,
                max_size=40))
def test_expected_series_is_row_order_invariant(rows):
    d = pd.DataFrame(rows, columns=["g", "v"])
    a = expected_series(d, {"groupby": "g", "value": "v"})
    b = expected_series(d.iloc[::-1], {"groupby": "g", "value": "v"})
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index(), rtol=1e-9)
