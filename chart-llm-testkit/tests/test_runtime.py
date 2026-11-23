"""Tests that start sandbox processes. Slower: each run imports pandas and matplotlib."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pandas as pd
import pytest

from chart_testkit import datasets
from chart_testkit.llm import ClaudeLLM, FakeLLM, LLMRefusal
from chart_testkit.sandbox import run_code

DF = pd.DataFrame({"sector": ["Energy", "Finance", "Health", "Energy"], "traded_value": [10.0, 20.0, 30.0, 5.0]})


def test_bar_extraction_and_png():
    code = "s = df.groupby('sector')['traded_value'].sum()\nfig, ax = plt.subplots()\nax.bar(s.index, s.values)"
    r = run_code(code, DF)
    assert r.ok and r.png_b64
    assert r.spec["chart_type"] == "bar"
    assert dict(zip(r.spec["categories"], r.spec["values"], strict=True)) == {"Health": 30, "Finance": 20, "Energy": 15}
    assert r.spec["value_axis"][0] == 0


def test_barh_reads_top_down():
    code = ("s = df.groupby('sector')['traded_value'].sum().sort_values()\n"
            "fig, ax = plt.subplots()\nax.barh(s.index, s.values)")
    r = run_code(code, DF)
    assert r.spec["chart_type"] == "barh"
    assert r.spec["categories"] == ["Health", "Finance", "Energy"]  # largest bar drawn at the top


def test_line_with_dates():
    df = datasets.build({"builtin": "trades", "seed": 1})
    code = "s = df.set_index('trade_date')['traded_value'].resample('MS').sum()\nplt.plot(s.index, s.values)"
    r = run_code(code, df)
    assert r.spec["chart_type"] == "line"
    assert r.spec["categories"][0] == "2024-01-01" and len(r.spec["values"]) == 12


def test_plotly_bar_and_pie():
    r = run_code("import plotly.express as px\nfig = px.pie(df, names='sector', values='traded_value')", DF)
    assert r.spec["chart_type"] == "pie"
    assert abs(sum(r.spec["values"]) - 1.0) < 1e-9


@pytest.mark.parametrize("code", [
    "import socket\nsocket.create_connection(('example.com', 80), timeout=2)",
    "import os\nos.system('echo hi')",
    "open('../escape.txt', 'w').write('x')",
])
def test_sandbox_blocks(code):
    r = run_code(code, DF, timeout_s=20)
    assert not r.ok and r.blocked


def test_sandbox_timeout():
    r = run_code("while True:\n    pass", DF, timeout_s=3)
    assert r.timed_out


def test_fake_llm_profiles_differ():
    text = "- sector: object\n- traded_value: float64\n\nRequest: Compare total traded value by sector."
    prompt = [{"role": "user", "content": text}]
    careful = FakeLLM("careful").complete("", prompt)
    buggy = FakeLLM("buggy").complete("", prompt)
    assert "set_ylim" not in careful and "set_ylim" in buggy


def test_claude_adapter_with_stub_client():
    llm = ClaudeLLM.__new__(ClaudeLLM)
    llm.model, llm.effort, llm.max_tokens, llm.fallbacks, llm.name = "claude-opus-5-5", "medium", 1000, True, "x"
    calls = {}

    def create(**kw):
        calls.update(kw)
        block = SimpleNamespace(type="text", text="```python\nx=1\n```")
        return SimpleNamespace(stop_reason="end_turn", content=[block])

    llm.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert "x=1" in llm.complete("sys", [{"role": "user", "content": "hi"}])
    assert calls["model"] == "claude-opus-5-5" and calls["fallbacks"] == "default"
    assert calls["output_config"] == {"effort": "medium"}

    def refuse(**kw):
        return SimpleNamespace(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"), content=[])

    llm.client.beta.messages.create = refuse
    with pytest.raises(LLMRefusal):
        llm.complete("sys", [{"role": "user", "content": "hi"}])


def test_cli_end_to_end(tmp_path):
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(textwrap.dedent(f"""
        name: e2e
        app: {{type: builtin, llm: {{type: fake, profile: buggy}}}}
        repeats: 2
        workers: 4
        cases: ["{(root / 'cases' / 'default.yaml').as_posix()}"]
        only: [TC-01, TC-02, TC-05]
    """), encoding="utf-8")
    out = tmp_path / "rep"
    proc = subprocess.run([sys.executable, "-m", "chart_testkit.cli", "run", str(cfg), "--out", str(out)],
                          capture_output=True, text=True, timeout=900, check=False)
    assert proc.returncode == 1, proc.stdout[-2000:] + proc.stderr[-2000:]
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    status = {c["id"]: c["status"] for c in rep["cases"]}
    assert status == {"TC-01": "pass", "TC-02": "fail", "TC-05": "fail"}
    assert (out / "report.html").read_text(encoding="utf-8").startswith("<!doctype html>")
