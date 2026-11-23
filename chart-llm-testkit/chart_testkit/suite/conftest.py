from __future__ import annotations

import os

import pytest

from chart_testkit import __version__, report
from chart_testkit.runner import Suite


def pytest_addoption(parser):
    g = parser.getgroup("chart-testkit")
    g.addoption("--chart-config", default=None, help="config YAML (or env CHART_TESTKIT_CONFIG)")
    g.addoption("--chart-out", default=None, help="report folder (or env CHART_TESTKIT_OUT)")


def _suite(config) -> Suite:
    if not hasattr(config, "_chart_suite"):
        path = config.getoption("--chart-config") or os.environ.get("CHART_TESTKIT_CONFIG")
        if not path:
            raise pytest.UsageError("no config: pass --chart-config PATH or set CHART_TESTKIT_CONFIG")
        config._chart_suite = Suite(path)
    return config._chart_suite


def pytest_generate_tests(metafunc):
    if "case_id" in metafunc.fixturenames:
        ids = [c["id"] for c in _suite(metafunc.config).cases]
        metafunc.parametrize("case_id", ids, ids=ids)


def pytest_collection_finish(session):
    session.config._chart_selected = [i.callspec.params["case_id"] for i in session.items if hasattr(i, "callspec")]


@pytest.fixture(scope="session")
def results(request):
    cfg = request.config
    suite = _suite(cfg)
    res = suite.run(getattr(cfg, "_chart_selected", None))
    yield res
    out = cfg.getoption("--chart-out") or os.environ.get("CHART_TESTKIT_OUT") or "chart-report"
    meta = {
        "name": suite.cfg["name"],
        "app": getattr(suite.app, "name", type(suite.app).__name__),
        "default repeats": suite.cfg["repeats"],
        "min pass rate": suite.cfg["min_pass_rate"],
        "cases run": len(res),
        "config": str(suite.config_path),
        "kit version": __version__,
    }
    cfg._chart_paths = report.write(res, meta, out)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    paths = getattr(config, "_chart_paths", None)
    if paths:
        terminalreporter.write_sep("-", "chart-testkit report")
        terminalreporter.write_line(f"HTML: {paths[1].resolve()}")
        terminalreporter.write_line(f"JSON: {paths[0].resolve()}")
