from __future__ import annotations

import os

import pytest

from od_testkit.context import SuiteContext
from od_testkit.metrics import SIZE_BUCKETS


def pytest_addoption(parser):
    group = parser.getgroup("od-testkit")
    group.addoption("--od-config", default=None, help="od-testkit YAML config (or env OD_TESTKIT_CONFIG)")
    group.addoption("--od-out", default=None, help="report output folder (or env OD_TESTKIT_OUT)")


@pytest.fixture(scope="session")
def ctx(request) -> SuiteContext:
    config = request.config.getoption("--od-config") or os.environ.get("OD_TESTKIT_CONFIG")
    if not config:
        raise pytest.UsageError("no config: pass --od-config PATH or set OD_TESTKIT_CONFIG")
    out = request.config.getoption("--od-out") or os.environ.get("OD_TESTKIT_OUT") or "od-report"
    context = SuiteContext(config, out)
    yield context
    _add_diagnostics(context)
    request.config._od_report_paths = context.report.write(context.out_dir)


def _add_diagnostics(ctx: SuiteContext) -> None:
    if "clean" not in ctx._preds:
        return
    ev = ctx.clean_eval
    total_fp = sum(ev.fp_breakdown.values())
    ctx.report.diagnostics["False positives by error type (clean, operating point)"] = [
        {"error type": k, "count": v, "share": v / total_fp if total_fp else float("nan")}
        for k, v in ev.fp_breakdown.items()
    ]
    rows = []
    for bucket, (tp, gt) in ev.size_recall.items():
        lo, hi = SIZE_BUCKETS[bucket]
        rows.append({"size": bucket, "area range px²": f"{lo:.0f} – {hi:.0f}", "objects": gt,
                     "recall": tp / gt if gt else float("nan")})
    ctx.report.diagnostics["Recall by object size (all classes)"] = rows
    ctx.report.diagnostics["Inference time per variant"] = [
        {"variant": k, "seconds": v, "ms / image": 1000 * v / len(ctx.samples)} for k, v in ctx.timings.items()
    ]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    paths = getattr(config, "_od_report_paths", None)
    if paths:
        terminalreporter.write_sep("-", "od-testkit report")
        terminalreporter.write_line(f"HTML: {paths[1].resolve()}")
        terminalreporter.write_line(f"JSON: {paths[0].resolve()}")
