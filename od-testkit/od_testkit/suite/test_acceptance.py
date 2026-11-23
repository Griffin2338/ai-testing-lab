"""Each test reads its own section under `tests:` in the config.

No section -> "not configured". Too little data -> pytest skip with the reason, shown as
"insufficient data" in the report.
"""

from __future__ import annotations

import pytest

from od_testkit import checks
from od_testkit.report import TestRecord


def _run(ctx, key: str) -> None:
    meta = checks.META[key]
    cfg = ctx.test_cfg(key)
    if cfg is None:
        ctx.report.add(TestRecord(meta.tc_id, meta.title, "skipped",
                                  f"Not configured: add 'tests.{key}' to the YAML config to enable it.",
                                  reference=meta.reference))
        pytest.skip(f"{meta.tc_id} not configured")
    try:
        rec = checks.CHECKS[key](ctx, cfg)
    except Exception as exc:
        ctx.report.add(TestRecord(meta.tc_id, meta.title, "error", f"{type(exc).__name__}: {exc}",
                                  reference=meta.reference))
        raise
    ctx.report.add(rec)
    if rec.status == "insufficient":
        pytest.skip(f"{meta.tc_id} insufficient data: {rec.summary}")
    assert rec.status != "fail", f"{meta.tc_id} {meta.title}: {rec.summary}"


def test_tc01_per_class_acceptance(ctx):
    _run(ctx, "per_class")


def test_tc02_brightness_robustness(ctx):
    _run(ctx, "brightness")


def test_tc03_corruption_robustness(ctx):
    _run(ctx, "corruptions")


def test_tc04_hflip_invariance(ctx):
    _run(ctx, "hflip")


def test_tc05_scale_and_pad_invariance(ctx):
    _run(ctx, "scale")


def test_tc06_object_insertion(ctx):
    _run(ctx, "insertion")


def test_tc07_negative_images(ctx):
    _run(ctx, "negatives")


def test_tc08_small_object_recall(ctx):
    _run(ctx, "small_objects")


def test_tc10_latency(ctx):
    _run(ctx, "latency")


def test_tc11_export_parity(ctx):
    _run(ctx, "parity")


def test_tc12_label_audit(ctx):
    _run(ctx, "label_audit")


def test_tc14_calibration(ctx):
    _run(ctx, "calibration")
