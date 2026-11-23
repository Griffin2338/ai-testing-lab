from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from od_testkit.dataset import load_yolo_dataset, read_class_names
from od_testkit.detectors import ShapeDetector
from od_testkit.metrics import evaluate
from od_testkit.synthetic import generate_dataset
from od_testkit.transforms import (
    CORRUPTIONS,
    adjust_brightness,
    corrupt,
    find_free_position,
    hflip,
    pad,
    rescale,
    unflip_boxes,
    unpad_boxes,
    unscale_boxes,
)

BOXES = np.array([[10, 20, 50, 60], [0, 0, 100, 30]], float)


def test_flip_roundtrip():
    img = np.random.default_rng(0).integers(0, 255, (60, 100, 3), dtype=np.uint8)
    assert np.array_equal(hflip(hflip(img)), img)
    assert np.allclose(unflip_boxes(unflip_boxes(BOXES, 100), 100), BOXES)
    assert unflip_boxes(np.array([[0, 0, 10, 10]]), 100).tolist() == [[90, 0, 100, 10]]


def test_scale_and_pad_mapping():
    img = np.zeros((60, 100, 3), np.uint8)
    small, sx, sy = rescale(img, 0.75)
    assert small.shape[:2] == (45, 75)
    assert np.allclose(unscale_boxes(BOXES * [sx, sy, sx, sy], sx, sy), BOXES)
    padded, p = pad(img, 0.1)
    assert p == 10 and padded.shape[:2] == (80, 120)
    assert np.allclose(unpad_boxes(BOXES + p, p), BOXES)


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_every_corruption_keeps_shape_and_dtype(name):
    img = np.random.default_rng(0).integers(0, 255, (64, 80, 3), dtype=np.uint8)
    for sev in (1, 5):
        out = corrupt(img, name, sev, seed=1)
        assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(corrupt(img, name, 3, seed=7), corrupt(img, name, 3, seed=7))  # reproducible


def test_corrupt_rejects_bad_input():
    img = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(KeyError):
        corrupt(img, "snow", 1)
    with pytest.raises(ValueError):
        corrupt(img, "fog", 6)


def test_find_free_position_respects_margin():
    rng = np.random.default_rng(0)
    occupied = np.array([[0, 0, 80, 100]], float)
    for _ in range(20):
        x, _ = find_free_position((100, 100, 3), (10, 10), occupied, 5, rng)
        assert x >= 85
    assert find_free_position((100, 100, 3), (10, 10), np.array([[0, 0, 100, 100]]), 0, rng) is None


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return generate_dataset(tmp_path_factory.mktemp("synth"), n_images=40, n_negatives=12, seed=5)


def test_synthetic_dataset_loads(synth):
    samples = load_yolo_dataset(synth / "images" / "val")
    assert len(samples) == 40
    assert sum(len(s.boxes) for s in samples) > 60
    assert read_class_names(synth / "data.yaml") == {0: "red", 1: "blue"}
    s = samples[0]
    assert (s.boxes[:, 2] <= s.width).all() and (s.boxes[:, 3] <= s.height).all()


def test_shape_detector_finds_large_objects_and_has_its_known_bugs(synth):
    samples = load_yolo_dataset(synth / "images" / "val")
    det = ShapeDetector()
    clean = [det.predict(s.load_image()) for s in samples]
    big = lambda s: np.minimum(s.boxes[:, 2] - s.boxes[:, 0], s.boxes[:, 3] - s.boxes[:, 1]) >= 32  # noqa: E731
    ev = evaluate(samples, clean, det.class_names, conf=0.4, gt_filter=big, n_boot=50)
    assert ev.pooled_recall() > 0.95
    dark = [det.predict(adjust_brightness(s.load_image(), 0.6)) for s in samples]
    ev_dark = evaluate(samples, dark, det.class_names, conf=0.4, gt_filter=big, n_boot=50)
    assert ev.pooled_recall() - ev_dark.pooled_recall() > 0.10  # v_min bug


def test_cli_end_to_end(synth, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(textwrap.dedent(f"""
        name: e2e
        dataset: {{images: "{(synth / 'images' / 'val').as_posix()}", names: "{(synth / 'data.yaml').as_posix()}"}}
        detector: {{type: shapes}}
        operating_point: {{conf: 0.4, iou: 0.5}}
        min_instances: 20
        bootstrap: {{n: 100, seed: 0}}
        tests:
          brightness: {{factors: [0.6], max_recall_drop: 0.05}}
          hflip: {{min_preserved: 0.95}}
          small_objects: {{min_instances: 500}}
          label_audit: {{}}
    """), encoding="utf-8")
    out = tmp_path / "report"
    proc = subprocess.run([sys.executable, "-m", "od_testkit.cli", "run", str(cfg), "--out", str(out)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 1, proc.stdout + proc.stderr  # TC-02 fails on the v_min bug
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    status = {r["tc_id"]: r["status"] for r in report["records"]}
    assert status["TC-02"] == "fail"
    assert status["TC-04"] == "pass"
    assert status["TC-08"] == "insufficient"
    assert status["TC-12"] == "info"
    assert status["TC-01"] == "skipped"
    assert (out / "report.html").read_text(encoding="utf-8").startswith("<!doctype html>")
