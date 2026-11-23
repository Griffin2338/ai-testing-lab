from __future__ import annotations

import doctest
import math

import numpy as np
import pytest

from od_testkit import stats
from od_testkit.geometry import iou_matrix
from od_testkit.matching import (
    BACKGROUND,
    CLASS_CONFUSION,
    DUPLICATE,
    LOCALIZATION,
    TP,
    match_image,
    preservation_rate,
)
from od_testkit.metrics import average_precision, calibration_error, evaluate
from od_testkit.structures import Detections, Sample


def dets(boxes, classes, scores):
    return Detections(np.array(boxes, float), np.array(classes), np.array(scores, float))


# ----------------------------------------------------------------- geometry


def test_iou_identical_disjoint_half():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]])
    assert iou_matrix(a, b) == pytest.approx(np.array([[1.0, 0.0, 50 / 150]]))


def test_iou_empty_inputs():
    assert iou_matrix(np.zeros((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)


# ----------------------------------------------------------------- matching


def test_match_classifies_every_error_type():
    gt_boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 110, 110]])
    gt_classes = np.array([0, 0, 1])
    pred = dets(
        [[0, 0, 10, 10],        # TP
         [0, 0, 10, 10],        # duplicate of the first
         [50, 50, 56, 56],      # same class, IoU 0.36: localisation error
         [100, 100, 110, 110],  # right place, wrong class: class confusion
         [200, 200, 210, 210]], # nothing there: background
        [0, 0, 0, 0, 0],
        [0.9, 0.8, 0.7, 0.6, 0.5],
    )
    m = match_image(pred, gt_boxes, gt_classes, 0.5)
    assert m.kind.tolist() == [TP, DUPLICATE, LOCALIZATION, CLASS_CONFUSION, BACKGROUND]
    assert m.gt_matched.tolist() == [True, False, False]


def test_match_is_greedy_by_score():
    gt = np.array([[0, 0, 10, 10]])
    pred = dets([[0, 0, 10, 9], [0, 0, 10, 10]], [0, 0], [0.9, 0.3])
    m = match_image(pred, gt, np.array([0]), 0.5)
    assert m.kind.tolist() == [TP, DUPLICATE]  # higher score wins even with lower IoU


def test_preservation_rate_uses_each_candidate_once():
    ref = dets([[0, 0, 10, 10], [0, 0, 10, 10]], [0, 0], [0.9, 0.8])
    cand = dets([[0, 0, 10, 10]], [0], [0.9])
    assert preservation_rate(ref, cand, 0.9) == (1, 2)
    assert preservation_rate(Detections.empty(), cand, 0.9) == (0, 0)


# ----------------------------------------------------------------- metrics


def _toy():
    samples = [
        Sample("a", "a.png", 100, 100, [[0, 0, 10, 10], [50, 50, 90, 90]], [0, 1]),
        Sample("b", "b.png", 100, 100, [[10, 10, 50, 50]], [0]),
    ]
    preds = [
        dets([[0, 0, 10, 10], [50, 50, 90, 90], [70, 0, 90, 20]], [0, 1, 1], [0.9, 0.8, 0.3]),
        dets([[10, 10, 50, 50]], [0], [0.6]),
    ]
    return samples, preds


def test_evaluate_operating_point_counts():
    samples, preds = _toy()
    ev = evaluate(samples, preds, {0: "a", 1: "b"}, conf=0.5, n_boot=200)
    assert (ev.per_class[0].tp, ev.per_class[0].fp, ev.per_class[0].fn) == (2, 0, 0)
    assert (ev.per_class[1].tp, ev.per_class[1].fp, ev.per_class[1].fn) == (1, 0, 0)
    ev_low = evaluate(samples, preds, {0: "a", 1: "b"}, conf=0.2, n_boot=200)
    assert ev_low.per_class[1].fp == 1  # the 0.3 box appears only below the threshold
    assert ev_low.fp_breakdown["background"] == 1


def test_perfect_detector_has_ap_one():
    samples, preds = _toy()
    ev = evaluate(samples, preds, {0: "a", 1: "b"}, conf=0.5, n_boot=50)
    assert ev.per_class[0].ap50 == pytest.approx(1.0)
    assert ev.per_class[0].ap50_95 == pytest.approx(1.0)
    assert ev.per_class[1].ap50 == pytest.approx(1.0)  # the extra FP ranks below the TP


def test_average_precision_edge_cases():
    assert math.isnan(average_precision(np.array([]), np.array([], bool), 0))
    assert average_precision(np.array([]), np.array([], bool), 3) == 0.0
    # one TP then one FP, 2 GT: recall reaches 0.5 with precision 1
    assert average_precision(np.array([0.9, 0.8]), np.array([True, False]), 2) == pytest.approx(51 / 101)


def test_size_buckets():
    samples, preds = _toy()
    ev = evaluate(samples, preds, {0: "a", 1: "b"}, conf=0.5, n_boot=50)
    assert ev.size_recall["small"] == (1, 1)   # 10x10
    assert ev.size_recall["medium"] == (2, 2)  # 40x40 boxes


def test_gt_filter_restricts_ground_truth():
    samples, preds = _toy()
    ev = evaluate(samples, preds, {0: "a", 1: "b"}, conf=0.5, n_boot=50,
                  gt_filter=lambda s: s.classes == 0)
    assert ev.per_class[1].instances == 0


def test_calibration_error_perfect_and_overconfident():
    samples, preds = _toy()
    ev = evaluate(samples, preds, {0: "a", 1: "b"}, conf=0.5, n_boot=50)
    ece, bins = calibration_error(ev.matches, preds, 0.5, n_bins=5)
    # all TP, scores 0.6-0.9: under-confident, ECE = 1 - mean score
    assert ece == pytest.approx(1 - np.mean([0.9, 0.8, 0.6]))
    assert sum(b["n"] for b in bins) == 3


# ----------------------------------------------------------------- statistics


def test_stats_doctests():
    failures, _ = doctest.testmod(stats)
    assert failures == 0


def test_wilson_matches_article():
    lo, hi = stats.wilson_interval(873, 900)
    assert (round(lo, 4), round(hi, 4)) == (0.9567, 0.9793)
    assert round(stats.wilson_interval(194, 200, sided="lower")[0], 3) == 0.943


def test_clopper_pearson_is_wider_than_wilson():
    w = stats.wilson_interval(873, 900)
    c = stats.clopper_pearson_interval(873, 900)
    assert c[0] <= w[0] + 1e-9 and c[1] >= w[1] - 1e-9
    assert stats.clopper_pearson_interval(0, 10)[0] == 0.0


def test_zero_failure_upper_bound_is_rule_of_three():
    assert stats.zero_failure_upper_bound(300) == pytest.approx(3 / 300, rel=0.01)


def test_bootstrap_ratio_is_deterministic_and_brackets_estimate():
    rng = np.random.default_rng(1)
    gt = rng.integers(1, 6, 200)
    tp = rng.binomial(gt, 0.8)
    r1 = stats.bootstrap_ratio(tp, gt, n_boot=500, seed=3)
    r2 = stats.bootstrap_ratio(tp, gt, n_boot=500, seed=3)
    assert (r1.low, r1.high) == (r2.low, r2.high)
    assert r1.low < r1.estimate < r1.high
    assert 0.02 < r1.high - r1.low < 0.15


def test_mcnemar():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(0, 10) < 0.01
    assert stats.mcnemar_exact(5, 5) == pytest.approx(1.0)


def test_detections_validation():
    with pytest.raises(ValueError):
        Detections(np.zeros((2, 4)), np.zeros(1), np.zeros(2))
