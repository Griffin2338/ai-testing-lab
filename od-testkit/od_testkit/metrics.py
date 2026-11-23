"""Metrics at the operating threshold, AP, size buckets and detection ECE."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from od_testkit.geometry import box_area
from od_testkit.matching import ERROR_NAMES, ImageMatch, match_image
from od_testkit.stats import bootstrap_ratio
from od_testkit.structures import Detections, Sample

SIZE_BUCKETS = {"small": (0.0, 32.0**2), "medium": (32.0**2, 96.0**2), "large": (96.0**2, float("inf"))}
COCO_IOUS = np.round(np.arange(0.5, 0.951, 0.05), 2)

GtFilter = Callable[[Sample], np.ndarray]  # bool mask over sample.boxes


@dataclass
class ClassStats:
    class_id: int
    name: str
    instances: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    recall_ci: tuple[float, float]
    precision_ci: tuple[float, float]
    ap50: float = float("nan")
    ap50_95: float = float("nan")


@dataclass
class EvalResult:
    conf: float
    iou_thr: float
    per_class: dict[int, ClassStats]
    per_image_tp: np.ndarray  # (n_images, n_classes)
    per_image_fp: np.ndarray
    per_image_gt: np.ndarray
    fp_breakdown: dict[str, int]
    size_recall: dict[str, tuple[int, int]]  # bucket -> (tp, gt)
    matches: list[ImageMatch] = field(default_factory=list)
    class_index: dict[int, int] = field(default_factory=dict)

    def pooled(self, classes: set[int] | None = None) -> tuple[int, int, int]:
        """(tp, fp, gt) summed over classes. None means all classes."""
        cols = self._cols(classes)
        return (
            int(self.per_image_tp[:, cols].sum()),
            int(self.per_image_fp[:, cols].sum()),
            int(self.per_image_gt[:, cols].sum()),
        )

    def pooled_recall(self, classes: set[int] | None = None) -> float:
        tp, _, gt = self.pooled(classes)
        return tp / gt if gt else float("nan")

    def pooled_precision(self, classes: set[int] | None = None) -> float:
        tp, fp, _ = self.pooled(classes)
        return tp / (tp + fp) if (tp + fp) else float("nan")

    def _cols(self, classes: set[int] | None) -> list[int]:
        if classes is None:
            return list(range(self.per_image_tp.shape[1]))
        return [self.class_index[c] for c in classes if c in self.class_index]


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def average_precision(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """COCO-style 101-point interpolated AP for one class."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(is_tp, dtype=bool)[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(~tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    q = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(q.mean())


def _gt_of(sample: Sample, gt_filter: GtFilter | None) -> tuple[np.ndarray, np.ndarray]:
    if gt_filter is None:
        return sample.boxes, sample.classes
    m = np.asarray(gt_filter(sample), dtype=bool)
    return sample.boxes[m], sample.classes[m]


def evaluate(
    samples: Sequence[Sample],
    predictions: Sequence[Detections],
    class_names: dict[int, str],
    conf: float,
    iou_thr: float = 0.5,
    *,
    gt_filter: GtFilter | None = None,
    compute_ap: bool = True,
    n_boot: int = 1000,
    seed: int = 0,
) -> EvalResult:
    if len(samples) != len(predictions):
        raise ValueError("one prediction set per sample is required")
    class_ids = sorted(class_names)
    cidx = {c: i for i, c in enumerate(class_ids)}
    n_img, n_cls = len(samples), len(class_ids)
    tp_m = np.zeros((n_img, n_cls))
    fp_m = np.zeros((n_img, n_cls))
    gt_m = np.zeros((n_img, n_cls))
    fp_breakdown = {name: 0 for name in ERROR_NAMES.values()}
    size_tp = {k: 0 for k in SIZE_BUCKETS}
    size_gt = {k: 0 for k in SIZE_BUCKETS}
    matches: list[ImageMatch] = []

    for i, (s, raw) in enumerate(zip(samples, predictions, strict=True)):
        gt_b, gt_c = _gt_of(s, gt_filter)
        pred = raw.above(conf)
        m = match_image(pred, gt_b, gt_c, iou_thr)
        matches.append(m)
        for c in gt_c:
            if c in cidx:
                gt_m[i, cidx[c]] += 1
        for c, k in zip(pred.classes, m.kind, strict=True):
            if c not in cidx:
                continue
            if k == 0:
                tp_m[i, cidx[c]] += 1
            else:
                fp_m[i, cidx[c]] += 1
                fp_breakdown[ERROR_NAMES[int(k)]] += 1
        areas = box_area(gt_b)
        for bucket, (lo, hi) in SIZE_BUCKETS.items():
            sel = (areas >= lo) & (areas < hi)
            size_gt[bucket] += int(sel.sum())
            size_tp[bucket] += int((sel & m.gt_matched).sum())

    rec_ci = bootstrap_ratio(tp_m, gt_m, n_boot=n_boot, seed=seed)
    prec_ci = bootstrap_ratio(tp_m, tp_m + fp_m, n_boot=n_boot, seed=seed)

    ap50: dict[int, float] = {}
    ap5095: dict[int, float] = {}
    if compute_ap:
        ap50, ap5095 = _ap_per_class(samples, predictions, class_ids, gt_filter)

    per_class: dict[int, ClassStats] = {}
    for c in class_ids:
        j = cidx[c]
        tp, fp, gt = int(tp_m[:, j].sum()), int(fp_m[:, j].sum()), int(gt_m[:, j].sum())
        p, r = _safe_div(tp, tp + fp), _safe_div(tp, gt)
        f1 = _safe_div(2 * p * r, p + r) if not (np.isnan(p) or np.isnan(r)) else float("nan")
        per_class[c] = ClassStats(
            class_id=c,
            name=class_names[c],
            instances=gt,
            tp=tp,
            fp=fp,
            fn=gt - tp,
            precision=p,
            recall=r,
            f1=f1,
            recall_ci=(float(rec_ci.low[j]), float(rec_ci.high[j])),
            precision_ci=(float(prec_ci.low[j]), float(prec_ci.high[j])),
            ap50=ap50.get(c, float("nan")),
            ap50_95=ap5095.get(c, float("nan")),
        )

    return EvalResult(
        conf=conf,
        iou_thr=iou_thr,
        per_class=per_class,
        per_image_tp=tp_m,
        per_image_fp=fp_m,
        per_image_gt=gt_m,
        fp_breakdown=fp_breakdown,
        size_recall={k: (size_tp[k], size_gt[k]) for k in SIZE_BUCKETS},
        matches=matches,
        class_index=cidx,
    )


def _ap_per_class(
    samples: Sequence[Sample],
    predictions: Sequence[Detections],
    class_ids: list[int],
    gt_filter: GtFilter | None,
) -> tuple[dict[int, float], dict[int, float]]:
    """AP at IoU 0.50:0.95 over all predictions, without the operating threshold."""
    gts = [_gt_of(s, gt_filter) for s in samples]
    n_gt = {c: sum(int((gc == c).sum()) for _, gc in gts) for c in class_ids}
    ap_by_iou: dict[float, dict[int, float]] = {}
    for t in COCO_IOUS:
        scores: dict[int, list[float]] = {c: [] for c in class_ids}
        hits: dict[int, list[bool]] = {c: [] for c in class_ids}
        for (gb, gc), pred in zip(gts, predictions, strict=True):
            m = match_image(pred, gb, gc, float(t))
            for c, sc, k in zip(pred.classes, pred.scores, m.kind, strict=True):
                if c in scores:
                    scores[c].append(float(sc))
                    hits[c].append(bool(k == 0))
        ap_by_iou[float(t)] = {
            c: average_precision(np.array(scores[c]), np.array(hits[c], dtype=bool), n_gt[c])
            for c in class_ids
        }
    ap50 = ap_by_iou[0.5]
    ap5095 = {}
    for c in class_ids:
        vals = [ap_by_iou[float(t)][c] for t in COCO_IOUS]
        ap5095[c] = float("nan") if np.isnan(vals[0]) else float(np.mean(vals))
    return ap50, ap5095


def mean_ap(result: EvalResult, classes: set[int] | None = None, key: str = "ap50") -> float:
    vals = [
        getattr(cs, key)
        for c, cs in result.per_class.items()
        if (classes is None or c in classes) and not np.isnan(getattr(cs, key))
    ]
    return float(np.mean(vals)) if vals else float("nan")


def calibration_error(
    matches: Sequence[ImageMatch],
    predictions: Sequence[Detections],
    conf: float,
    n_bins: int = 10,
    classes: set[int] | None = None,
) -> tuple[float, list[dict]]:
    """Detection ECE for boxes with score >= conf.

    Scores are binned between conf and 1; each bin compares mean score with precision.
    matches must come from evaluate() run with the same conf.
    """
    scores, hits = [], []
    for m, raw in zip(matches, predictions, strict=True):
        pred = raw.above(conf)
        keep = np.ones(len(pred), dtype=bool) if classes is None else np.isin(pred.classes, list(classes))
        scores.extend(pred.scores[keep].tolist())
        hits.extend((m.kind[keep] == 0).tolist())
    s, h = np.array(scores), np.array(hits, dtype=float)
    if len(s) == 0:
        return float("nan"), []
    edges = np.linspace(conf, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_bins - 1)
    ece, bins = 0.0, []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        mean_conf, prec = float(s[sel].mean()), float(h[sel].mean())
        ece += n / len(s) * abs(prec - mean_conf)
        bins.append({"bin": f"{edges[b]:.2f}-{edges[b + 1]:.2f}", "n": n,
                     "mean_score": mean_conf, "precision": prec})
    return float(ece), bins
