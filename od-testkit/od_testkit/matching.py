"""Greedy class-aware matching (COCO style) and TIDE error types for false positives.

TIDE: Bolya et al., ECCV 2020.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from od_testkit.geometry import iou_matrix
from od_testkit.structures import Detections

TP = 0
DUPLICATE = 1
LOCALIZATION = 2
CLASS_CONFUSION = 3
BACKGROUND = 4

ERROR_NAMES = {
    DUPLICATE: "duplicate",
    LOCALIZATION: "localization",
    CLASS_CONFUSION: "class_confusion",
    BACKGROUND: "background",
}

LOCALIZATION_MIN_IOU = 0.1


@dataclass
class ImageMatch:
    kind: np.ndarray  # per prediction: TP or an error code
    gt_matched: np.ndarray  # per ground-truth box: bool
    matched_gt: np.ndarray  # per prediction: index of matched GT or -1
    ious: np.ndarray  # (n_pred, n_gt)

    @property
    def is_tp(self) -> np.ndarray:
        return self.kind == TP


def match_image(
    pred: Detections,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_thr: float = 0.5,
) -> ImageMatch:
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    gt_classes = np.asarray(gt_classes, dtype=np.int64).reshape(-1)
    n_pred, n_gt = len(pred), len(gt_boxes)
    ious = iou_matrix(pred.boxes, gt_boxes)
    kind = np.full(n_pred, BACKGROUND, dtype=np.int64)
    matched_gt = np.full(n_pred, -1, dtype=np.int64)
    gt_matched = np.zeros(n_gt, dtype=bool)

    for i in np.argsort(-pred.scores, kind="stable"):
        if n_gt == 0:
            continue
        same = gt_classes == pred.classes[i]
        cand = np.where(same & ~gt_matched, ious[i], -1.0)
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            gt_matched[j] = True
            kind[i] = TP
            matched_gt[i] = j
            continue
        if np.any(same & gt_matched & (ious[i] >= iou_thr)):
            kind[i] = DUPLICATE
        elif np.any(same & (ious[i] >= LOCALIZATION_MIN_IOU)):
            kind[i] = LOCALIZATION
        elif np.any(~same & (ious[i] >= iou_thr)):
            kind[i] = CLASS_CONFUSION
        else:
            kind[i] = BACKGROUND
    return ImageMatch(kind=kind, gt_matched=gt_matched, matched_gt=matched_gt, ious=ious)


def preservation_rate(
    reference: Detections, candidate: Detections, iou_thr: float
) -> tuple[int, int]:
    """Count reference boxes found again in candidate (same class, IoU >= iou_thr).

    Returns (preserved, total). A candidate box can match only once.
    """
    if len(reference) == 0:
        return 0, 0
    ious = iou_matrix(reference.boxes, candidate.boxes)
    used = np.zeros(len(candidate), dtype=bool)
    preserved = 0
    for i in np.argsort(-reference.scores, kind="stable"):
        if len(candidate) == 0:
            break
        same = candidate.classes == reference.classes[i]
        cand = np.where(same & ~used, ious[i], -1.0)
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            used[j] = True
            preserved += 1
    return preserved, len(reference)
