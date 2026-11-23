"""Box helpers. Boxes are x1, y1, x2, y2 in pixels."""

from __future__ import annotations

import numpy as np


def box_area(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, shape (len(a), len(b))."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, width)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, height)
    return b


def expand_boxes(boxes: np.ndarray, margin: float) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, :2] -= margin
    b[:, 2:] += margin
    return b
