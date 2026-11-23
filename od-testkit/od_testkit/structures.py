from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


def _as_boxes(a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    return arr.reshape(-1, 4) if arr.size else np.zeros((0, 4), dtype=np.float64)


@dataclass
class Detections:
    """Detections for one image. Boxes are x1, y1, x2, y2 in pixels."""

    boxes: np.ndarray
    classes: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        self.boxes = _as_boxes(self.boxes)
        self.classes = np.asarray(self.classes, dtype=np.int64).reshape(-1)
        self.scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        n = len(self.boxes)
        if len(self.classes) != n or len(self.scores) != n:
            raise ValueError(
                f"boxes/classes/scores length mismatch: {n}/{len(self.classes)}/{len(self.scores)}"
            )

    @classmethod
    def empty(cls) -> Detections:
        return cls(np.zeros((0, 4)), np.zeros(0, dtype=np.int64), np.zeros(0))

    def __len__(self) -> int:
        return len(self.boxes)

    def select(self, mask: np.ndarray) -> Detections:
        return Detections(self.boxes[mask], self.classes[mask], self.scores[mask])

    def above(self, conf: float) -> Detections:
        return self.select(self.scores >= conf)

    def of_classes(self, classes: set[int] | None) -> Detections:
        if classes is None:
            return self
        return self.select(np.isin(self.classes, list(classes)))

    def with_boxes(self, boxes: np.ndarray) -> Detections:
        return Detections(boxes, self.classes.copy(), self.scores.copy())


@dataclass
class Sample:
    """A labelled test image. GT boxes are x1, y1, x2, y2 in pixels."""

    image_id: str
    image_path: Path
    width: int
    height: int
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    classes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def __post_init__(self) -> None:
        self.boxes = _as_boxes(self.boxes)
        self.classes = np.asarray(self.classes, dtype=np.int64).reshape(-1)

    def load_image(self) -> np.ndarray:
        return read_image(self.image_path)


def read_image(path: str | Path) -> np.ndarray:
    # cv2.imread can't open non-ASCII paths on Windows
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"cannot decode image: {path}")
    return img


def write_image(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise OSError(f"cannot encode image: {path}")
    buf.tofile(str(path))
