"""Detector adapters.

An adapter needs predict(image_bgr) -> Detections and a class_names dict. Return
everything above a low score floor; the checks apply the operating threshold, and AP
needs the low-score boxes. For other runtimes or a model behind an API, use
`type: python` in the config (see examples/http_detector.py).
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from od_testkit.structures import Detections


@runtime_checkable
class Detector(Protocol):
    class_names: dict[int, str]

    def predict(self, image: np.ndarray) -> Detections: ...


class UltralyticsDetector:
    """Anything ultralytics.YOLO can load: .pt, .onnx, .engine, ..."""

    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        conf_floor: float = 0.001,
        nms_iou: float = 0.7,
        max_det: int = 300,
        device: str | None = None,
        half: bool = False,
    ) -> None:
        from ultralytics import YOLO  # optional dependency

        self.model = YOLO(weights)
        self.kwargs = dict(imgsz=imgsz, conf=conf_floor, iou=nms_iou, max_det=max_det, verbose=False)
        if device is not None:
            self.kwargs["device"] = device
        if half:
            self.kwargs["half"] = True
        names = self.model.names
        self.class_names = {int(k): str(v) for k, v in (names.items() if isinstance(names, dict)
                                                        else enumerate(names))}

    def predict(self, image: np.ndarray) -> Detections:
        r = self.model.predict(image, **self.kwargs)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return Detections.empty()
        return Detections(
            b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(np.int64), b.conf.cpu().numpy()
        )


class ShapeDetector:
    """HSV colour-blob detector for the synthetic demo.

    Known bugs, kept on purpose so the demo has something to find:
      - v_min drops dark pixels, so recall falls apart on darker images
      - min_area drops small blobs
      - the red hue band (<= 12) also matches orange
    """

    class_names = {0: "red", 1: "blue"}

    def __init__(self, v_min: int = 140, s_min: int = 100, min_area: int = 250, downsample: int = 1):
        self.v_min, self.s_min, self.min_area, self.downsample = v_min, s_min, min_area, downsample

    def predict(self, image: np.ndarray) -> Detections:
        f = max(1, int(self.downsample))
        work = image
        if f > 1:
            h, w = image.shape[:2]
            work = cv2.resize(image, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        base = (sat >= self.s_min) & (val >= self.v_min)
        masks = {0: base & ((hue <= 12) | (hue >= 170)), 1: base & (hue >= 100) & (hue <= 130)}
        kernel = np.ones((3, 3), np.uint8)
        boxes, classes, scores = [], [], []
        for cls, m in masks.items():
            m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h * f * f
                if area < self.min_area:
                    continue
                fill = cv2.contourArea(cnt) / max(w * h, 1)
                score = float(np.clip(0.45 + 0.5 * min(1.0, area / 3000.0) + 0.05 * fill, 0.0, 0.99))
                boxes.append([x * f, y * f, (x + w) * f, (y + h) * f])
                classes.append(cls)
                scores.append(score)
        if not boxes:
            return Detections.empty()
        return Detections(np.array(boxes, float), np.array(classes), np.array(scores))


def _load_target(target: str, base_dir: Path) -> Any:
    """'package.module:attr' or 'path/to/file.py:attr'"""
    module_part, _, attr = target.partition(":")
    if not attr:
        raise ValueError(f"python detector target must look like 'module:attr', got {target!r}")
    if module_part.endswith(".py"):
        path = Path(module_part)
        if not path.is_absolute():
            path = base_dir / path
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)
    return getattr(module, attr)


def build_detector(cfg: dict, base_dir: Path) -> Detector:
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind == "ultralytics":
        weights = cfg.pop("weights")
        wp = Path(weights)
        if not wp.is_absolute() and (base_dir / wp).exists():
            weights = str(base_dir / wp)
        return UltralyticsDetector(weights, **cfg)
    if kind == "shapes":
        return ShapeDetector(**cfg)
    if kind == "python":
        factory = _load_target(cfg.pop("target"), base_dir)
        det = factory(**cfg.pop("kwargs", {}), **cfg)
        if not isinstance(det, Detector):
            raise TypeError(f"{factory!r} must return an object with predict() and class_names")
        return det
    raise ValueError(f"unknown detector type {kind!r} (use ultralytics, shapes or python)")
