"""Adapter for a detector behind an HTTP endpoint.

Config:
    detector:
      type: python
      target: examples/http_detector.py:HttpDetector
      kwargs:
        url: http://localhost:8000/detect
        class_names: [car, truck, bus, pedestrian]

Response format (change _parse for your API):
    {"detections": [{"box": [x1, y1, x2, y2], "class_id": 0, "score": 0.93}, ...]}

AP needs low-score boxes too. If the service filters by score, ask for a test flag that
returns everything above ~0.001.
"""

from __future__ import annotations

import json
import urllib.request

import cv2
import numpy as np

from od_testkit.structures import Detections


class HttpDetector:
    def __init__(self, url: str, class_names: list[str] | dict, timeout: float = 30.0):
        self.url = url
        self.timeout = timeout
        self.class_names = (dict(enumerate(class_names)) if isinstance(class_names, list)
                            else {int(k): v for k, v in class_names.items()})

    def predict(self, image: np.ndarray) -> Detections:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        req = urllib.request.Request(self.url, data=buf.tobytes(), headers={"Content-Type": "image/jpeg"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return self._parse(json.loads(r.read()))

    @staticmethod
    def _parse(payload: dict) -> Detections:
        dets = payload.get("detections", [])
        if not dets:
            return Detections.empty()
        return Detections(
            np.array([d["box"] for d in dets], dtype=float),
            np.array([d["class_id"] for d in dets], dtype=int),
            np.array([d["score"] for d in dets], dtype=float),
        )
