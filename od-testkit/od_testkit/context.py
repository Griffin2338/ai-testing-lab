"""Per-run state: config, dataset, detector, and predictions cached per image variant."""

from __future__ import annotations

import time
import zlib
from collections import OrderedDict
from collections.abc import Callable, Sequence
from functools import cached_property
from pathlib import Path

import numpy as np

from od_testkit import __version__
from od_testkit.config import load_config, resolve
from od_testkit.dataset import load_unlabelled_images, load_yolo_dataset, read_class_names
from od_testkit.detectors import Detector, build_detector
from od_testkit.metrics import EvalResult, evaluate
from od_testkit.report import Report
from od_testkit.structures import Detections, Sample

ImageTransform = Callable[[np.ndarray, Sample], np.ndarray]
BoxUnmap = Callable[[np.ndarray, Sample], np.ndarray]


def stable_seed(*parts: object) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


class SuiteContext:
    def __init__(self, config_path: str | Path, out_dir: str | Path) -> None:
        self.cfg, self.base_dir = load_config(config_path)
        self.config_path = Path(config_path).resolve()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        op = self.cfg["operating_point"]
        self.conf = float(op["conf"])
        self.iou = float(op["iou"])
        self.min_instances = int(self.cfg["min_instances"])
        self.n_boot = int(self.cfg["bootstrap"]["n"])
        self.seed = int(self.cfg["bootstrap"]["seed"])
        self.confidence_level = float(self.cfg["confidence_level"])

        self.detector: Detector = build_detector(self.cfg["detector"], self.base_dir)
        ds = self.cfg["dataset"]
        self.samples: list[Sample] = load_yolo_dataset(
            resolve(self.base_dir, ds["images"]), resolve(self.base_dir, ds.get("labels")), ds.get("limit")
        )
        if not self.samples:
            raise ValueError("the dataset contains no images")
        names = ds.get("names")
        if isinstance(names, str):
            names = resolve(self.base_dir, names)
        self.class_names: dict[int, str] = read_class_names(names) if names else dict(self.detector.class_names)

        coi = self.cfg.get("classes_of_interest")
        if coi:
            by_name = {v: k for k, v in self.class_names.items()}
            missing = [c for c in coi if c not in by_name]
            if missing:
                raise ValueError(f"classes_of_interest not in class names: {missing}")
            self.classes_of_interest: set[int] | None = {by_name[c] for c in coi}
        else:
            self.classes_of_interest = None

        self._images: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_limit = int(self.cfg["cache_images"])
        self._preds: dict[str, list[Detections]] = {}
        self.timings: dict[str, float] = {}

        self.report = Report(
            name=self.cfg["name"],
            meta={
                "kit_version": __version__,
                "config": str(self.config_path),
                "detector": self.cfg["detector"].get("weights") or self.cfg["detector"]["type"],
                "images": len(self.samples),
                "objects": int(sum(len(s.boxes) for s in self.samples)),
                "operating_conf": self.conf,
                "iou_threshold": self.iou,
                "classes_of_interest": ", ".join(coi) if coi else "all",
                "min_instances_per_class": self.min_instances,
            },
        )
        self._check_names()

    # ------------------------------------------------------------------ helpers

    def _check_names(self) -> None:
        det_names = getattr(self.detector, "class_names", {}) or {}
        clash = [
            f"{k}: dataset={v!r} detector={det_names[k]!r}"
            for k, v in self.class_names.items()
            if k in det_names and det_names[k] != v
        ]
        if clash:
            self.report.warnings.append(
                "Dataset and detector use different names for the same class id (matching uses ids): "
                + "; ".join(clash[:10])
            )

    def image(self, idx: int) -> np.ndarray:
        if idx in self._images:
            self._images.move_to_end(idx)
            return self._images[idx]
        img = self.samples[idx].load_image()
        if self._cache_limit > 0:
            self._images[idx] = img
            while len(self._images) > self._cache_limit:
                self._images.popitem(last=False)
        return img

    def test_cfg(self, key: str) -> dict | None:
        """Config section of a test, or None if it isn't enabled."""
        tests = self.cfg.get("tests") or {}
        if key not in tests or tests[key] is False:
            return None
        return dict(tests[key] or {})

    def name_of(self, class_id: int) -> str:
        return self.class_names.get(class_id, str(class_id))

    # -------------------------------------------------------------- predictions

    def predictions(
        self,
        key: str,
        transform: ImageTransform | None = None,
        unmap: BoxUnmap | None = None,
        detector: Detector | None = None,
    ) -> list[Detections]:
        """Predict on every image after transform; unmap moves boxes back to original coords."""
        if key in self._preds:
            return self._preds[key]
        det = detector or self.detector
        out: list[Detections] = []
        t0 = time.perf_counter()
        for i, s in enumerate(self.samples):
            img = self.image(i)
            if transform is not None:
                img = transform(img, s)
            d = det.predict(img)
            if unmap is not None and len(d):
                d = d.with_boxes(unmap(d.boxes, s))
            out.append(d)
        self.timings[key] = time.perf_counter() - t0
        self._preds[key] = out
        return out

    @property
    def clean(self) -> list[Detections]:
        return self.predictions("clean")

    @cached_property
    def clean_eval(self) -> EvalResult:
        return self.evaluate(self.clean)

    def evaluate(
        self,
        preds: Sequence[Detections],
        gt_filter=None,
        compute_ap: bool = False,
        samples: Sequence[Sample] | None = None,
    ) -> EvalResult:
        return evaluate(
            samples or self.samples,
            preds,
            self.class_names,
            self.conf,
            self.iou,
            gt_filter=gt_filter,
            compute_ap=compute_ap,
            n_boot=self.n_boot,
            seed=self.seed,
        )

    @cached_property
    def clean_eval_with_ap(self) -> EvalResult:
        return self.evaluate(self.clean, compute_ap=True)

    def negatives(self, cfg: dict) -> list[Sample]:
        if cfg.get("dir"):
            return load_unlabelled_images(resolve(self.base_dir, cfg["dir"]), cfg.get("limit"))
        return []
