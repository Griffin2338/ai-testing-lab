"""YOLO-format loader: one .txt per image, lines of `class cx cy w h` in 0..1.

An image without a label file has no objects.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from od_testkit.structures import Sample

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_class_names(spec) -> dict[int, str]:
    """Accepts a list, an {id: name} dict or a path to a YOLO data.yaml."""
    if isinstance(spec, (str, Path)):
        data = yaml.safe_load(Path(spec).read_text(encoding="utf-8"))
        spec = data["names"]
    if isinstance(spec, dict):
        return {int(k): str(v) for k, v in spec.items()}
    return {i: str(n) for i, n in enumerate(spec)}


def _image_size(path: Path) -> tuple[int, int]:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"cannot decode image: {path}")
    return img.shape[1], img.shape[0]


def default_labels_dir(images_dir: Path) -> Path:
    """.../images/<split> -> .../labels/<split>"""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_dir.parent / "labels"


def load_yolo_dataset(
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[Sample]:
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    labels_dir = Path(labels_dir) if labels_dir else default_labels_dir(images_dir)
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if limit:
        paths = paths[:limit]
    samples: list[Sample] = []
    for p in paths:
        w, h = _image_size(p)
        boxes, classes = [], []
        label_file = labels_dir / f"{p.stem}.txt"
        if label_file.exists():
            for line_no, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) < 5:
                    raise ValueError(f"{label_file}:{line_no}: expected 'class cx cy w h', got {line!r}")
                c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                classes.append(c)
        samples.append(
            Sample(image_id=p.stem, image_path=p, width=w, height=h,
                   boxes=np.array(boxes, dtype=np.float64), classes=np.array(classes, dtype=np.int64))
        )
    return samples


def load_unlabelled_images(images_dir: str | Path, limit: int | None = None) -> list[Sample]:
    """Negative images: no target objects expected."""
    images_dir = Path(images_dir)
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if limit:
        paths = paths[:limit]
    out = []
    for p in paths:
        w, h = _image_size(p)
        out.append(Sample(image_id=p.stem, image_path=p, width=w, height=h))
    return out
