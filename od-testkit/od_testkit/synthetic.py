"""Synthetic dataset for the demo and CI.

Red and blue rectangles/discs on noisy grey backgrounds. negatives/ holds images with
clutter and orange/purple shapes only.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from od_testkit.structures import write_image
from od_testkit.transforms import find_free_position

CLASS_NAMES = {0: "red", 1: "blue"}


def _background(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    level = int(rng.integers(90, 200))
    img = np.full((h, w, 3), level, np.float32) + rng.normal(0, 8, (h, w, 3))
    for _ in range(int(rng.integers(0, 4))):  # low-saturation clutter
        g = int(rng.integers(40, 230))
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, (g, g, g), int(rng.integers(1, 4)))
    return np.clip(img, 0, 255).astype(np.uint8)


def _colour(rng: np.random.Generator, kind: str) -> tuple[int, int, int]:
    v = int(rng.integers(170, 251))
    lo = lambda: int(rng.integers(15, 60))  # noqa: E731
    if kind == "red":
        return (lo(), lo(), v)
    if kind == "blue":
        return (v, int(rng.integers(40, 80)), lo())
    if kind == "orange":
        return (lo(), int(v * 0.45), v)
    if kind == "purple":
        return (v, lo(), int(v * 0.75))
    raise ValueError(kind)


def _draw_shape(img, rng, kind: str, occupied: list) -> np.ndarray | None:
    s = int(round(np.exp(rng.uniform(np.log(12), np.log(96)))))  # log-uniform 12..96 px
    aspect = rng.uniform(0.7, 1.4) if rng.random() < 0.5 else 1.0
    w, h = max(8, int(s * aspect)), s
    disc = rng.random() < 0.4
    if disc:
        w = h = s if s % 2 else s + 1
    pos = find_free_position(img.shape, (h, w), np.array(occupied).reshape(-1, 4), 4, rng)
    if pos is None:
        return None
    x, y = pos
    colour = _colour(rng, kind)
    if disc:
        r = w // 2
        cv2.circle(img, (x + r, y + r), r, colour, -1, lineType=cv2.LINE_8)
        box = np.array([x, y, x + 2 * r + 1, y + 2 * r + 1], float)
    else:
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), colour, -1)
        box = np.array([x, y, x + w, y + h], float)
    occupied.append(box)
    return box


def generate_dataset(
    out_dir: str | Path,
    n_images: int = 200,
    n_negatives: int = 60,
    size: tuple[int, int] = (320, 320),
    seed: int = 0,
) -> Path:
    """Writes images/val, labels/val, negatives/ and data.yaml under out_dir."""
    out = Path(out_dir)
    rng = np.random.default_rng(seed)
    h, w = size
    for i in range(n_images):
        img = _background(rng, size)
        occupied: list = []
        lines = []
        for _ in range(int(rng.integers(1, 7))):
            cls = int(rng.integers(0, 2))
            box = _draw_shape(img, rng, CLASS_NAMES[cls], occupied)
            if box is None:
                continue
            cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
            bw, bh = (box[2] - box[0]) / w, (box[3] - box[1]) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        write_image(out / "images" / "val" / f"syn_{i:04d}.png", img)
        label = out / "labels" / "val" / f"syn_{i:04d}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    for i in range(n_negatives):
        img = _background(rng, size)
        occupied = []
        for _ in range(int(rng.integers(1, 4))):
            _draw_shape(img, rng, "orange" if rng.random() < 0.5 else "purple", occupied)
        write_image(out / "negatives" / f"neg_{i:04d}.png", img)

    (out / "data.yaml").write_text(
        yaml.safe_dump({"path": ".", "val": "images/val", "names": CLASS_NAMES}, sort_keys=False),
        encoding="utf-8",
    )
    return out
