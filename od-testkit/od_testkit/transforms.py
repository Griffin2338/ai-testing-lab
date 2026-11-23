"""Image transforms for the robustness and metamorphic checks.

Each geometric transform has a matching function that maps boxes back to the original
image. Corruptions follow ImageNet-C / COCO-C (Hendrycks & Dietterich 2019, Michaelis
et al. 2019), five severities each, reimplemented with numpy and OpenCV. Compare scores
between runs of this kit only; they won't match published ImageNet-C numbers.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

# ---------------------------------------------------------------- photometric


def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """0.6 = 40 % darker, 1.4 = 40 % brighter."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _to01(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def _from01(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _scale_for(img: np.ndarray) -> float:
    # kernel sizes below assume ~320 px images
    return max(1.0, min(img.shape[:2]) / 320.0)


def gaussian_noise(img, severity, rng):
    sigma = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    x = _to01(img)
    return _from01(x + rng.normal(0, sigma, x.shape))


def shot_noise(img, severity, rng):
    lam = [60, 25, 12, 5, 3][severity - 1]
    x = _to01(img)
    return _from01(rng.poisson(x * lam) / lam)


def impulse_noise(img, severity, rng):
    amount = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    out = img.copy()
    mask = rng.random(img.shape[:2])
    out[mask < amount / 2] = 0
    out[mask > 1 - amount / 2] = 255
    return out


def defocus_blur(img, severity, rng):
    radius = int(round([3, 4, 6, 8, 10][severity - 1] * _scale_for(img)))
    k = 2 * radius + 1
    kernel = np.zeros((k, k), np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def motion_blur(img, severity, rng):
    length = int(round([10, 15, 15, 15, 20][severity - 1] * _scale_for(img)))
    angle = float(rng.uniform(-45, 45))
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def jpeg_compression(img, severity, rng):
    quality = [25, 18, 15, 10, 7][severity - 1]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def pixelate(img, severity, rng):
    f = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def contrast(img, severity, rng):
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    x = _to01(img)
    mean = x.mean(axis=(0, 1), keepdims=True)
    return _from01((x - mean) * c + mean)


def fog(img, severity, rng):
    alpha = [0.3, 0.4, 0.5, 0.6, 0.7][severity - 1]
    h, w = img.shape[:2]
    coarse = rng.random((max(2, h // 32), max(2, w // 32))).astype(np.float32)
    fogmap = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    fogmap = cv2.GaussianBlur(fogmap, (0, 0), sigmaX=max(h, w) / 30)
    fogmap = (fogmap - fogmap.min()) / max(float(np.ptp(fogmap)), 1e-6)
    x = _to01(img)
    return _from01(x * (1 - alpha) + alpha * (0.6 + 0.4 * fogmap[..., None]))


def low_light(img, severity, rng):
    factor = [0.7, 0.55, 0.4, 0.3, 0.2][severity - 1]
    x = _to01(img) ** 1.2 * factor
    return _from01(x + rng.normal(0, 0.01 * severity, x.shape))


CorruptionFn = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]

CORRUPTIONS: dict[str, CorruptionFn] = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "defocus_blur": defocus_blur,
    "motion_blur": motion_blur,
    "jpeg_compression": jpeg_compression,
    "pixelate": pixelate,
    "contrast": contrast,
    "fog": fog,
    "low_light": low_light,
}


def corrupt(img: np.ndarray, name: str, severity: int, seed: int = 0) -> np.ndarray:
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption {name!r}; available: {sorted(CORRUPTIONS)}")
    if not 1 <= severity <= 5:
        raise ValueError("severity must be 1..5")
    return CORRUPTIONS[name](img, severity, np.random.default_rng(seed))


# ---------------------------------------------------------------- geometric


def hflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[:, ::-1])


def unflip_boxes(boxes: np.ndarray, width: int) -> np.ndarray:
    """Undo hflip() for predicted boxes."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = width - b[:, [2, 0]]
    return b


def rescale(img: np.ndarray, factor: float) -> tuple[np.ndarray, float, float]:
    """Returns (image, sx, sy). sx/sy differ slightly from factor after rounding."""
    h, w = img.shape[:2]
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp), nw / w, nh / h


def unscale_boxes(boxes: np.ndarray, sx: float, sy: float) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] /= sx
    b[:, [1, 3]] /= sy
    return b


def pad(img: np.ndarray, fraction: float, value: int = 114) -> tuple[np.ndarray, int]:
    """Constant border of fraction * max(h, w) px on each side. Returns (image, offset)."""
    p = int(round(fraction * max(img.shape[:2])))
    out = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(value, value, value))
    return out, p


def unpad_boxes(boxes: np.ndarray, offset: int) -> np.ndarray:
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4) - offset


# ---------------------------------------------------------------- object insertion


def find_free_position(
    image_shape: tuple[int, ...],
    patch_shape: tuple[int, ...],
    occupied: np.ndarray,
    margin: int,
    rng: np.random.Generator,
    tries: int = 100,
) -> tuple[int, int] | None:
    """Random top-left corner at least margin px from all occupied boxes, or None."""
    h, w = image_shape[:2]
    ph, pw = patch_shape[:2]
    if ph > h or pw > w:
        return None
    occ = np.asarray(occupied, dtype=np.float64).reshape(-1, 4)
    for _ in range(tries):
        x = int(rng.integers(0, w - pw + 1))
        y = int(rng.integers(0, h - ph + 1))
        cand = np.array([x - margin, y - margin, x + pw + margin, y + ph + margin], dtype=np.float64)
        if len(occ) == 0:
            return x, y
        overlap_x = (occ[:, 0] < cand[2]) & (occ[:, 2] > cand[0])
        overlap_y = (occ[:, 1] < cand[3]) & (occ[:, 3] > cand[1])
        if not np.any(overlap_x & overlap_y):
            return x, y
    return None


def paste_patch(img: np.ndarray, patch: np.ndarray, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (new image, pasted box)."""
    out = img.copy()
    ph, pw = patch.shape[:2]
    out[y : y + ph, x : x + pw] = patch
    return out, np.array([x, y, x + pw, y + ph], dtype=np.float64)
