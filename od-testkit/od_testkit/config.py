"""Config loading. Relative paths resolve against the config file's folder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "name": "Object detection acceptance run",
    "operating_point": {"conf": 0.25, "iou": 0.5},
    "classes_of_interest": None,
    "min_instances": 200,
    "bootstrap": {"n": 1000, "seed": 0},
    "confidence_level": 0.95,
    "cache_images": 600,
    "tests": {},
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in ("dataset", "detector"):
        if key not in raw:
            raise ValueError(f"{path}: missing required section '{key}'")
    return _merge(DEFAULTS, raw), path.parent


def resolve(base_dir: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p).resolve()
