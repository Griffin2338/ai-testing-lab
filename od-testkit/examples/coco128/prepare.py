"""Fetch COCO128 and yolo11n.pt (Ultralytics assets, ~12 MB) and export ONNX.

ONNX export needs: pip install onnx onnxruntime onnxslim
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def download_coco128() -> None:
    if (DATA / "coco128" / "images").exists():
        print("coco128 already present")
        return
    DATA.mkdir(parents=True, exist_ok=True)
    zip_path = DATA / "coco128.zip"
    print(f"downloading {COCO128_URL}")
    with urllib.request.urlopen(COCO128_URL, timeout=120) as r, open(zip_path, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():  # no path traversal
            if member.startswith("/") or ".." in Path(member).parts:
                raise RuntimeError(f"unsafe path in archive: {member}")
        z.extractall(DATA)
    zip_path.unlink()


def prepare_model() -> None:
    from ultralytics import YOLO

    pt = HERE / "yolo11n.pt"
    model = YOLO(str(pt) if pt.exists() else "yolo11n.pt")
    if not pt.exists():
        src = Path("yolo11n.pt")
        if src.exists() and src.resolve() != pt:
            shutil.move(str(src), pt)
    onnx = HERE / "yolo11n.onnx"
    if not onnx.exists():
        out = Path(YOLO(str(pt)).export(format="onnx", imgsz=640, dynamic=False, simplify=True))
        if out.resolve() != onnx:
            shutil.move(str(out), onnx)
    print(f"model ready: {pt.name}, {onnx.name}")
    del model


if __name__ == "__main__":
    download_coco128()
    prepare_model()
