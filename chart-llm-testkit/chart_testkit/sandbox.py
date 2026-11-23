"""Run generated code in a separate Python process and collect the chart spec."""

from __future__ import annotations

import base64
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_MPL_CACHE = Path(tempfile.gettempdir()) / "chart_testkit_mplcache"


@dataclass
class RunResult:
    ok: bool
    error: str | None
    spec: dict = field(default_factory=dict)
    png_b64: str | None = None
    blocked: list[str] = field(default_factory=list)
    seconds: float = 0.0
    timed_out: bool = False
    stderr_tail: str = ""


def _child_env() -> dict[str, str]:
    # no API keys, tokens or proxies reach the generated code
    keep = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "LANG", "LC_ALL")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["MPLCONFIGDIR"] = str(_MPL_CACHE)
    env["MPLBACKEND"] = "Agg"
    return env


def _limits(memory_mb: int):
    if os.name != "posix" or not memory_mb:
        return None

    def apply():
        import resource

        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return apply


def run_code(code: str, df: pd.DataFrame, timeout_s: float = 30.0, memory_mb: int = 2048) -> RunResult:
    """Execute code with `df` in scope. Memory limit applies on POSIX only."""
    _MPL_CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="chart_run_") as wd:
        Path(wd, "code.py").write_text(code, encoding="utf-8")
        with open(Path(wd, "input.pkl"), "wb") as f:
            pickle.dump(df, f)
        cmd = [sys.executable, "-I", "-m", "chart_testkit._child", wd]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=_child_env(),
                                  cwd=wd, preexec_fn=_limits(memory_mb), check=False)
        except subprocess.TimeoutExpired:
            return RunResult(False, f"timeout after {timeout_s:.0f}s", timed_out=True,
                             seconds=time.perf_counter() - t0)
        res_file = Path(wd, "result.json")
        if not res_file.exists():
            tail = (proc.stderr or "")[-800:]
            return RunResult(False, f"child crashed (exit {proc.returncode})", stderr_tail=tail,
                             seconds=time.perf_counter() - t0)
        data = json.loads(res_file.read_text(encoding="utf-8"))
        png = Path(wd, "chart.png")
        return RunResult(
            ok=bool(data.get("ok")),
            error=data.get("error") or data.get("extract_error"),
            spec=data.get("spec") or {},
            png_b64=base64.b64encode(png.read_bytes()).decode() if png.exists() else None,
            blocked=list(data.get("blocked") or []),
            seconds=time.perf_counter() - t0,
            stderr_tail=(proc.stderr or "")[-800:],
        )
