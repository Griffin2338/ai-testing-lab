"""Sandbox child process: run generated code under an audit-hook guard, dump the chart spec.

Usage: python -I -m chart_testkit._child WORKDIR

The guard is a test harness, not a security boundary. Code that wants to escape can find
ways (gc, native extensions). Run production code in a container without network.
"""

from __future__ import annotations

import json
import os
import pickle
import site
import sys
import time

BLOCKED_EVENTS = {
    "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto",
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.startfile",
    "os.fork", "os.forkpty", "os.kill", "ctypes.dlopen", "ctypes.dlsym", "webbrowser.open",
    "urllib.Request", "http.client.connect", "ftplib.connect", "smtplib.connect",
    "winreg.OpenKey", "winreg.CreateKey",
}
PATH_EVENTS = {"os.remove", "os.rename", "os.rmdir", "os.mkdir", "shutil.rmtree", "os.chmod", "os.truncate"}
LIST_EVENTS = {"os.listdir", "os.scandir"}


def _norm(p) -> str:
    return os.path.normcase(os.path.abspath(os.fsdecode(p)))


def _under(path: str, roots: list[str]) -> bool:
    return any(path == r or path.startswith(r + os.sep) for r in roots)


def _font_dirs() -> list[str]:
    dirs = ["/usr/share/fonts", "/usr/local/share/fonts", "/Library/Fonts", "/System/Library/Fonts"]
    windir = os.environ.get("WINDIR") or os.environ.get("SYSTEMROOT")
    if windir:
        dirs.append(os.path.join(windir, "Fonts"))
    return dirs


def install_guard(workdir: str, read_roots: list[str], write_roots: list[str], blocked: list[str]) -> None:
    reads = [_norm(r) for r in read_roots]
    writes = [_norm(w) for w in write_roots]

    def deny(event: str, detail: str = "") -> None:
        msg = f"{event} {detail}".strip()
        blocked.append(msg[:200])
        raise PermissionError(f"blocked by sandbox: {msg}")

    def hook(event, args):
        if event in BLOCKED_EVENTS:
            deny(event, str(args[0])[:80] if args else "")
        elif event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            if path is None or isinstance(path, int):
                return
            p = _norm(path)
            writing = bool(mode and any(c in str(mode) for c in "wax+")) or bool(
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND)
            )
            if writing and not _under(p, writes):
                deny("open-write", p)
            if not writing and not _under(p, reads + writes):
                deny("open-read", p)
        elif event in PATH_EVENTS and args:
            if not _under(_norm(args[0]), writes):
                deny(event, _norm(args[0]))
        elif event in LIST_EVENTS and args and args[0] is not None:
            if not _under(_norm(args[0]), reads + writes):
                deny(event, _norm(args[0]))

    sys.addaudithook(hook)


def _find_plotly_figure(ns: dict):
    figs = [v for v in ns.values() if type(v).__module__.startswith("plotly.graph_objs")]
    return figs[-1] if figs else None


def main(workdir: str) -> int:
    wd = os.path.abspath(workdir)
    sys.dont_write_bytecode = True

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.collections  # noqa: F401  (loaded before the guard; extract uses them)
    import matplotlib.container  # noqa: F401
    import matplotlib.dates  # noqa: F401
    import matplotlib.patches  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from chart_testkit import extract

    with open(os.path.join(wd, "input.pkl"), "rb") as f:
        df = pickle.load(f)
    with open(os.path.join(wd, "code.py"), encoding="utf-8") as f:
        code = f.read()

    # these two are slow to import, load them only when the code mentions them
    if "plotly" in code:
        try:
            import plotly.express  # noqa: F401
            import plotly.graph_objects as go

            go.Figure.show = lambda self, *a, **k: None
        except ImportError:
            pass
    if "seaborn" in code or "sns." in code:
        try:
            import seaborn  # noqa: F401
        except ImportError:
            pass

    warm = plt.figure()
    warm.canvas.draw()
    plt.close(warm)

    mpl_cache = os.environ.get("MPLCONFIGDIR", wd)
    # import locations stay readable: lazy imports scan sys.path (editable installs add the project root)
    read_roots = [sys.prefix, sys.base_prefix, sys.exec_prefix, *site.getsitepackages(), matplotlib.get_data_path(),
                  *[p for p in sys.path if p and os.path.isdir(p)], *_font_dirs()]
    blocked: list[str] = []
    install_guard(wd, read_roots, [wd, mpl_cache], blocked)

    ns = {"__name__": "__main__", "df": df, "pd": pd, "np": np, "plt": plt}
    result: dict = {"ok": False, "error": None}
    t0 = time.perf_counter()
    try:
        exec(compile(code, "<generated>", "exec"), ns)
        result["ok"] = True
    except BaseException as exc:  # SystemExit and KeyboardInterrupt count as failures too
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    result["exec_seconds"] = round(time.perf_counter() - t0, 3)

    try:
        pfig = _find_plotly_figure(ns)
        if pfig is not None:
            spec = extract.extract_plotly(pfig)
        elif plt.get_fignums():
            mfig = plt.gcf()
            spec = extract.extract_matplotlib(mfig)
            mfig.savefig(os.path.join(wd, "chart.png"), dpi=60)
        else:
            spec = {"axes": []}
        result["spec"] = extract.summarise(spec)
    except BaseException as exc:
        result["spec"] = {"axes": [], "chart_type": "none"}
        result["extract_error"] = f"{type(exc).__name__}: {exc}"[:300]
    result["blocked"] = blocked
    with open(os.path.join(wd, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
