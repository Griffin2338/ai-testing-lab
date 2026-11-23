"""chart-testkit run | try"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _run(args) -> int:
    import pytest

    suite = Path(__file__).parent / "suite"
    os.environ["CHART_TESTKIT_CONFIG"] = str(Path(args.config).resolve())
    os.environ["CHART_TESTKIT_OUT"] = str(Path(args.out).resolve())
    pargs = [str(suite), "-c", str(suite / "pytest.ini"), "--rootdir", str(suite), "-p", "no:cacheprovider",
             "-v" if args.verbose else "-q"]
    if args.k:
        pargs += ["-k", args.k]
    if args.junit:
        pargs.append(f"--junitxml={Path(args.junit).resolve()}")
    return int(pytest.main(pargs + args.pytest_args))


def _try(args) -> int:
    import yaml

    from chart_testkit import datasets, policy
    from chart_testkit.app import build_app
    from chart_testkit.sandbox import run_code

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    app = build_app(cfg["app"], Path(args.config).resolve().parent)
    df = datasets.build({"builtin": args.data})
    gen = app.generate(df, args.request)
    if gen.code is None:
        print(f"CLARIFY: {gen.clarification}")
        return 0
    print(gen.code, "\n")
    pol = policy.check(gen.code, list(df.columns))
    print("policy:", "ok" if pol.ok else pol.violations, "| fabrication:", pol.fabrication or "none")
    rr = run_code(gen.code, df)
    spec = rr.spec
    print("ran:", rr.ok, rr.error or "", "| blocked:", rr.blocked or "none")
    keys = ("chart_type", "categories", "values", "value_axis")
    print(json.dumps({k: spec.get(k) for k in keys}, default=str)[:1500])
    if args.png and rr.png_b64:
        import base64

        Path(args.png).write_bytes(base64.b64decode(rr.png_b64))
        print("chart saved to", Path(args.png).resolve())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="chart-testkit", description="Tests for LLM chart generation")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the case suite for a config")
    r.add_argument("config")
    r.add_argument("--out", default="chart-report")
    r.add_argument("-k", default=None, help="pytest -k expression, e.g. 'TC-01 or TC-05'")
    r.add_argument("--junit", default=None)
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=_run, pytest_args=[])

    t = sub.add_parser("try", help="send one request to the app and show code, policy and extracted chart")
    t.add_argument("config")
    t.add_argument("request")
    t.add_argument("--data", default="trades", help="builtin dataset name (trades, long_labels)")
    t.add_argument("--png", default=None, help="save the rendered chart here")
    t.set_defaults(func=_try)

    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    args = p.parse_args(argv)
    if extra:
        args.pytest_args = extra
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
