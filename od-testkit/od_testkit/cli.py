"""od-testkit run | synth | stats"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _run(args: argparse.Namespace) -> int:
    import pytest

    suite = Path(__file__).parent / "suite"
    os.environ["OD_TESTKIT_CONFIG"] = str(Path(args.config).resolve())
    os.environ["OD_TESTKIT_OUT"] = str(Path(args.out).resolve())
    pargs = [str(suite), "-c", str(suite / "pytest.ini"), "--rootdir", str(suite), "-p", "no:cacheprovider",
             "-v" if args.verbose else "-q"]
    if args.k:
        pargs += ["-k", args.k]
    if args.junit:
        pargs += [f"--junitxml={Path(args.junit).resolve()}"]
    return int(pytest.main(pargs + list(args.pytest_args)))


def _synth(args: argparse.Namespace) -> int:
    from od_testkit.synthetic import generate_dataset

    out = generate_dataset(args.out, n_images=args.images, n_negatives=args.negatives, seed=args.seed)
    print(f"synthetic dataset written to {out.resolve()}")
    return 0


def _stats(args: argparse.Namespace) -> int:
    from od_testkit import stats

    if args.what == "sample-size":
        n = stats.sample_size_for_margin(args.p, args.margin, args.confidence)
        print(f"{n} labelled cases to estimate p≈{args.p} within ±{args.margin} at {args.confidence:.0%}")
    elif args.what == "ci":
        lo, hi = stats.wilson_interval(args.k, args.n, args.confidence)
        low1 = stats.wilson_interval(args.k, args.n, args.confidence, sided="lower")[0]
        print(f"p̂ = {args.k / args.n:.4f}; {args.confidence:.0%} Wilson CI [{lo:.4f}, {hi:.4f}]; "
              f"one-sided lower bound {low1:.4f}")
    elif args.what == "zero-failure":
        n = stats.zero_failure_sample_size(args.rate, args.confidence)
        print(f"{n} failure-free trials to claim failure rate ≤ {args.rate} at {args.confidence:.0%}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="od-testkit", description="Object-detection acceptance testing for QA teams")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the acceptance suite for a YAML config")
    r.add_argument("config")
    r.add_argument("--out", default="od-report", help="report folder (default: od-report)")
    r.add_argument("-k", default=None, help="pytest -k expression to select tests, e.g. 'tc01 or tc04'")
    r.add_argument("--junit", default=None, help="also write JUnit XML (for CI dashboards)")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=_run, pytest_args=[])

    s = sub.add_parser("synth", help="generate the synthetic demo dataset")
    s.add_argument("out")
    s.add_argument("--images", type=int, default=200)
    s.add_argument("--negatives", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=_synth)

    st = sub.add_parser("stats", help="sample-size and confidence-interval calculator")
    st.add_argument("what", choices=["sample-size", "ci", "zero-failure"])
    st.add_argument("--p", type=float, default=0.9, help="expected proportion (sample-size)")
    st.add_argument("--margin", type=float, default=0.03, help="± margin (sample-size)")
    st.add_argument("--k", type=int, default=0, help="successes (ci)")
    st.add_argument("--n", type=int, default=1, help="trials (ci)")
    st.add_argument("--rate", type=float, default=0.01, help="max failure rate (zero-failure)")
    st.add_argument("--confidence", type=float, default=0.95)
    st.set_defaults(func=_stats)

    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:  # rest goes to pytest as-is
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    args = p.parse_args(argv)
    if extra:
        if args.cmd != "run":
            p.error("extra arguments after '--' are only accepted by 'run'")
        args.pytest_args = extra
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
