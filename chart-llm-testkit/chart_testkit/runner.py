"""Load a config and its cases, run every generation, and aggregate per case."""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from chart_testkit import datasets, oracle, policy
from chart_testkit.app import build_app
from chart_testkit.oracle import Evidence, Outcome
from chart_testkit.sandbox import run_code

DEFAULTS = {
    "name": "Chart generation acceptance run",
    "repeats": 5,
    "workers": 4,
    "min_pass_rate": 0.8,
    "sandbox": {"timeout_s": 30, "memory_mb": 2048},
    "cases": [],
}


@dataclass
class RunRecord:
    variant: str
    repeat: int
    passed: bool
    outcomes: list[Outcome]
    code: str | None = None
    clarification: str | None = None
    png_b64: str | None = None
    seconds: float = 0.0

    @property
    def failed(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.passed]


@dataclass
class CaseResult:
    case_id: str
    title: str
    category: str
    status: str
    summary: str
    runs: list[RunRecord] = field(default_factory=list)
    variants: dict[str, tuple[int, int]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def _mentions_clarification(expect) -> bool:
    if isinstance(expect, dict):
        return "clarification" in expect or any(_mentions_clarification(v) for v in expect.values())
    if isinstance(expect, list):
        return any(_mentions_clarification(v) for v in expect)
    return False


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    out.update(b or {})
    return out


class Suite:
    def __init__(self, config_path: str | Path):
        path = Path(config_path).resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self.cfg = {**DEFAULTS, **raw, "sandbox": {**DEFAULTS["sandbox"], **(raw.get("sandbox") or {})}}
        self.base_dir = path.parent
        self.config_path = path
        self.cases: list[dict] = []
        for f in self.cfg["cases"]:
            p = Path(f)
            p = p if p.is_absolute() else self.base_dir / p
            self.cases += yaml.safe_load(p.read_text(encoding="utf-8")) or []
        ids = [c["id"] for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate case ids")
        if self.cfg.get("only"):
            self.cases = [c for c in self.cases if c["id"] in set(self.cfg["only"])]
        self.app = build_app(self.cfg["app"], self.base_dir)
        self.timeout = float(self.cfg["sandbox"]["timeout_s"])
        self.memory = int(self.cfg["sandbox"]["memory_mb"])
        self._frames: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------ data

    def frame(self, case: dict) -> pd.DataFrame:
        if case["id"] not in self._frames:
            self._frames[case["id"]] = datasets.build(case["dataset"])
        return self._frames[case["id"]]

    # ------------------------------------------------------------ one generation

    def _execute(self, df: pd.DataFrame, request: str):
        """Generate, check policy, run. Returns (generation, outcomes, run_result or None)."""
        try:
            gen = self.app.generate(df, request)
        except Exception as exc:
            return None, [Outcome("generate", False, f"{type(exc).__name__}: {exc}"[:300])], None
        if gen.code is None:
            return gen, [], None
        pol = policy.check(gen.code, [str(c) for c in df.columns])
        outs = [Outcome("policy", pol.ok, pol.syntax_error or "; ".join(pol.violations) or "clean")]
        rr = run_code(gen.code, df, timeout_s=self.timeout, memory_mb=self.memory)
        outs.append(Outcome("executes", rr.ok, rr.error or "ran"))
        outs.append(Outcome("sandbox", not rr.blocked, "; ".join(rr.blocked[:3]) or "nothing blocked"))
        rr.policy = pol  # type: ignore[attr-defined]
        return gen, outs, rr

    def _single(self, case: dict, variant: str, request: str, expect: dict, repeat: int) -> RunRecord:
        t0 = time.perf_counter()
        df = self.frame(case)
        gen, outs, rr = self._execute(df, request)
        if gen is None:
            return RunRecord(variant, repeat, False, outs, seconds=time.perf_counter() - t0)
        if gen.code is None:
            ev = Evidence(df, {}, policy.PolicyResult(), gen.clarification)
            if not _mentions_clarification(expect):
                outs.append(Outcome("answered", False, f"asked instead of drawing: {gen.clarification}"[:200]))
        else:
            ev = Evidence(df, rr.spec, rr.policy)
        outs += oracle.evaluate(ev, expect)
        return RunRecord(variant, repeat, all(o.passed for o in outs), outs, gen.code, gen.clarification,
                         rr.png_b64 if rr else None, time.perf_counter() - t0)

    def _metamorphic(self, case: dict, rel: dict, repeat: int) -> RunRecord:
        t0 = time.perf_counter()
        df = self.frame(case)
        kind = rel["type"]
        if kind == "shuffle_rows":
            tdf = df.sample(frac=1.0, random_state=repeat).reset_index(drop=True)
        elif kind == "scale":
            tdf = df.assign(**{rel["column"]: df[rel["column"]] * float(rel["factor"])})
        elif kind == "rename_value":
            tdf = df.assign(**{rel["column"]: df[rel["column"]].replace({rel["from"]: rel["to"]})})
        else:
            raise ValueError(f"unknown metamorphic relation {kind!r}")
        g1, o1, r1 = self._execute(df, case["request"])
        g2, o2, r2 = self._execute(tdf, case["request"])
        outs = [Outcome(f"base {o.name}", o.passed, o.detail) for o in o1]
        outs += [Outcome(f"transformed {o.name}", o.passed, o.detail) for o in o2]
        if r1 is None or r2 is None:
            outs.append(Outcome(kind, False, "no chart in one of the runs"))
        else:
            outs.append(Outcome(kind, *_relation(kind, rel, r1.spec, r2.spec)))
        return RunRecord(kind, repeat, all(o.passed for o in outs), outs, g2.code if g2 else None,
                         png_b64=r2.png_b64 if r2 else None, seconds=time.perf_counter() - t0)

    def _probe(self, name: str, spec: dict) -> RunRecord:
        t0 = time.perf_counter()
        rr = run_code(spec["code"], pd.DataFrame({"a": [1, 2]}), timeout_s=min(self.timeout, 10), memory_mb=self.memory)
        if spec.get("expect", "blocked") == "blocked":
            ok = (not rr.ok) and (bool(rr.blocked) or rr.timed_out)
            detail = "timed out" if rr.timed_out else ("; ".join(rr.blocked[:2]) or f"not blocked: {rr.error or 'ran'}")
        else:
            ok, detail = rr.ok, rr.error or "ran cleanly"
        return RunRecord(name, 0, ok, [Outcome(name, ok, detail)], spec["code"], seconds=time.perf_counter() - t0)

    # ------------------------------------------------------------ plan and run

    def _units(self, case: dict):
        repeats = int(case.get("repeats", self.cfg["repeats"]))
        if "probe" in case:
            for name, spec in case["probe"].items():
                yield lambda n=name, s=spec: self._probe(n, s)
            return
        if "metamorphic" in case:
            for rel in case["metamorphic"]:
                for i in range(repeats):
                    yield lambda r=rel, i=i: self._metamorphic(case, r, i)
            return
        variants = case.get("variants") or {"default": {}}
        for vname, v in variants.items():
            request = v.get("request") or (case.get("request", "") + v.get("suffix", ""))
            expect = _merge(case.get("expect") or {}, v.get("expect") or {})
            for i in range(repeats):
                yield lambda vn=vname, rq=request, ex=expect, i=i: self._single(case, vn, rq, ex, i)

    def run(self, case_ids: list[str] | None = None) -> dict[str, CaseResult]:
        os.environ["CHART_TESTKIT_SENTINEL"] = "parent-only"  # TC-04 checks it never reaches the child
        cases = [c for c in self.cases if case_ids is None or c["id"] in case_ids]
        plan = [(c, u) for c in cases for u in self._units(c)]
        with ThreadPoolExecutor(max_workers=int(self.cfg["workers"])) as pool:
            records = list(pool.map(lambda cu: cu[1](), plan))
        by_case: dict[str, list[RunRecord]] = {}
        for (c, _), rec in zip(plan, records, strict=True):
            by_case.setdefault(c["id"], []).append(rec)
        return {c["id"]: self._aggregate(c, by_case.get(c["id"], [])) for c in cases}

    def _aggregate(self, case: dict, runs: list[RunRecord]) -> CaseResult:
        category = case.get("category", "quality")
        need_all = bool(case.get("require_all")) or category == "safety"
        min_rate = float(case.get("min_pass_rate", self.cfg["min_pass_rate"]))
        variants: dict[str, tuple[int, int]] = {}
        for r in runs:
            k, n = variants.get(r.variant, (0, 0))
            variants[r.variant] = (k + r.passed, n + 1)
        failing = []
        for v, (k, n) in variants.items():
            if (need_all and k < n) or (not need_all and k / max(n, 1) < min_rate):
                failing.append(v)
        k_all, n_all = sum(k for k, _ in variants.values()), sum(n for _, n in variants.values())
        lo, hi = wilson(k_all, n_all)
        rule = "every run must pass" if need_all else f"pass rate ≥ {min_rate:.0%}"
        summary = f"{k_all}/{n_all} runs passed (95% CI {lo:.0%}–{hi:.0%}); rule: {rule}."
        if failing and len(variants) > 1:
            summary += f" Failing: {', '.join(failing)}."
        notes = []
        reasons: dict[str, int] = {}
        for r in runs:
            for o in r.failed:
                reasons[o.name] = reasons.get(o.name, 0) + 1
        if reasons:
            ranked = sorted(reasons.items(), key=lambda x: -x[1])
            notes.append("Failed checks: " + ", ".join(f"{k} ×{v}" for k, v in ranked))
        return CaseResult(case["id"], case.get("title", case["id"]), category, "fail" if failing else "pass",
                          summary, runs, variants, notes)


def _relation(kind: str, rel: dict, a: dict, b: dict, tol: float = 0.005) -> tuple[bool, str]:
    ca = {str(c): v for c, v in zip(a.get("categories") or [], a.get("values") or [], strict=False)}
    cb = {str(c): v for c, v in zip(b.get("categories") or [], b.get("values") or [], strict=False)}
    if not ca or not cb:
        return False, "no data series in one of the charts"
    if kind == "rename_value":
        ca = {(rel["to"] if k == rel["from"] else k): v for k, v in ca.items()}
    factor = float(rel.get("factor", 1.0)) if kind == "scale" else 1.0
    if set(ca) != set(cb):
        return False, f"categories changed: {sorted(set(ca) ^ set(cb))[:5]}"
    empty = sorted(k for k in ca if ca[k] is None or cb[k] is None)
    if empty:
        return False, f"no value for {empty[:5]} in one of the charts"
    bad = [k for k in ca if abs(cb[k] - ca[k] * factor) > tol * max(abs(ca[k] * factor), 1e-9)]
    if bad:
        return False, f"{len(bad)} value(s) broke the relation, e.g. {bad[0]}: {ca[bad[0]]:.4g} -> {cb[bad[0]]:.4g}"
    return True, "relation holds"
