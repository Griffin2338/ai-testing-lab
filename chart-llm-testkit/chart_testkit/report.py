"""Case results -> report.json and a single-file report.html."""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from chart_testkit.runner import CaseResult

LABEL = {"pass": "Pass", "fail": "Fail", "error": "Error", "skipped": "Not run"}


def _e(s) -> str:
    return html.escape(str(s))


def _clean(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def to_dict(results: dict[str, CaseResult], meta: dict) -> dict:
    cases = []
    for r in results.values():
        cases.append({
            "id": r.case_id, "title": r.title, "category": r.category, "status": r.status, "summary": r.summary,
            "variants": {k: {"passed": a, "runs": b} for k, (a, b) in r.variants.items()},
            "notes": r.notes,
            "runs": [{"variant": x.variant, "repeat": x.repeat, "passed": x.passed, "seconds": round(x.seconds, 2),
                      "clarification": x.clarification, "code": x.code,
                      "checks": [{"name": o.name, "passed": o.passed, "detail": o.detail} for o in x.outcomes]}
                     for x in r.runs],
        })
    return _clean({"meta": meta, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "cases": cases})


def write(results: dict[str, CaseResult], meta: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp, hp = out / "report.json", out / "report.html"
    jp.write_text(json.dumps(to_dict(results, meta), indent=2, ensure_ascii=False), encoding="utf-8")
    hp.write_text(_html(results, meta), encoding="utf-8")
    return jp, hp


def _case_html(r: CaseResult) -> str:
    rows = ""
    for v, (k, n) in r.variants.items():
        rows += f"<tr><td>{_e(v)}</td><td>{k}/{n}</td><td>{k / max(n, 1):.0%}</td></tr>"
    table = f"<table><thead><tr><th>variant</th><th>passed</th><th>rate</th></tr></thead><tbody>{rows}</tbody></table>"
    sample = next((x for x in r.runs if not x.passed), r.runs[0] if r.runs else None)
    detail = ""
    if sample:
        checks = "".join(
            f"<li class='{'ok' if o.passed else 'bad'}'><b>{_e(o.name)}</b>: {_e(o.detail)}</li>"
            for o in sample.outcomes
        )
        img = f"<img alt='chart from this run' src='data:image/png;base64,{sample.png_b64}'>" if sample.png_b64 else ""
        body = sample.code or (f"CLARIFY: {sample.clarification}" if sample.clarification else "(no code)")
        which = "first failing run" if not sample.passed else "first run"
        detail = (f"<details><summary>{which}: {_e(sample.variant)} #{sample.repeat + 1}</summary>"
                  f"<div class='run'><ul class='checks'>{checks}</ul>{img}</div><pre>{_e(body)}</pre></details>")
    notes = "".join(f"<li>{_e(n)}</li>" for n in r.notes)
    return (f"<section class='tc' id='{_e(r.case_id)}'><div class='head'><span class='badge s-{r.status}'>"
            f"{LABEL[r.status]}</span><h3>{_e(r.case_id)} · {_e(r.title)}</h3>"
            f"<span class='cat'>{_e(r.category)}</span></div><p>{_e(r.summary)}</p>{table}"
            f"{'<ul class=notes>' + notes + '</ul>' if notes else ''}{detail}</section>")


def _html(results: dict[str, CaseResult], meta: dict) -> str:
    counts = {s: sum(r.status == s for r in results.values()) for s in LABEL}
    cards = "".join(f"<div class='card s-{s}'><div class='n'>{counts[s]}</div><div class='l'>{LABEL[s]}</div></div>"
                    for s in ("pass", "fail"))
    meta_rows = "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in meta.items())
    nav = "".join(
        f"<a href='#{_e(r.case_id)}'><span class='dot s-{r.status}'></span>{_e(r.case_id)} {_e(r.title)}</a>"
        for r in results.values()
    )
    sections = "".join(_case_html(r) for r in results.values())
    return _TEMPLATE.format(title=_e(meta.get("name", "Chart generation report")), cards=cards, meta=meta_rows,
                            nav=nav, sections=sections,
                            created=_e(datetime.now(timezone.utc).isoformat(timespec="seconds")))


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#f7f7f5; --panel:#fff; --text:#1d1d1b; --muted:#6b6b66; --line:#e3e3de;
  --pass:#1f7a4d; --fail:#b42318; --error:#7a1fb4; --skipped:#7a7a74; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#161615; --panel:#1f1f1d; --text:#ecece8; --muted:#a3a39c; --line:#33332f;
    --pass:#4cc38a; --fail:#ff6b5e; --error:#c58bff; --skipped:#9a9a93; }}
}}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,"Segoe UI",Roboto,sans-serif }}
.wrap {{ max-width:1100px; margin:0 auto; padding:24px 16px 64px }}
h1 {{ font-size:24px; margin:0 0 4px }} h2 {{ font-size:18px; margin:32px 0 12px }} h3 {{ font-size:16px; margin:0 }}
.muted {{ color:var(--muted) }}
.cards {{ display:flex; gap:12px; flex-wrap:wrap; margin:20px 0 }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 16px;
  min-width:120px }}
.card .n {{ font-size:28px; font-weight:650 }} .card .l {{ color:var(--muted); font-size:13px }}
.card.s-pass .n {{ color:var(--pass) }} .card.s-fail .n {{ color:var(--fail) }}
.grid {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,2fr); gap:16px }}
@media (max-width:760px) {{ .grid {{ grid-template-columns:1fr }} }}
.panel,.tc {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px }}
.tc {{ margin:0 0 14px }}
nav a {{ display:block; color:var(--text); text-decoration:none; padding:3px 0; font-size:14px }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px; background:var(--skipped) }}
.head {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap }}
.cat {{ color:var(--muted); font-size:13px }}
.badge {{ font-size:12px; font-weight:600; padding:2px 8px; border-radius:999px; color:#fff;
  background:var(--skipped) }}
.s-pass.dot,.badge.s-pass {{ background:var(--pass) }} .s-fail.dot,.badge.s-fail {{ background:var(--fail) }}
.s-error.dot,.badge.s-error {{ background:var(--error) }}
table {{ border-collapse:collapse; font-size:13.5px; margin:6px 0 }}
th,td {{ text-align:left; padding:5px 12px 5px 0; border-bottom:1px solid var(--line) }}
th {{ color:var(--muted); font-weight:600 }}
.meta td {{ overflow-wrap:anywhere }}
.notes {{ color:var(--muted); font-size:13.5px; margin:8px 0 }}
details {{ margin-top:10px }} summary {{ cursor:pointer; color:var(--muted) }}
.run {{ display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; margin:10px 0 }}
.run img {{ max-width:100%; width:420px; border:1px solid var(--line); border-radius:6px; background:#fff }}
.checks {{ margin:0; padding-left:18px; font-size:13.5px; flex:1; min-width:260px }}
.checks .bad {{ color:var(--fail) }}
pre {{ background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:10px; overflow-x:auto;
  font-size:12.5px }}
</style></head>
<body><div class="wrap">
<h1>{title}</h1><div class="muted">Generated {created} by chart-llm-testkit</div>
<div class="cards">{cards}</div>
<div class="grid">
  <div class="panel"><h3>Run</h3><table class="meta">{meta}</table></div>
  <div class="panel"><h3>Cases</h3><nav>{nav}</nav></div>
</div>
<h2>Cases</h2>
{sections}
</div></body></html>
"""
