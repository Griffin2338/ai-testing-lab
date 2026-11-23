"""Test records -> report.json and a single-file report.html."""

from __future__ import annotations

import html
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUSES = ("pass", "fail", "insufficient", "info", "skipped", "error")


@dataclass
class TestRecord:
    __test__ = False  # keep pytest from collecting it

    tc_id: str
    title: str
    status: str
    summary: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reference: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"bad status {self.status!r}")

    @property
    def passed(self) -> bool:
        return self.status in ("pass", "info", "insufficient", "skipped")


@dataclass
class Report:
    name: str
    meta: dict[str, Any] = field(default_factory=dict)
    records: list[TestRecord] = field(default_factory=list)
    diagnostics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def add(self, record: TestRecord) -> TestRecord:
        self.records = [r for r in self.records if r.tc_id != record.tc_id] + [record]
        return record

    def counts(self) -> dict[str, int]:
        c = {s: 0 for s in STATUSES}
        for r in self.records:
            c[r.status] += 1
        return c

    # ------------------------------------------------------------------ output

    def write(self, out_dir: str | Path) -> tuple[Path, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.records.sort(key=lambda r: r.tc_id)
        jp = out / "report.json"
        jp.write_text(json.dumps(_clean(asdict(self)), indent=2, ensure_ascii=False), encoding="utf-8")
        hp = out / "report.html"
        hp.write_text(self.to_html(), encoding="utf-8")
        return jp, hp

    def to_html(self) -> str:
        c = self.counts()
        cards = "".join(
            f'<div class="card s-{s}"><div class="n">{c[s]}</div><div class="l">{_LABEL[s]}</div></div>'
            for s in ("pass", "fail", "insufficient", "info", "skipped", "error")
            if c[s] or s in ("pass", "fail")
        )
        meta = "".join(f"<tr><th>{_e(k)}</th><td>{_e(_fmt(v))}</td></tr>" for k, v in self.meta.items())
        warns = "".join(f'<div class="warn">{_e(w)}</div>' for w in self.warnings)
        nav = "".join(
            f'<a href="#{_e(r.tc_id)}"><span class="dot s-{r.status}"></span>{_e(r.tc_id)} {_e(r.title)}</a>'
            for r in self.records
        )
        sections = "".join(self._record_html(r) for r in self.records)
        diag = "".join(
            f'<section class="tc"><h3>{_e(title)}</h3>{_table(rows)}</section>'
            for title, rows in self.diagnostics.items()
            if rows
        )
        return _TEMPLATE.format(
            title=_e(self.name), created=_e(self.created), cards=cards, meta=meta, warns=warns,
            nav=nav, sections=sections, diag=diag or "<p class='muted'>No diagnostics.</p>",
        )

    @staticmethod
    def _record_html(r: TestRecord) -> str:
        notes = "".join(f"<li>{_e(n)}</li>" for n in r.notes)
        ref = f'<p class="ref">Based on: {_e(r.reference)}</p>' if r.reference else ""
        return (
            f'<section class="tc" id="{_e(r.tc_id)}"><div class="tc-head">'
            f'<span class="badge s-{r.status}">{_LABEL[r.status]}</span>'
            f"<h3>{_e(r.tc_id)} · {_e(r.title)}</h3></div>"
            f'<p class="summary">{_e(r.summary)}</p>{_table(r.rows)}'
            f'{"<ul class=notes>" + notes + "</ul>" if notes else ""}{ref}</section>'
        )


_LABEL = {
    "pass": "Pass", "fail": "Fail", "insufficient": "Insufficient data",
    "info": "Info", "skipped": "Not configured", "error": "Error",
}


def _e(s: Any) -> str:
    return html.escape(str(s))


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "–"
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, float) for x in v):
        return f"[{_fmt(v[0])}, {_fmt(v[1])}]"
    return str(v)


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    head = "".join(f"<th>{_e(c)}</th>" for c in cols)
    body = ""
    for r in rows:
        cells = ""
        for c in cols:
            v = r.get(c, "")
            cls = ""
            if c == "verdict":
                cls = f' class="v-{_e(str(v).split()[0].lower())}"' if v else ""
            cells += f"<td{cls}>{_e(_fmt(v))}</td>"
        body += f"<tr>{cells}</tr>"
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _clean(o: Any) -> Any:
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):  # numpy scalar
        return _clean(o.item())
    return o


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg:#f7f7f5; --panel:#fff; --text:#1d1d1b; --muted:#6b6b66; --line:#e3e3de;
  --pass:#1f7a4d; --fail:#b42318; --insufficient:#9a6700; --info:#2f5fb3; --skipped:#7a7a74; --error:#7a1fb4;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#161615; --panel:#1f1f1d; --text:#ecece8; --muted:#a3a39c; --line:#33332f;
    --pass:#4cc38a; --fail:#ff6b5e; --insufficient:#e0b341; --info:#7aa7ff; --skipped:#9a9a93; --error:#c58bff; }}
}}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--text);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif }}
.wrap {{ max-width:1100px; margin:0 auto; padding:24px 16px 64px }}
h1 {{ font-size:24px; margin:0 0 4px }} h2 {{ font-size:18px; margin:32px 0 12px }}
h3 {{ font-size:16px; margin:0 }}
.muted,.ref {{ color:var(--muted) }} .ref {{ font-size:13px; margin:8px 0 0 }}
.cards {{ display:flex; gap:12px; flex-wrap:wrap; margin:20px 0 }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:12px 16px; min-width:120px }}
.card .n {{ font-size:28px; font-weight:650; font-variant-numeric:tabular-nums }}
.card .l {{ color:var(--muted); font-size:13px }}
.card.s-pass .n {{ color:var(--pass) }} .card.s-fail .n {{ color:var(--fail) }}
.card.s-insufficient .n {{ color:var(--insufficient) }} .card.s-info .n {{ color:var(--info) }}
.card.s-error .n {{ color:var(--error) }}
.grid {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,2fr); gap:16px }}
@media (max-width:760px) {{ .grid {{ grid-template-columns:1fr }} }}
.panel,.tc {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px }}
.tc {{ margin:0 0 14px }}
nav a {{ display:block; color:var(--text); text-decoration:none; padding:3px 0; font-size:14px }}
nav a:hover {{ text-decoration:underline }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px; background:var(--skipped) }}
.tc-head {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap }}
.badge {{ font-size:12px; font-weight:600; padding:2px 8px; border-radius:999px; color:#fff;
  background:var(--skipped) }}
.s-pass.dot,.badge.s-pass {{ background:var(--pass) }} .s-fail.dot,.badge.s-fail {{ background:var(--fail) }}
.s-insufficient.dot,.badge.s-insufficient {{ background:var(--insufficient) }}
.s-info.dot,.badge.s-info {{ background:var(--info) }} .s-error.dot,.badge.s-error {{ background:var(--error) }}
.summary {{ margin:10px 0 }}
.tw {{ overflow-x:auto }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; font-variant-numeric:tabular-nums }}
th,td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); white-space:nowrap }}
th {{ color:var(--muted); font-weight:600 }}
.v-pass {{ color:var(--pass); font-weight:600 }} .v-fail {{ color:var(--fail); font-weight:600 }}
.v-insufficient {{ color:var(--insufficient); font-weight:600 }}
.meta th {{ width:45% }} .meta td {{ white-space:normal; overflow-wrap:anywhere }}
.warn {{ border-left:4px solid var(--insufficient); background:var(--panel); padding:8px 12px; margin:8px 0 }}
.notes {{ margin:8px 0 0; color:var(--muted); font-size:13.5px }}
</style></head>
<body><div class="wrap">
<h1>{title}</h1><div class="muted">Generated {created} by od-testkit</div>
<div class="cards">{cards}</div>
{warns}
<div class="grid">
  <div class="panel"><h3>Run</h3><table class="meta">{meta}</table></div>
  <div class="panel"><h3>Tests</h3><nav>{nav}</nav></div>
</div>
<h2>Test results</h2>
{sections}
<h2>Diagnostics</h2>
{diag}
</div></body></html>
"""
