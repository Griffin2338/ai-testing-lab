"""Static checks on generated code, before anything runs."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

ALLOWED_IMPORTS = {
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "math", "datetime", "calendar",
    "statistics", "textwrap", "itertools", "collections", "decimal", "re",
}
FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "__import__", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "input", "breakpoint", "exit", "quit",
}
# writing files or pulling data from elsewhere: the chart must come from `df`
FORBIDDEN_METHODS = {
    "savefig", "write_image", "write_html", "write_json", "to_csv", "to_excel", "to_pickle",
    "to_parquet", "to_json", "to_sql", "read_csv", "read_excel", "read_json", "read_sql",
    "read_html", "read_parquet", "read_pickle", "read_clipboard", "show_in_browser",
}


@dataclass
class PolicyResult:
    violations: list[str] = field(default_factory=list)
    fabrication: list[str] = field(default_factory=list)
    columns_read: set[str] = field(default_factory=set)
    unknown_columns: set[str] = field(default_factory=set)
    syntax_error: str | None = None

    @property
    def ok(self) -> bool:
        return not self.violations and self.syntax_error is None


def _df_aliases(tree: ast.AST) -> set[str]:
    """`df` plus names assigned from expressions that use it (d = df.copy(), ...)."""
    cached = getattr(tree, "_df_aliases", None)
    if cached is not None:
        return cached
    aliases = {"df"}
    for _ in range(3):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(n, ast.Name) and n.id in aliases for n in ast.walk(node.value)
            ):
                aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
    tree._df_aliases = aliases  # type: ignore[attr-defined]
    return aliases


def _string_args(node: ast.AST) -> list[str]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append(n.value)
    return out


def check(code: str, columns: list[str] | None = None) -> PolicyResult:
    res = PolicyResult()
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        res.syntax_error = f"line {exc.lineno}: {exc.msg}"
        return res
    cols = set(columns or [])
    assigned: set[str] = set()
    referenced: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    res.violations.append(f"import not allowed: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod not in ALLOWED_IMPORTS:
                res.violations.append(f"import not allowed: {node.module}")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in FORBIDDEN_CALLS:
                res.violations.append(f"call not allowed: {f.id}()")
            if isinstance(f, ast.Attribute) and f.attr in FORBIDDEN_METHODS:
                res.violations.append(f"call not allowed: .{f.attr}()")
            if isinstance(f, ast.Attribute) and f.attr in ("DataFrame", "Series") and node.args:
                if isinstance(node.args[0], (ast.Dict, ast.List)):
                    res.fabrication.append(f"builds a {f.attr} from literals")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                res.violations.append(f"dunder access: {node.attr}")
            if node.attr == "random" and isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy"):
                res.fabrication.append("uses np.random")
        elif isinstance(node, ast.Name) and node.id == "random":
            res.fabrication.append("uses the random module")
        elif isinstance(node, ast.List) and len(node.elts) >= 3:
            if all(isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) for e in node.elts):
                res.fabrication.append(f"numeric literal list of length {len(node.elts)}")
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            key = node.slice.value
            if isinstance(key, str) and isinstance(node.value, ast.Name) and node.value.id in _df_aliases(tree):
                if isinstance(node.ctx, ast.Store):
                    assigned.add(key)
                else:
                    referenced.add(key)

    # column names passed as strings, e.g. groupby("sector"), x="traded_value"
    for s in _string_args(tree):
        if s in cols:
            res.columns_read.add(s)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "assign":
            assigned.update(k.arg for k in node.keywords if k.arg)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "rename":
            for kw in node.keywords:
                if kw.arg == "columns" and isinstance(kw.value, ast.Dict):
                    assigned.update(v.value for v in kw.value.values
                                    if isinstance(v, ast.Constant) and isinstance(v.value, str))
    res.columns_read |= referenced & cols
    if cols:
        res.unknown_columns = {r for r in referenced if r not in cols and r not in assigned}
    res.fabrication = sorted(set(res.fabrication))
    return res
