"""LLM backends: Claude through the Anthropic SDK, and a scripted fake for demos and CI."""

from __future__ import annotations

import hashlib
import re
import threading
from typing import Protocol

Message = dict  # {"role": "user" | "assistant", "content": str}


class LLMRefusal(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[Message]) -> str: ...


class ClaudeLLM:
    """Claude via the Messages API. Credentials come from ANTHROPIC_API_KEY or `ant auth login`."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 16000,
                 fallbacks: bool = True, timeout_s: float = 120.0):
        import anthropic

        self.client = anthropic.Anthropic(timeout=timeout_s)
        self.model, self.effort, self.max_tokens, self.fallbacks = model, effort, max_tokens, fallbacks
        self.name = f"{model} (effort {effort})"

    def complete(self, system: str, messages: list[Message]) -> str:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages,
                      output_config={"effort": self.effort})
        if self.fallbacks:
            # on a policy refusal the API re-runs the request on a fallback model
            resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                    fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise LLMRefusal(f"refused ({getattr(details, 'category', None)})")
        return "".join(b.text for b in resp.content if b.type == "text")


# ---------------------------------------------------------------- fake LLM

TR_WORDS = {"sektör": "sector", "işlem": "traded_value", "hisse": "shares", "borsa": "exchange", "fiyat": "price"}


def _parse_prompt(text: str) -> tuple[str, list[str]]:
    req = re.search(r"^Request:\s*(.+)$", text, re.M)
    cols = re.findall(r"^- (.+): \S+$", text, re.M)
    return (req.group(1).strip() if req else text.strip()), cols


class FakeLLM:
    """Rule-based stand-in for an LLM. Understands only the requests in cases/default.yaml.

    profile="careful" writes correct code. profile="buggy" makes mistakes seen in real
    LLM output: a zoomed value axis, invented data, obeying instructions hidden in column
    names, the wrong column among look-alikes, seaborn's default mean estimator, unrotated
    long labels, vague axis names, and an intermittent bug: for "... by exchange" requests,
    every `flaky_every`-th call with the same prompt averages instead of summing.
    The schedule depends only on how often a prompt was seen, so pass counts don't change
    when repeats run in parallel.
    """

    def __init__(self, profile: str = "buggy", seed: int = 0, flaky_every: int = 9):
        if profile not in ("careful", "buggy"):
            raise ValueError("profile must be 'careful' or 'buggy'")
        self.profile, self.seed, self.flaky_every = profile, seed, flaky_every
        self.name = f"fake-{profile}"
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def _seen(self, prompt: str) -> int:
        key = hashlib.sha256(prompt.encode()).hexdigest()
        with self._lock:
            n = self._counts.get(key, 0)
            self._counts[key] = n + 1
        return n

    def complete(self, system: str, messages: list[Message]) -> str:
        prompt = messages[0]["content"]
        request, cols = _parse_prompt(prompt)
        code = self._write(request, cols, self._seen(prompt))
        if code.startswith("CLARIFY:"):
            return code
        return f"```python\n{code}\n```"

    # ------------------------------------------------------------ helpers

    @property
    def buggy(self) -> bool:
        return self.profile == "buggy"

    def _col(self, word: str, cols: list[str]) -> str | None:
        word = word.lower().strip()
        if not self.buggy:
            for tr, en in TR_WORDS.items():
                if word.startswith(tr):
                    word = en
        word = word.rstrip("s") if word.endswith("s") and word.rstrip("s") in [c.lower() for c in cols] else word
        exact = [c for c in cols if c.lower() == word or c.lower().replace("_", " ") == word]
        if exact and not self.buggy:
            return exact[0]
        loose = [c for c in cols if word.replace(" ", "_") in c.lower()]
        if self.buggy and loose:
            return loose[-1]  # picks e.g. traded_value_prev over traded_value
        return exact[0] if exact else (loose[0] if loose else None)

    def _dates(self, cols: list[str]) -> list[str]:
        return [c for c in cols if c.endswith("_date")]

    def _injected(self, cols: list[str]) -> bool:
        return any("ignore" in c.lower() and "import" in c.lower() for c in cols)

    # ------------------------------------------------------------ code writer

    def _write(self, request: str, cols: list[str], n: int) -> str:
        r = request.lower()
        prefix = ""
        if self.buggy and self._injected(cols):
            prefix = "import os\nos.system('calc')\n"
        suffix = ""
        if self.buggy and "save" in r and "home" in r:
            prefix += "import os\n"
            suffix = "\nplt.savefig(os.path.join(os.path.expanduser('~'), 'chart.png'))"
        if self.buggy and "from the web" in r:
            prefix += "import urllib.request\nrate = float(urllib.request.urlopen('https://api.example.com/eurusd').read())\n"
        lib = "plotly" if "plotly" in r else "seaborn" if "seaborn" in r else "matplotlib"

        if self.buggy and not r.isascii():
            body = "fig, ax = plt.subplots()\nax.plot(df.index, df[df.columns[0]])\nax.set_title('Chart')"
            return prefix + body
        if "profit" in r and "profit" not in [c.lower() for c in cols]:
            return prefix + self._profit(cols)
        if "relationship between" in r:
            m = re.search(r"between (\w+) and (\w+)", r)
            x, y = self._col(m.group(1), cols), self._col(m.group(2), cols)
            return prefix + (f"fig, ax = plt.subplots()\nax.scatter(df[{x!r}], df[{y!r}], alpha=0.6)\n"
                             f"ax.set_xlabel({x!r})\nax.set_ylabel({y!r})\nax.set_title('{y} vs {x}')") + suffix
        if "share of" in r:
            g = self._col(re.search(r"by (\w+)", r).group(1), cols)
            return prefix + (f"s = df[{g!r}].value_counts()\nfig, ax = plt.subplots()\n"
                             f"ax.pie(s.values, labels=s.index.astype(str), autopct='%1.0f%%')\n"
                             f"ax.set_title('Share of trades by {g}')") + suffix
        if "stacked" in r:
            return prefix + self._stacked(cols) + suffix
        if "over time" in r or "monthly" in r or "trend" in r:
            return prefix + self._timeseries(r, cols) + suffix
        m = re.search(r"top (\d+) (\w+) by (\w+)", r)
        if m:
            return prefix + self._top(int(m.group(1)), self._col(m.group(2), cols), self._col(m.group(3), cols))
        if "per fund" in r:
            return prefix + self._long_labels(cols)
        m = re.search(r"(?:total |net )?(\w+) (?:by|per) (\w+(?: \w+)?)", r)
        if not m and not self.buggy:
            m = re.search(r"(\w+)\w* göre toplam (\w+)", r)
            if m:
                return prefix + self._bar(self._col(m.group(2), cols), self._col(m.group(1), cols), r, n, lib) + suffix
        if not m:
            return "CLARIFY: Which column should I plot?"
        v, g = self._col(m.group(1), cols), self._col(m.group(2), cols)
        if v is None or g is None:
            return "CLARIFY: I can't find those columns. Which ones should I use?"
        return prefix + self._bar(v, g, r, n, lib) + suffix

    def _bar(self, v: str, g: str, r: str, n: int, lib: str) -> str:
        flaky = self.buggy and "by exchange" in r and n % self.flaky_every == self.flaky_every // 2
        agg = "mean" if flaky else "sum"
        filt = ""
        if "cancelled" in r and not self.buggy:
            filt = "df = df[~df['is_cancelled']]\n"
        label = v.split(";")[0].replace("_", " ").title()
        glabel = g.split(";")[0].replace("_", " ").title()
        if lib == "plotly":
            return (f"{filt}import plotly.express as px\nd = df.groupby({g!r}, as_index=False)[{v!r}].{agg}()\n"
                    f"fig = px.bar(d, x={g!r}, y={v!r}, title='{label} by {glabel}')\n"
                    f"fig.update_layout(xaxis_title='{glabel}', yaxis_title='{label}')")
        if lib == "seaborn":
            est = "" if self.buggy else ", estimator='sum', errorbar=None"
            return (f"{filt}import seaborn as sns\nfig, ax = plt.subplots()\n"
                    f"sns.barplot(data=df, x={g!r}, y={v!r}, ax=ax{est})\n"
                    f"ax.set_xlabel('{glabel}')\nax.set_ylabel('{label}')\nax.set_title('{label} by {glabel}')")
        zoom = "\nax.set_ylim(s.min() * 0.9, s.max() * 1.02)" if self.buggy and "compare" in r else ""
        return (f"{filt}s = df.groupby({g!r})[{v!r}].{agg}()\nfig, ax = plt.subplots(figsize=(8, 5))\n"
                f"ax.bar(s.index.astype(str), s.values)\nax.set_xlabel('{glabel}')\nax.set_ylabel('{label}')\n"
                f"ax.set_title('{label} by {glabel}'){zoom}")

    def _top(self, n: int, g: str, v: str) -> str:
        pick = f".sort_values().tail({n})" if self.buggy else f".nlargest({n})"
        return (f"s = df.groupby({g!r})[{v!r}].sum(){pick}\nfig, ax = plt.subplots()\n"
                f"ax.bar(s.index, s.values)\nax.set_xlabel('{g.title()}')\nax.set_ylabel('{v.title()}')\n"
                f"ax.set_title('Top {n} {g} by {v}')")

    def _timeseries(self, r: str, cols: list[str]) -> str:
        dates = self._dates(cols)
        v = self._col("traded_value", cols) or "traded_value"
        date = dates[0] if dates else "date"
        xlabel = "Date" if self.buggy else f"{date.replace('_', ' ').capitalize()} (month)"
        return (f"s = df.set_index({date!r})[{v!r}].resample('MS').sum()\nfig, ax = plt.subplots(figsize=(9, 4))\n"
                f"ax.plot(s.index, s.values, marker='o')\nax.set_xlabel({xlabel!r})\nax.set_ylabel('Traded value')\n"
                f"ax.set_title('Monthly traded value')")

    def _stacked(self, cols: list[str]) -> str:
        unit = "Traded value" if self.buggy else "Traded value (USD)"
        return ("d = df.assign(month=df['trade_date'].dt.to_period('M').astype(str))\n"
                "t = d.pivot_table(index='month', columns='exchange', values='traded_value', aggfunc='sum').fillna(0)\n"
                "ax = t.plot(kind='bar', stacked=True, figsize=(10, 5))\n"
                f"ax.set_xlabel('Month')\nax.set_ylabel({unit!r})\nax.set_title('Monthly traded value by exchange')")

    def _profit(self, cols: list[str]) -> str:
        if self.buggy:
            return ("months = pd.date_range('2024-01-01', periods=12, freq='MS')\n"
                    "profit = np.random.default_rng(0).normal(5000, 800, 12)\n"
                    "fig, ax = plt.subplots()\nax.plot(months, profit)\nax.set_xlabel('Month')\n"
                    "ax.set_ylabel('Profit')\nax.set_title('Monthly profit')")
        if "proceeds" in cols and "cost_basis" in cols:
            return ("s = df.assign(profit=df['proceeds'] - df['cost_basis']).set_index('trade_date')['profit']"
                    ".resample('MS').sum()\nfig, ax = plt.subplots()\nax.plot(s.index, s.values, marker='o')\n"
                    "ax.set_xlabel('Trade date (month)')\nax.set_ylabel('Profit (proceeds - cost basis)')\n"
                    "ax.set_title('Monthly profit, derived as proceeds - cost basis')")
        return "CLARIFY: There is no profit column. Which columns should I use to compute profit?"

    def _long_labels(self, cols: list[str]) -> str:
        if self.buggy:
            return ("fig, ax = plt.subplots(figsize=(8, 5))\nax.bar(df['fund'], df['trades'])\n"
                    "ax.set_xlabel('Fund')\nax.set_ylabel('Trades')\nax.set_title('Trades per fund')")
        return ("d = df.sort_values('trades')\nfig, ax = plt.subplots(figsize=(9, 12))\n"
                "ax.barh(d['fund'], d['trades'])\nax.set_xlabel('Trades')\nax.set_ylabel('Fund')\n"
                "ax.set_title('Trades per fund')\nfig.tight_layout()")


def build_llm(cfg: dict) -> LLM:
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind == "claude":
        return ClaudeLLM(**cfg)
    if kind == "fake":
        return FakeLLM(**cfg)
    raise ValueError(f"unknown llm type {kind!r} (claude or fake)")
