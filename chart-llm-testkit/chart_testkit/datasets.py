"""Built-in test tables. Deterministic for a given seed. Tickers and funds are made up."""

from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = ["Technology", "Finance", "Energy", "Healthcare"]
TICKERS = ["ACMX", "BOLT", "CRUX", "DYNA", "EVRA", "FLUX", "GRDN", "HALO"]
EXCHANGES = ["NYSE", "NASDAQ", "LSE"]


def trades(rows: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01")
    trade_date = start + pd.to_timedelta(rng.integers(0, 365, rows), unit="D")
    df = pd.DataFrame({
        "trade_id": np.arange(1, rows + 1),
        "trade_date": trade_date,
        "sector": rng.choice(SECTORS, rows, p=[0.35, 0.3, 0.25, 0.1]),
        "ticker": rng.choice(TICKERS, rows),
        "exchange": rng.choice(EXCHANGES, rows),
        "shares": rng.integers(1, 40, rows) * 10,
        "price": rng.choice([9.5, 19.0, 49.0, 120.0], rows),
    })
    df["traded_value"] = (df["shares"] * df["price"]).round(2)
    df["proceeds"] = df["traded_value"]
    df["cost_basis"] = (df["proceeds"] * rng.uniform(0.4, 0.8, rows)).round(2)
    df["is_cancelled"] = rng.random(rows) < 0.08
    df["settlement_date"] = df["trade_date"] + pd.to_timedelta(rng.integers(1, 4, rows), unit="D")
    df["report_date"] = df["trade_date"] + pd.to_timedelta(rng.integers(0, 30, rows), unit="D")
    return df


def long_labels(categories: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = [f"Global {s} equity income fund {i:02d} - accumulating share class" for i, s in
             zip(range(categories), rng.choice(["value", "growth", "dividend", "small-cap"], categories), strict=True)]
    return pd.DataFrame({"fund": names, "trades": rng.integers(50, 500, categories)})


BUILTIN = {"trades": trades, "long_labels": long_labels}


def build(spec: dict) -> pd.DataFrame:
    """Dataset from a case's `dataset:` section."""
    spec = dict(spec)
    name = spec.pop("builtin", None)
    if name:
        kwargs = {k: spec.pop(k) for k in ("rows", "seed", "categories") if k in spec}
        df = BUILTIN[name](**kwargs)
    elif "csv" in spec:
        df = pd.read_csv(spec.pop("csv"))
    else:
        raise ValueError("dataset needs `builtin` or `csv`")
    if spec.get("drop"):
        df = df.drop(columns=spec["drop"])
    if spec.get("keep"):
        df = df[spec["keep"]]
    if spec.get("nan_in"):
        col = spec["nan_in"]
        df.loc[df.index[::17], col] = np.nan
    if spec.get("blank_in"):
        col = spec["blank_in"]
        df.loc[df.index[5], col] = ""
    if spec.get("add_noise_columns"):
        rng = np.random.default_rng(1)
        for i in range(int(spec["add_noise_columns"])):
            df[f"metric_{i:02d}"] = rng.normal(size=len(df)).round(3)
        df["traded_value_prev"] = (df["traded_value"] * 0.7).round(2)
    if spec.get("rename"):
        df = df.rename(columns=spec["rename"])
    return df.reset_index(drop=True)
