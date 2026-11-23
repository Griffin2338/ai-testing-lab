"""Binomial intervals, sample sizes and a bootstrap helper used by the checks."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats as _st


def _z(confidence: float, sided: str) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if sided == "two":
        return float(_st.norm.ppf(1 - (1 - confidence) / 2))
    if sided in ("lower", "upper"):
        return float(_st.norm.ppf(confidence))
    raise ValueError("sided must be 'two', 'lower' or 'upper'")


def wilson_interval(k: int, n: int, confidence: float = 0.95, sided: str = "two") -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    >>> lo, hi = wilson_interval(873, 900)
    >>> round(lo, 4), round(hi, 4)
    (0.9567, 0.9793)
    >>> round(wilson_interval(873, 900, sided="lower")[0], 4)
    0.9591

    sided="lower" returns (lo, 1.0), sided="upper" returns (0.0, hi).
    """
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    z = _z(confidence, sided)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    if sided == "lower":
        return (lo, 1.0)
    if sided == "upper":
        return (0.0, hi)
    return (lo, hi)


def clopper_pearson_interval(
    k: int, n: int, confidence: float = 0.95, sided: str = "two"
) -> tuple[float, float]:
    """Exact binomial interval. Wider than Wilson."""
    if n <= 0:
        return (0.0, 1.0)
    alpha = 1 - confidence
    a = alpha / 2 if sided == "two" else alpha
    lo = 0.0 if k == 0 else float(_st.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(_st.beta.ppf(1 - a, k + 1, n - k))
    if sided == "lower":
        return (lo, 1.0)
    if sided == "upper":
        return (0.0, hi)
    return (lo, hi)


def sample_size_for_margin(p: float, margin: float, confidence: float = 0.95) -> int:
    """Cases needed to estimate a proportion within +/- margin.

    >>> sample_size_for_margin(0.90, 0.02)
    865
    >>> sample_size_for_margin(0.5, 0.03)
    1068
    """
    z = _z(confidence, "two")
    return math.ceil(z * z * p * (1 - p) / (margin * margin))


def sample_size_one_sided(p0: float, p1: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Cases needed to show accuracy >= p0 if the true accuracy is p1.

    >>> sample_size_one_sided(0.95, 0.97)
    631
    """
    if not p1 > p0:
        raise ValueError("p1 must be greater than p0")
    za = float(_st.norm.ppf(1 - alpha))
    zb = float(_st.norm.ppf(power))
    num = (za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) ** 2
    return math.ceil(num / (p1 - p0) ** 2)


def zero_failure_sample_size(max_failure_rate: float, confidence: float = 0.95) -> int:
    """Failure-free trials needed to claim failure rate <= max_failure_rate (rule of three).

    >>> zero_failure_sample_size(0.01)
    299
    >>> zero_failure_sample_size(0.05)
    59
    """
    return math.ceil(math.log(1 - confidence) / math.log(1 - max_failure_rate))


def zero_failure_upper_bound(n: int, confidence: float = 0.95) -> float:
    """Upper bound on the failure rate after n clean trials. About 3/n at 95 %."""
    if n <= 0:
        return 1.0
    return 1 - (1 - confidence) ** (1 / n)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value.

    b: cases model A passes and model B fails. c: the opposite.
    """
    n = b + c
    if n == 0:
        return 1.0
    return float(_st.binomtest(min(b, c), n, 0.5).pvalue)


@dataclass
class BootstrapResult:
    estimate: float
    low: float
    high: float


def bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap CI of sum(num) / sum(den), resampling rows.

    Rows are images, so boxes from the same image stay together. With 2-D input
    each column (class) gets its own interval.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    if num.shape != den.shape:
        raise ValueError("numerator and denominator must have the same shape")
    squeeze = num.ndim == 1
    if squeeze:
        num, den = num[:, None], den[:, None]
    n = num.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        est = num.sum(0) / den.sum(0)
    if n == 0:
        nan = np.full(num.shape[1], np.nan)
        res = (nan, nan, nan)
    else:
        rng = np.random.default_rng(seed)
        weights = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = (weights @ num) / (weights @ den)
        a = (1 - confidence) / 2
        with warnings.catch_warnings():  # all-NaN column for classes with no GT
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = np.nanquantile(ratios, a, axis=0)
            hi = np.nanquantile(ratios, 1 - a, axis=0)
        res = (est, lo, hi)
    if squeeze:
        return BootstrapResult(float(res[0][0]), float(res[1][0]), float(res[2][0]))
    return BootstrapResult(res[0], res[1], res[2])  # type: ignore[arg-type]
