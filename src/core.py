"""Metrics, intermittency classification, and the baselines everything is scored against.

Three deliberate choices live here:

1. MAPE is not implemented. On a panel that is ~40% zeros it is not a weak
   metric, it is an undefined one -- the per-observation denominator is zero on
   two-fifths of the evaluation set. Any codebase that reports MAPE on M5-shaped
   data has not looked at the series. WMAPE (denominator summed, not
   per-observation) and MASE are used instead.

2. Bias is a first-class metric next to accuracy, not a footnote. Replenishment
   punishes error asymmetrically: over-forecast costs holding and markdown,
   under-forecast costs a stockout and possibly the customer. A model with 30%
   WMAPE and 0% bias is a better replenishment input than one with 28% WMAPE
   and +9% bias.

3. Every accuracy number is reported as skill against a named baseline
   (Forecast Value Added). An absolute WMAPE is not interpretable; "-4.1% WMAPE
   vs seasonal naive" is.
"""
from __future__ import annotations

import numpy as np

SEASON = 7  # daily retail data: the season that matters is the week


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def wmape(y, yhat) -> float:
    """Weighted MAPE = sum|e| / sum(y). Defined when individual y are zero."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y - yhat).sum() / denom)


def bias_pct(y, yhat) -> float:
    """Signed forecast bias. Positive = systematically over-forecasting."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y).sum()
    if denom == 0:
        return float("nan")
    return float((yhat - y).sum() / denom)


def mase(y, yhat, y_train, season: int = SEASON) -> float:
    """MASE with a SEASONAL naive denominator (in-sample, one full season lag).

    Scale-free and finite on intermittent series, which is exactly why it
    replaces MAPE here. Reading: 0.85 means the model's average absolute error
    is 85% of what you would have got by saying "same as this day last week",
    i.e. a 15% improvement. Above 1.0 means the naive rule was better.
    """
    y_train = np.asarray(y_train, float)
    if len(y_train) <= season:
        return float("nan")
    scale = np.abs(y_train[season:] - y_train[:-season]).mean()
    if scale == 0:
        return float("nan")
    return float(np.abs(np.asarray(y, float) - np.asarray(yhat, float)).mean() / scale)


def rmsse(y, yhat, y_train, season: int = SEASON) -> float:
    """M5's own metric (the squared-error sibling of MASE), for comparability."""
    y_train = np.asarray(y_train, float)
    if len(y_train) <= season:
        return float("nan")
    scale = np.mean((y_train[season:] - y_train[:-season]) ** 2)
    if scale == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2) / scale))


# --------------------------------------------------------------------------
# intermittency: the Syntetos-Boylan classification
# --------------------------------------------------------------------------
ADI_CUT = 1.32
CV2_CUT = 0.49


def adi_cv2(y) -> tuple[float, float]:
    """Average Demand Interval and squared coefficient of variation of nonzero demand."""
    y = np.asarray(y, float)
    nz = y[y > 0]
    if len(nz) == 0:
        return float("inf"), 0.0
    adi = len(y) / len(nz)
    cv2 = float((nz.std() / nz.mean()) ** 2) if nz.mean() > 0 else 0.0
    return float(adi), cv2


def classify(y) -> str:
    """Syntetos-Boylan quadrants at the standard ADI=1.32 / CV^2=0.49 cuts.

    smooth       -> regular demand, regular size: ordinary methods work
    erratic      -> regular timing, wild sizes
    intermittent -> irregular timing, regular sizes: Croston territory
    lumpy        -> irregular timing AND wild sizes: nothing works well, and
                    saying so is more useful than pretending otherwise
    """
    adi, cv2 = adi_cv2(y)
    if adi < ADI_CUT and cv2 < CV2_CUT:
        return "smooth"
    if adi < ADI_CUT and cv2 >= CV2_CUT:
        return "erratic"
    if adi >= ADI_CUT and cv2 < CV2_CUT:
        return "intermittent"
    return "lumpy"


# --------------------------------------------------------------------------
# baselines -- built first, reported forever after
# --------------------------------------------------------------------------
def naive(y_train, h: int) -> np.ndarray:
    """Last observed value, carried forward."""
    return np.repeat(float(y_train[-1]), h)


def seasonal_naive(y_train, h: int, season: int = SEASON) -> np.ndarray:
    """Same day last week. The baseline that is genuinely hard to beat."""
    y_train = np.asarray(y_train, float)
    if len(y_train) < season:
        return naive(y_train, h)
    last = y_train[-season:]
    return np.array([last[i % season] for i in range(h)])


def ses(y_train, h: int, alpha: float | None = None) -> np.ndarray:
    """Simple exponential smoothing; alpha grid-searched on in-sample SSE if unset."""
    y_train = np.asarray(y_train, float)
    if len(y_train) == 0:
        return np.zeros(h)
    if alpha is None:
        best, best_sse = 0.1, np.inf
        for a in np.arange(0.05, 0.96, 0.05):
            level, sse = y_train[0], 0.0
            for v in y_train[1:]:
                sse += (v - level) ** 2
                level = a * v + (1 - a) * level
            if sse < best_sse:
                best, best_sse = a, sse
        alpha = best
    level = y_train[0]
    for v in y_train[1:]:
        level = alpha * v + (1 - alpha) * level
    return np.repeat(float(level), h)


def croston(y_train, h: int, alpha: float = 0.1, variant: str = "sba") -> np.ndarray:
    """Croston / Syntetos-Boylan Approximation for intermittent demand.

    Croston decomposes the series into demand SIZE and demand INTERVAL, smooths
    each separately, and forecasts size/interval. The original estimator is
    biased upward (Syntetos & Boylan 2001); SBA applies the (1 - alpha/2)
    correction. On a replenishment system that bias is not academic -- it is
    systematically ordering too much of every slow mover in the assortment, so
    'sba' is the default here.
    """
    y = np.asarray(y_train, float)
    nz_idx = np.flatnonzero(y > 0)
    if len(nz_idx) == 0:
        return np.zeros(h)
    if len(nz_idx) == 1:
        return np.repeat(y[nz_idx[0]] / len(y), h)

    z = y[nz_idx[0]]            # smoothed demand size
    p = float(nz_idx[0] + 1)    # smoothed inter-arrival interval
    last = nz_idx[0]
    for i in nz_idx[1:]:
        z = alpha * y[i] + (1 - alpha) * z
        p = alpha * (i - last) + (1 - alpha) * p
        last = i
    rate = z / max(p, 1e-9)
    if variant == "sba":
        rate *= (1 - alpha / 2)
    return np.repeat(float(rate), h)


BASELINES = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "ses": ses,
    "croston_sba": croston,
}
