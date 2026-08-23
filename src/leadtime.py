"""Lead-time demand quantiles -- the conversion the last pass said it could not do.

THE PROBLEM, STATED PRECISELY
-----------------------------
`quantiles.py` produces a calibrated P95 for *one day*. Replenishment does not
order for one day; it orders to cover a lead time L, so what it needs is the P95
of the SUM of demand over the next L days.

    P95( D_1 + ... + D_L )   is NOT   P95(D_1) + ... + P95(D_L)

Summing daily quantiles assumes every day goes wrong together -- perfect
dependence. Unless demand is perfectly correlated across days (it is not), the
sum of quantiles is an OVERSTATEMENT and orders too much. But the equally common
alternative, treating the days as independent and scaling by sqrt(L), is an
UNDERSTATEMENT whenever demand is positively autocorrelated -- which it is,
because promotions, weather and weekday effects all persist. The two standard
shortcuts err in opposite directions, and which one is worse depends on the
series.

WHAT THIS DOES INSTEAD
----------------------
It resamples **contiguous blocks of forecast-error paths** and adds them to the
point forecast. Blocks rather than individual days is the entire point: a block
carries whatever cross-day dependence the errors actually have, without anyone
having to name a correlation structure or fit a copula.

The errors are standardised by the point forecast before resampling and
re-inflated after, so a series averaging 0.3 units/day and one averaging 30 can
share the same error pool. That is what makes this estimable at all -- per-series
error paths at the bottom level are far too few to bootstrap on their own.

THIS IS THE ML-1 -> DATA-2 JOIN
-------------------------------
DATA-2 sized its safety stock from HISTORICAL demand mean and sd, and its own
report named that as the single most valuable missing piece: what you actually
want is the distribution of demand over the lead time implied by *the forecast
and its errors*. That distribution is what this module returns, and DATA-2 now
consumes it.
"""
from __future__ import annotations

import numpy as np

DEFAULT_BLOCK = 7      # one week: long enough to carry weekday and promo runs
MIN_POOL_PATHS = 30


def standardise(errors: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Scale errors by the forecast level so series of different sizes pool.

    Divided by sqrt(point) rather than point itself, because the target is a
    count: for Poisson-ish demand the standard deviation grows like the square
    root of the mean, so sqrt is the transform that actually makes the pooled
    errors homoscedastic. Dividing by the mean would over-correct and make the
    big series look artificially calm.
    """
    scale = np.sqrt(np.clip(point, 0.25, None))
    return errors / scale


def _blocks(pool: np.ndarray, length: int, block: int,
            rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw n paths of `length` standardised errors by tiling random blocks."""
    T = pool.shape[0]
    if T < block:
        block = max(1, T)
    n_blocks = int(np.ceil(length / block))
    starts = rng.integers(0, max(T - block + 1, 1), size=(n, n_blocks))
    idx = starts[:, :, None] + np.arange(block)[None, None, :]
    idx = np.clip(idx, 0, T - 1)
    return pool[idx].reshape(n, -1)[:, :length]


def leadtime_samples(point_path: np.ndarray, error_pool: np.ndarray,
                     n_samples: int = 4000, block: int = DEFAULT_BLOCK,
                     seed: int = 0) -> np.ndarray:
    """Monte-Carlo the total demand over a lead time.

    point_path : (L,)  point forecast for each of the L lead-time days
    error_pool : (T,)  standardised residuals, in TIME ORDER (order is the
                       information the block bootstrap exists to use -- shuffling
                       this array silently turns the method back into iid)
    """
    point_path = np.asarray(point_path, float)
    L = len(point_path)
    pool = np.asarray(error_pool, float)
    pool = pool[np.isfinite(pool)]
    if len(pool) < MIN_POOL_PATHS:
        raise ValueError("error pool too small to bootstrap: %d" % len(pool))

    rng = np.random.default_rng(seed)
    z = _blocks(pool, L, block, rng, n_samples)          # (n, L)
    scale = np.sqrt(np.clip(point_path, 0.25, None))[None, :]
    daily = np.clip(point_path[None, :] + z * scale, 0, None)
    return daily.sum(axis=1)


def leadtime_quantile(point_path, error_pool, tau: float, **kw) -> float:
    return float(np.quantile(leadtime_samples(point_path, error_pool, **kw), tau))


def sum_of_daily_quantiles(daily_quantiles: np.ndarray) -> float:
    """The shortcut this module exists to price. Assumes perfect dependence."""
    return float(np.sum(daily_quantiles))


def iid_scaled_quantile(point_path, sigma_daily: float, tau: float) -> float:
    """The other shortcut: normal approximation with independent days.

    sd over L days = sigma_daily * sqrt(L). Understates whenever demand is
    positively autocorrelated, which is the usual direction.
    """
    from scipy.stats import norm
    L = len(point_path)
    return float(np.sum(point_path) + norm.ppf(tau) * sigma_daily * np.sqrt(L))


def autocorrelation(pool: np.ndarray, lag: int = 1) -> float:
    """The statistic that decides which shortcut is worse. Reported, not assumed."""
    x = np.asarray(pool, float)
    x = x[np.isfinite(x)]
    if len(x) <= lag + 2:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def compare_methods(point_path, error_pool, daily_quantiles, tau: float,
                    **kw) -> dict:
    """All three answers side by side, which is the only way the gap is visible."""
    boot = leadtime_quantile(point_path, error_pool, tau, **kw)
    pool = np.asarray(error_pool, float)
    sigma = float(np.nanstd(pool) * np.sqrt(np.clip(np.mean(point_path), 0.25, None)))
    return dict(
        tau=tau,
        mean_leadtime_demand=float(np.sum(point_path)),
        block_bootstrap=boot,
        sum_of_daily_quantiles=sum_of_daily_quantiles(daily_quantiles),
        iid_normal=iid_scaled_quantile(point_path, sigma, tau),
        lag1_autocorr=autocorrelation(pool, 1),
    )
