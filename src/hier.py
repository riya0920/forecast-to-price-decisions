"""The hierarchy, and reconciliation across it.

A retail forecast is not one number, it is a coherent set: what the item-store
forecasts add up to has to be what the store forecast says, or replenishment and
finance are working off two different plans. Bottom-up is the free way to get
coherence. MinT is the way that also uses the information in the aggregate
series -- aggregates are smoother and often more forecastable than the noisy
bottom level, and MinT lets that information flow back down.

Levels (a strict tree, 353 series):

    Total                     1
    State                     2
    Store                     5
    Store x Category         15
    Store x Department       30
    Store x Item            300   <- bottom
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = ["total", "state", "store", "store_cat", "store_dept", "store_item"]


def series_key(df: pd.DataFrame, level: str) -> pd.Series:
    if level == "total":
        return pd.Series("TOTAL", index=df.index)
    if level == "state":
        return df.state_id
    if level == "store":
        return df.store_id
    if level == "store_cat":
        return df.store_id + "|" + df.cat_id
    if level == "store_dept":
        return df.store_id + "|" + df.dept_id
    if level == "store_item":
        return df.store_id + "|" + df.item_id
    raise ValueError(level)


def build_panels(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One wide (date x series) frame of unit sales per level."""
    out = {}
    for lv in LEVELS:
        k = series_key(df, lv)
        g = df.assign(_k=k).groupby(["date", "_k"], observed=True).sales.sum().unstack("_k")
        out[lv] = g.sort_index()
    return out


def build_exog(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Level-appropriate price/promo exogenous features.

    Price only exists at the bottom. Rolling it up is a modelling decision, not
    an accident: an aggregate 'price' is a unit-weighted average sell price and
    an aggregate 'promo' is the share of that group's units currently on deal.
    Both are what a merchant actually looks at for a department.
    """
    out = {}
    for lv in LEVELS:
        k = series_key(df, lv)
        d = df.assign(_k=k, _rev=df.sales * df.sell_price,
                      _pu=df.sales * df.promo)
        agg = d.groupby(["date", "_k"], observed=True).agg(
            units=("sales", "sum"), rev=("_rev", "sum"), promo_units=("_pu", "sum"),
            price_mean=("sell_price", "mean"), promo_mean=("promo", "mean"))

        # Unit-weighted where units sold; plain mean of the shelf price where
        # nothing sold. The fallback matters: sell price and promo flag are KNOWN
        # on a zero-sales day, and a units-weighted-only definition would discard
        # them on exactly the intermittent series that need them most.
        u = agg.units.replace(0, np.nan)
        price = ((agg.rev / u).fillna(agg.price_mean)
                 .unstack("_k").sort_index().ffill().bfill())
        promo_share = ((agg.promo_units / u).fillna(agg.promo_mean)
                       .unstack("_k").sort_index().fillna(0.0))
        snap = d.groupby(["date", "_k"], observed=True).snap.max().unstack("_k").sort_index()
        out[lv] = {"price": price, "promo": promo_share, "snap": snap.fillna(0)}
    return out


def summing_matrix(panels: dict[str, pd.DataFrame], df: pd.DataFrame) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """S: (n_all x n_bottom) 0/1 matrix mapping bottom series to every level."""
    bottom_cols = list(panels["store_item"].columns)
    b_index = {c: i for i, c in enumerate(bottom_cols)}

    # map each bottom series to its ancestors
    meta = (df[["store_id", "item_id", "dept_id", "cat_id", "state_id"]]
            .drop_duplicates()
            .assign(bkey=lambda x: x.store_id + "|" + x.item_id))
    meta = meta.set_index("bkey")

    rows, names = [], []
    for lv in LEVELS:
        for col in panels[lv].columns:
            r = np.zeros(len(bottom_cols))
            for bkey, m in meta.iterrows():
                if lv == "total":
                    hit = True
                elif lv == "state":
                    hit = m.state_id == col
                elif lv == "store":
                    hit = m.store_id == col
                elif lv == "store_cat":
                    hit = (m.store_id + "|" + m.cat_id) == col
                elif lv == "store_dept":
                    hit = (m.store_id + "|" + m.dept_id) == col
                else:
                    hit = bkey == col
                if hit:
                    r[b_index[bkey]] = 1.0
            rows.append(r)
            names.append((lv, col))
    return np.vstack(rows), names


def bottom_up(S: np.ndarray, bottom_fc: np.ndarray) -> np.ndarray:
    """Coherent by construction. bottom_fc: (n_bottom x h) -> (n_all x h)."""
    return S @ bottom_fc


def _shrink_cov(resid: np.ndarray) -> np.ndarray:
    """Schafer-Strimmer style shrinkage of the residual covariance to its diagonal.

    The raw covariance of 353 series estimated from a few hundred residuals is
    rank-deficient and cannot be inverted; shrinking toward the diagonal is the
    standard MinT(shrink) fix and is why MinT is usable at all at retail widths.
    """
    n, T = resid.shape
    resid = resid - resid.mean(axis=1, keepdims=True)
    S_hat = (resid @ resid.T) / max(T - 1, 1)
    d = np.diag(np.diag(S_hat))
    # shrinkage intensity from the variance of the sample correlations
    sd = np.sqrt(np.diag(S_hat))
    sd[sd == 0] = 1e-9
    R = S_hat / np.outer(sd, sd)
    off = ~np.eye(n, dtype=bool)
    var_r = np.var(R[off]) if off.sum() else 0.0
    lam = float(np.clip(var_r / (np.sum(R[off] ** 2) / max(off.sum(), 1) + 1e-12), 0.05, 1.0))
    return lam * d + (1 - lam) * S_hat


def mint(S: np.ndarray, base_fc: np.ndarray, resid: np.ndarray,
         method: str = "shrink") -> np.ndarray:
    """MinT reconciliation. base_fc: (n_all x h) incoherent base forecasts.

    yhat_tilde = S (S' W^-1 S)^-1 S' W^-1 yhat_base
    """
    n_all = S.shape[0]
    if method == "ols":
        W = np.eye(n_all)
    elif method == "wls":
        v = resid.var(axis=1)
        v[v <= 0] = 1e-6
        W = np.diag(v)
    else:
        W = _shrink_cov(resid)
        W = W + np.eye(n_all) * (np.trace(W) / n_all) * 1e-6  # ridge for safety

    Wi = np.linalg.pinv(W)
    M = S @ np.linalg.pinv(S.T @ Wi @ S) @ S.T @ Wi
    return M @ base_fc


def coherence_error(S: np.ndarray, fc: np.ndarray) -> float:
    """Max absolute violation of the adding-up constraints. 0 == coherent."""
    bottom = fc[-S.shape[1]:]
    return float(np.abs(fc - S @ bottom).max())
