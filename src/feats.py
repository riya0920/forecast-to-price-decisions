"""Feature construction for the direct multi-horizon GBM.

The horizon is 28 days, so every lag used is >= 28 days old. That is not a
stylistic choice, it is what makes the model *direct* rather than recursive: at
forecast origin t, the feature for target day t+27 uses information from t-1 at
the newest. No recursion, no error compounding, and no way for a day inside the
horizon to leak into its own feature row.

Prices and promotions ARE included for the forecast period, and that is
realistic rather than cheating: a retailer's promo calendar and price plan are
set weeks ahead of the ship date. If the planned promo calendar were unknown at
forecast time the whole pricing half of this project would be unbuildable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_LAG = 28
LAGS = [28, 29, 30, 35, 42, 56, 364]
ROLL_WINDOWS = [7, 28, 56]

EVENTS = ["ThanksgivingRunup", "Thanksgiving", "BlackFriday", "ChristmasRunup",
          "Christmas", "NewYearEve", "July4", "SuperBowl"]


def long_frame(panel: pd.DataFrame, exog: dict, calendar: pd.DataFrame) -> pd.DataFrame:
    """Wide (date x series) -> long feature table."""
    df = panel.stack().rename("y").reset_index()
    df.columns = ["date", "key", "y"]
    for name in ("price", "promo", "snap"):
        s = exog[name].stack().rename(name).reset_index()
        s.columns = ["date", "key", name]
        df = df.merge(s, on=["date", "key"], how="left")
    df = df.merge(calendar[["date", "dow", "doy", "month", "event"]], on="date", how="left")
    return df.sort_values(["key", "date"], ignore_index=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("key", observed=True).y

    for L in LAGS:
        df["lag_%d" % L] = g.shift(L)
    base = g.shift(MIN_LAG)
    for w in ROLL_WINDOWS:
        df["rmean_%d" % w] = base.groupby(df.key, observed=True).transform(
            lambda s, w=w: s.rolling(w, min_periods=max(2, w // 4)).mean())
        df["rstd_%d" % w] = base.groupby(df.key, observed=True).transform(
            lambda s, w=w: s.rolling(w, min_periods=max(2, w // 4)).std())
    # share of the trailing window with zero demand: the intermittency signal
    # the model would otherwise have to rediscover from lags
    df["zero_share_56"] = base.groupby(df.key, observed=True).transform(
        lambda s: s.rolling(56, min_periods=14).apply(lambda v: (v == 0).mean(), raw=True))

    # Mean of the SAME WEEKDAY over the previous 4 available weeks. Shifted 4
    # positions within the (key, dow) series, which is 28 calendar days -- so it
    # respects the same origin cutoff as every other lag here.
    df["dow_mean_4w"] = (df.groupby(["key", "dow"], observed=True).y
                           .transform(lambda s: s.shift(4).rolling(4, min_periods=2).mean()))

    # calendar as cyclical, not integer-coded
    df["doy_sin"] = np.sin(2 * np.pi * df.doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df.doy / 365.25)
    df["is_weekend"] = df.dow.isin([5, 6]).astype(int)

    # price relative to the item's own recent shelf price -- the level a
    # demand model can actually use; absolute price is a series id in disguise
    pg = df.groupby("key", observed=True).price
    df["price_rel_56"] = df.price / pg.transform(
        lambda s: s.rolling(56, min_periods=7).mean())
    df["price_chg_7"] = df.price / pg.shift(7) - 1
    df["log_price"] = np.log(df.price.clip(lower=0.01))

    for e in EVENTS:
        df["ev_" + e] = (df.event == e).astype(np.int8)
    # a demand model that only knows the holiday itself misses the run-up, which
    # is where the volume actually is
    df["days_to_xmas"] = (pd.Timestamp("2000-12-25").dayofyear - df.doy).abs().clip(upper=45)
    return df


FEATURE_COLS = (
    ["lag_%d" % L for L in LAGS]
    + ["rmean_%d" % w for w in ROLL_WINDOWS]
    + ["rstd_%d" % w for w in ROLL_WINDOWS]
    + ["zero_share_56", "dow_mean_4w", "dow", "month", "doy_sin", "doy_cos",
       "is_weekend", "promo", "snap", "price_rel_56", "price_chg_7", "log_price",
       "days_to_xmas"]
    + ["ev_" + e for e in EVENTS]
)
