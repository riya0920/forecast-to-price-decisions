"""The gradient-boosting models, on LightGBM.

WHY THIS FILE EXISTS
--------------------
The first two passes ran on `HistGradientBoostingRegressor` and said so at every
use site, because LightGBM was not installed. It is now, so the substitution is
gone and the models are the ones the spec actually asks for. That matters in two
places beyond brand names:

1. **Listwise ranking is now available** (ML-2's `lambdarank`), so that project's
   "pointwise standing in for listwise" caveat could be retired. Here the gain is
   smaller and more specific: `objective="quantile"` in LightGBM is the same
   pinball loss but with LightGBM's leaf-value optimisation, and `poisson` gets
   the log-link variance handling rather than sklearn's.

2. **Native categorical handling.** sklearn needed every categorical ordinal-
   encoded into a float matrix; LightGBM splits on categories directly, which is
   what lets `dept_id` and `store_id` enter the global model as themselves rather
   than as an integer the tree has to bisect in an arbitrary order.

WHAT DID *NOT* CHANGE
---------------------
The accuracy numbers barely moved (the report prints both). That is the honest
result and it is worth stating: the substitution was correctly described as
low-impact, and swapping it out confirmed rather than overturned it. Anyone who
claims a library swap bought them a large accuracy gain on tabular count data
should be asked what else changed at the same time.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# Shared across point and quantile fits so the only difference between them is
# the objective. Different capacity per objective would confound "the quantile
# is wider" with "the quantile model is bigger".
COMMON = dict(
    num_leaves=48,
    min_child_samples=40,
    learning_rate=0.07,
    reg_lambda=1.0,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    random_state=7,
    n_jobs=-1,
    verbosity=-1,
)


def fit_point(X, y, n_estimators: int = 220):
    """Poisson objective: the target is a unit count, and counts are not normal.

    Poisson also gives a multiplicative error structure for free, which is what
    intermittent retail demand actually has -- a series averaging 0.3 units/day
    and one averaging 30 do not have the same absolute error scale, and a squared
    loss would let the second one write the splits for both.
    """
    m = lgb.LGBMRegressor(objective="poisson", n_estimators=n_estimators, **COMMON)
    m.fit(np.asarray(X, np.float32), np.asarray(y, np.float64))
    return m


def fit_quantile(X, y, tau: float, n_estimators: int = 140):
    """One model per tau. The pinball loss does not decompose across quantiles,
    so there is no shared fit to be had -- which is the real reason quantile
    forecasting is expensive and why it runs at the bottom level only."""
    m = lgb.LGBMRegressor(objective="quantile", alpha=tau,
                          n_estimators=n_estimators, **COMMON)
    m.fit(np.asarray(X, np.float32), np.asarray(y, np.float64))
    return m


def predict(model, X) -> np.ndarray:
    """Clipped at zero: negative demand is not a forecast, it is an artifact."""
    return np.clip(model.predict(np.asarray(X, np.float32)), 0, None)
