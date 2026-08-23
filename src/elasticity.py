"""Elasticity below the category: per item, per segment, and across items.

WHAT CHANGED UNDER THIS
-----------------------
The generator used to give every item in a category one elasticity, so "estimate
elasticity per item" was a question with no right answer -- any dispersion an
estimator reported was noise by construction. It now draws each item's own
elasticity around its category's (lognormal, so the sign survives) and makes each
item's demand depend on its department siblings' prices. Both are recorded in
TRUTH.json, so both are scored rather than asserted.

THE THREE QUESTIONS
-------------------
1. **Is per-item elasticity recoverable?** Not "can I fit 60 regressions" -- of
   course I can -- but does the resulting dispersion track the real dispersion,
   or is it noise dressed as heterogeneity? The test is the correlation with
   truth and, more usefully, whether SHRINKING each item toward its category mean
   beats the raw estimate. If shrinkage all the way to the category mean wins,
   the honest answer is that this panel cannot support per-item pricing.

2. **Does elasticity vary by segment?** Promo vs non-promo weeks, and by demand
   tier. A single number per item is itself an aggregation.

3. **Cross-price.** Recovering the substitution coefficient is what makes the
   markdown optimiser's cannibalisation term more than decoration.

WHY STATSMODELS NOW
-------------------
`statsmodels` was unavailable for the first two passes, so OLS with HC0 errors
was written out by hand and GLMs went through `PoissonRegressor` (which is a
penalised fit with no inference attached). It is installed now, so these are
real GLMs with real standard errors -- which matters here specifically, because
question 1 is about whether the dispersion exceeds its own standard error, and
that question cannot be asked of a point estimate with no se.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

MIN_WEEKS = 60          # below this an item's own regression is not worth fitting


def _weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly aggregation. The price decision is weekly, and it keeps the
    fixed-effect design small enough to fit 60 times."""
    d = df.copy()
    d["key"] = d.store_id + "|" + d.item_id
    w = (d.groupby(["item_id", "dept_id", "cat_id", "key",
                    pd.Grouper(key="date", freq="W")])
          .agg(sales=("sales", "sum"), price=("sell_price", "mean"),
               promo=("promo", "mean"), snap=("snap", "mean"))
          .reset_index())
    w["woy"] = w.date.dt.isocalendar().week.astype(int)
    w["log_p"] = np.log(w.price.clip(lower=0.01))
    return w


def _fit_poisson(y, X):
    """Poisson GLM, log link, HC0 covariance.

    HC0 rather than the model-based covariance because the gamma mixing in the
    generator makes the counts over-dispersed relative to Poisson -- the point
    estimate is still consistent under the mean specification, but the
    model-based standard errors would be too small by roughly the square root of
    the dispersion. Reporting an se that is confidently wrong is worse than
    reporting none, because question 1 is decided on that se.
    """
    m = sm.GLM(np.asarray(y, float), np.asarray(X, float),
               family=sm.families.Poisson())
    return m.fit(cov_type="HC0", maxiter=200)


def _design(sub: pd.DataFrame, extra: dict | None = None):
    """log price + promo + snap + store FE + week-of-year FE + intercept."""
    parts = {"log_p": sub.log_p.to_numpy(float),
             "promo": sub.promo.to_numpy(float),
             "snap": sub.snap.to_numpy(float)}
    if extra:
        parts.update(extra)
    X = pd.DataFrame(parts, index=sub.index)
    X = pd.concat([X,
                   pd.get_dummies(sub.key, prefix="k", drop_first=True).astype(float),
                   pd.get_dummies(sub.woy, prefix="w", drop_first=True).astype(float)],
                  axis=1)
    X.insert(0, "const", 1.0)
    return X


# --------------------------------------------------------------------------
# 1. per item
# --------------------------------------------------------------------------
def per_item(df: pd.DataFrame, truth_by_item: dict) -> pd.DataFrame:
    w = _weekly(df)
    rows = []
    for item, sub in w.groupby("item_id", observed=True):
        if len(sub) < MIN_WEEKS:
            continue
        X = _design(sub)
        try:
            res = _fit_poisson(sub.sales, X)
        except Exception:
            continue
        i = list(X.columns).index("log_p")
        rows.append(dict(item_id=item, cat_id=sub.cat_id.iloc[0],
                         dept_id=sub.dept_id.iloc[0],
                         estimate=float(res.params[i]), se=float(res.bse[i]),
                         n=len(sub), truth=float(truth_by_item[item])))
    return pd.DataFrame(rows)


def shrink(est: np.ndarray, se: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Empirical-Bayes shrinkage of each item toward its group's mean.

    weight = tau^2 / (tau^2 + se^2), where tau^2 is the estimated BETWEEN-item
    variance: total observed dispersion minus the average sampling variance. If
    the observed dispersion is no larger than the noise, tau^2 clamps to zero and
    every item collapses to the group mean -- which is the estimator saying, in
    its own arithmetic, that there is no heterogeneity here to price on.
    """
    est, se = np.asarray(est, float), np.asarray(se, float)
    out = np.empty_like(est)
    for g in np.unique(group):
        m = group == g
        mu = est[m].mean()
        tau2 = max(est[m].var(ddof=1) - np.mean(se[m] ** 2), 0.0)
        wt = tau2 / (tau2 + se[m] ** 2) if tau2 > 0 else np.zeros(m.sum())
        out[m] = mu + wt * (est[m] - mu)
    return out


def score_item_estimates(tab: pd.DataFrame) -> pd.DataFrame:
    """Raw vs shrunk vs category-mean, scored against the per-item truth.

    The category-mean row is the one that matters: it is what the previous pass
    did, and if it wins then per-item elasticity is not estimable on this panel
    however sophisticated the fit.
    """
    cat_mean = tab.groupby("cat_id", observed=True).estimate.transform("mean")
    shrunk = shrink(tab.estimate.values, tab.se.values, tab.cat_id.values)
    truth = tab.truth.values
    rows = []
    for name, est in (("raw_per_item", tab.estimate.values),
                      ("shrunk_to_category", shrunk),
                      ("category_mean_only", cat_mean.values)):
        rows.append(dict(
            estimator=name,
            mae=float(np.mean(np.abs(est - truth))),
            rmse=float(np.sqrt(np.mean((est - truth) ** 2))),
            corr_with_truth=float(np.corrcoef(est, truth)[0, 1])
            if np.std(est) > 1e-12 else float("nan"),
            sd_of_estimates=float(np.std(est)),
        ))
    # WITHIN-category metrics. The pooled correlation is close to useless here
    # and it is important to say why: category elasticities are -2.1 / -1.2 /
    # -0.6, so almost all the variance in the truth is BETWEEN categories, and
    # any estimator that gets the three category means roughly right scores a
    # high correlation while knowing nothing about any individual SKU. The
    # within-category number is the one that answers "can I price this item
    # differently from its neighbour on the same shelf".
    dev_t = truth - tab.groupby("cat_id", observed=True).truth.transform("mean").values
    for i, (name, est) in enumerate((("raw_per_item", tab.estimate.values),
                                     ("shrunk_to_category", shrunk),
                                     ("category_mean_only", cat_mean.values))):
        e = pd.Series(est, index=tab.index)
        dev_e = (e - e.groupby(tab.cat_id).transform("mean")).values
        rows[i]["within_cat_mae"] = float(np.mean(np.abs(dev_e - dev_t)))
        rows[i]["within_cat_corr"] = (float(np.corrcoef(dev_e, dev_t)[0, 1])
                                      if np.std(dev_e) > 1e-12 else 0.0)
    out = pd.DataFrame(rows)
    out.attrs["sd_of_truth"] = float(np.std(truth))
    out.attrs["sd_of_truth_within"] = float(np.std(dev_t))
    return out


# --------------------------------------------------------------------------
# 2. by segment
# --------------------------------------------------------------------------
def by_segment(df: pd.DataFrame, truth_by_item: dict) -> pd.DataFrame:
    """Elasticity estimated separately on promo and non-promo weeks, and by tier.

    A promo-week elasticity is not the same object as a shelf-price elasticity --
    it bundles the display lift with the price cut, which is exactly the
    confound the promo control exists to remove. Splitting rather than
    controlling makes the size of that bundling visible.
    """
    w = _weekly(df)
    w["segment_promo"] = np.where(w.promo > 0.3, "promo_weeks", "shelf_weeks")
    vol = w.groupby("item_id", observed=True).sales.mean()
    w["segment_tier"] = np.where(
        w.item_id.map(vol) >= vol.median(), "high_volume", "low_volume")

    rows = []
    for col in ("segment_promo", "segment_tier"):
        for (cat, seg), sub in w.groupby(["cat_id", col], observed=True):
            if len(sub) < MIN_WEEKS:
                continue
            X = _design(sub)
            try:
                res = _fit_poisson(sub.sales, X)
            except Exception:
                continue
            i = list(X.columns).index("log_p")
            tr = float(np.mean([truth_by_item[k] for k in sub.item_id.unique()]))
            rows.append(dict(dimension=col, category=cat, segment=seg,
                             estimate=float(res.params[i]), se=float(res.bse[i]),
                             truth_mean=tr, bias=float(res.params[i]) - tr,
                             n=len(sub)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. cross-price
# --------------------------------------------------------------------------
def rival_index(df: pd.DataFrame) -> pd.DataFrame:
    """Mean log relative price of an item's DEPARTMENT SIBLINGS, per store-day.

    Computed as (department total of log-relative-price minus mine) / (k-1),
    which is O(n) rather than the O(k) per row a groupby-apply would cost -- and
    more importantly is exactly the quantity the generator used, so a failure to
    recover the coefficient is a failure of the estimator and not of the feature.
    """
    d = df.copy()
    base = d.groupby("item_id", observed=True).sell_price.transform("median")
    d["log_rel"] = np.log(d.sell_price / base)
    g = d.groupby(["date", "store_id", "dept_id"], observed=True).log_rel
    tot, k = g.transform("sum"), g.transform("count")
    d["rival_log_rel"] = np.where(k > 1, (tot - d.log_rel) / (k - 1), 0.0)
    return d


def _daily_design(s: pd.DataFrame, with_rival: bool, calendar_fe: bool):
    """Daily design matrix. Cross-price has to be daily -- see `cross_price`."""
    parts = {"log_p": np.log(s.sell_price.clip(lower=0.01)).to_numpy(float),
             "promo": s.promo.to_numpy(float),
             "snap": s.snap.to_numpy(float)}
    if with_rival:
        parts["rival_log_rel"] = s.rival_log_rel.to_numpy(float)
    X = [pd.DataFrame(parts, index=s.index),
         pd.get_dummies(s.key, prefix="k", drop_first=True).astype(float),
         pd.get_dummies(s.woy, prefix="w", drop_first=True).astype(float)]
    if calendar_fe:
        X.append(pd.get_dummies(s.event, prefix="e", drop_first=True).astype(float))
        X.append(pd.get_dummies(s.dow_, prefix="d", drop_first=True).astype(float))
    out = pd.concat(X, axis=1)
    out.insert(0, "const", 1.0)
    return out


def _prep_daily(df: pd.DataFrame, category: str | None) -> pd.DataFrame:
    d = rival_index(df)
    if category:
        d = d[d.cat_id == category]
    d = d.copy()
    d["key"] = d.store_id + "|" + d.item_id
    d["woy"] = d.date.dt.isocalendar().week.astype(int)
    d["dow_"] = d.date.dt.dayofweek
    return d


def cross_price(df: pd.DataFrame, category: str | None = None,
                calendar_fe: bool = True) -> dict:
    """Own-price and cross-price elasticity in one specification.

    DAILY, not weekly, and that is the identifying choice rather than a
    performance footnote. The confound is the one this project already documents
    for own-price, arriving somewhere new: promotions are scheduled into strong
    periods, and what makes a period strong here is EVENTS and DAY OF WEEK, both
    of which are shared across an entire department. A rival's promotion
    therefore lands disproportionately on days my own demand is high for reasons
    that have nothing to do with the rival, and the regression charges that
    co-movement to the cross term -- with a NEGATIVE sign, because the rival's
    price is low exactly when my demand is high.

    Week-of-year fixed effects cannot absorb it. Thanksgiving moves within the
    ISO week from year to year and day-of-week variation is inside the week by
    definition, so weekly aggregation destroys precisely the variation the
    control needs. `calendar_fe=False` reproduces the attenuated version, and
    the two side by side are the finding.
    """
    s = _prep_daily(df, category)
    X = _daily_design(s, True, calendar_fe)
    res = _fit_poisson(s.sales, X)
    cols = list(X.columns)
    return dict(
        own=float(res.params[cols.index("log_p")]),
        own_se=float(res.bse[cols.index("log_p")]),
        cross=float(res.params[cols.index("rival_log_rel")]),
        cross_se=float(res.bse[cols.index("rival_log_rel")]),
        n=int(len(s)), calendar_fe=bool(calendar_fe),
    )


def cross_price_omitted(df: pd.DataFrame, category: str | None = None,
                        calendar_fe: bool = True) -> dict:
    """The same model WITHOUT the rival term -- the specification everyone runs.

    Omitting a regressor correlated with price biases the own-price coefficient
    by (cross elasticity) x (regression of rival price on own price). Department
    prices co-move -- shared cost shocks, shared promo calendars -- so that
    second factor is nonzero and the bias has a predictable sign. The pair of
    numbers prices the omission instead of arguing about it.
    """
    s = _prep_daily(df, category)
    X = _daily_design(s, False, calendar_fe)
    res = _fit_poisson(s.sales, X)
    i = list(X.columns).index("log_p")
    return dict(own=float(res.params[i]), own_se=float(res.bse[i]),
                n=int(len(s)), calendar_fe=bool(calendar_fe))
