"""Guards on the claims this project makes.

The leakage test is the one that matters. Everything else in a forecasting repo
can be inspected by reading it; whether a feature row for a date inside the
horizon secretly depends on that date's own future cannot. So it is tested by
corrupting the future and asserting the features do not move.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import core, feats, hier  # noqa: E402


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def test_wmape_is_defined_on_zero_heavy_series():
    y = np.array([0, 0, 0, 5, 0, 0, 3])
    yhat = np.array([1, 0, 0, 4, 0, 1, 2])
    assert np.isfinite(core.wmape(y, yhat))
    # the equivalent MAPE would divide by zero on 5 of 7 observations
    assert (y == 0).sum() == 5


def test_wmape_matches_hand_computation():
    y = np.array([10.0, 0.0, 5.0])
    yhat = np.array([8.0, 2.0, 5.0])
    assert core.wmape(y, yhat) == pytest.approx((2 + 2 + 0) / 15)


def test_bias_sign_convention():
    y = np.array([10.0, 10.0])
    assert core.bias_pct(y, np.array([11.0, 11.0])) > 0   # over-forecast
    assert core.bias_pct(y, np.array([9.0, 9.0])) < 0     # under-forecast


def test_mase_of_seasonal_naive_is_about_one():
    """A MASE near 1.0 for the method that DEFINES the denominator is the check
    that the denominator is what it claims to be.

    Averaged over seeds rather than asserted on one: on a single draw the last
    observed week can happen to sit near the series mean, which makes seasonal
    naive look far better than it is (the first version of this test asserted on
    seed 0 and read 0.55). Averaging is the honest form of the claim.
    """
    vals = []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        y_train = rng.poisson(4, size=400).astype(float)
        y_true = rng.poisson(4, size=28).astype(float)
        vals.append(core.mase(y_true, core.seasonal_naive(y_train, 28), y_train))
    assert 0.9 < float(np.mean(vals)) < 1.1


# --------------------------------------------------------------------------
# intermittency
# --------------------------------------------------------------------------
def test_syntetos_boylan_quadrants():
    rng = np.random.default_rng(1)
    smooth = rng.poisson(30, 400).astype(float)
    assert core.classify(smooth) == "smooth"

    intermittent = np.zeros(400)
    intermittent[::5] = 3.0          # regular size, irregular timing
    assert core.classify(intermittent) == "intermittent"

    # Irregular timing AND wildly irregular size. Sizes must be dispersed enough
    # to clear CV^2 = 0.49: uniform(1,60) only reaches CV^2 ~ 0.33 and classifies
    # as intermittent, which is the classifier being right and the first version
    # of this fixture being wrong.
    lumpy = np.zeros(400)
    idx = rng.choice(400, 60, replace=False)
    lumpy[idx] = rng.exponential(20.0, 60)
    assert core.adi_cv2(lumpy)[1] > 0.49
    assert core.classify(lumpy) == "lumpy"

    erratic = rng.exponential(20.0, 400)   # every day, wild sizes
    assert core.classify(erratic) == "erratic"


def test_croston_sba_is_below_classic_croston():
    """SBA exists because Croston is biased high. If our SBA is not strictly
    below classic Croston, the correction is not wired up."""
    y = np.zeros(300)
    y[::4] = 6.0
    classic = core.croston(y, 7, variant="classic")[0]
    sba = core.croston(y, 7, variant="sba")[0]
    assert sba < classic
    assert sba == pytest.approx(classic * (1 - 0.1 / 2))


def test_croston_handles_all_zero_series():
    assert core.croston(np.zeros(100), 5).tolist() == [0.0] * 5


# --------------------------------------------------------------------------
# LEAKAGE -- the load-bearing test
# --------------------------------------------------------------------------
def _toy_panel(n_days=500, n_series=4, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rows = []
    for s in range(n_series):
        rows.append(pd.DataFrame({
            "date": dates,
            "store_id": "S%d" % (s % 2), "state_id": "CA",
            "item_id": "I%d" % s, "dept_id": "D0", "cat_id": "C0",
            "sales": rng.poisson(3, n_days).astype(np.int32),
            "sell_price": np.round(rng.uniform(2, 5, n_days), 2).astype(np.float32),
            "promo": rng.integers(0, 2, n_days).astype(np.int8),
            "snap": rng.integers(0, 2, n_days).astype(np.int8),
        }))
    df = pd.concat(rows, ignore_index=True)
    cal = pd.DataFrame({"date": dates})
    cal["dow"] = cal.date.dt.dayofweek
    cal["doy"] = cal.date.dt.dayofyear
    cal["month"] = cal.date.dt.month
    cal["event"] = "none"
    return df, cal


def test_no_target_leakage_across_the_forecast_origin():
    """Corrupt every sale AT OR AFTER the origin, rebuild the features, and assert
    the feature rows for dates inside the horizon are bit-identical.

    If any lag or rolling window reached forward -- the single most common way a
    retail forecast is silently broken -- these frames would differ.
    """
    df, cal = _toy_panel()
    origin = df.date.max() - pd.Timedelta(27, "D")

    def build(frame):
        panels = hier.build_panels(frame)
        exog = hier.build_exog(frame)
        L = feats.long_frame(panels["store_item"], exog["store_item"], cal)
        return feats.add_features(L)

    clean = build(df)

    corrupted_df = df.copy()
    mask = corrupted_df.date >= origin
    corrupted_df.loc[mask, "sales"] = corrupted_df.loc[mask, "sales"] * 1000 + 777
    corrupted = build(corrupted_df)

    # only the LAG/ROLLING features are at risk; price and calendar columns are
    # known in advance by design and are excluded from the claim
    target_derived = [c for c in feats.FEATURE_COLS
                      if c.startswith(("lag_", "rmean_", "rstd_", "zero_share",
                                       "dow_mean"))]
    a = clean[clean.date >= origin].sort_values(["key", "date"])[target_derived]
    b = corrupted[corrupted.date >= origin].sort_values(["key", "date"])[target_derived]
    assert a.shape == b.shape
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_every_target_derived_feature_respects_the_min_lag():
    """A structural restatement of the same claim: no lag shorter than the horizon."""
    lags = [int(c.split("_")[1]) for c in feats.FEATURE_COLS if c.startswith("lag_")]
    assert lags and min(lags) >= feats.MIN_LAG == 28


# --------------------------------------------------------------------------
# hierarchy / reconciliation
# --------------------------------------------------------------------------
def test_summing_matrix_and_coherence():
    df, _ = _toy_panel()
    panels = hier.build_panels(df)
    S, names = hier.summing_matrix(panels, df)
    n_bottom = panels["store_item"].shape[1]
    assert S.shape == (len(names), n_bottom)

    # the total row must sum every bottom series exactly once
    total_row = S[names.index(("total", "TOTAL"))]
    assert total_row.sum() == n_bottom

    rng = np.random.default_rng(5)
    bottom = rng.random((n_bottom, 7)) * 10
    bu = hier.bottom_up(S, bottom)
    assert hier.coherence_error(S, bu) == pytest.approx(0.0, abs=1e-9)


def test_mint_output_is_coherent():
    df, _ = _toy_panel()
    panels = hier.build_panels(df)
    S, names = hier.summing_matrix(panels, df)
    rng = np.random.default_rng(6)
    base = rng.random((S.shape[0], 7)) * 10          # deliberately incoherent
    resid = rng.normal(size=(S.shape[0], 60))
    assert hier.coherence_error(S, base) > 1e-6      # the starting point is NOT coherent
    rec = hier.mint(S, base, resid, method="shrink")
    assert hier.coherence_error(S, rec) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------
def test_markdown_schedules_only_deepen():
    import run_pricing
    for combo in run_pricing.monotone_schedules():
        assert all(combo[i] <= combo[i + 1] for i in range(len(combo) - 1))


def test_inelastic_demand_is_never_marked_down():
    """The economic sanity check: if cutting price does not buy enough volume,
    the optimiser must refuse. A markdown tool that always recommends a markdown
    is a discount button, not an optimiser."""
    import run_pricing
    sched, _ = run_pricing.optimise(base_daily=2.0, ref_price=10.0, inventory=60,
                                    elasticity=-0.2, phase_len=14)
    assert sched == (0.0, 0.0, 0.0)


def test_elastic_demand_with_excess_stock_is_marked_down():
    import run_pricing
    sched, _ = run_pricing.optimise(base_daily=2.0, ref_price=10.0, inventory=120,
                                    elasticity=-2.5, phase_len=14)
    assert max(sched) > 0.0
