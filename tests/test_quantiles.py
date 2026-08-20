"""Tests for the quantile forecasts and the replenishment handoff."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import quantiles as Q  # noqa: E402


# --------------------------------------------------------------------------
# pinball loss
# --------------------------------------------------------------------------
def test_pinball_is_minimised_by_the_true_quantile():
    """THE property of a proper scoring rule. If this fails, every quantile
    number in the report is scored by something that does not measure it."""
    rng = np.random.default_rng(0)
    y = rng.gamma(2.0, 3.0, 40_000)
    for tau in (0.1, 0.5, 0.9, 0.95):
        truth = float(np.quantile(y, tau))
        best = Q.pinball_loss(y, np.full_like(y, truth), tau)
        for off in (-2.0, -0.5, 0.5, 2.0):
            assert Q.pinball_loss(y, np.full_like(y, truth + off), tau) >= best


def test_pinball_penalises_asymmetrically():
    """At tau=0.9 an under-forecast must cost ~9x an over-forecast of the same
    size -- that asymmetry IS what a 90% service level asserts."""
    y = np.array([10.0])
    under = Q.pinball_loss(y, np.array([9.0]), 0.9)     # forecast too low by 1
    over = Q.pinball_loss(y, np.array([11.0]), 0.9)     # too high by 1
    assert under == pytest.approx(0.9)
    assert over == pytest.approx(0.1)
    assert under / over == pytest.approx(9.0)


def test_pinball_at_median_is_half_absolute_error():
    y = np.array([1.0, 5.0, 9.0])
    f = np.array([5.0, 5.0, 5.0])
    assert Q.pinball_loss(y, f, 0.5) == pytest.approx(np.mean(np.abs(y - f)) / 2)


def test_mae_would_rank_the_p90_worse_than_the_p50():
    """Why pinball and not MAE, demonstrated: MAE is minimised by the median, so
    scoring a P90 on MAE calls the right answer wrong."""
    rng = np.random.default_rng(1)
    y = rng.gamma(2.0, 3.0, 20_000)
    p50 = float(np.quantile(y, 0.5))
    p90 = float(np.quantile(y, 0.9))
    mae_50 = np.mean(np.abs(y - p50))
    mae_90 = np.mean(np.abs(y - p90))
    assert mae_90 > mae_50                                    # MAE says P90 is worse
    assert (Q.pinball_loss(y, np.full_like(y, p90), 0.9)
            < Q.pinball_loss(y, np.full_like(y, p50), 0.9))   # pinball says otherwise


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------
def test_coverage_of_a_true_quantile_matches_tau():
    rng = np.random.default_rng(2)
    y = rng.normal(50, 10, 60_000)
    for tau in (0.5, 0.9, 0.95):
        q = float(np.quantile(y, tau))
        assert Q.coverage(y, np.full_like(y, q)) == pytest.approx(tau, abs=0.01)


def test_coverage_inflates_on_zero_heavy_data():
    """The P50 artifact the report explains: on a zero-inflated series the median
    forecast is 0, and every zero day counts as covered because 0 <= 0."""
    y = np.array([0.0] * 60 + [5.0] * 40)
    assert Q.coverage(y, np.zeros_like(y)) == pytest.approx(0.60)


# --------------------------------------------------------------------------
# the newsvendor handoff
# --------------------------------------------------------------------------
def test_newsvendor_round_trip():
    for sl in (0.5, 0.75, 0.9, 0.95, 0.98):
        ratio = Q.implied_cost_ratio(sl)
        assert Q.newsvendor_quantile(ratio, 1.0) == pytest.approx(sl)


def test_implied_cost_ratio_matches_the_reported_numbers():
    """The claims the report makes in prose, pinned as arithmetic."""
    assert Q.implied_cost_ratio(0.90) == pytest.approx(9.0)
    assert Q.implied_cost_ratio(0.95) == pytest.approx(19.0)
    assert Q.implied_cost_ratio(0.98) == pytest.approx(49.0)


def test_higher_service_implies_a_higher_cost_ratio():
    prev = -1.0
    for sl in (0.5, 0.75, 0.9, 0.95, 0.99):
        cur = Q.implied_cost_ratio(sl)
        assert cur > prev
        prev = cur


def test_order_up_to_nets_off_on_hand_and_rounds_to_case():
    fc = np.array([3.0, 3.0, 3.0])          # 9 units of lead-time demand
    assert Q.order_up_to(fc, on_hand=0.0, case_pack=1) == 9
    assert Q.order_up_to(fc, on_hand=4.0, case_pack=1) == 5
    assert Q.order_up_to(fc, on_hand=4.0, case_pack=4) == 8      # rounds up
    assert Q.order_up_to(fc, on_hand=99.0) == 0                  # never negative


# --------------------------------------------------------------------------
# the evaluation table
# --------------------------------------------------------------------------
def test_evaluate_quantiles_produces_a_winning_diagonal():
    """End to end: fit true quantiles of a known distribution and check each one
    wins its own pinball loss -- which is exactly the diagonal the report reads."""
    rng = np.random.default_rng(3)
    y = rng.gamma(2.0, 4.0, 30_000)
    preds = {tau: np.full_like(y, float(np.quantile(y, tau)))
             for tau in Q.SERVICE_LEVELS}
    rows = Q.evaluate_quantiles(y, preds)
    for row in rows:
        tau = row["tau"]
        own = row["pinball@%.2f" % tau]
        others = [r["pinball@%.2f" % tau] for r in rows if r["tau"] != tau]
        assert own <= min(others) + 1e-12


def test_quantile_forecasts_are_ordered():
    """A P95 below a P50 is incoherent no matter what any loss says."""
    rng = np.random.default_rng(4)
    y = rng.gamma(2.0, 4.0, 5_000)
    vals = [float(np.quantile(y, t)) for t in Q.SERVICE_LEVELS]
    assert vals == sorted(vals)
