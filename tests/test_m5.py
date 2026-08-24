"""The real M5 panel, and the scale claim the deep arm rested on.

Skips when the cached panel is absent, so the suite passes without 226 MB on
disk.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import m5 as M  # noqa: E402


needs_m5 = pytest.mark.skipif(not M.available(),
                              reason="no M5 panel in .vendor/kaggle/")


@pytest.fixture(scope="module")
def panel():
    if not M.available():
        pytest.skip("no M5 panel")
    return M.load_panel(n_series=400, seed=0)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
@needs_m5
def test_the_panel_has_m5s_shape(panel):
    """30,490 series x 1,941 days is M5. A loader that quietly produced fewer
    days would make every backtest origin wrong without failing."""
    assert panel["panel"].shape[0] == 400
    assert panel["panel"].shape[1] == 1941
    assert len(panel["keys"]) == 400


@needs_m5
def test_series_are_sampled_not_rows(panel):
    """M5's long format is one row per series-day. Taking the first N rows gives
    a handful of series with complete histories rather than N series, and every
    intermittency statistic computed on it describes the wrong thing."""
    other = M.load_panel(n_series=400, seed=1)
    assert set(panel["keys"]) != set(other["keys"])
    assert other["panel"].shape == panel["panel"].shape


@needs_m5
def test_keys_look_like_m5_ids(panel):
    """`id` is dictionary-encoded in this parquet, and np.asarray on a
    dictionary column yields INDICES rather than labels -- which would silently
    key the panel by dictionary position."""
    for k in panel["keys"][:20]:
        assert "_" in k
        assert any(k.startswith(p) for p in ("FOODS", "HOBBIES", "HOUSEHOLD"))


@needs_m5
def test_sales_are_non_negative_integers(panel):
    p = panel["panel"]
    assert (p >= 0).all()
    assert np.allclose(p, np.round(p))


@needs_m5
def test_the_same_seed_gives_the_same_sample(panel):
    again = M.load_panel(n_series=400, seed=0)
    assert again["keys"] == panel["keys"]
    assert np.array_equal(again["panel"], panel["panel"])


# --------------------------------------------------------------------------
# the property the generator was built to imitate
# --------------------------------------------------------------------------
@needs_m5
def test_m5_is_mostly_zeros(panel):
    """Two thirds of it. That is the property the generator imitates and the
    reason WMAPE misbehaves -- worth confirming on the real panel before
    trusting any conclusion drawn from a simulated one."""
    inter = M.intermittency(panel["panel"])
    assert 0.4 < inter["zero_share"] < 0.85


@needs_m5
def test_intermittency_reports_the_syntetos_boylan_inputs(panel):
    inter = M.intermittency(panel["panel"])
    assert inter["median_adi"] > 1.0
    assert inter["median_cv2"] >= 0.0
    assert inter["series"] == 400


def test_intermittency_handles_an_all_zero_series():
    """A series that never sold has no inter-demand interval at all, and a
    divide-by-zero there would poison the median for the whole panel."""
    p = np.zeros((3, 50), dtype=np.float32)
    p[0, ::5] = 1.0
    inter = M.intermittency(p)
    assert np.isfinite(inter["median_adi"])
    assert inter["zero_share"] > 0.9


def test_intermittency_of_a_dense_series_is_a_daily_interval():
    p = np.ones((2, 40), dtype=np.float32)
    inter = M.intermittency(p)
    assert inter["median_adi"] == pytest.approx(1.0)
    assert inter["zero_share"] == 0.0


# --------------------------------------------------------------------------
# the metric trap the sweep is built around
# --------------------------------------------------------------------------
def test_wmape_is_not_comparable_across_different_series_samples():
    """The mistake this section made first. Each arm of the sweep is a DIFFERENT
    sample of series and they are not equally hard, so a flat absolute WMAPE
    across two samples can mean the model got better. FVA against a baseline
    computed on the SAME series is the comparison that survives."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import run_m5 as R

    easy = np.array([[10.0, 10.0, 10.0, 10.0]])
    hard = np.array([[10.0, 0.0, 30.0, 0.0]])
    flat = np.full((1, 4), 10.0)
    # identical absolute error, very different baselines
    assert R.wmape(easy, flat) < R.wmape(hard, flat)


def test_bias_is_signed_and_wmape_is_not():
    """The original finding was about BIAS, not accuracy: a model a fifth low on
    every order is a different and worse decision, and WMAPE cannot see the
    difference between low and high."""
    import run_m5 as R

    actual = np.array([[10.0, 10.0]])
    low = np.array([[8.0, 8.0]])
    high = np.array([[12.0, 12.0]])
    assert R.wmape(actual, low) == pytest.approx(R.wmape(actual, high))
    assert R.bias(actual, low) < 0 < R.bias(actual, high)


def test_seasonal_naive_repeats_the_last_week():
    import run_m5 as R

    hist = np.arange(14, dtype=float).reshape(1, 14)
    out = R.seasonal_naive(hist, horizon=10, period=7)
    assert out.shape == (1, 10)
    assert list(out[0, :7]) == list(hist[0, -7:])
    assert out[0, 7] == hist[0, -7]
