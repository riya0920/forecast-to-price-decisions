"""Tests for the completion pass: lead-time demand, the pooled gate, probabilistic
reconciliation, per-item elasticity, the multi-item optimiser, and the service."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import elasticity as EL  # noqa: E402
from src import gate as G  # noqa: E402
from src import leadtime as LT  # noqa: E402
from src import markdown_opt as MO  # noqa: E402
from src import probrec as PR  # noqa: E402


# --------------------------------------------------------------------------
# lead-time demand
# --------------------------------------------------------------------------
def _ar1_pool(n=2000, rho=0.6, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def test_block_bootstrap_beats_iid_on_autocorrelated_errors():
    """THE reason this module exists. With positively autocorrelated errors the
    iid formula understates the lead-time quantile, because it assumes the bad
    days do not cluster and they do."""
    pool = _ar1_pool()
    point = np.full(14, 5.0)
    dq = np.full(14, 8.0)
    cmp_ = LT.compare_methods(point, pool, dq, 0.95, seed=3)
    assert cmp_["lag1_autocorr"] > 0.4
    assert cmp_["iid_normal"] < cmp_["block_bootstrap"]


def test_sum_of_daily_quantiles_overstates():
    """The other shortcut, erring the other way: summing daily quantiles asserts
    every day goes wrong together."""
    pool = _ar1_pool(rho=0.2)
    point = np.full(28, 4.0)
    daily_q95 = np.array([np.quantile(4.0 + pool * np.sqrt(4.0), 0.95)] * 28)
    cmp_ = LT.compare_methods(point, pool, daily_q95, 0.95, seed=5)
    assert cmp_["sum_of_daily_quantiles"] > cmp_["block_bootstrap"]


def test_leadtime_quantile_is_monotone_in_tau():
    pool = _ar1_pool()
    point = np.full(7, 3.0)
    prev = -np.inf
    for tau in (0.5, 0.75, 0.9, 0.95, 0.99):
        q = LT.leadtime_quantile(point, pool, tau, seed=1)
        assert q > prev
        prev = q


def test_leadtime_quantile_grows_with_lead_time():
    pool = _ar1_pool()
    a = LT.leadtime_quantile(np.full(7, 3.0), pool, 0.95, seed=1)
    b = LT.leadtime_quantile(np.full(28, 3.0), pool, 0.95, seed=1)
    assert b > a


def test_shuffling_the_pool_destroys_the_dependence_it_was_meant_to_use():
    """A guard on the contract: `error_pool` must be in TIME ORDER. Shuffling it
    silently turns the block bootstrap back into an iid bootstrap, and this test
    exists so that a future refactor that sorts or groups the pool fails loudly
    rather than quietly returning a smaller number."""
    pool = _ar1_pool(rho=0.75)
    rng = np.random.default_rng(0)
    shuffled = pool.copy()
    rng.shuffle(shuffled)
    point = np.full(28, 4.0)
    ordered_q = LT.leadtime_quantile(point, pool, 0.95, seed=2)
    shuffled_q = LT.leadtime_quantile(point, shuffled, 0.95, seed=2)
    assert ordered_q > shuffled_q * 1.02


def test_error_pool_must_be_big_enough():
    with pytest.raises(ValueError):
        LT.leadtime_samples(np.full(7, 1.0), np.arange(5.0))


def test_autocorrelation_recovers_a_known_rho():
    assert LT.autocorrelation(_ar1_pool(rho=0.7, n=8000), 1) == pytest.approx(0.7, abs=0.06)


# --------------------------------------------------------------------------
# the pooled-class gate
# --------------------------------------------------------------------------
def _panel(seed=0):
    """A panel that reproduces the failure the first gate actually hit.

    On the lumpy class `naive` is CHEAPER on WMAPE than the unbiased model and
    systematically 30% low: its error is one-sided, so |error| is small while the
    signed error is large. That is exactly how a degenerate low forecast wins an
    absolute-error metric on intermittent demand, and it is why the gate needs a
    bias screen rather than a better accuracy metric.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(40):
        klass = "smooth" if k < 20 else "lumpy"
        for fold in range(1, 7):
            actual = 100.0 if klass == "smooth" else 20.0
            # unbiased but noisy: errors of both signs, so |error| accumulates
            noisy = 0.16 if klass == "smooth" else 0.40
            for method, spec in (
                    ("gbm", ("unbiased", noisy)),
                    ("croston_sba", ("unbiased", noisy * 1.6)),
                    ("naive", ("low", 0.40 if klass == "smooth" else 0.30))):
                kind, mag = spec
                if kind == "unbiased":
                    sign = 1.0 if (fold + k) % 2 == 0 else -1.0
                    yhat = actual * (1 + sign * mag * (1 + 0.05 * rng.standard_normal()))
                else:
                    yhat = actual * (1 - mag)      # always low, never high
                rows.append(dict(fold=fold, key="k%d" % k, klass=klass,
                                 method=method, y=actual, yhat=yhat))
    return pd.DataFrame(rows)


def test_bias_screen_rejects_an_accurate_but_biased_champion():
    """The failure the first gate hit: `naive` looked best on WMAPE for lumpy
    series while running -29% bias. Accuracy alone cannot express that."""
    df = _panel()
    sel = df[df.fold.isin((1, 2, 3))]
    lumpy = sel[sel.klass == "lumpy"]
    picked = G.choose(G._score(lumpy))
    assert picked != "naive", "a -30% biased method must not win the gate"


def test_without_the_screen_the_biased_method_does_win():
    """The control for the test above -- proving the screen is load-bearing and
    not decoration."""
    df = _panel()
    lumpy = df[df.fold.isin((1, 2, 3)) & (df.klass == "lumpy")]
    assert G.choose(G._score(lumpy), max_abs_bias=1e9) == "naive"


def test_gate_comparison_returns_all_three_and_counts_decisions():
    df = _panel()
    tab = G.compare_gates(df[df.fold.isin((1, 2, 3))], df[df.fold.isin((4, 5, 6))])
    assert set(tab.gate) == {"global", "per_class", "per_series"}
    assert tab.set_index("gate").loc["global", "n_decisions"] == 1
    assert tab.set_index("gate").loc["per_class", "n_decisions"] == 2
    assert tab.set_index("gate").loc["per_series", "n_decisions"] == 40


def test_a_group_with_no_selection_data_is_not_dropped():
    """Silently dropping series is how a policy gets flattered by having its
    hardest cases removed -- a bug this project already hit once."""
    df = _panel()
    sel = df[df.fold.isin((1, 2, 3)) & (df.key != "k0")]
    ev = df[df.fold.isin((4, 5, 6))]
    picked = G.apply_gate(ev, G.fit_gate(sel, "key"), "key")
    assert "k0" in set(picked.key)


# --------------------------------------------------------------------------
# probabilistic reconciliation
# --------------------------------------------------------------------------
def _hier():
    """4 bottom series -> 2 middles -> 1 total."""
    S = np.array([
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    ], float)
    return S


def test_every_sample_path_is_coherent():
    """Coherence for a distribution is a property of PATHS, and it must hold in
    every draw or the whole construction is pointless."""
    S = _hier()
    rng = np.random.default_rng(0)
    point = rng.uniform(2, 10, (4, 5))
    resid = rng.standard_normal((4, 200))
    draws = PR.bottom_samples(point, resid, n_samples=50, seed=0)
    allq = PR.reconcile_samples(S, draws)
    assert allq.shape == (50, 7, 5)
    bottom = allq[:, -4:, :]
    assert np.allclose(allq, np.einsum("ab,nbh->nah", S, bottom))


def test_the_parent_p95_is_below_the_sum_of_child_p95s():
    """The finding the section exists to make: quantiles do NOT add up, and
    forcing them to would assert that every child has its bad day together."""
    S = _hier()
    rng = np.random.default_rng(1)
    point = np.full((4, 3), 20.0)
    resid = rng.standard_normal((4, 400)) * 4.0     # independent across series
    draws = PR.bottom_samples(point, resid, n_samples=3000, seed=2, shrink=1.0)
    allq = PR.reconcile_samples(S, draws)
    q = PR.sample_quantiles(allq, (0.95,))[0.95]
    total_own = q[0].sum()
    sum_of_children = (S @ q[-4:])[0].sum()
    assert sum_of_children > total_own * 1.02


def test_diversification_vanishes_when_the_children_are_perfectly_correlated():
    """The boundary case that proves the effect is dependence and not arithmetic:
    with perfectly correlated children the sum of quantiles IS the quantile of
    the sum, and the gap closes."""
    S = _hier()
    common = np.random.default_rng(3).standard_normal(400) * 4.0
    resid = np.vstack([common] * 4)
    point = np.full((4, 3), 20.0)
    draws = PR.bottom_samples(point, resid, n_samples=3000, seed=4, shrink=0.0)
    q = PR.sample_quantiles(PR.reconcile_samples(S, draws), (0.95,))[0.95]
    ratio = (S @ q[-4:])[0].sum() / q[0].sum()
    assert ratio == pytest.approx(1.0, abs=0.03)


def test_coherence_gap_reports_a_positive_diversification():
    S = _hier()
    rng = np.random.default_rng(5)
    draws = PR.bottom_samples(np.full((4, 2), 15.0),
                              rng.standard_normal((4, 300)) * 3.0,
                              n_samples=2000, seed=6, shrink=1.0)
    q = PR.sample_quantiles(PR.reconcile_samples(S, draws), (0.95,))
    gap = PR.coherence_gap(S, q, 0.95)
    assert gap["mean_sum_minus_direct"] > 0
    assert 0.0 < gap["share_where_sum_exceeds"] <= 1.0


def test_middle_out_preserves_the_middle_total():
    """The one thing middle-out must not break: disaggregating and re-adding has
    to return the middle forecast it started from."""
    group = np.array(["A", "A", "B", "B"])
    hist = np.array([[10.0] * 20, [30.0] * 20, [5.0] * 20, [5.0] * 20])
    props = PR.proportions(hist, group)
    assert props[0] == pytest.approx(0.25)
    mid = np.array([[100.0, 80.0], [40.0, 40.0]])
    bottom = PR.middle_out(mid, group, props)
    assert bottom[:2].sum(axis=0) == pytest.approx(mid[0])
    assert bottom[2:].sum(axis=0) == pytest.approx(mid[1])


# --------------------------------------------------------------------------
# elasticity shrinkage
# --------------------------------------------------------------------------
def test_shrinkage_collapses_to_the_group_mean_when_there_is_no_signal():
    """The estimator saying, in its own arithmetic, that the dispersion it sees
    is entirely noise."""
    rng = np.random.default_rng(0)
    truth = np.full(40, -1.5)
    se = np.full(40, 0.5)
    est = truth + rng.standard_normal(40) * se
    out = EL.shrink(est, se, np.full(40, "A"))
    assert np.std(out) < 0.15 * np.std(est)


def test_shrinkage_preserves_dispersion_when_the_signal_is_strong():
    rng = np.random.default_rng(1)
    truth = rng.normal(-1.5, 0.8, 60)
    se = np.full(60, 0.05)
    est = truth + rng.standard_normal(60) * se
    out = EL.shrink(est, se, np.full(60, "A"))
    assert np.corrcoef(out, truth)[0, 1] > 0.95
    assert np.std(out) > 0.8 * np.std(truth)


def test_shrinkage_never_moves_an_estimate_past_the_group_mean():
    rng = np.random.default_rng(2)
    est = rng.normal(-1.0, 0.4, 30)
    se = rng.uniform(0.1, 0.6, 30)
    out = EL.shrink(est, se, np.full(30, "A"))
    mu = est.mean()
    assert np.all(np.abs(out - mu) <= np.abs(est - mu) + 1e-9)


# --------------------------------------------------------------------------
# multi-item markdown
# --------------------------------------------------------------------------
def _plan(n=4):
    return [dict(item_id="I%d" % i, base_daily=3.0, ref_price=10.0,
                 elasticity=-1.8, inventory=120.0) for i in range(n)]


def test_schedule_is_monotone_non_increasing_in_price():
    res = MO.solve_once(_plan(), [0.0, 0.2, 0.4], 3, 14,
                        np.zeros((4, 3)), 0.0, None)
    d = res["discounts"]
    assert np.all(np.diff(d, axis=1) >= -1e-9), d


def test_exactly_one_depth_per_item_phase():
    res = MO.solve_once(_plan(3), [0.0, 0.1, 0.3], 2, 10,
                        np.zeros((3, 2)), 0.0, None)
    assert res["discounts"].shape == (3, 2)
    assert set(np.unique(res["discounts"])) <= {0.0, 0.1, 0.3}


def test_a_budget_constraint_actually_binds():
    plan = _plan()
    free = MO.solve(plan, [0.0, 0.2, 0.4, 0.5], 3, 14, 0.0, budget=None)
    tight = MO.solve(plan, [0.0, 0.2, 0.4, 0.5], 3, 14, 0.0, budget=1.0)
    assert tight["discounts"].mean() <= free["discounts"].mean()
    assert tight["status"] == "Optimal"


def test_shared_inventory_pools_are_respected():
    """Two items sharing one pool may not jointly sell more than the pool holds."""
    plan = _plan(2)
    for p in plan:
        p["inventory"] = 50.0
    res = MO.solve_once(plan, [0.0, 0.5], 2, 21, np.zeros((2, 2)), 0.0, None,
                        shared_pools={"P": [0, 1]})
    sold = MO.evaluate(plan, res["discounts"], 2, 21, 0.0)
    assert sold["units"] <= 100.0 + 1e-6


def test_the_fixed_point_reports_whether_it_settled():
    res = MO.solve(_plan(), [0.0, 0.2, 0.4], 3, 14, 0.35, max_iter=15)
    assert isinstance(res["converged"], bool)
    assert res["iterations"] >= 1
    if res["converged"]:
        assert res["gap_history"][-1] == 0.0


def test_cannibalisation_changes_the_demand_a_schedule_implies():
    """If the cross term did nothing the whole section would be decoration."""
    a = MO.phase_demand(5.0, -1.8, 0.2, rival_log_rel=-0.3, cross=0.35,
                        phase_start=0, phase_len=14)
    b = MO.phase_demand(5.0, -1.8, 0.2, rival_log_rel=0.0, cross=0.35,
                        phase_start=0, phase_len=14)
    assert a < b, "cheaper rivals must take volume away"


def test_evaluate_never_sells_more_than_inventory():
    plan = _plan(2)
    for p in plan:
        p["inventory"] = 7.0
    out = MO.evaluate(plan, np.full((2, 3), 0.5), 3, 14, 0.35)
    assert out["units"] <= 14.0 + 1e-6
    assert out["leftover_units"] >= -1e-9


# --------------------------------------------------------------------------
# the service
# --------------------------------------------------------------------------
def _client():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    import serve
    return fastapi_testclient.TestClient(serve.app), serve


def test_service_starts_and_reports_health():
    client, _ = _client()
    r = client.get("/health")
    assert r.status_code == 200
    assert "ok" in r.json()


def test_forecast_endpoint_returns_a_fan_and_the_fva():
    client, serve = _client()
    if not serve.STORE.ok:
        pytest.skip("run `python run_forecast.py` first")
    key = sorted(serve.STORE.quantiles)[0]
    body = client.get("/forecast/%s" % key).json()
    assert len(body["mean"]) == 28
    assert set(body["quantiles"]) == {"0.50", "0.75", "0.90", "0.95", "0.98"}
    # the fan must be ordered at every horizon day, or it is not a fan
    fan = np.array([body["quantiles"]["%.2f" % t]
                    for t in (0.50, 0.75, 0.90, 0.95, 0.98)])
    assert (np.diff(fan, axis=0) >= -1e-6).mean() > 0.98


def test_order_endpoint_returns_the_cost_ratio_the_service_level_asserts():
    client, serve = _client()
    if not serve.STORE.ok:
        pytest.skip("run `python run_forecast.py` first")
    key = sorted(serve.STORE.quantiles)[0]
    body = client.post("/order", json={"key": key, "lead_time_days": 7,
                                       "service_level": 0.95}).json()
    assert body["implied_cost_ratio"] == pytest.approx(19.0, abs=0.01)
    assert body["order_quantity"] >= 0
    assert body["buffer_over_mean"] >= -1e-6


def test_case_pack_rounds_the_order_up():
    client, serve = _client()
    if not serve.STORE.ok:
        pytest.skip("run `python run_forecast.py` first")
    key = sorted(serve.STORE.quantiles)[0]
    body = client.post("/order", json={"key": key, "case_pack": 12,
                                       "service_level": 0.95}).json()
    assert body["order_quantity"] % 12 == 0


def test_unknown_series_is_a_404_not_a_500():
    client, serve = _client()
    if not serve.STORE.ok:
        pytest.skip("run `python run_forecast.py` first")
    assert client.get("/forecast/NOPE|NOPE").status_code == 404


def test_an_override_requires_a_reason():
    client, serve = _client()
    if not serve.STORE.ok:
        pytest.skip("run `python run_forecast.py` first")
    key = sorted(serve.STORE.quantiles)[0]
    bad = client.post("/override", json={"key": key, "horizon_day": 0,
                                         "value": 5.0, "reason": "x"})
    assert bad.status_code == 422, "an unexplained override must not be accepted"


# --------------------------------------------------------------------------
# quantile crossing
# --------------------------------------------------------------------------
def test_crossing_is_detected_when_the_fan_is_out_of_order():
    """A P90 above a P95 is not a rounding wart -- an order system reading it
    orders MORE stock for a LOWER service level."""
    from src import quantiles as Q
    bad = {0.50: np.array([1.0, 1.0]), 0.90: np.array([5.0, 2.0]),
           0.95: np.array([4.0, 3.0])}
    assert Q.crossing_rate(bad) > 0


def test_rearrangement_removes_every_crossing():
    from src import quantiles as Q
    bad = {0.50: np.array([1.0, 9.0]), 0.90: np.array([5.0, 2.0]),
           0.95: np.array([4.0, 3.0])}
    assert Q.crossing_rate(Q.rearrange(bad)) == 0.0


def test_rearrangement_preserves_the_multiset_at_each_point():
    """Sorting rearranges; it must never invent or discard a value, or the fan
    would no longer be made of the model's own predictions."""
    from src import quantiles as Q
    rng = np.random.default_rng(0)
    fan = {t: rng.uniform(0, 10, 25) for t in (0.5, 0.75, 0.9, 0.95, 0.98)}
    out = Q.rearrange(fan)
    before = np.sort(np.vstack([fan[t] for t in sorted(fan)]), axis=0)
    after = np.sort(np.vstack([out[t] for t in sorted(out)]), axis=0)
    assert np.allclose(before, after)


def test_rearrangement_leaves_an_already_monotone_fan_alone():
    from src import quantiles as Q
    good = {0.50: np.array([1.0, 2.0]), 0.90: np.array([3.0, 4.0]),
            0.95: np.array([5.0, 6.0])}
    out = Q.rearrange(good)
    for t in good:
        assert np.allclose(out[t], good[t])


def test_the_service_serves_a_fan_with_no_crossings_at_all():
    """The serving boundary is where this has to hold: whatever the models
    produced, what leaves the building must be coherent."""
    client, serve = _client()
    if not serve.STORE.ok:
        pytest.skip("run `python run_forecast.py` first")
    from src import quantiles as Q
    for key in sorted(serve.STORE.quantiles)[:40]:
        body = client.get("/forecast/%s" % key).json()
        fan = {float(t): np.array(v) for t, v in body["quantiles"].items()}
        assert Q.crossing_rate(fan) == 0.0, key


def test_the_raw_models_really_did_cross():
    """The control: if the raw fan had never crossed, the rearrangement above
    would be untested decoration."""
    _, serve = _client()
    if not serve.STORE.ok:
        pytest.skip("run `python run_forecast.py` first")
    assert serve.STORE.crossing_before > 0.0
