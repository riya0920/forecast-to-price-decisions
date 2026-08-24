

def test_daily_paths_sum_to_the_leadtime_samples():
    """`leadtime_samples` delegates to `leadtime_daily_paths`, so there is one
    bootstrap and not two. If these ever diverge, two consumers of this join are
    reasoning about different distributions."""
    import numpy as np
    from src import leadtime as LT
    rng = np.random.default_rng(0)
    pool = rng.normal(0, 1, 900)
    point = np.linspace(2.0, 5.0, 14)
    daily = LT.leadtime_daily_paths(point, pool, n_samples=500, seed=3)
    total = LT.leadtime_samples(point, pool, n_samples=500, seed=3)
    assert daily.shape == (500, 14)
    assert np.allclose(daily.sum(axis=1), total)


def test_daily_paths_are_never_negative():
    """Demand is clipped at zero, and a consumer that walks the path day by day
    to find a stockout date would otherwise see stock RECOVER on a negative day."""
    import numpy as np
    from src import leadtime as LT
    rng = np.random.default_rng(1)
    pool = rng.normal(0, 3, 900)
    daily = LT.leadtime_daily_paths(np.full(10, 0.5), pool, n_samples=400, seed=1)
    assert (daily >= 0).all()
