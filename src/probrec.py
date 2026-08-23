"""Middle-out reconciliation, and probabilistic reconciliation across the hierarchy.

TWO GAPS, ONE FILE
------------------
The last pass reconciled POINT forecasts four ways and left two things unbuilt:
middle-out, and any notion of reconciling the *distribution*. The second is the
one that actually bites, and its statement in the last README was the clue:

    "the quantiles are not reconciled across the hierarchy at all, so the P95s
     do not add up even though the point forecasts do"

THE P95s SHOULD NOT ADD UP
--------------------------
This is the part worth getting right, because the obvious "fix" is wrong. A
quantile is not a linear functional of the distribution:

    P95(A + B)  !=  P95(A) + P95(B)

unless A and B are perfectly dependent. Forcing the store P95 to equal the sum of
its items' P95s would not make the forecast coherent, it would make it wrong --
you would be asserting that every item in the store has its bad week
simultaneously. Adding-up is a property of REALISATIONS, not of quantiles.

So the coherence requirement has to be restated for distributions: what must add
up is every sample PATH. Draw a joint sample of the bottom level, push it through
the summing matrix S, and every level is coherent in every draw by construction.
Read quantiles off the resulting coherent sample set and the store P95 is
automatically *below* the sum of item P95s, by exactly the diversification the
data supports.

That is what `reconcile_samples` does. The samples are drawn with the residual
CORRELATION across series preserved (a Gaussian copula on the empirical residual
covariance) -- because if the draws were independent across series, the
diversification benefit would be an artifact of the sampler rather than a
measurement of the assortment.

MIDDLE-OUT
----------
Forecast at a middle level, aggregate up, and disaggregate down by historical
proportions. It exists because it is what large retailers actually run: the
middle level is where the series are smooth enough to model well and still
granular enough that the proportions below are stable. Its weakness is that
those proportions are assumed constant over the horizon, which is exactly what
fails during a promotion.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# middle-out
# --------------------------------------------------------------------------
def proportions(history_bottom: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """Each bottom series' historical share of its middle-level parent.

    Shares are computed over the whole history rather than the recent tail: a
    short window makes the shares track the last promotion, which is the
    behaviour middle-out is least able to justify.
    """
    tot = np.asarray(history_bottom, float).sum(axis=1)
    out = np.zeros_like(tot)
    for g in np.unique(group_ids):
        m = group_ids == g
        s = tot[m].sum()
        out[m] = tot[m] / s if s > 0 else 1.0 / max(m.sum(), 1)
    return out


def middle_out(middle_fc: np.ndarray, group_ids: np.ndarray,
               props: np.ndarray) -> np.ndarray:
    """Push a middle-level forecast down to the bottom by fixed proportions.

    middle_fc : (n_groups x h) in the order of np.unique(group_ids)
    returns   : (n_bottom x h)
    """
    groups = list(np.unique(group_ids))
    idx = {g: i for i, g in enumerate(groups)}
    rows = np.array([idx[g] for g in group_ids])
    return middle_fc[rows] * props[:, None]


# --------------------------------------------------------------------------
# probabilistic reconciliation
# --------------------------------------------------------------------------
def _nearest_psd(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    w, V = np.linalg.eigh((C + C.T) / 2.0)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T


def bottom_samples(point_bottom: np.ndarray, resid_bottom: np.ndarray,
                   n_samples: int = 500, seed: int = 0,
                   shrink: float = 0.2) -> np.ndarray:
    """Joint samples of bottom-level demand with the cross-series correlation kept.

    point_bottom : (n_bottom x h)
    resid_bottom : (n_bottom x T) in-sample residuals

    Returns (n_samples x n_bottom x h).

    The correlation is shrunk toward the identity because 300 series estimated
    from a few hundred residual days gives a correlation matrix that is
    rank-deficient and wildly overfitted in its off-diagonal -- the same problem
    MinT(shrink) exists to solve, and the same fix.
    """
    rng = np.random.default_rng(seed)
    n_b, h = point_bottom.shape
    R = resid_bottom - resid_bottom.mean(axis=1, keepdims=True)
    sd = R.std(axis=1)
    sd[sd <= 0] = 1e-6
    C = np.corrcoef(R / sd[:, None])
    C = np.nan_to_num(C, nan=0.0)
    C = (1 - shrink) * C + shrink * np.eye(n_b)
    L = np.linalg.cholesky(_nearest_psd(C))

    z = rng.standard_normal((n_samples, h, n_b)) @ L.T        # correlated across series
    draws = point_bottom[None, :, :] + np.transpose(z, (0, 2, 1)) * sd[None, :, None]
    return np.clip(draws, 0, None)


def reconcile_samples(S: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Push bottom draws up. Coherent in EVERY draw, by construction.

    draws : (n_samples x n_bottom x h)  ->  (n_samples x n_all x h)
    """
    return np.einsum("ab,nbh->nah", S, draws)


def sample_quantiles(all_draws: np.ndarray, taus) -> dict:
    """Quantiles per series per horizon-day, read off the coherent sample set."""
    return {float(t): np.quantile(all_draws, t, axis=0) for t in taus}


def coherence_gap(S: np.ndarray, q_by_tau: dict, tau: float) -> dict:
    """How far the *quantiles* are from adding up -- which is the point.

    A positive `sum_minus_direct` is the diversification benefit: the sum of the
    children's tail quantiles exceeds the parent's own tail quantile, because the
    children do not all have their bad day together. Reporting it as a number is
    the difference between knowing your hierarchy diversifies and assuming it.
    """
    q = q_by_tau[float(tau)]
    n_b = S.shape[1]
    bottom = q[-n_b:]
    naive_sum = S @ bottom
    direct = q
    diff = naive_sum - direct
    return dict(
        tau=float(tau),
        max_abs_gap=float(np.abs(diff).max()),
        mean_sum_minus_direct=float(diff.mean()),
        share_where_sum_exceeds=float((diff > 1e-9).mean()),
    )
