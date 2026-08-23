"""Quantile forecasts, and the handoff to replenishment.

THE QUESTION THIS ANSWERS
-------------------------
"Replenishment asks for a single number per item x store x day. Your model
produces a distribution. What do you hand them?"

Handing them the MEAN is the naive answer and it is wrong in a specific,
expensive way: ordering to the mean means you are short roughly half the time.
Demand above the mean costs a stockout, demand below it costs holding -- and
those costs are not equal, so the right order point is not the middle.

What you hand them is a QUANTILE chosen from the item's target service level.
The newsvendor result says the optimal quantile is

    q* = Cu / (Cu + Co)

where Cu is the cost of being one unit short (lost margin, and possibly the
customer) and Co is the cost of one unit too many (holding, and eventually
markdown). A 95% service target IS the claim that understocking is 19x worse
than overstocking. Saying it that way turns a service-level argument into a
cost argument, which is the conversation worth having.

WHY PINBALL LOSS
----------------
A quantile forecast cannot be scored with MAE -- MAE is minimised by the median,
so it would score every quantile as worse than the P50 by construction. The
pinball (quantile) loss is the proper scoring rule: it is minimised by the true
quantile, so a P90 that is well calibrated beats a P50 on P90 pinball loss.
"""
from __future__ import annotations

import numpy as np

# Service level -> the implied cost ratio, so the table can be read either way.
SERVICE_LEVELS = (0.50, 0.75, 0.90, 0.95, 0.98)


def pinball_loss(y_true, y_pred, tau: float) -> float:
    """Proper scoring rule for the tau-quantile. Lower is better.

    loss = mean( tau * (y - yhat)      where y >= yhat
                 (1-tau) * (yhat - y)  where y <  yhat )

    The asymmetry IS the metric: at tau=0.9 an under-forecast is penalised 9x
    more than an over-forecast of the same size, which is exactly the asymmetry
    a 90% service level is asserting.
    """
    y = np.asarray(y_true, float)
    f = np.asarray(y_pred, float)
    diff = y - f
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def coverage(y_true, y_pred) -> float:
    """Share of actuals at or below the forecast. For a calibrated tau-quantile
    this should be tau."""
    y = np.asarray(y_true, float)
    f = np.asarray(y_pred, float)
    return float(np.mean(y <= f + 1e-9))


def implied_cost_ratio(service_level: float) -> float:
    """The newsvendor inverse: what a service level SAYS about your costs.

    q* = Cu / (Cu + Co)  =>  Cu/Co = q* / (1 - q*)

    Useful in a planning meeting, because "95% service" sounds like a policy and
    "understocking is 19x worse than overstocking" sounds like a claim someone
    has to defend.
    """
    q = float(np.clip(service_level, 1e-6, 1 - 1e-6))
    return q / (1 - q)


def newsvendor_quantile(cost_under: float, cost_over: float) -> float:
    """The other direction: given the costs, what quantile should you order to?"""
    tot = cost_under + cost_over
    return float(cost_under / tot) if tot > 0 else 0.5


def order_up_to(quantile_forecast: np.ndarray, on_hand: float = 0.0,
                case_pack: int = 1) -> int:
    """Turn a quantile forecast over the lead time into an order quantity."""
    need = max(0.0, float(np.sum(quantile_forecast)) - on_hand)
    if case_pack <= 1:
        return int(np.ceil(need))
    return int(np.ceil(need / case_pack) * case_pack)


def evaluate_quantiles(y_true, preds_by_tau: dict) -> list[dict]:
    """Score every quantile on its OWN pinball loss plus coverage.

    Reporting each quantile's loss against every tau (not just its own) is what
    shows the diagonal: a well-calibrated set of quantiles should have each
    forecast winning the loss for the tau it was fitted to.
    """
    rows = []
    for tau, pred in sorted(preds_by_tau.items()):
        row = dict(tau=tau, coverage=coverage(y_true, pred),
                   mean_forecast=float(np.mean(pred)))
        for other in sorted(preds_by_tau):
            row["pinball@%.2f" % other] = pinball_loss(y_true, pred, other)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# quantile crossing
# --------------------------------------------------------------------------
def crossing_rate(preds_by_tau: dict) -> float:
    """Share of points where a higher tau predicts a LOWER value than a lower tau.

    Fitting one model per tau means nothing in the estimation couples them, so
    nothing prevents the fitted P90 from landing above the fitted P95 on some
    input. It is not a rare pathology -- it happens wherever the trees split
    differently, which is most often in the sparse regions of feature space where
    the quantiles matter most.

    An order system reading a crossed fan orders MORE stock for a LOWER service
    level, which is not a small numerical wart: it inverts the meaning of the
    parameter a planner is turning.
    """
    taus = sorted(preds_by_tau)
    if len(taus) < 2:
        return 0.0
    fan = np.vstack([np.asarray(preds_by_tau[t], float).ravel() for t in taus])
    return float((np.diff(fan, axis=0) < -1e-9).mean())


def rearrange(preds_by_tau: dict) -> dict:
    """Monotone rearrangement: sort the fan at each point.

    This is the Chernozhukov-Fernandez-Val-Galichon rearrangement, and the reason
    to prefer it to "fit a fancier joint model" is that it is FREE and provably
    safe: sorting an estimated quantile curve can only reduce (never increase)
    the estimation error against the true, necessarily monotone, quantile
    function. So it cannot make the pinball loss worse, and the report checks
    that rather than taking it on faith.

    It is a post-processing step, not a fix to the cause. The cause is that five
    independent fits have no idea the others exist.
    """
    taus = sorted(preds_by_tau)
    fan = np.vstack([np.asarray(preds_by_tau[t], float).ravel() for t in taus])
    fan = np.sort(fan, axis=0)
    out = {}
    for i, t in enumerate(taus):
        out[t] = fan[i].reshape(np.asarray(preds_by_tau[t], float).shape)
    return out
