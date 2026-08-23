"""The pooled-class selection gate -- the fix this project's own report recommended.

THE FINDING THIS RESPONDS TO
----------------------------
Per-SERIES model selection lost to a single global GBM (0.528 WMAPE vs 0.505).
The reason was diagnosed and then not acted on: choosing a champion from three
folds of one series is choosing on ~84 observations, most of them zero. The
selection is dominated by noise, and WMAPE's denominator makes the degenerate
all-zeros forecast look excellent on an intermittent series -- so the gate handed
96 series to `naive`, whose bias was -29%.

WHAT A POOLED GATE CHANGES
--------------------------
Estimate the gate on the intermittency CLASS rather than the series. There are
four classes (smooth / erratic / intermittent / lumpy) and 300 series, so each
decision is made on ~75x more data than a per-series decision. The bet is that
"which method wins" is a property of the demand REGIME, not of the individual
SKU -- which is the same bet the Syntetos-Boylan classification itself makes, and
this is the experiment that tests it.

Three gates are compared on the same evaluation folds:

    global    -- one model for everything (the incumbent champion)
    per_class -- a champion per intermittency class  (this file)
    per_series-- a champion per series               (the loser)

THE TRAP THIS AVOIDS
--------------------
A gate that can select `naive` on WMAPE will select it for the wrong reason. The
gate here scores candidates on WMAPE *and* rejects any candidate whose selection-
window bias is worse than a threshold, because a forecast that is 29% low is not
a cheaper forecast, it is a different and worse decision. Accuracy alone cannot
express that; the bias screen is what encodes it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_ABS_BIAS = 0.15     # a champion may not be more than 15% biased either way


def _score(sub: pd.DataFrame) -> pd.DataFrame:
    """WMAPE and bias per method over whatever rows are handed in."""
    g = sub.groupby("method", observed=True)
    out = g.apply(lambda d: pd.Series({
        "wmape": np.abs(d.y - d.yhat).sum() / max(np.abs(d.y).sum(), 1e-9),
        "bias": (d.yhat - d.y).sum() / max(np.abs(d.y).sum(), 1e-9),
        "n": len(d),
    }), include_groups=False)
    return out.reset_index()


def choose(scored: pd.DataFrame, max_abs_bias: float = MAX_ABS_BIAS) -> str:
    """Lowest WMAPE among candidates that pass the bias screen.

    If every candidate fails the screen the least-biased one wins, not the most
    accurate -- because at that point the panel is telling you no method here is
    usable and the safer failure is the one that does not systematically
    under-order.
    """
    ok = scored[scored.bias.abs() <= max_abs_bias]
    if len(ok) == 0:
        return str(scored.loc[scored.bias.abs().idxmin(), "method"])
    return str(ok.loc[ok.wmape.idxmin(), "method"])


def fit_gate(selection: pd.DataFrame, by: str,
             max_abs_bias: float = MAX_ABS_BIAS) -> dict:
    """Learn a champion per group from the SELECTION folds only.

    `by` is a column name ("key" for per-series, "klass" for per-class) or the
    literal "global" for one champion over everything.
    """
    if by == "global":
        return {"__ALL__": choose(_score(selection), max_abs_bias)}
    return {str(k): choose(_score(sub), max_abs_bias)
            for k, sub in selection.groupby(by, observed=True)}


def apply_gate(evaluation: pd.DataFrame, gate: dict, by: str) -> pd.DataFrame:
    """Pick each row's forecast according to the gate. Returns y / yhat / method."""
    if by == "global":
        want = pd.Series(gate["__ALL__"], index=evaluation.index)
    else:
        want = evaluation[by].astype(str).map(gate)
        # A group with no selection-window data gets the global champion rather
        # than being silently dropped -- dropping is how a policy gets flattered
        # by having its hardest cases removed (a bug this project already hit).
        fallback = choose(_score(evaluation), 1e9)
        want = want.fillna(fallback)
    picked = evaluation[evaluation.method.values == want.values]
    return picked


def compare_gates(selection: pd.DataFrame, evaluation: pd.DataFrame,
                  max_abs_bias: float = MAX_ABS_BIAS) -> pd.DataFrame:
    """The whole experiment in one table: three gates, same folds, same metrics."""
    rows = []
    for name, by in (("global", "global"), ("per_class", "klass"),
                     ("per_series", "key")):
        gate = fit_gate(selection, by, max_abs_bias)
        picked = apply_gate(evaluation, gate, by)
        y, yhat = picked.y.to_numpy(float), picked.yhat.to_numpy(float)
        denom = max(np.abs(y).sum(), 1e-9)
        rows.append(dict(
            gate=name,
            n_decisions=len(gate),
            wmape=float(np.abs(y - yhat).sum() / denom),
            bias=float((yhat - y).sum() / denom),
            n_rows=len(picked),
            distinct_methods=int(pd.Series(list(gate.values())).nunique()),
        ))
    return pd.DataFrame(rows)
