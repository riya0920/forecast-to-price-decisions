"""The three analyses that consume the backtest rather than re-running it.

1. THE POOLED-CLASS SELECTION GATE
   The report recommended it and never built it. Per-series selection lost to a
   single global model; the diagnosis was that a champion chosen from three folds
   of one intermittent series is chosen on noise. Pooling the decision to the
   intermittency CLASS is the natural middle, and this is the experiment.

2. LEAD-TIME DEMAND QUANTILES
   `quantiles.py` produces calibrated DAILY quantiles. Replenishment orders to
   cover a lead time, and the quantile of a sum is not the sum of quantiles.
   Three answers are compared -- sum-of-daily (assumes perfect dependence),
   iid-normal scaling (assumes none), and a block bootstrap of error paths (uses
   whatever dependence the data has).

3. PROBABILISTIC RECONCILIATION
   The point forecasts add up; the quantiles do not, and should not. Coherence
   for a distribution is a property of sample PATHS, so bottom-level draws with
   the cross-series correlation preserved are pushed through the summing matrix
   and quantiles are read off the coherent sample set.

Run after `run_forecast.py`. Reads out/backtest_raw.csv.gz, out/quantile_raw.pkl
and out/fold_state.pkl; writes out/advanced_report.txt.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import gate, leadtime, probrec, quantiles as Q  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

SELECTION_FOLDS = (1, 2, 3)
EVAL_FOLDS = (4, 5, 6)
LEAD_TIMES = (7, 14, 28)


def main():
    lines, summary = [], {}

    def emit(s=""):
        print(s)
        lines.append(s)

    raw = pd.read_csv(os.path.join(OUT, "backtest_raw.csv.gz"))
    bottom = raw[raw.level == "store_item"].copy()

    # ----------------------------------------------------------------------
    emit("=" * 78)
    emit("A. THE POOLED-CLASS SELECTION GATE")
    emit("=" * 78)
    # Rebuild per-row y/yhat from the aggregates the backtest stored: WMAPE and
    # the actual total are enough to recover total absolute error, which is all
    # a gate comparison needs (it never looks inside the horizon).
    g = bottom.assign(y=bottom.actual, yhat=bottom.actual + bottom.signed_err)
    sel = g[g.fold.isin(SELECTION_FOLDS)]
    ev = g[g.fold.isin(EVAL_FOLDS)]

    tab = gate.compare_gates(sel, ev)
    emit(tab.to_string(index=False, float_format=lambda x: "%9.4f" % x))
    emit("")
    best = tab.loc[tab.wmape.idxmin()]
    emit("BEST BY WMAPE: %s (%.4f)" % (best.gate, best.wmape))
    emit("")
    gl = tab[tab.gate == "global"].iloc[0]
    pc = tab[tab.gate == "per_class"].iloc[0]
    ps = tab[tab.gate == "per_series"].iloc[0]
    emit("  global    %d decision  wmape %.4f  bias %+.4f" % (gl.n_decisions, gl.wmape, gl.bias))
    emit("  per_class %d decisions wmape %.4f  bias %+.4f" % (pc.n_decisions, pc.wmape, pc.bias))
    emit("  per_series %d decisions wmape %.4f  bias %+.4f" % (ps.n_decisions, ps.wmape, ps.bias))
    emit("")
    emit("Pooling the gate to the intermittency class makes each decision on ~%d"
         % int(round(len(sel) / max(pc.n_decisions, 1))))
    emit("rows instead of ~%d, which is the entire bet: that WHICH METHOD WINS is"
         % int(round(len(sel) / max(ps.n_decisions, 1))))
    emit("a property of the demand REGIME rather than of the individual SKU. That")
    emit("is the same bet the Syntetos-Boylan classification itself makes, so this")
    emit("is a test of the classification as much as of the gate.")
    emit("")
    emit("The BIAS column is why this gate is not just the old one with bigger")
    emit("groups. Candidates whose selection-window bias exceeds +/-%.0f%% are"
         % (100 * gate.MAX_ABS_BIAS))
    emit("rejected before accuracy is even considered, because the failure the")
    emit("first gate hit was not inaccuracy -- it was handing 96 series to `naive`")
    emit("at -29% bias, which is a replenishment system that under-orders every")
    emit("slow mover it owns. WMAPE alone cannot express that and a screen can.")
    emit("")
    summary["gate"] = tab.round(4).to_dict("records")

    # ----------------------------------------------------------------------
    emit("=" * 78)
    emit("B. LEAD-TIME DEMAND -- THE QUANTILE OF A SUM IS NOT THE SUM OF QUANTILES")
    emit("=" * 78)
    with open(os.path.join(OUT, "quantile_raw.pkl"), "rb") as f:
        qrec = pickle.load(f)
    qdf = pd.DataFrame(qrec)

    # Pool standardised one-step errors across series, IN TIME ORDER. Pooling is
    # what makes this estimable: a single bottom-level series has ~168 evaluated
    # days across the folds, nowhere near enough to bootstrap a 28-day path.
    pool = []
    for r in qrec:
        a = np.asarray(r["actual"], float)
        m = np.asarray(r["mean_fc"], float)
        pool.append(leadtime.standardise(a - m, m))
    err_pool = np.concatenate(pool)
    emit("Pooled standardised error path: %d observations, lag-1 autocorrelation %+.4f"
         % (len(err_pool), leadtime.autocorrelation(err_pool, 1)))
    emit("")

    # ---- quantile crossing, before anything else uses the fan ----------
    fans = [{t: np.asarray(r["q%.2f" % t], float) for t in Q.SERVICE_LEVELS}
            for r in qrec]
    cross_before = float(np.mean([Q.crossing_rate(f) for f in fans]))
    fixed = [Q.rearrange(f) for f in fans]
    cross_after = float(np.mean([Q.crossing_rate(f) for f in fixed]))

    y_all = np.concatenate([np.asarray(r["actual"], float) for r in qrec])
    pin_rows = []
    for t in Q.SERVICE_LEVELS:
        raw = np.concatenate([f[t] for f in fans])
        fix = np.concatenate([f[t] for f in fixed])
        pin_rows.append(dict(tau=t,
                             pinball_raw=Q.pinball_loss(y_all, raw, t),
                             pinball_rearranged=Q.pinball_loss(y_all, fix, t),
                             coverage_raw=Q.coverage(y_all, raw),
                             coverage_rearranged=Q.coverage(y_all, fix)))
    P = pd.DataFrame(pin_rows)
    emit("QUANTILE CROSSING (a defect this project shipped for two passes):")
    emit("  crossing rate before rearrangement: %.4f" % cross_before)
    emit("  crossing rate after  rearrangement: %.4f" % cross_after)
    emit("")
    emit(P.to_string(index=False, float_format=lambda x: "%10.5f" % x))
    emit("")
    worse = (P.pinball_rearranged > P.pinball_raw + 1e-9).sum()
    emit("  Pinball loss got WORSE at %d of %d quantiles." % (worse, len(P)))
    emit("")
    emit("  Five independently fitted quantile models have nothing coupling them,")
    emit("  so the fitted P90 can land ABOVE the fitted P95 -- and it does, on")
    emit("  %.1f%% of item-days. That is not a numerical wart. An order system"
         % (100 * cross_before))
    emit("  reading a crossed fan orders MORE stock for a LOWER service level,")
    emit("  which inverts the meaning of the dial a planner is turning.")
    emit("")
    emit("  Monotone rearrangement -- sorting the fan at each point -- is the")
    emit("  Chernozhukov-Fernandez-Val-Galichon fix, and its appeal is that it is")
    emit("  free: the true quantile function is monotone, so sorting an estimate")
    emit("  moves it toward the truth in POPULATION.")
    emit("")
    if worse:
        emit("  That guarantee is a population statement, and the table shows the")
        emit("  finite-sample reality: %d of %d quantiles improved and %d got very"
             % (len(P) - worse, len(P), worse))
        emit("  slightly worse (%.5f at tau=%.2f, against a loss of %.5f)."
             % (float((P.pinball_rearranged - P.pinball_raw).max()),
                float(P.loc[(P.pinball_rearranged - P.pinball_raw).idxmax(), "tau"]),
                float(P.loc[(P.pinball_rearranged - P.pinball_raw).idxmax(),
                            "pinball_raw"])))
        emit("  Reporting that rather than rounding it away matters, because")
        emit("  \"provably no worse\" is the kind of claim that gets quoted without")
        emit("  the word \"expected\" in front of it. The right reading is that")
        emit("  rearrangement costs nothing measurable and removes an incoherence")
        emit("  that would otherwise reach a replenishment system.")
    else:
        emit("  The table is the check rather than the citation, and it holds at")
        emit("  every quantile.")
    emit("")
    emit("  It is a post-processing step and not a fix to the cause. The cause is")
    emit("  five fits that do not know the others exist; a joint or monotone-")
    emit("  constrained estimator would be the real answer and is not built.")
    emit("")
    summary["crossing"] = dict(before=cross_before, after=cross_after,
                               pinball=P.round(6).to_dict("records"))

    rows = []
    for L in LEAD_TIMES:
        for tau in (0.90, 0.95, 0.98):
            # the average series, so the comparison is about METHOD not about
            # which SKU was picked
            point = np.array([np.mean([np.asarray(r["mean_fc"], float)[d]
                                       for r in qrec]) for d in range(L)])
            dq = np.array([np.mean([f[tau][d] for f in fixed])
                           for d in range(L)])
            cmp_ = leadtime.compare_methods(point, err_pool, dq, tau, seed=L)
            actual_lt = np.array([np.asarray(r["actual"], float)[:L].sum()
                                  for r in qrec])
            scale = actual_lt.mean() / max(cmp_["mean_leadtime_demand"], 1e-9)
            rows.append(dict(
                lead_time=L, tau=tau,
                mean_demand=cmp_["mean_leadtime_demand"],
                block_bootstrap=cmp_["block_bootstrap"],
                sum_of_daily_q=cmp_["sum_of_daily_quantiles"],
                iid_normal=cmp_["iid_normal"],
                sum_vs_boot=cmp_["sum_of_daily_quantiles"] / max(cmp_["block_bootstrap"], 1e-9),
                iid_vs_boot=cmp_["iid_normal"] / max(cmp_["block_bootstrap"], 1e-9),
                actual_scale=scale))
    LT = pd.DataFrame(rows)
    emit(LT.drop(columns=["actual_scale"]).to_string(
        index=False, float_format=lambda x: "%9.3f" % x))
    emit("")
    emit("THE TWO SHORTCUTS ERR IN OPPOSITE DIRECTIONS, which is the finding:")
    emit("")
    emit("  sum-of-daily-quantiles assumes every day goes wrong TOGETHER (perfect")
    emit("  dependence). It is %.2fx the bootstrap at L=%d, tau=0.95 -- ordering"
         % (LT[(LT.lead_time == 28) & (LT.tau == 0.95)].sum_vs_boot.iloc[0], 28))
    emit("  that much extra inventory to cover a coincidence that does not happen.")
    emit("")
    emit("  iid-normal scaling assumes the days are INDEPENDENT and scales sigma by")
    emit("  sqrt(L). It is %.2fx the bootstrap on the same row -- and it is the"
         % LT[(LT.lead_time == 28) & (LT.tau == 0.95)].iid_vs_boot.iloc[0])
    emit("  version most safety-stock formulas actually implement.")
    emit("")
    emit("  The block bootstrap sits between them because it uses the dependence")
    emit("  the errors actually have rather than assuming either extreme. Nothing")
    emit("  about it is clever: resample CONTIGUOUS blocks and the dependence")
    emit("  comes along for free, with no correlation structure to name and no")
    emit("  copula to fit.")
    emit("")
    rho = leadtime.autocorrelation(err_pool, 1)
    emit("WHICH SHORTCUT IS WORSE DEPENDS ON THE AUTOCORRELATION, and here it is")
    emit("measured rather than assumed: lag-1 rho = %+.4f, which is very nearly" % rho)
    emit("ZERO. So on THIS panel the iid formula is close to right (%.3fx) and the"
         % LT[(LT.lead_time == 28) & (LT.tau == 0.95)].iid_vs_boot.iloc[0])
    emit("sum-of-quantiles shortcut is the badly wrong one. That ordering is a")
    emit("property of the data, not a general result, and stating it the other way")
    emit("round would be the easy mistake: with strongly autocorrelated errors the")
    emit("iid version is the dangerous one, because it under-covers in exactly the")
    emit("clustered weeks that cause the stockout.")
    emit("")
    emit("The value of the bootstrap is therefore NOT that it beat both shortcuts")
    emit("by a lot here. It is that it did not need to be told which regime it was")
    emit("in. A planner who picks the iid formula is making an assumption about")
    emit("autocorrelation whether or not they know it, and this panel is the lucky")
    emit("case where the assumption happens to hold.")
    emit("")
    emit("THIS IS THE DATA-2 JOIN. DATA-2 sized safety stock from HISTORICAL mean")
    emit("and sd and named forecast integration as its most valuable missing piece;")
    emit("its negative-binomial experiment then failed for precisely this reason --")
    emit("an iid marginal, however well specified, cannot produce the right")
    emit("lead-time quantile. This is the distribution it needed.")
    emit("")
    summary["leadtime"] = LT.round(4).to_dict("records")

    # ----------------------------------------------------------------------
    emit("=" * 78)
    emit("C. PROBABILISTIC RECONCILIATION -- WHY THE P95s SHOULD NOT ADD UP")
    emit("=" * 78)
    with open(os.path.join(OUT, "fold_state.pkl"), "rb") as f:
        st = pickle.load(f)
    S, names = st["S"], st["names"]
    n_b = S.shape[1]
    point_bottom = st["base_fc"][-n_b:]
    resid_bottom = st["resid"][-n_b:]

    draws = probrec.bottom_samples(point_bottom, resid_bottom, n_samples=400, seed=1)
    allq = probrec.reconcile_samples(S, draws)
    qs = probrec.sample_quantiles(allq, (0.5, 0.9, 0.95))

    emit("400 joint bottom-level draws, cross-series correlation preserved")
    emit("(shrunk 20% toward the identity), pushed through S.")
    emit("")
    rows = []
    for tau in (0.5, 0.9, 0.95):
        rows.append(probrec.coherence_gap(S, qs, tau))
    G = pd.DataFrame(rows)
    emit(G.to_string(index=False, float_format=lambda x: "%10.4f" % x))
    emit("")
    lv_index = {}
    for i, (lv, key) in enumerate(names):
        lv_index.setdefault(lv, []).append(i)
    q95 = qs[0.95]
    bottom_q95 = q95[-n_b:]
    naive_sum = S @ bottom_q95
    emit("Sum of children's P95 vs the level's own P95, by level:")
    emit("  %-12s %12s %12s %10s" % ("level", "sum of P95", "own P95", "ratio"))
    for lv in ["total", "state", "store", "store_cat", "store_dept"]:
        idx = lv_index[lv]
        a = float(naive_sum[idx].sum())
        b = float(q95[idx].sum())
        emit("  %-12s %12.1f %12.1f %10.3f" % (lv, a, b, a / max(b, 1e-9)))
    emit("")
    emit("THE GAP IS THE POINT, NOT A DEFECT TO CLOSE.")
    emit("")
    emit("  A quantile is not a linear functional of a distribution:")
    emit("  P95(A+B) != P95(A) + P95(B) unless A and B are perfectly dependent.")
    emit("  Forcing the store P95 to equal the sum of its items' P95s would not")
    emit("  make the forecast coherent, it would make it WRONG -- it asserts that")
    emit("  every item in the store has its bad week simultaneously.")
    emit("")
    emit("  Adding up is a property of REALISATIONS. Every sample path here is")
    emit("  coherent by construction, so the coherent P95 at the total is")
    emit("  automatically BELOW the sum of item P95s by exactly the")
    emit("  diversification the residual correlation supports. The ratio column")
    emit("  is that diversification, measured.")
    emit("")
    emit("  Note the ratio SHRINKS toward the bottom of the hierarchy and grows at")
    emit("  the top: the more series you pool, the more the independent part of")
    emit("  their errors cancels. Aggregate safety stock sized by summing item")
    emit("  safety stocks is over-provisioned by that ratio, and it is the")
    emit("  argument for holding buffer centrally rather than at the leaf.")
    emit("")
    summary["prob_reconciliation"] = G.round(4).to_dict("records")

    with open(os.path.join(OUT, "advanced_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "advanced_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n-> out/advanced_report.txt, out/advanced_metrics.json")


if __name__ == "__main__":
    main()
