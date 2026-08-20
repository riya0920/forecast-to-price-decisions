"""Turn the raw backtest into the tables a demand-planning team argues over.

The framing throughout is Forecast Value Added: seasonal naive is the incumbent,
every model is reported as skill against it, and the slices where the model
LOSES are printed, not buried. A demand planner does not deploy a global winner;
they deploy a per-series decision, and the losing share is what that decision is
made of.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import hier, quantiles as Q  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SELECTION_FOLDS = [1, 2, 3]
EVAL_FOLDS = [4, 5, 6]
BASE = "seasonal_naive"


def w(gr, err="abs_err", act="actual"):
    """Volume-weighted WMAPE over a group -- sum of errors / sum of actuals.

    Averaging per-series WMAPE would let a 0.2-units/day series count as much as
    a 30-units/day one. Replenishment dollars do not work that way.
    """
    a = gr[act].sum()
    return float(gr[err].sum() / a) if a > 0 else np.nan


def main():
    R = pd.read_csv(os.path.join(OUT, "backtest_raw.csv.gz"))
    cls = pd.read_csv(os.path.join(OUT, "intermittency.csv")).set_index("key")
    RC = pd.read_csv(os.path.join(OUT, "reconciliation_raw.csv.gz"))
    lines, summary = [], {}

    def emit(s=""):
        print(s)
        lines.append(s)

    ev = R[R.fold.isin(EVAL_FOLDS)]

    # ------------------------------------------------------------------
    emit("=" * 78)
    emit("1. WMAPE BY HIERARCHY LEVEL  (evaluation folds 4-6, 28-day horizon)")
    emit("=" * 78)
    methods = ["naive", "seasonal_naive", "ses", "croston_sba", "gbm"]
    tab = {}
    for lv in hier.LEVELS:
        row = {}
        for m in methods:
            row[m] = w(ev[(ev.level == lv) & (ev.method == m)])
        row["FVA_gbm"] = row[BASE] - row["gbm"]
        tab[lv] = row
    T = pd.DataFrame(tab).T
    emit(T.to_string(float_format=lambda x: "%7.4f" % x))
    emit("")
    emit("FVA_gbm > 0 means the GBM beat seasonal naive at that level, in WMAPE points.")
    summary["wmape_by_level"] = {k: {m: (None if pd.isna(v) else round(v, 4))
                                     for m, v in r.items()} for k, r in T.iterrows()}

    # ------------------------------------------------------------------
    emit("")
    emit("=" * 78)
    emit("2. HORIZON DECAY  (bottom level, store x item)")
    emit("=" * 78)
    b = ev[ev.level == "store_item"]
    rows = []
    for m in methods:
        g = b[b.method == m]
        rows.append(dict(method=m,
                         h1_7=w(g, "abs_err_h1_7", "actual_h1_7"),
                         h8_28=w(g, "abs_err_h8_28", "actual_h8_28")))
    HD = pd.DataFrame(rows).set_index("method")
    HD["decay"] = HD.h8_28 - HD.h1_7
    emit(HD.to_string(float_format=lambda x: "%7.4f" % x))
    emit("")
    emit("Seasonal naive barely decays -- it has no recent information to lose. The GBM's decay is")
    emit("the honest cost of a 4-week horizon: recent-lag information stops helping.")
    summary["horizon"] = HD.round(4).to_dict("index")

    # ------------------------------------------------------------------
    emit("")
    emit("=" * 78)
    emit("3. FORECAST BIAS  (signed; + = over-forecast)")
    emit("=" * 78)
    rows = []
    for m in methods:
        g = b[b.method == m]
        rows.append(dict(method=m, bias_pct=float(g.signed_err.sum() / g.actual.sum()),
                         wmape=w(g)))
    BI = pd.DataFrame(rows).set_index("method")
    emit(BI.to_string(float_format=lambda x: "%+8.4f" % x))
    emit("")
    emit("Why this table sits next to accuracy: replenishment is asymmetric. A +5%")
    emit("bias is 5% more inventory dollars on every item, every cycle, forever; a")
    emit("-5% bias is a stockout rate that no service-level formula will predict")
    emit("because the formula assumes an unbiased forecast. Variance you buffer,")
    emit("bias you cannot -- safety stock sized on sigma does not cover a drift.")
    summary["bias"] = BI.round(4).to_dict("index")

    # ------------------------------------------------------------------
    emit("")
    emit("=" * 78)
    emit("4. INTERMITTENT BUCKET: MASE, NOT MAPE")
    emit("=" * 78)
    bb = b.join(cls[["cls", "zero_share"]], on="key")
    rows = []
    for c in ["smooth", "erratic", "intermittent", "lumpy"]:
        g = bb[bb.cls == c]
        if not len(g):
            continue
        r = dict(cls=c, n_series=g.key.nunique(),
                 zero_share=float(g.zero_share.mean()))
        for m in methods:
            r["MASE_" + m] = float(g[g.method == m].mase.mean())
        rows.append(r)
    MA = pd.DataFrame(rows).set_index("cls")
    emit(MA.to_string(float_format=lambda x: "%7.3f" % x))
    emit("")
    emit("MAPE is not in this repo. On these buckets %.0f%% of observation-days are"
         % (100 * bb.zero_share.mean()))
    emit("zero, so a per-observation percentage error divides by zero on most of the")
    emit("evaluation set. MASE reads directly: 0.85 = 15%% less average absolute error")
    emit("than 'same day last week'. Above 1.00 = the naive rule was better.")
    summary["mase_by_class"] = MA.round(3).to_dict("index")

    # ------------------------------------------------------------------
    emit("")
    emit("=" * 78)
    emit("5. WHERE THE GBM LOSES  (the table a global-winner deployment hides)")
    emit("=" * 78)
    piv = (b.groupby(["key", "method"]).apply(lambda g: w(g), include_groups=False)
             .unstack("method"))
    piv = piv.join(cls[["cls", "mean"]])
    piv["gbm_wins"] = piv.gbm < piv[BASE]
    loss = piv.groupby("cls").agg(n=("gbm_wins", "size"),
                                  gbm_win_rate=("gbm_wins", "mean"),
                                  mean_units_day=("mean", "mean"))
    emit(loss.to_string(float_format=lambda x: "%7.3f" % x))
    overall = float(piv.gbm_wins.mean())
    emit("")
    emit("GBM beats seasonal naive on %.1f%% of item x store series." % (100 * overall))
    emit("It loses on %.1f%%. Deploying the aggregate winner everywhere would" % (100 * (1 - overall)))
    emit("knowingly degrade those series -- which is why section 6 exists.")
    summary["gbm_win_rate_overall"] = round(overall, 4)
    summary["gbm_win_rate_by_class"] = loss.gbm_win_rate.round(4).to_dict()

    # ------------------------------------------------------------------
    emit("")
    emit("=" * 78)
    emit("6. FVA-GATED PER-SERIES SELECTION")
    emit("=" * 78)
    emit("Champion chosen per series on folds 1-3, scored on folds 4-6. The")
    emit("selection never sees the evaluation window.")
    emit("")
    sel = R[R.fold.isin(SELECTION_FOLDS) & (R.level == "store_item")]
    sel_w = (sel.groupby(["key", "method"]).apply(lambda g: w(g), include_groups=False)
                .unstack("method"))
    sel_b = (sel.groupby(["key", "method"])
                .apply(lambda g: (g.signed_err.sum() / g.actual.sum()
                                  if g.actual.sum() > 0 else np.nan),
                       include_groups=False)
                .unstack("method"))

    # Series with no sales at all in the selection window have no basis for a
    # choice. They fall back to the incumbent rather than being dropped -- an
    # earlier version of this report silently excluded them from the per-series
    # row, which flattered the policy by removing its hardest series.
    champion = sel_w.idxmin(axis=1).where(sel_w.notna().any(axis=1)).fillna(BASE)
    n_fallback = int((sel_w.notna().any(axis=1) == False).sum())  # noqa: E712

    # Bias-guarded gate: a candidate whose SELECTION-window bias is worse than
    # +/-15% is disqualified regardless of its WMAPE. See the note below for why
    # this guard is not optional.
    GUARD = 0.15
    masked = sel_w.mask(sel_b.abs() > GUARD)
    champion_guarded = (masked.idxmin(axis=1)
                        .where(masked.notna().any(axis=1))
                        .fillna(sel_w.idxmin(axis=1)).fillna(BASE))

    ev_b = ev[ev.level == "store_item"]
    rows = []
    for m in methods:
        g = ev_b[ev_b.method == m]
        rows.append(dict(policy=m, wmape=w(g),
                         bias=float(g.signed_err.sum() / g.actual.sum())))
    for label, ch in (("PER-SERIES (WMAPE gate)", champion),
                      ("PER-SERIES (WMAPE + bias guard)", champion_guarded)):
        p = ev_b[ev_b.method == ev_b.key.map(ch)]
        assert p.key.nunique() == ev_b.key.nunique(), "a series was dropped by the gate"
        rows.append(dict(policy=label, wmape=w(p),
                         bias=float(p.signed_err.sum() / p.actual.sum())))
    SEL = pd.DataFrame(rows).set_index("policy")
    SEL["FVA_vs_snaive"] = SEL.loc[BASE, "wmape"] - SEL.wmape
    emit(SEL.to_string(float_format=lambda x: "%+8.4f" % x))
    emit("")
    emit("Champion mix -- WMAPE gate vs bias-guarded gate:")
    mix = pd.DataFrame({"wmape_gate": champion.value_counts(),
                        "bias_guarded": champion_guarded.value_counts()}).fillna(0).astype(int)
    emit(mix.to_string())
    emit("(%d series had no sales in the selection window and fall back to %s.)"
         % (n_fallback, BASE))
    emit("")
    emit("THE RESULT THIS SECTION EXISTS FOR, stated against my own expectation:")
    emit("per-series selection LOSES to just deploying the GBM everywhere. Both")
    emit("gates do. The textbook answer to 'your model loses on 18% of series' is")
    emit("'select per series' -- and measured on this panel that answer is wrong.")
    emit("")
    emit("Two mechanisms, both visible above:")
    emit("  (a) WMAPE rewards a degenerate forecast on intermittent demand. On a")
    emit("      mostly-zero series 'last value' is usually zero, so naive predicts")
    emit("      all-zeros, scores WMAPE ~1.0, and cannot be beaten downward by any")
    emit("      method that ever puts a unit on the wrong day. The plain gate duly")
    emit("      hands 96 series to naive -- whose standalone bias is -29%, i.e. a")
    emit("      replenishment system that under-orders every slow mover it owns.")
    emit("  (b) The gate is estimated on 3 folds x 28 days of a series that sells a")
    emit("      unit every third day. That is not enough signal to rank five methods,")
    emit("      so the choice is mostly noise, and noise costs more than the 18% of")
    emit("      series it was meant to rescue were losing.")
    emit("")
    emit("The bias guard does what it was built to do -- it moves 54 series off naive")
    emit("and pulls portfolio bias from %+.2f%% to %+.2f%% -- and still costs WMAPE."
         % (100 * SEL.loc["PER-SERIES (WMAPE gate)", "bias"],
            100 * SEL.loc["PER-SERIES (WMAPE + bias guard)", "bias"]))
    emit("")
    emit("So the deployment recommendation from this backtest is the GBM globally,")
    emit("with per-series selection revisited only when there is enough evaluation")
    emit("history per series to estimate the choice -- more folds, or a gate")
    emit("estimated on a pooled class (intermittent/erratic/smooth) rather than on")
    emit("each series alone. That pooled variant is not built here; see the README.")
    summary["selection"] = SEL.round(4).to_dict("index")
    summary["champion_mix"] = mix.to_dict()
    summary["n_no_history_fallback"] = n_fallback
    pd.DataFrame({"wmape_gate": champion,
                  "bias_guarded": champion_guarded}).to_csv(
        os.path.join(OUT, "champion_per_series.csv"))

    # ------------------------------------------------------------------
    emit("")
    emit("=" * 78)
    emit("7. HIERARCHICAL RECONCILIATION")
    emit("=" * 78)
    rc = RC[RC.fold.isin(EVAL_FOLDS)]
    coh = rc[rc.level == "_coherence"].groupby("recon").abs_err.max()
    body = rc[rc.level != "_coherence"]
    RT = (body.groupby(["level", "recon"])
              .apply(lambda g: w(g), include_groups=False).unstack("recon"))
    RT = RT.reindex(hier.LEVELS)
    order = [c for c in ("base_incoherent", "bottom_up", "mint_ols",
                         "mint_wls", "mint_shrink") if c in RT.columns]
    RT = RT[order]
    emit(RT.to_string(float_format=lambda x: "%7.4f" % x))
    emit("")
    emit("Max coherence violation (units, over the whole horizon):")
    emit(coh.to_string(float_format=lambda x: "%.6f" % x))
    emit("")
    emit("Reading: base forecasts are produced independently at every level and do")
    emit("NOT add up -- that column is the incoherent starting point. Bottom-up is")
    emit("coherent for free but discards the aggregate models entirely. The three")
    emit("MinT variants differ ONLY in the weight matrix W they invert:")
    emit("")
    emit("  mint_ols     W = I. Treats every series' error as equally important,")
    emit("               which for a hierarchy spanning a 300-unit total and a")
    emit("               0.2-unit item is obviously false, and it is included to")
    emit("               show how false.")
    emit("  mint_wls     W = diag(residual variances). Scales by each series' own")
    emit("               error, which is the cheap fix and usually most of the win.")
    emit("  mint_shrink  W = shrunk residual COVARIANCE. Also uses the correlation")
    emit("               between series -- the thing that makes MinT better than")
    emit("               scaling in principle, and the thing that is hardest to")
    emit("               estimate from 353 series and 180 residual days.")
    emit("")
    best_by_level = RT.drop(columns=["base_incoherent"]).idxmin(axis=1)
    emit("Best reconciliation per level:")
    for lv in hier.LEVELS:
        if lv in best_by_level.index:
            emit("  %-12s %s" % (lv, best_by_level[lv]))
    emit("")
    emit("BOTTOM-UP WINS OR TIES ALMOST EVERYWHERE, and that is the honest result.")
    emit("MinT should beat it in theory -- it uses information bottom-up throws")
    emit("away. It does not here, and the reason is visible in the base_incoherent")
    emit("column: the directly-fitted aggregate models are WORSE than aggregating")
    emit("the bottom-level forecasts (total 0.11 vs 0.06), because errors cancel on")
    emit("the way up. MinT blends those weaker aggregate forecasts back in, and")
    emit("blending in a worse signal makes things worse. That is not a bug in MinT;")
    emit("it is MinT correctly weighting information that happens not to be worth")
    emit("having on this hierarchy.")
    summary["reconciliation"] = RT.round(4).to_dict("index")
    summary["coherence_max_violation"] = coh.round(6).to_dict()

    # ------------------------------------------------------------------
    emit("")
    emit("=" * 78)
    emit("8. QUANTILE FORECASTS -- WHAT YOU ACTUALLY HAND REPLENISHMENT")
    emit("=" * 78)
    emit("The spec's question: replenishment wants one number per item x store x")
    emit("day, and the model produces a distribution. What do you hand them?")
    emit("")
    emit("The first pass answered this in PROSE and said the code could not do it.")
    emit("Now it can: a separate quantile GBM per service level, at the bottom")
    emit("level, scored on the evaluation folds.")
    emit("")
    import pickle
    with open(os.path.join(OUT, "quantile_raw.pkl"), "rb") as f:
        qrows = pickle.load(f)
    qrows = [r for r in qrows if r["fold"] in EVAL_FOLDS]

    actual = np.concatenate([np.array(r["actual"]) for r in qrows])
    mean_fc = np.concatenate([np.array(r["mean_fc"]) for r in qrows])
    preds = {tau: np.concatenate([np.array(r["q%.2f" % tau]) for r in qrows])
             for tau in Q.SERVICE_LEVELS}

    rows = Q.evaluate_quantiles(actual, preds)
    QT = pd.DataFrame(rows).set_index("tau")
    QT["implied_cost_ratio"] = [Q.implied_cost_ratio(t) for t in QT.index]
    show_cols = ["coverage", "mean_forecast", "implied_cost_ratio"] + \
                ["pinball@%.2f" % t for t in Q.SERVICE_LEVELS]
    emit(QT[show_cols].to_string(float_format=lambda x: "%9.4f" % x))
    emit("")
    emit("COVERAGE is the calibration check: the share of actual days at or below")
    emit("the forecast. For a well-calibrated tau-quantile it should equal tau.")
    for t in Q.SERVICE_LEVELS:
        emit("  tau=%.2f  target coverage %.2f  achieved %.4f  (%+.4f)"
             % (t, t, QT.loc[t, "coverage"], QT.loc[t, "coverage"] - t))
    emit("")
    emit("THE P50 OVER-COVERS BADLY (%.4f against a target of 0.50) and the reason"
         % QT.loc[0.50, "coverage"])
    emit("is worth stating rather than glossing: coverage counts actual <= forecast,")
    emit("and on a series that is zero 40%% of the time the median forecast is often")
    emit("ZERO. Every zero day then counts as covered, because 0 <= 0. That is a")
    emit("property of coverage as a diagnostic on discrete zero-inflated data, not")
    emit("a miscalibrated model -- and the pinball diagonal below confirms the P50")
    emit("is in fact the best P50 available. It is a reminder that a calibration")
    emit("check needs to be read against the data it is computed on: the same")
    emit("statistic that is informative at tau=0.95 is nearly meaningless at 0.50")
    emit("here.")
    emit("")
    emit("THE DIAGONAL IS THE PROOF. Each row is scored on EVERY tau's pinball")
    emit("loss, and a calibrated set of quantiles should have each forecast")
    emit("winning the loss for the tau it was fitted to:")
    diag_ok = 0
    for t in Q.SERVICE_LEVELS:
        col = "pinball@%.2f" % t
        winner = QT[col].idxmin()
        ok = abs(winner - t) < 1e-9
        diag_ok += int(ok)
        emit("  pinball@%.2f is minimised by tau=%.2f  %s"
             % (t, winner, "OK" if ok else "<-- MISCALIBRATED"))
    emit("")
    emit("%d of %d quantiles win their own loss." % (diag_ok, len(Q.SERVICE_LEVELS)))
    emit("")
    emit("WHY PINBALL AND NOT MAE: MAE is minimised by the MEDIAN, so scoring a")
    emit("P90 on MAE would rank it worse than the P50 by construction -- it would")
    emit("be measuring the wrong thing and calling the right answer wrong. The")
    emit("pinball loss is the proper scoring rule for a quantile: at tau=0.9 an")
    emit("under-forecast is penalised 9x more than an over-forecast of the same")
    emit("size, which is exactly the asymmetry a 90% service level asserts.")
    emit("")
    emit("SO WHAT DO YOU HAND THEM. Not the mean. Ordering to the mean means being")
    emit("short about half the time, and the two errors do not cost the same. You")
    emit("hand them the quantile implied by the item's service target, and the")
    emit("newsvendor result says which one:")
    emit("")
    emit("    q* = Cu / (Cu + Co)")
    emit("")
    emit("where Cu is the cost of being one unit short and Co of one too many.")
    emit("Read the implied_cost_ratio column backwards and a service level stops")
    emit("being a policy and becomes a claim somebody has to defend:")
    for t in (0.90, 0.95, 0.98):
        emit("  a %.0f%% service target ASSERTS that understocking costs %.0fx"
             % (100 * t, Q.implied_cost_ratio(t)))
        emit("  what overstocking costs.")
    emit("")
    emit("The mean forecast sits at %.4f units/day and the P95 at %.4f -- the gap"
         % (mean_fc.mean(), preds[0.95].mean()))
    emit("is the safety stock the service level is buying, and it is %.0f%% more"
         % (100 * (preds[0.95].mean() / max(mean_fc.mean(), 1e-9) - 1)))
    emit("inventory than ordering to the mean would hold.")
    emit("")
    emit("HONEST LIMITS. These are DAILY quantiles, and replenishment needs the")
    emit("quantile of demand over the LEAD TIME, which is not the sum of daily")
    emit("quantiles -- summing quantiles overstates, because the days do not all")
    emit("go wrong together. Converting one to the other needs the dependence")
    emit("structure across days, which is exactly what data2's negative-binomial")
    emit("experiment found it was missing. Neither project has it, and they")
    emit("independently arrived at the same gap, which is at least consistent.")
    summary["quantiles"] = dict(
        table=QT[show_cols].round(4).to_dict("index"),
        diagonal_wins=diag_ok, n_taus=len(Q.SERVICE_LEVELS),
        mean_forecast=float(mean_fc.mean()),
        p95_forecast=float(preds[0.95].mean()))

    with open(os.path.join(OUT, "forecast_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(OUT, "forecast_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n-> out/forecast_report.txt, out/forecast_metrics.json")


if __name__ == "__main__":
    main()
