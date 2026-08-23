"""Rolling-origin backtest: baselines first, GBM second, FVA reported for both.

Protocol
--------
Six folds, 28-day horizon, origins stepping back 28 days at a time. Training is
everything strictly before the origin. There is no random split anywhere in this
file, because a random split on a time series is not a weak evaluation, it is a
broken one -- it trains on the future of the very series it scores.

Folds 1-3 are the SELECTION window (used to decide which method wins per series);
folds 4-6 are the EVALUATION window. The per-series champion is therefore chosen
without seeing the data it is scored on.

Everything is reported as Forecast Value Added against seasonal naive, and the
places the GBM LOSES to seasonal naive are printed as prominently as the places
it wins, because on a real assortment that share is never zero and a planner
needs to know which items to leave on the simple rule.

Writes out/backtest_raw.csv.gz and out/reconciliation_raw.csv.gz; report.py turns
those into the tables.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import core, deep, feats, gbm, hier, probrec  # noqa: E402
from src import quantiles as Q  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")

H = 28
N_FOLDS = 6
RESID_TAIL = 180  # days of in-sample residual used to estimate MinT's W


def fit_predict_quantile(train, test, tau: float):
    """A separate model per quantile. The pinball loss does not decompose across
    taus, so there is no shared fit to be had -- which is why this runs only at
    the bottom level, where replenishment actually places orders."""
    m = gbm.fit_quantile(train[feats.FEATURE_COLS], train.y, tau)
    return gbm.predict(m, test[feats.FEATURE_COLS])


def fit_predict_gbm(train, test):
    """One global model per level. Poisson loss: the target is a unit count."""
    m = gbm.fit_point(train[feats.FEATURE_COLS], train.y)
    return (gbm.predict(m, test[feats.FEATURE_COLS]), m,
            gbm.predict(m, train[feats.FEATURE_COLS]))


def fit_predict_nbeats(bottom_panel, origin, h_dates):
    """The global deep arm, trained per fold on everything before the origin.

    Trained on the BOTTOM level only. N-BEATS is a global model over series that
    share a shape; aggregate levels have a different shape and only 30 of them,
    so pooling them in would be adding noise to buy nothing.
    """
    hist = bottom_panel[bottom_panel.index < origin]
    arr = hist.to_numpy(float).T                      # (n_series x T)
    X, Y, _ = deep.make_windows(arr)
    model = deep.train(X, Y, epochs=8)
    fc = deep.forecast(model, arr)[:, :len(h_dates)]
    return {c: fc[i] for i, c in enumerate(bottom_panel.columns)}


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(os.path.join(DATA, "sales.parquet"))
    cal = pd.read_csv(os.path.join(DATA, "calendar.csv"), parse_dates=["date"])

    panels = hier.build_panels(df)
    exog = hier.build_exog(df)
    dates = panels["store_item"].index
    print("levels:", {k: v.shape[1] for k, v in panels.items()})

    long = {}
    for lv in hier.LEVELS:
        long[lv] = feats.add_features(feats.long_frame(panels[lv], exog[lv], cal))
    print("features built in %.1fs" % (time.time() - t0))

    # ---- Syntetos-Boylan classification, bottom level ----
    bottom = panels["store_item"]
    cls = pd.DataFrame([
        dict(key=c, adi=core.adi_cv2(bottom[c].to_numpy())[0],
             cv2=core.adi_cv2(bottom[c].to_numpy())[1],
             cls=core.classify(bottom[c].to_numpy()),
             mean=float(bottom[c].mean()), zero_share=float((bottom[c] == 0).mean()))
        for c in bottom.columns]).set_index("key")
    cls.to_csv(os.path.join(OUT, "intermittency.csv"))
    print("\nSyntetos-Boylan classes (bottom level, n=%d):" % len(cls))
    print(cls.cls.value_counts().to_string())
    print()

    S, names = hier.summing_matrix(panels, df)
    name_pos = {n: i for i, n in enumerate(names)}
    origins = [dates[-(k * H)] for k in range(N_FOLDS, 0, -1)]

    records, recon_records, qrecords = [], [], []

    for fi, origin in enumerate(origins, start=1):
        h_dates = dates[dates >= origin][:H]
        base_fc = np.zeros((len(names), H))
        resid = np.zeros((len(names), RESID_TAIL))
        nbeats_fc = fit_predict_nbeats(panels["store_item"], origin, h_dates)

        for lv in hier.LEVELS:
            L = long[lv]
            ok = L[feats.FEATURE_COLS].notna().all(axis=1)
            tr = L[(L.date < origin) & ok]
            te = L[L.date.isin(h_dates)].copy()
            pred, _model, fitted = fit_predict_gbm(tr, te)
            te["gbm"] = pred

            # Quantiles only at the bottom level: that is where replenishment
            # places an order, and one model per tau per level would triple a
            # six-minute backtest to buy numbers nobody would use.
            if lv == "store_item":
                for tau in Q.SERVICE_LEVELS:
                    te["q%.2f" % tau] = fit_predict_quantile(tr, te, tau)

            # residual tail per key, taken in one pass (not a per-key filter)
            rdf = pd.DataFrame({"key": tr.key.to_numpy(), "r": tr.y.to_numpy() - fitted})
            rmap = {k: g.r.to_numpy()[-RESID_TAIL:] for k, g in rdf.groupby("key", observed=True)}

            for key, grp in te.groupby("key", observed=True):
                grp = grp.sort_values("date")
                y_true = grp.y.to_numpy(float)
                hist = panels[lv][key]
                y_train = hist[hist.index < origin].to_numpy(float)

                fc = {"gbm": grp.gbm.to_numpy(float)}
                for bname, fn in core.BASELINES.items():
                    fc[bname] = fn(y_train, len(y_true))
                if lv == "store_item" and key in nbeats_fc:
                    fc["nbeats"] = np.asarray(nbeats_fc[key], float)[:len(y_true)]

                for meth, yhat in fc.items():
                    records.append(dict(
                        fold=fi, level=lv, key=key, method=meth,
                        klass=(cls.cls.get(key, "n/a") if lv == "store_item"
                               else "n/a"),
                        wmape=core.wmape(y_true, yhat),
                        bias=core.bias_pct(y_true, yhat),
                        mase=core.mase(y_true, yhat, y_train),
                        rmsse=core.rmsse(y_true, yhat, y_train),
                        abs_err=float(np.abs(y_true - yhat).sum()),
                        abs_err_h1_7=float(np.abs(y_true[:7] - yhat[:7]).sum()),
                        abs_err_h8_28=float(np.abs(y_true[7:] - yhat[7:]).sum()),
                        signed_err=float((yhat - y_true).sum()),
                        actual=float(y_true.sum()),
                        actual_h1_7=float(y_true[:7].sum()),
                        actual_h8_28=float(y_true[7:].sum()),
                    ))

                if lv == "store_item":
                    qrow = dict(fold=fi, key=key)
                    for tau in Q.SERVICE_LEVELS:
                        col = "q%.2f" % tau
                        qrow[col] = grp[col].to_numpy(float).tolist()
                    qrow["actual"] = y_true.tolist()
                    qrow["mean_fc"] = fc["gbm"].tolist()
                    qrecords.append(qrow)

                i = name_pos[(lv, key)]
                base_fc[i] = fc["gbm"]
                r = rmap.get(key, np.zeros(RESID_TAIL))
                resid[i] = np.pad(r, (max(0, RESID_TAIL - len(r)), 0))[-RESID_TAIL:]

        bu = hier.bottom_up(S, base_fc[-S.shape[1]:])

        # MIDDLE-OUT: forecast at store x department (30 series), then push down
        # by each item's historical share of its department and aggregate back
        # up. The proportions are held FIXED over the horizon, which is the
        # method's real weakness and the reason it is scored rather than
        # asserted -- a promotion changes exactly those shares.
        mid_lv = "store_dept"
        mid_names = list(panels[mid_lv].columns)
        mid_pos = {c: i for i, c in enumerate(mid_names)}
        bottom_cols = list(panels["store_item"].columns)
        meta_dept = (df.assign(bkey=df.store_id + "|" + df.item_id,
                               dkey=df.store_id + "|" + df.dept_id)
                       [["bkey", "dkey"]].drop_duplicates().set_index("bkey").dkey)
        group_ids = np.array([meta_dept[c] for c in bottom_cols])
        hist_bottom = panels["store_item"][panels["store_item"].index < origin]
        props = probrec.proportions(hist_bottom.to_numpy(float).T, group_ids)
        mid_fc = np.vstack([base_fc[name_pos[(mid_lv, c)]] for c in mid_names])
        mo_bottom = probrec.middle_out(
            mid_fc[[mid_pos[g] for g in np.unique(group_ids)]],
            group_ids, props)
        mo = hier.bottom_up(S, mo_bottom)

        mt = hier.mint(S, base_fc, resid, method="shrink")
        mt_ols = hier.mint(S, base_fc, resid, method="ols")
        mt_wls = hier.mint(S, base_fc, resid, method="wls")
        actual = np.vstack([panels[lv][key].reindex(h_dates).to_numpy(float)
                            for lv, key in names])

        for label, fcm in (("base_incoherent", base_fc), ("bottom_up", bu),
                           ("middle_out", mo),
                           ("mint_ols", mt_ols), ("mint_wls", mt_wls),
                           ("mint_shrink", mt)):
            for i, (lv, key) in enumerate(names):
                recon_records.append(dict(
                    fold=fi, level=lv, key=key, recon=label,
                    abs_err=float(np.abs(actual[i] - fcm[i]).sum()),
                    signed_err=float((fcm[i] - actual[i]).sum()),
                    actual=float(actual[i].sum())))
            recon_records.append(dict(
                fold=fi, level="_coherence", key="_max_violation", recon=label,
                abs_err=hier.coherence_error(S, fcm), signed_err=np.nan, actual=np.nan))

        print("fold %d  origin %s  (%.0fs elapsed)" % (fi, origin.date(), time.time() - t0))

    pd.DataFrame(records).to_csv(os.path.join(OUT, "backtest_raw.csv.gz"),
                                 index=False, compression="gzip")
    pd.DataFrame(recon_records).to_csv(os.path.join(OUT, "reconciliation_raw.csv.gz"),
                                       index=False, compression="gzip")
    import pickle
    with open(os.path.join(OUT, "quantile_raw.pkl"), "wb") as f:
        pickle.dump(qrecords, f)
    # The last fold's bottom-level point forecasts, residual history and summing
    # matrix -- everything run_advanced.py needs for lead-time quantiles and
    # probabilistic reconciliation, so that analysis can be re-run without
    # paying for the backtest again.
    with open(os.path.join(OUT, "fold_state.pkl"), "wb") as f:
        pickle.dump(dict(S=S, names=names, base_fc=base_fc, resid=resid,
                         bottom_cols=list(panels["store_item"].columns),
                         actual=actual), f)
    print("\nbacktest complete in %.0fs -> out/backtest_raw.csv.gz" % (time.time() - t0))


if __name__ == "__main__":
    main()
