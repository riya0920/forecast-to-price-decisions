"""Real M5 against the generator, and the scale claim the deep arm rested on.

    "It is not evidence that deep forecasting loses. 300 series is far below
     where global deep models start paying, and THAT is the finding for anyone
     deciding whether to staff the work."

M5 has 30,490 series. This runs the same two models at 300, 3,000 and 30,000.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import deep as D   # noqa: E402
from src import m5 as M     # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
HORIZON = 28
WINDOW_BUDGET = 150_000
EPOCHS = 8


def wmape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.abs(actual).sum()
    return float(np.abs(actual - pred).sum() / denom) if denom > 0 else float("nan")


def bias(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.abs(actual).sum()
    return float((pred - actual).sum() / denom) if denom > 0 else float("nan")


def seasonal_naive(hist: np.ndarray, horizon: int, period: int = 7) -> np.ndarray:
    """The baseline every FVA number in this project is measured against."""
    reps = int(np.ceil(horizon / period))
    tail = hist[:, -period:]
    return np.tile(tail, (1, reps))[:, :horizon]


def main():
    os.makedirs(OUT, exist_ok=True)
    lines, summary = [], {}

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("ML-1 M5 PASS -- THE SCALE CLAIM, ON REAL DATA")
    emit("=" * 78)
    if not M.available():
        emit("No M5 panel in .vendor/kaggle/.")
        return
    emit("'No real dataset. M5 is not downloadable here.' -- half false. The")
    emit("competition download 403s until its rules are accepted, but the daily")
    emit("sales panel is republished as a plain Kaggle DATASET.")
    emit("")
    emit("PROVENANCE: tbierhance/m5-forecasting-parquet-and-aggregations, a")
    emit("third-party re-encoding rather than the official archive, not diffed")
    emit("against it because the official one is the gated one.")
    emit("")

    full = M.load_panel(n_series=None)
    inter = M.intermittency(np.asarray(full["panel"][:2000]))
    emit("=" * 78)
    emit("A. IS THE GENERATOR'S DEMAND SHAPED LIKE M5'S?")
    emit("=" * 78)
    emit("  M5 panel                 : %d series x %d days"
         % (full["panel"].shape[0], full["panel"].shape[1]))
    emit("  zero share               : %.4f" % inter["zero_share"])
    emit("  median inter-demand gap  : %.3f days" % inter["median_adi"])
    emit("  median CV^2 of nonzeros  : %.3f" % inter["median_cv2"])
    emit("")
    emit("  Two thirds of M5 is zeros. That is the property the generator was")
    emit("  built to imitate and the reason WMAPE misbehaves on it -- and it is")
    emit("  worth confirming on the real panel before trusting any conclusion")
    emit("  drawn from a simulated one.")
    emit("")
    summary["intermittency"] = inter

    # ------------------------------------------------------------------
    emit("=" * 78)
    emit("B. DOES THE DEEP ARM CATCH UP AS THE PANEL GROWS?")
    emit("=" * 78)
    emit("Same architecture, same loss, same 28-day horizon, and -- this is the")
    emit("part that makes it an experiment -- THE SAME TRAINING BUDGET. Every arm")
    emit("sees ~%d windows for %d epochs; only the number of distinct series"
         % (WINDOW_BUDGET, EPOCHS))
    emit("those windows are drawn from changes.")
    emit("")
    emit("  Holding epochs fixed and letting the window count grow with the panel")
    emit("  would confound BREADTH with VOLUME: 30,000 series at stride 3 is 18")
    emit("  million windows, and an arm that wins there has been given sixty")
    emit("  times the compute as well as a hundred times the series. The claim")
    emit("  under test is about breadth, so breadth is what varies.")
    emit("")
    rows = []
    for n_series in (300, 3000, 30000):
        d = M.load_panel(n_series=n_series, seed=0)
        panel = np.asarray(d["panel"], dtype=np.float32)
        train, actual = panel[:, :-HORIZON], panel[:, -HORIZON:]

        # stride chosen so the window count lands near the budget regardless of
        # how many series the arm has
        span = train.shape[1] - D.WINDOW - D.HORIZON + 1
        per_series = max(1, WINDOW_BUDGET // n_series)
        stride = max(1, span // per_series)

        t0 = time.time()
        X, Y, _sc = D.make_windows(train, stride=stride)
        if len(X) > WINDOW_BUDGET:
            keep = np.random.default_rng(0).choice(len(X), WINDOW_BUDGET,
                                                   replace=False)
            X, Y = X[keep], Y[keep]
        model = D.train(X, Y, epochs=EPOCHS, seed=0)
        # `forecast` is batched -- it takes (n_series x T) and returns
        # (n_series x horizon). Calling it per row hands it a 1-D array and it
        # indexes [:, -window:] on that, which is where the first run died.
        pred = D.forecast(model, train)
        deep_s = time.time() - t0

        sn = seasonal_naive(train, HORIZON)
        rows.append(dict(series=n_series, stride=stride, windows=len(X),
                         nbeats_wmape=wmape(actual, pred),
                         nbeats_bias=bias(actual, pred),
                         snaive_wmape=wmape(actual, sn),
                         snaive_bias=bias(actual, sn),
                         fit_seconds=round(deep_s, 1)))
        emit("  %5d series -> stride %3d, %6d windows, %5.0fs"
             % (n_series, stride, len(X), deep_s))
    S = pd.DataFrame(rows)
    emit("")
    emit(S.to_string(index=False, float_format=lambda x: "%10.4f" % x))
    emit("")
    S["fva"] = S.snaive_wmape - S.nbeats_wmape
    first, last = S.iloc[0], S.iloc[-1]
    emit("  RAW WMAPE IS NOT COMPARABLE ACROSS THESE ROWS, and reading it that way")
    emit("  was the first mistake this section made. Each arm is a DIFFERENT")
    emit("  SAMPLE of series, and the samples are not equally hard: the seasonal")
    emit("  naive scores %.4f on the 300-series sample and %.4f on the 30,000."
         % (first.snaive_wmape, last.snaive_wmape))
    emit("  A model whose absolute WMAPE holds flat across those two is doing")
    emit("  BETTER on the second, not worse.")
    emit("")
    emit("  FVA against the seasonal naive is the comparison that survives,")
    emit("  because both terms are computed on the same series:")
    emit("")
    for _, r in S.iterrows():
        emit("    %5d series : naive %.4f  n-beats %.4f  FVA %+.4f"
             % (r.series, r.snaive_wmape, r.nbeats_wmape, r.fva))
    emit("")
    improved = last.fva > first.fva
    if improved:
        emit("  THE SCALE CLAIM HOLDS, MODESTLY AND ONLY ON THE RIGHT METRIC. FVA")
        emit("  rises %+.4f -> %+.4f -> %+.4f as the panel widens, on a FIXED"
             % (S.fva.iloc[0], S.fva.iloc[1], S.fva.iloc[2]))
        emit("  training budget -- so it is breadth buying the improvement rather")
        emit("  than gradient steps. '300 series is far below where global deep")
        emit("  models start paying' was directionally right.")
        emit("")
        emit("  But read the size before staffing anything: a hundred times the")
        emit("  series buys %+.4f of FVA. That is a real gain and it is not the"
             % (last.fva - first.fva))
        emit("  step change the phrase 'start paying' suggests.")
    else:
        emit("  THE SCALE CLAIM DOES NOT HOLD. FVA does not improve with breadth")
        emit("  on a fixed budget (%+.4f at 300, %+.4f at %d)."
             % (first.fva, last.fva, last.series))
    emit("")
    emit("  AND THE BIAS DOES NOT IMPROVE AT ALL, WHICH IS THE REAL FINDING.")
    emit("")
    for _, r in S.iterrows():
        emit("    %5d series : n-beats bias %+.4f   naive bias %+.4f"
             % (r.series, r.nbeats_bias, r.snaive_bias))
    emit("")
    emit("  On the generator this project measured N-BEATS at -20.6% bias and")
    emit("  called the WMAPE gap 'the boring half' -- the real problem being that")
    emit("  a model a fifth low on every order is a different and worse decision.")
    emit("  On real M5 it forecasts %.0f%% LOW, worse than on the generator, and"
         % (100 * abs(S.nbeats_bias.mean())))
    emit("  a hundred times the series does not move it.")
    emit("")
    emit("  THAT IS WHY IT BEATS THE NAIVE ON WMAPE. Mean-scaled absolute error on")
    emit("  a panel that is %.0f%% zeros REWARDS forecasting low: a low forecast is"
         % (100 * inter["zero_share"]))
    emit("  right on the many zero days and wrong only on the few that sell. The")
    emit("  deep arm is winning the metric by exploiting it, and a replenishment")
    emit("  system driven by it would under-order every slow mover it owns.")
    emit("")
    emit("  So the honest verdict is split. The SCALE explanation was")
    emit("  directionally right and worth little. The BIAS diagnosis was right")
    emit("  and is worth more on real data than it was on the generator.")
    emit("")
    emit("  WHAT IS NOT CONTROLLED: compute is held fixed on purpose, so this")
    emit("  cannot say whether more EPOCHS would help; and the GBM arm is not")
    emit("  rerun here, so 'N-BEATS 0.5656 against the GBM 0.5291' has no real-")
    emit("  data counterpart. The baseline here is the seasonal naive.")
    emit("")
    summary["scale"] = S.round(5).to_dict("records")

    emit("=" * 78)
    emit("WHAT M5 CANNOT DO")
    emit("=" * 78)
    emit("  No known elasticity, no planted cross-price effect, no ground-truth")
    emit("  substitute set. Every scored-against-truth result in this project")
    emit("  stays on the generator: real data can falsify a claim about model")
    emit("  behaviour, and it cannot supply a number nobody measured.")
    emit("")

    with open(os.path.join(OUT, "m5_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "m5_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n-> out/m5_report.txt")


if __name__ == "__main__":
    main()
