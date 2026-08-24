"""The real M5 panel — the dataset this project said it could not have.

WHAT THIS PROJECT SAID, IN EVERY PASS
-------------------------------------
"No real dataset. M5 is not downloadable here, and the generator is the point
anyway."

Half true, and the half that was false took five passes to check. M5 IS
downloadable with a Kaggle account. The competition's own download endpoint
returns 403 until the rules are accepted in a browser, but the daily sales panel
has been republished as a plain Kaggle DATASET, and datasets carry no rules gate.

PROVENANCE, STATED BECAUSE IT MATTERS
-------------------------------------
This is `tbierhance/m5-forecasting-parquet-and-aggregations`, a third-party
re-encoding of the competition files, not the official archive. 59,181,090 rows =
30,490 series x 1,941 days, which is M5's exact shape, and the calendar and
sell_prices come with it. It has not been diffed against the official CSVs
because those are the thing that is gated. **If the numbers below ever matter to
somebody, the official archive is one rules-acceptance away and should be used.**

THE EXPERIMENT THIS IS FOR
--------------------------
This project's deep arm lost -- N-BEATS 0.5656 WMAPE against the GBM's 0.5291 --
and it explained the loss by SCALE:

    "It is not evidence that deep forecasting loses. 300 series is far below
     where global deep models start paying, and THAT is the finding for anyone
     deciding whether to staff the work."

That is a falsifiable claim about a regime, and M5 has 30,490 series. The sweep
in `run_m5.py` runs the same two models at 300, 3,000 and 30,000 series on real
data. If the deep arm catches up as the panel grows, the explanation was right.
If it does not, "we needed more series" was a story.

WHAT THE GENERATOR STILL DOES THAT M5 CANNOT
--------------------------------------------
M5 has no known elasticity, no planted cross-price effect and no ground-truth
substitute set, so every scored-against-truth result in this project stays on the
generator. Real data can falsify a claim about model behaviour; it cannot supply
a number nobody measured.
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(HERE, os.pardir, ".vendor", "kaggle")
SALES = os.path.join(VENDOR, "daily_sales_items_all.parquet")
CALENDAR = os.path.join(VENDOR, "calendar.csv")
PRICES = os.path.join(VENDOR, "sell_prices.parquet")


def available() -> bool:
    return os.path.exists(SALES) and os.path.exists(CALENDAR)


def _full_panel_cached() -> dict:
    """Build the whole 30,490 x 1,941 panel once and cache it as .npy.

    The long format is 59 million rows and a pass over it costs ~6 minutes. The
    sweep below wants 300, 3,000 and 30,000 series, and re-reading the parquet
    for each would spend half an hour re-deriving the same matrix. Cached, the
    whole panel is 236 MB of float32 and every subsample is a slice.
    """
    # UNCOMPRESSED .npy plus a small json sidecar. savez_compressed on a 236 MB
    # float32 array spends minutes in zlib for a file that is not scarce here,
    # and the first attempt at this was killed mid-write with nothing cached --
    # so the expensive pass had to be repeated for no benefit.
    cache = os.path.join(VENDOR, "m5_panel.npy")
    meta = os.path.join(VENDOR, "m5_panel_meta.json")
    if os.path.exists(cache) and os.path.exists(meta):
        import json
        with open(meta) as f:
            m = json.load(f)
        return dict(panel=np.load(cache, mmap_mode="r"), keys=m["keys"],
                    days=m["days"])

    import pyarrow as pa
    import pyarrow.parquet as pq

    # `id` is DICTIONARY-encoded and `d` is an int16 day number, not the "d_123"
    # string the competition CSVs use. Both were assumed wrong first time round:
    # np.asarray on a dictionary column yields INDICES, not labels, which would
    # have silently keyed the panel by dictionary position.
    pf = pq.ParquetFile(SALES)
    ids_all, d_all, v_all = [], [], []
    for batch in pf.iter_batches(batch_size=2_000_000,
                                 columns=["id", "d", "value"]):
        t = pa.Table.from_batches([batch])
        ids_all.append(np.asarray(t["id"].cast(pa.string()).to_pylist(),
                                  dtype=object))
        d_all.append(np.asarray(t["d"], dtype=np.int32))
        v_all.append(np.asarray(t["value"], dtype=np.float32))
    sid = np.concatenate(ids_all)
    sd = np.concatenate(d_all)
    sv = np.concatenate(v_all)

    keys = np.array(sorted(set(sid.tolist())), dtype=object)
    days = np.sort(np.unique(sd))
    kpos = {k: i for i, k in enumerate(keys)}
    dpos = {int(d): i for i, d in enumerate(days)}
    panel = np.zeros((len(keys), len(days)), dtype=np.float32)
    panel[[kpos[k] for k in sid], [dpos[int(d)] for d in sd]] = sv
    import json
    tmp = cache + ".part"
    np.save(tmp, panel)
    os.replace(tmp + ".npy" if os.path.exists(tmp + ".npy") else tmp, cache)
    with open(meta, "w") as f:
        json.dump(dict(keys=[str(k) for k in keys],
                       days=[str(d) for d in days]), f)
    return dict(panel=panel, keys=[str(k) for k in keys],
                days=[str(d) for d in days])


def load_panel(n_series: int | None = 300, seed: int = 0,
               min_days: int = 1000) -> dict:
    """A (n_series x T) matrix of daily unit sales, plus the series keys.

    Sampled by SERIES, never by row. M5's long format has one row per
    series-day; taking the first N rows would give a handful of series with
    complete histories rather than N series, and every intermittency statistic
    computed on it would describe the wrong thing.
    """
    full = _full_panel_cached()
    panel, keys = full["panel"], full["keys"]
    if min_days and panel.shape[1] < min_days:
        raise RuntimeError("panel has only %d days" % panel.shape[1])
    if n_series is None or n_series >= panel.shape[0]:
        return dict(panel=panel, keys=keys, days=full["days"])
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(panel.shape[0], size=n_series, replace=False))
    return dict(panel=panel[idx], keys=[keys[i] for i in idx],
                days=full["days"])


def intermittency(panel: np.ndarray) -> dict:
    """Zero share and the Syntetos-Boylan inputs, on real demand.

    The generator's intermittency is a parameter. M5's is a fact, and the two
    being close is the only reason any generator-based conclusion has a claim on
    reality.
    """
    zero = float((panel == 0).mean())
    nz = [np.flatnonzero(row) for row in panel]
    adi, cv2 = [], []
    for row, idx in zip(panel, nz):
        if len(idx) < 2:
            continue
        gaps = np.diff(idx)
        adi.append(float(gaps.mean()))
        v = row[idx]
        cv2.append(float((v.std() / v.mean()) ** 2) if v.mean() > 0 else 0.0)
    return dict(zero_share=zero,
                median_adi=float(np.median(adi)) if adi else float("nan"),
                median_cv2=float(np.median(cv2)) if cv2 else float("nan"),
                series=int(panel.shape[0]), days=int(panel.shape[1]))
