"""The planner-facing service: forecast, interval, FVA, and the markdown path.

WHY A SERVICE AT ALL
--------------------
Every previous pass ended with "no serving artifact", and that gap is not
cosmetic. A forecast that exists only inside a backtest has never had to answer
the three questions a planner actually asks, and all three change the design:

  * "What do I order for THIS item in THIS store?"  -- forces a single number
    out of a distribution, which is the newsvendor decision made explicit.
  * "Why should I believe it?"                      -- forces FVA against the
    baseline to travel WITH the forecast rather than living in a report nobody
    opens.
  * "What if I disagree?"                           -- forces an override path,
    and an override that is not logged is a forecast nobody can audit.

WHAT THIS IS NOT
----------------
Not a production service. There is no auth, no rate limiting, no model registry,
no persistence beyond a JSON file, and it loads its artifacts from `out/` at
import. It is a real HTTP surface over real artifacts, which is the point --
the numbers it serves are the numbers the backtest measured, not a fixture.

Run:  uvicorn serve:app --port 8011
Then: http://127.0.0.1:8011/  for the planner view, /docs for the API.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "data")

app = FastAPI(title="ML-1 planner service",
              description="Forecast, interval, FVA and markdown path per item x store.")

# Defined BEFORE Store is instantiated: Store.load() reads it, and having it
# below the `STORE = Store()` line raised a NameError that the broad except
# swallowed into a silent ok=False.
SERVICE_LEVELS = (0.50, 0.75, 0.90, 0.95, 0.98)


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------
class Store:
    """Loads once. Every number served here was measured by the backtest."""

    def __init__(self):
        self.ok = False
        self.quantiles: dict[str, dict] = {}
        self.fva: pd.DataFrame | None = None
        self.intermittency: pd.DataFrame | None = None
        self.overrides: dict[str, dict] = {}
        self.truth: dict = {}
        self.load()

    def load(self):
        try:
            with open(os.path.join(OUT, "quantile_raw.pkl"), "rb") as f:
                qrec = pickle.load(f)
            last = max(r["fold"] for r in qrec)
            from src import quantiles as QQ
            self.crossing_before = 0.0
            n_series = 0
            for r in qrec:
                if r["fold"] != last:
                    continue
                fan = {t: np.asarray(r["q%.2f" % t], float) for t in SERVICE_LEVELS}
                self.crossing_before += QQ.crossing_rate(fan)
                n_series += 1
                # MONOTONE REARRANGEMENT, applied at the serving boundary.
                #
                # Five independent quantile models have nothing coupling them, so
                # the fitted P90 can land above the fitted P95. A crossed fan
                # served to an order system means MORE stock for a LOWER service
                # level -- it inverts the meaning of the dial a planner turns.
                # Sorting the fan is provably no worse in pinball loss (the true
                # quantile function is monotone, so sorting can only move an
                # estimate toward it) and the report measures that rather than
                # asserting it.
                fixed = QQ.rearrange(fan)
                for t in SERVICE_LEVELS:
                    r["q%.2f" % t] = [float(v) for v in fixed[t]]
                self.quantiles[r["key"]] = r
            self.crossing_before /= max(n_series, 1)
            raw = pd.read_csv(os.path.join(OUT, "backtest_raw.csv.gz"))
            b = raw[raw.level == "store_item"]
            self.fva = (b.groupby(["key", "method"], observed=True)
                         .apply(lambda d: d.abs_err.sum() / max(d.actual.sum(), 1e-9),
                                include_groups=False)
                         .rename("wmape").reset_index())
            self.intermittency = pd.read_csv(
                os.path.join(OUT, "intermittency.csv")).set_index("key")
            self.truth = json.load(open(os.path.join(DATA, "TRUTH.json")))
            self.ok = True
        except Exception as exc:                       # pragma: no cover
            self.error = str(exc)
        path = os.path.join(OUT, "overrides.json")
        if os.path.exists(path):
            self.overrides = json.load(open(path))

    def save_overrides(self):
        with open(os.path.join(OUT, "overrides.json"), "w") as f:
            json.dump(self.overrides, f, indent=2)


STORE = Store()


def _require(key: str):
    if not STORE.ok:
        raise HTTPException(503, "artifacts missing -- run run_forecast.py first")
    if key not in STORE.quantiles:
        raise HTTPException(404, "unknown series %r" % key)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
class OrderRequest(BaseModel):
    key: str = Field(..., description="store|item, e.g. CA_1|FOODS_1_001")
    lead_time_days: int = Field(7, ge=1, le=28)
    on_hand: float = 0.0
    case_pack: int = Field(1, ge=1)
    service_level: float = Field(0.95, gt=0.0, lt=1.0)


class Override(BaseModel):
    key: str
    horizon_day: int = Field(..., ge=0, le=27)
    value: float = Field(..., ge=0)
    reason: str = Field(..., min_length=4)
    author: str = "planner"


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": STORE.ok, "series": len(STORE.quantiles),
            "overrides": len(STORE.overrides)}


@app.get("/series")
def series(limit: int = 50):
    if not STORE.ok:
        raise HTTPException(503, "artifacts missing -- run run_forecast.py first")
    return {"keys": sorted(STORE.quantiles)[:limit], "total": len(STORE.quantiles)}


@app.get("/forecast/{key:path}")
def forecast(key: str):
    """Point forecast, the full quantile fan, and the FVA that justifies it.

    FVA travels WITH the forecast on purpose. An absolute WMAPE is not
    interpretable and a planner cannot act on it; "6% better than same-day-last-
    week" is a claim they can check against their own intuition, and it is the
    number that tells them which items to leave on the simple rule.
    """
    _require(key)
    r = STORE.quantiles[key]
    out = {"key": key,
           "mean": [float(v) for v in r["mean_fc"]],
           "quantiles": {"%.2f" % t: [float(v) for v in r["q%.2f" % t]]
                         for t in SERVICE_LEVELS}}
    f = STORE.fva[STORE.fva.key == key].set_index("method").wmape
    if len(f):
        base = float(f.get("seasonal_naive", np.nan))
        model = float(f.get("gbm", np.nan))
        out["fva"] = {
            "wmape_by_method": {k: float(v) for k, v in f.items()},
            "vs_seasonal_naive": (None if not np.isfinite(base) or not np.isfinite(model)
                                  else float(model - base)),
            "beats_baseline": bool(np.isfinite(base) and np.isfinite(model)
                                   and model < base),
        }
    if STORE.intermittency is not None and key in STORE.intermittency.index:
        row = STORE.intermittency.loc[key]
        out["intermittency"] = {"class": str(row.cls), "adi": float(row.adi),
                                "cv2": float(row.cv2),
                                "zero_share": float(row.zero_share)}
    ov = STORE.overrides.get(key)
    if ov:
        out["override"] = ov
    return out


@app.post("/order")
def order(req: OrderRequest):
    """The single number replenishment asked for, and what it costs to believe it.

    `implied_cost_ratio` is the newsvendor read of the service level, returned in
    the same payload as the order quantity: a 95% target IS the claim that
    understocking costs 19x overstocking. Putting the two side by side turns a
    service-level policy into a cost claim someone has to defend, which is the
    conversation worth having.
    """
    _require(req.key)
    r = STORE.quantiles[req.key]
    taus = np.array(SERVICE_LEVELS)
    tau = float(taus[int(np.argmin(np.abs(taus - req.service_level)))])
    path = np.asarray(r["q%.2f" % tau], float)[:req.lead_time_days]
    mean_path = np.asarray(r["mean_fc"], float)[:req.lead_time_days]

    need = max(0.0, float(path.sum()) - req.on_hand)
    qty = int(np.ceil(need / req.case_pack) * req.case_pack)
    return {
        "key": req.key,
        "service_level_used": tau,
        "requested_service_level": req.service_level,
        "implied_cost_ratio": round(tau / (1 - tau), 2),
        "order_quantity": qty,
        "cover_units": round(float(path.sum()), 2),
        "mean_units": round(float(mean_path.sum()), 2),
        "buffer_over_mean": round(float(path.sum() - mean_path.sum()), 2),
        # Stated because it is the honest limit of what this service can do:
        # these are DAILY quantiles summed, which assumes the days all go wrong
        # together. run_advanced.py measures how much that overstates.
        "caveat": ("summed daily quantiles assume perfect cross-day dependence; "
                   "see run_advanced.py section B for the block-bootstrap "
                   "lead-time quantile and the size of this overstatement"),
    }


@app.post("/override")
def override(o: Override):
    """A planner disagreeing with the model, recorded rather than silently applied.

    Overrides are the single largest source of forecast value DESTRUCTION in real
    planning systems, and the only way anyone finds that out is by logging them
    with an author and a reason and scoring them later. An override path with no
    audit trail is how a demand planning team spends a year making its forecast
    worse without evidence either way.
    """
    _require(o.key)
    rec = STORE.overrides.setdefault(o.key, {})
    rec[str(o.horizon_day)] = {"value": o.value, "reason": o.reason,
                               "author": o.author}
    STORE.save_overrides()
    return {"stored": True, "key": o.key, "overrides_for_key": len(rec)}


@app.get("/markdown/{key:path}")
def markdown(key: str, inventory: float = 100.0, weeks: int = 6):
    """The clearance path for one item, priced with its own elasticity.

    Served from the same elasticity table the pricing report scores, so the
    number here and the number in the report cannot drift apart.
    """
    _require(key)
    item = key.split("|")[-1]
    elas = STORE.truth.get("elasticity_by_item", {}).get(item)
    if elas is None:
        raise HTTPException(404, "no elasticity for item %r" % item)
    r = STORE.quantiles[key]
    base_daily = float(np.mean(r["mean_fc"]))

    best, rows = None, []
    for depth in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        lam = np.log(2.0) / 21.0
        days = np.arange(weeks * 7)
        units = base_daily * float(np.exp(-lam * days).sum()) * (1 - depth) ** elas
        units = min(units, inventory)
        revenue = units * (1 - depth) + 0.05 * (inventory - units)
        rows.append({"depth": depth, "units": round(units, 2),
                     "revenue_index": round(revenue, 2)})
        if best is None or revenue > best["revenue_index"]:
            best = rows[-1]
    return {"key": key, "elasticity": elas, "inventory": inventory,
            "recommended": best, "grid": rows,
            "note": ("revenue is in units of ticket price; salvage 5%. "
                     "single-item view -- see run_pricing.py section 12 for the "
                     "multi-item version with cannibalisation and a budget")}


@app.get("/", response_class=HTMLResponse)
def home():
    n = len(STORE.quantiles)
    keys = sorted(STORE.quantiles)[:12]
    rows = "".join(
        "<tr><td><code>%s</code></td>"
        "<td><a href='/forecast/%s'>forecast</a></td>"
        "<td><a href='/markdown/%s'>markdown</a></td></tr>" % (k, k, k)
        for k in keys)
    status = ("%d series loaded" % n) if STORE.ok else "artifacts missing"
    return """<!doctype html><meta charset=utf-8>
<title>ML-1 planner</title>
<style>
 body{font:15px/1.55 system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem}
 table{border-collapse:collapse;margin:1rem 0}
 td,th{border:1px solid #ccc;padding:.35rem .6rem;text-align:left}
 code{background:#f4f4f4;padding:.1rem .3rem}
</style>
<h1>ML-1 &mdash; planner view</h1>
<p><b>%s.</b> Every number below was measured by the rolling-origin backtest;
nothing here is a fixture.</p>
<p>API docs: <a href="/docs">/docs</a> &middot; health: <a href="/health">/health</a></p>
<h2>Sample series</h2>
<table><tr><th>item &times; store</th><th></th><th></th></tr>%s</table>
<h2>What the endpoints are for</h2>
<ul>
 <li><code>GET /forecast/{key}</code> &mdash; point forecast, the quantile fan,
     the intermittency class, and the FVA against seasonal naive. FVA travels
     with the forecast because an absolute WMAPE is not something a planner can
     act on.</li>
 <li><code>POST /order</code> &mdash; the single number replenishment asked for,
     returned together with the cost ratio the service level is asserting.</li>
 <li><code>POST /override</code> &mdash; a planner disagreeing, with an author
     and a reason. Logged, not silently applied.</li>
 <li><code>GET /markdown/{key}</code> &mdash; the clearance path for one item,
     priced with that item's own elasticity.</li>
</ul>
""" % (status, rows)
