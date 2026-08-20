"""M5-shaped retail demand generator with a known data-generating process.

The real M5 competition data is not downloadable in this offline environment, so
this generates a dataset with M5's *structure* (item -> dept -> category
hierarchy, store -> state, daily unit sales, per-day sell prices, a SNAP
calendar, holiday events) and, critically, a KNOWN ground truth for the
quantities the pricing layer later estimates:

  - true per-category price elasticity   (TRUTH.elasticity_by_category)
  - true promo display lift, separate from the price effect

Having the truth is the point. An elasticity estimated off observational price
variation in this panel is biased, and because truth is known the bias can be
*measured* rather than disclaimed. See docs/ELASTICITY.md.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260818)

START = pd.Timestamp("2022-01-01")
DAYS = 1460  # 4 years

CATEGORIES = {
    # category: (n_depts, true elasticity, base price band)
    "FOODS": (2, -2.1, (1.5, 6.0)),
    "HOUSEHOLD": (2, -1.2, (3.0, 18.0)),
    "HOBBIES": (2, -0.6, (5.0, 40.0)),
}
ITEMS_PER_DEPT = 10
STORES = {
    "CA_1": "CA", "CA_2": "CA", "CA_3": "CA", "TX_1": "TX", "TX_2": "TX",
}
STORE_MULT = {"CA_1": 1.35, "CA_2": 1.0, "CA_3": 0.72, "TX_1": 1.1, "TX_2": 0.85}

# SNAP issuance days differ by state in M5; CA runs 1-10, TX odd days 1-15.
SNAP_DAYS = {"CA": set(range(1, 11)), "TX": {d for d in range(1, 16) if d % 2 == 1}}

PROMO_DISPLAY_LIFT = 0.35  # lift a promo gives BEYOND its own price cut


def _calendar() -> pd.DataFrame:
    dates = pd.date_range(START, periods=DAYS, freq="D")
    cal = pd.DataFrame({"date": dates})
    cal["dow"] = cal.date.dt.dayofweek
    cal["doy"] = cal.date.dt.dayofyear
    cal["year"] = cal.date.dt.year
    cal["month"] = cal.date.dt.month
    cal["day"] = cal.date.dt.day

    ev = pd.Series("none", index=cal.index)
    mult = pd.Series(1.0, index=cal.index)

    def mark(mask, name, m):
        ev[mask & (ev == "none")] = name
        mult[mask] *= m

    # Thanksgiving: 4th Thursday of November. The week BEFORE it is the largest
    # grocery demand event of the US retail year; the day itself is a trough for
    # store traffic. Both are modelled, because a demand model that treats the
    # holiday as one bump gets the sign wrong on the day itself.
    for y in cal.year.unique():
        nov = cal[(cal.year == y) & (cal.month == 11) & (cal.dow == 3)]
        if len(nov) >= 4:
            tg = nov.iloc[3].date
            mark(cal.date.eq(tg), "Thanksgiving", 0.55)
            mark(cal.date.between(tg - pd.Timedelta(6, "D"), tg - pd.Timedelta(1, "D")),
                 "ThanksgivingRunup", 1.9)
            mark(cal.date.eq(tg + pd.Timedelta(1, "D")), "BlackFriday", 1.45)
        mark(cal.date.eq(pd.Timestamp(y, 12, 25)), "Christmas", 0.05)
        mark(cal.date.between(pd.Timestamp(y, 12, 18), pd.Timestamp(y, 12, 24)),
             "ChristmasRunup", 1.8)
        mark(cal.date.eq(pd.Timestamp(y, 12, 31)), "NewYearEve", 1.3)
        mark(cal.date.eq(pd.Timestamp(y, 7, 4)), "July4", 1.35)
        feb = cal[(cal.year == y) & (cal.month == 2) & (cal.dow == 6)]
        if len(feb):
            mark(cal.date.eq(feb.iloc[0].date), "SuperBowl", 1.5)
    cal["event"] = ev
    cal["event_mult"] = mult
    return cal


def _items() -> pd.DataFrame:
    rows = []
    for cat, (n_dept, elas, band) in CATEGORIES.items():
        for d in range(1, n_dept + 1):
            dept = "%s_%d" % (cat, d)
            for i in range(1, ITEMS_PER_DEPT + 1):
                # Base rate spans three orders of magnitude so the panel really
                # contains smooth, intermittent AND lumpy series -- the whole
                # point of the ADI/CV^2 classification downstream.
                tier = RNG.choice(["fast", "mid", "slow", "very_slow"],
                                  p=[0.20, 0.35, 0.30, 0.15])
                base = {"fast": RNG.uniform(8, 30), "mid": RNG.uniform(1.5, 6),
                        "slow": RNG.uniform(0.25, 1.0),
                        "very_slow": RNG.uniform(0.03, 0.18)}[tier]
                rows.append(dict(
                    item_id="%s_%03d" % (dept, i), dept_id=dept, cat_id=cat,
                    tier=tier, base_rate=base, elasticity=elas,
                    base_price=round(float(RNG.uniform(*band)), 2),
                    seasonal_amp=float(RNG.uniform(0.05, 0.45)),
                    seasonal_phase=float(RNG.uniform(0, 2 * np.pi)),
                    snap_sens=float(RNG.uniform(0.25, 0.55) if cat == "FOODS"
                                    else RNG.uniform(0.0, 0.10)),
                ))
    return pd.DataFrame(rows)


def _price_path(n_days, base, demand_index):
    """Sell-price path + promo flags.

    Endogeneity is planted deliberately: promotions are scheduled preferentially
    into weeks the retailer already expects to be strong. That is what real
    merchants do, and it is exactly why an elasticity regressed off observational
    price variation is biased. The generator makes that bias reproducible
    instead of hypothetical.
    """
    price = np.full(n_days, base, dtype=float)
    promo = np.zeros(n_days, dtype=int)

    resets = np.sort(RNG.choice(n_days, size=int(RNG.integers(3, 8)), replace=False))
    cur, prev = base, 0
    for r in resets:
        price[prev:r] = cur
        cur = round(cur * float(RNG.uniform(0.94, 1.09)), 2)
        prev = r
    price[prev:] = cur

    n_weeks = n_days // 7
    wk = np.arange(n_weeks)
    wk_demand = np.array([demand_index[w * 7:(w + 1) * 7].mean() for w in wk])
    z = (wk_demand - wk_demand.mean()) / (wk_demand.std() + 1e-9)
    p_promo = np.clip(0.07 + 0.09 * z, 0.01, 0.55)
    for w in wk[RNG.random(n_weeks) < p_promo]:
        s, e = w * 7, min(w * 7 + 7, n_days)
        depth = float(RNG.choice([0.10, 0.15, 0.20, 0.25, 0.33],
                                 p=[0.25, 0.25, 0.25, 0.15, 0.10]))
        price[s:e] = np.round(price[s:e] * (1 - depth), 2)
        promo[s:e] = 1
    return price, promo


def build():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "data")
    os.makedirs(out, exist_ok=True)

    cal = _calendar()
    items = _items()
    n = len(cal)

    dow_mult = np.array([0.86, 0.80, 0.83, 0.92, 1.15, 1.42, 1.32])  # Mon..Sun
    seasonal_base = cal.event_mult.to_numpy() * dow_mult[cal.dow.to_numpy()]

    frames = []
    for store, state in STORES.items():
        smult = STORE_MULT[store]
        snap = cal.day.isin(SNAP_DAYS[state]).to_numpy().astype(float)
        for it in items.itertuples():
            seas = 1 + it.seasonal_amp * np.sin(
                2 * np.pi * cal.doy.to_numpy() / 365.25 + it.seasonal_phase)
            demand_index = seasonal_base * seas
            price, promo = _price_path(n, it.base_price, demand_index)

            lam = (it.base_rate * smult * demand_index
                   * (1 + it.snap_sens * snap)
                   * (price / it.base_price) ** it.elasticity
                   * (1 + PROMO_DISPLAY_LIFT * promo))
            # gamma mixing -> negative-binomial counts; retail sales are not Poisson
            lam = lam * RNG.gamma(shape=4.0, scale=0.25, size=n)
            sales = RNG.poisson(lam)

            frames.append(pd.DataFrame({
                "date": cal.date.to_numpy(), "store_id": store, "state_id": state,
                "item_id": it.item_id, "dept_id": it.dept_id, "cat_id": it.cat_id,
                "sales": sales.astype(np.int32),
                "sell_price": price.astype(np.float32),
                "promo": promo.astype(np.int8), "snap": snap.astype(np.int8),
            }))

    df = pd.concat(frames, ignore_index=True)
    df = df.merge(cal[["date", "event", "dow"]], on="date", how="left")
    df.to_parquet(os.path.join(out, "sales.parquet"), index=False)
    items.to_csv(os.path.join(out, "items.csv"), index=False)
    cal.to_csv(os.path.join(out, "calendar.csv"), index=False)

    truth = {
        "elasticity_by_category": {c: v[1] for c, v in CATEGORIES.items()},
        "promo_display_lift": PROMO_DISPLAY_LIFT,
        "n_series": int(df.groupby(["item_id", "store_id"]).ngroups),
        "n_rows": int(len(df)),
        "days": DAYS,
        "date_min": str(df.date.min().date()),
        "date_max": str(df.date.max().date()),
        "zero_share": round(float((df.sales == 0).mean()), 4),
    }
    with open(os.path.join(out, "TRUTH.json"), "w") as f:
        json.dump(truth, f, indent=2)
    print(json.dumps(truth, indent=2))


if __name__ == "__main__":
    build()
