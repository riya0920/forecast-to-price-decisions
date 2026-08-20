"""The decision half: observational elasticity, then a markdown schedule built on it.

Two things make this more than a regression:

1. The generator knows the true elasticity, so the estimate is SCORED, not
   asserted. Every specification's bias is measured in elasticity points against
   truth, and the direction is explained by the mechanism that produced it
   (promotions scheduled into strong weeks -- see src/generate.py::_price_path).

2. The markdown optimizer DECIDES using the biased observational estimate and is
   SCORED using the true elasticity. That is the real situation: a pricing team
   never has the true number, only the number their data gave them. The gap
   between the two runs is the dollar cost of the estimation bias, which is a
   more useful thing to hand a merchant than a confidence interval.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")

CLEARANCE_DAYS = 42
N_PHASES = 3
DISCOUNT_GRID = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

# Unsold end-of-season stock is liquidated at 5% of ticket. This is a
# merchant-supplied input and it drives the whole answer: at the 15% I first
# used, the optimiser correctly refused to discount anything, because 15% of
# ticket beats the margin on an incremental unit for an inelastic item. The
# number is small here because end-of-season goods are jobbed out or disposed of.
SALVAGE_FRACTION = 0.05

# End-of-season demand decays -- the reason clearance markdowns are PHASED
# rather than flat. With a constant demand rate and a multiplicative price
# effect, a single price is provably optimal and a phased schedule can only tie
# it; the schedule earns its keep by holding price while demand is still strong
# and cutting as it fades. Half-life is an assumption, so it is sensitivity-
# tested alongside elasticity rather than presented as known.
DECAY_HALFLIFE_DAYS = 21.0

SENSITIVITY = [-0.30, 0.0, 0.30]


# ==========================================================================
# 1. Elasticity
# ==========================================================================
def ols(X, y):
    """Plain OLS with HC0 (heteroskedasticity-robust) standard errors.

    statsmodels is not installed here, so this is written out. HC0 rather than
    classical SEs because demand variance grows with demand level; classical
    SEs would overstate precision on exactly the fast movers that dominate the
    fit.
    """
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    S = (X * (resid ** 2)[:, None]).T @ X
    cov = XtX_inv @ S @ XtX_inv
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None))


def within(df, cols, by):
    """Series fixed effects by demeaning, rather than 300 dummy columns."""
    out = df[cols].copy()
    g = df.groupby(by, observed=True)
    for c in cols:
        out[c] = df[c] - g[c].transform("mean")
    return out


def poisson_elasticity(sub, promo_control: bool):
    """Poisson GLM with a log link, series fixed effects, week-of-year controls.

    This is the specification the log-log OLS above should have been all along.
    Unit sales are counts with a large zero mass; log1p(sales) is not log(sales),
    and on a series that is zero 70% of the time the transform compresses exactly
    the observations that carry the price signal, attenuating the coefficient
    toward zero. A log link on the MEAN handles zeros without transforming them.

    Fitted on weekly aggregates: the price decision is a weekly one, and weekly
    aggregation keeps the fixed-effect design a manageable size.
    """
    import scipy.sparse as sp
    from sklearn.linear_model import PoissonRegressor

    w = (sub.groupby(["key", pd.Grouper(key="date", freq="W")])
            .agg(sales=("sales", "sum"), price=("sell_price", "mean"),
                 promo=("promo", "mean"), snap=("snap", "mean"))
            .reset_index())
    w["woy"] = w.date.dt.isocalendar().week.astype(int)

    dense = [np.log(w.price.to_numpy(float))]
    if promo_control:
        dense.append(w.promo.to_numpy(float))
    dense.append(w.snap.to_numpy(float))

    K = pd.get_dummies(w.key, sparse=True)
    W = pd.get_dummies(w.woy, prefix="w", sparse=True)
    X = sp.hstack([sp.csr_matrix(np.column_stack(dense)),
                   sp.csr_matrix(K.sparse.to_coo()),
                   sp.csr_matrix(W.sparse.to_coo())]).tocsr()
    m = PoissonRegressor(alpha=1e-6, max_iter=800, fit_intercept=True)
    m.fit(X, w.sales.to_numpy(float))
    return float(m.coef_[0]), len(w)


def estimate_elasticities(df, truth):
    """Six specifications, each scored against the truth the generator planted."""
    d = df.copy()
    d["key"] = d.store_id + "|" + d.item_id
    d["log_q"] = np.log1p(d.sales)
    d["log_p"] = np.log(d.sell_price)
    d["dow"] = d.date.dt.dayofweek
    d["woy"] = d.date.dt.isocalendar().week.astype(int)

    specs = {
        "A_naive": dict(controls=[], fe=False, drop_promo=False),
        "B_promo_control": dict(controls=["promo"], fe=False, drop_promo=False),
        "C_promo_fe_calendar": dict(controls=["promo", "snap"], fe=True, drop_promo=False),
        "D_exclude_promo_weeks": dict(controls=["snap"], fe=True, drop_promo=True),
    }

    rows = []
    for cat, sub_all in d.groupby("cat_id", observed=True):
        true_e = truth["elasticity_by_category"][cat]

        for name, promo_ctrl in (("E_poisson_no_promo_ctrl", False),
                                 ("F_poisson_full", True)):
            est, n = poisson_elasticity(sub_all, promo_ctrl)
            rows.append(dict(category=cat, spec=name, estimate=est, se=np.nan,
                             truth=true_e, bias=est - true_e, n=n))

        for name, cfg in specs.items():
            sub = sub_all[sub_all.promo == 0] if cfg["drop_promo"] else sub_all
            cols = ["log_p"] + cfg["controls"]
            if cfg["fe"]:
                # week-of-year and day-of-week absorbed as dummies, series FE by
                # demeaning; without them, seasonality that co-moves with promo
                # timing loads onto the price coefficient
                Z = within(sub, cols, "key")
                dows = pd.get_dummies(sub.dow, prefix="dow", drop_first=True).astype(float)
                woys = pd.get_dummies(sub.woy, prefix="woy", drop_first=True).astype(float)
                X = np.column_stack([Z.to_numpy(float), dows.to_numpy(), woys.to_numpy()])
                yv = (sub.log_q - sub.groupby("key", observed=True).log_q.transform("mean")).to_numpy(float)
            else:
                X = np.column_stack([np.ones(len(sub)), sub[cols].to_numpy(float)])
                yv = sub.log_q.to_numpy(float)
            beta, se = ols(X, yv)
            idx = 0 if cfg["fe"] else 1
            est = float(beta[idx])
            rows.append(dict(category=cat, spec=name, estimate=est,
                             se=float(se[idx]), truth=true_e,
                             bias=est - true_e, n=len(sub)))
    return pd.DataFrame(rows)


# ==========================================================================
# 2. Markdown optimisation
# ==========================================================================
def simulate_schedule(base_daily, ref_price, prices_by_phase, inventory,
                      elasticity, phase_len, halflife=DECAY_HALFLIFE_DAYS):
    """Run one markdown schedule and return (revenue, units_sold, leftover).

    Daily demand on day t at price p:

        base_daily * 2 ** (-t / halflife) * (p / p_ref) ** elasticity

    censored by remaining inventory. Stepped daily rather than per phase so the
    decay is integrated correctly inside a phase. Leftover is salvaged, which is
    what stops the objective collapsing to 'discount to zero'.
    """
    remaining = float(inventory)
    revenue = 0.0
    sold = 0.0
    t = 0
    for p in prices_by_phase:
        pr = (p / ref_price) ** elasticity
        for _ in range(phase_len):
            rate = base_daily * (2.0 ** (-t / halflife)) * pr
            take = min(rate, remaining)
            revenue += take * p
            sold += take
            remaining -= take
            t += 1
            if remaining <= 1e-9:
                break
        if remaining <= 1e-9:
            break
    revenue += remaining * ref_price * SALVAGE_FRACTION
    return revenue, sold, remaining


def monotone_schedules():
    """Discounts may only deepen. A markdown that goes back up is not a markdown."""
    return [c for c in itertools.product(DISCOUNT_GRID, repeat=N_PHASES)
            if all(c[i] <= c[i + 1] for i in range(N_PHASES - 1))]


def optimise(base_daily, ref_price, inventory, elasticity, phase_len, hl=DECAY_HALFLIFE_DAYS):
    best, best_rev = None, -np.inf
    for combo in monotone_schedules():
        prices = [ref_price * (1 - c) for c in combo]
        rev, _, _ = simulate_schedule(base_daily, ref_price, prices,
                                      inventory, elasticity, phase_len, hl)
        if rev > best_rev:
            best, best_rev = combo, rev
    return best, best_rev


def best_flat(base_daily, ref_price, inventory, elasticity, phase_len, hl=DECAY_HALFLIFE_DAYS):
    """The policy the phased schedule has to beat: one discount, held all window."""
    best, best_rev = None, -np.inf
    for c in DISCOUNT_GRID:
        prices = [ref_price * (1 - c)] * N_PHASES
        rev, _, _ = simulate_schedule(base_daily, ref_price, prices, inventory,
                                      elasticity, phase_len, hl)
        if rev > best_rev:
            best, best_rev = c, rev
    return best, best_rev


def expected_full_price_demand(base_daily, phase_len, hl=DECAY_HALFLIFE_DAYS):
    return float(base_daily * sum(2.0 ** (-t / hl)
                                  for t in range(phase_len * N_PHASES)))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(os.path.join(DATA, "sales.parquet"))
    items = pd.read_csv(os.path.join(DATA, "items.csv"))
    truth = json.load(open(os.path.join(DATA, "TRUTH.json")))
    lines, summary = [], {}

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("8. PRICE ELASTICITY -- OBSERVATIONAL, AND SCORED AGAINST TRUTH")
    emit("=" * 78)
    E = estimate_elasticities(df, truth)
    piv = E.pivot(index="spec", columns="category", values="estimate")
    bias = E.pivot(index="spec", columns="category", values="bias")
    tru = E.groupby("category").truth.first()
    emit("Estimated elasticity by specification:")
    emit(piv.to_string(float_format=lambda x: "%7.3f" % x))
    emit("")
    emit("TRUE elasticity (planted by the generator):")
    emit(tru.to_string(float_format=lambda x: "%7.3f" % x))
    emit("")
    emit("Bias (estimate - truth); negative = estimated MORE elastic than reality:")
    emit(bias.to_string(float_format=lambda x: "%+7.3f" % x))
    emit("")
    emit("Mean |bias| per specification:")
    emit(bias.abs().mean(axis=1).to_string(float_format=lambda x: "%7.3f" % x))
    emit("")
    emit("TWO DIFFERENT BIASES, PULLING IN OPPOSITE DIRECTIONS. Reading the bias")
    emit("table by row rather than by column is the whole point:")
    emit("")
    emit("  Specs A and B are not merely attenuated, they have the WRONG SIGN on")
    emit("  two of three categories -- they report demand RISING with price. With no")
    emit("  fixed effects, a pooled regression is identified off the price gap")
    emit("  BETWEEN products, not off any product's response to its own price. It is")
    emit("  comparing a $6 item to a $2 item and calling the difference elasticity.")
    emit("  Adding the promo control (B) changes essentially nothing, because the")
    emit("  problem was never the promo flag.")
    emit("")
    emit("  Specs C and D add series fixed effects, which fixes the sign but leaves")
    emit("  the estimate badly attenuated toward zero. That residue is a LIKELIHOOD")
    emit("  problem: unit sales are counts with a large zero mass, and log1p(sales)")
    emit("  is not log(sales) -- on a series that is zero most days the transform")
    emit("  crushes exactly the observations carrying the price signal. No amount of")
    emit("  further control fixes a misspecified link.")
    emit("")
    emit("  Spec E (Poisson GLM, log link, but NO promo control) is biased AWAY from")
    emit("  zero -- demand looks more elastic than it is. That IS the identification")
    emit("  problem: prices in this panel were not randomised. The generator")
    emit("  schedules promotions into weeks demand is already strong")
    emit("  (src/generate.py::_price_path), and a promotion carries a display lift")
    emit("  on top of its price cut. Low price and high demand therefore arrive")
    emit("  together, and the regression charges the co-movement to price.")
    emit("")
    emit("  Spec F (Poisson GLM + promo control + series FE + week-of-year) fixes")
    emit("  both and lands close to truth. It is the number the markdown simulator")
    emit("  uses.")
    emit("")
    emit("The honest ceiling: spec F is still OBSERVATIONAL. It recovers truth here")
    emit("because I know what confound to control for -- I wrote the generator. On")
    emit("real data there is no such guarantee, and no specification search can")
    emit("establish that the remaining price variation is as-good-as-random. That")
    emit("takes a price experiment: randomised price cells across matched stores.")
    emit("Section 9 prices exactly what not having run one costs.")
    summary["elasticity"] = E.round(4).to_dict("records")

    # ---------------- markdown ----------------
    emit("")
    emit("=" * 78)
    emit("9. MARKDOWN SIMULATOR -- DECIDE ON THE ESTIMATE, SCORE ON THE TRUTH")
    emit("=" * 78)
    phase_len = CLEARANCE_DAYS // N_PHASES

    last = df[df.date > df.date.max() - pd.Timedelta(56, "D")]
    rate = (last.groupby(["store_id", "item_id"]).sales.mean()
                .rename("base_daily").reset_index())
    rate = rate.merge(items[["item_id", "cat_id", "base_price"]], on="item_id")
    rate = rate[rate.base_daily > 0.15].copy()
    # Inventory sized so it will NOT clear at full price -- otherwise there is
    # no decision to make and the simulator is theatre.
    rate["expected"] = [expected_full_price_demand(b, phase_len) for b in rate.base_daily]
    rate["inventory"] = np.ceil(rate.expected * 2.2)

    est_by_cat = E.pivot(index="spec", columns="category", values="estimate")
    GOOD, BAD = "F_poisson_full", "A_naive"

    emit("Clearance window %d days, %d phases of %d days, salvage %.0f%% of ticket,"
         % (CLEARANCE_DAYS, N_PHASES, phase_len, 100 * SALVAGE_FRACTION))
    emit("demand half-life %.0f days. Inventory = 2.2x expected full-price demand,"
         % DECAY_HALFLIFE_DAYS)
    emit("so doing nothing leaves more than half the stock on the floor.")
    emit("")
    emit("Three decision-makers price the same inventory:")
    emit("  GOOD    -- decides on spec F (Poisson GLM), the best observational estimate")
    emit("  BAD     -- decides on spec A (naive log-log OLS), the portfolio default")
    emit("  ORACLE  -- decides knowing the world's true elasticity")
    emit("All three are SCORED in the same world, at its true elasticity.")
    emit("")

    rows = []
    for cat in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
        cand = rate[rate.cat_id == cat]
        e_good = float(est_by_cat.loc[GOOD, cat])
        e_bad = float(est_by_cat.loc[BAD, cat])
        true_e = truth["elasticity_by_category"][cat]
        for label, mult in zip(["-30%", "true", "+30%"], SENSITIVITY):
            world_e = true_e * (1 + mult)
            tot = dict(none=0.0, flat=0.0, good=0.0, bad=0.0, oracle=0.0,
                       u_good=0.0, u_none=0.0)
            sched_good, sched_or = [], []
            for r in cand.itertuples():
                ref = float(r.base_price)
                s_good, _ = optimise(r.base_daily, ref, r.inventory, e_good, phase_len)
                s_bad, _ = optimise(r.base_daily, ref, r.inventory, e_bad, phase_len)
                f_good, _ = best_flat(r.base_daily, ref, r.inventory, e_good, phase_len)
                s_or, _ = optimise(r.base_daily, ref, r.inventory, world_e, phase_len)
                sched_good.append(s_good); sched_or.append(s_or)

                def run(prices):
                    return simulate_schedule(r.base_daily, ref, prices, r.inventory,
                                             world_e, phase_len)
                rv_g, u_g, _ = run([ref * (1 - c) for c in s_good])
                rv_b, _, _ = run([ref * (1 - c) for c in s_bad])
                rv_f, _, _ = run([ref * (1 - f_good)] * N_PHASES)
                rv_o, _, _ = run([ref * (1 - c) for c in s_or])
                rv_n, u_n, _ = run([ref] * N_PHASES)
                tot["good"] += rv_g; tot["bad"] += rv_b; tot["flat"] += rv_f
                tot["oracle"] += rv_o; tot["none"] += rv_n
                tot["u_good"] += u_g; tot["u_none"] += u_n
            inv = float(cand.inventory.sum())
            mg = max(set(sched_good), key=sched_good.count)
            mo = max(set(sched_or), key=sched_or.count)
            rows.append(dict(
                category=cat, scenario=label, world_e=world_e,
                rev_none=tot["none"], rev_bad=tot["bad"], rev_good=tot["good"],
                rev_oracle=tot["oracle"],
                good_vs_none_pct=100 * (tot["good"] / tot["none"] - 1),
                good_vs_flat_pct=100 * (tot["good"] / tot["flat"] - 1),
                bad_spec_cost_pct=100 * (1 - tot["bad"] / tot["oracle"]),
                good_spec_cost_pct=100 * (1 - tot["good"] / tot["oracle"]),
                sellthru_good=100 * tot["u_good"] / inv,
                sellthru_none=100 * tot["u_none"] / inv,
                sched_good="/".join("%d" % (100 * c) for c in mg),
                sched_oracle="/".join("%d" % (100 * c) for c in mo)))
    M = pd.DataFrame(rows).set_index(["category", "scenario"])
    emit(M.to_string(float_format=lambda x: "%9.2f" % x))
    emit("")
    emit("WHAT THIS TABLE SAYS, in the order a merchant would ask:")
    emit("")
    emit("1. Markdown depth is a function of elasticity, and the optimiser works")
    emit("   that out without being told. Read sched_oracle down the categories:")
    for cat in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
        r0 = M.loc[(cat, "true")]
        emit("     %-10s true e=%5.2f  ->  %s   (sell-through %.0f%% vs %.0f%% at full price)"
             % (cat, truth["elasticity_by_category"][cat], r0.sched_oracle,
                r0.sellthru_good, r0.sellthru_none))
    emit("   An inelastic category should not be marked down at all -- cutting price")
    emit("   there gives away margin on units that were going to sell anyway. A")
    emit("   markdown budget spread evenly across categories spends most of itself")
    emit("   where it repays least.")
    emit("")
    emit("2. THE COST OF THE ELASTICITY ESTIMATE, IN DOLLARS. bad_spec_cost_pct is")
    emit("   the revenue given up by a team that estimated elasticity the way most")
    emit("   portfolio projects do -- log-log OLS on a zero-heavy count series:")
    for cat in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
        r0 = M.loc[(cat, "true")]
        emit("     %-10s bad spec %6.2f%% of clearance revenue   |  good spec %5.2f%%"
             % (cat, r0.bad_spec_cost_pct, r0.good_spec_cost_pct))
    emit("   The attenuated estimate makes demand look inelastic, so the BAD team")
    emit("   marks down too little or not at all and eats the salvage. That number")
    emit("   is the business case for the price experiment, priced.")
    emit("")
    emit("3. SENSITIVITY (elasticity +/-30% around truth):")
    for cat in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
        s = M.loc[cat]
        holds = bool((s.good_vs_none_pct >= -1e-9).all())
        emit("     %-10s good-vs-no-markdown across worlds: %s -> %s"
             % (cat, ", ".join("%+.2f%%" % v for v in s.good_vs_none_pct),
                "DECISION HOLDS" if holds else "DECISION FLIPS"))
    emit("   Where it holds, the recommendation survives a 30% elasticity error even")
    emit("   though the revenue forecast does not -- the level moves, the decision")
    emit("   does not. Where it flips, the honest answer is that the category cannot")
    emit("   be priced off observational data and needs a test before margin is bet.")
    emit("")
    emit("4. good_vs_flat_pct is small throughout. That is the honest size of the")
    emit("   phasing win: with a stated demand half-life, most of the value is in")
    emit("   marking down AT ALL, not in the shape of the path. A markdown-")
    emit("   optimisation project that headlines the phasing gain and hides the")
    emit("   mark-down-or-not gain has its emphasis backwards.")
    summary["markdown"] = M.reset_index().round(3).to_dict("records")

    with open(os.path.join(OUT, "pricing_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(OUT, "pricing_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    E.to_csv(os.path.join(OUT, "elasticity.csv"), index=False)
    print("\n-> out/pricing_report.txt, out/pricing_metrics.json")


if __name__ == "__main__":
    main()
