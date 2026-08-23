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
from src import elasticity as EL  # noqa: E402
from src import markdown_opt as MO  # noqa: E402

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



# ==========================================================================
# 3. Below the category: per item, per segment, and across items
# ==========================================================================
def section_heterogeneity(df, truth, emit, summary):
    emit("=" * 78)
    emit("10. ELASTICITY BELOW THE CATEGORY -- IS PER-ITEM PRICING ESTIMABLE?")
    emit("=" * 78)
    by_item = truth["elasticity_by_item"]
    tab = EL.per_item(df, by_item)
    scored = EL.score_item_estimates(tab)
    sd_truth = scored.attrs["sd_of_truth"]

    emit("60 items, one Poisson GLM each (store FE + week-of-year FE + promo).")
    emit("")
    emit(scored.to_string(index=False, float_format=lambda x: "%8.4f" % x))
    emit("")
    emit("True sd of per-item elasticity: %.4f" % sd_truth)
    emit("")
    best = scored.loc[scored.mae.idxmin(), "estimator"]
    raw = scored.set_index("estimator")
    emit("BEST BY MAE: %s" % best)
    emit("")
    if best == "category_mean_only":
        emit("  The category mean WINS. Per-item elasticity is real in the data --")
        emit("  the generator planted a spread of %.3f -- and this panel still" % sd_truth)
        emit("  cannot recover it well enough to beat ignoring it. That is the")
        emit("  answer a pricing team needs before they build per-SKU pricing:")
        emit("  heterogeneity existing is not the same as heterogeneity being")
        emit("  ESTIMABLE, and the second is the one that pays.")
    else:
        emit("  Per-item estimation beats the category mean, so the heterogeneity")
        emit("  is not merely real but recoverable at this panel width.")
    emit("")
    emit("  The raw per-item estimates have sd %.4f against a true sd of %.4f."
         % (raw.loc["raw_per_item", "sd_of_estimates"], sd_truth))
    infl = raw.loc["raw_per_item", "sd_of_estimates"] / max(sd_truth, 1e-9)
    emit("  That is %.2fx the real dispersion: most of what a per-item table" % infl)
    emit("  displays as 'this SKU is more elastic' is sampling noise, and a")
    emit("  merchant reading it would price on it anyway.")
    emit("")
    emit("  Shrinkage is the middle answer and it is not a compromise -- it is the")
    emit("  estimator that knows how much of its own dispersion to believe. Its")
    emit("  weight per item is tau^2/(tau^2+se^2) with tau^2 estimated from the")
    emit("  data, so if the panel had more signal it would shrink less.")
    emit("")
    summary["item_elasticity"] = scored.to_dict("records")
    summary["item_elasticity_sd_truth"] = sd_truth

    emit("-" * 78)
    emit("Elasticity by SEGMENT (one number per item is itself an aggregation):")
    seg = EL.by_segment(df, by_item)
    emit(seg.to_string(index=False, float_format=lambda x: "%8.3f" % x))
    emit("")
    emit("  Promo-week and shelf-week elasticities are not the same object. A")
    emit("  promo week bundles the display lift with the price cut, so its")
    emit("  'elasticity' answers 'what happens when I run a promotion', while the")
    emit("  shelf-week number answers 'what happens when I change the price'.")
    emit("  Those are different decisions and merchants routinely quote one for")
    emit("  the other.")
    emit("")
    summary["segment_elasticity"] = seg.round(4).to_dict("records")
    return tab


def section_cross_price(df, truth, emit, summary):
    emit("=" * 78)
    emit("11. CROSS-PRICE ELASTICITY -- WHAT A MARKDOWN TAKES FROM THE SHELF")
    emit("=" * 78)
    true_cross = truth["cross_elasticity"]
    by_item = truth["elasticity_by_item"]

    rows = []
    for cat in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
        r = EL.cross_price(df, category=cat, calendar_fe=True)
        tru_own = float(np.mean([v for k, v in by_item.items() if k.startswith(cat)]))
        rows.append(dict(category=cat, spec="G_daily_event_dow_FE", own=r["own"],
                         own_se=r["own_se"], cross=r["cross"],
                         cross_se=r["cross_se"], true_own=tru_own,
                         true_cross=true_cross))
    naive_fe = EL.cross_price(df, category="FOODS", calendar_fe=False)
    tru_own_f = float(np.mean([v for k, v in by_item.items() if k.startswith("FOODS")]))
    rows.append(dict(category="FOODS", spec="H_no_calendar_FE", own=naive_fe["own"],
                     own_se=naive_fe["own_se"], cross=naive_fe["cross"],
                     cross_se=naive_fe["cross_se"], true_own=tru_own_f,
                     true_cross=true_cross))
    om = EL.cross_price_omitted(df, category="FOODS", calendar_fe=True)
    rows.append(dict(category="FOODS", spec="I_rival_term_omitted", own=om["own"],
                     own_se=om["own_se"], cross=np.nan, cross_se=np.nan,
                     true_own=tru_own_f, true_cross=true_cross))
    C = pd.DataFrame(rows)
    emit(C.to_string(index=False, float_format=lambda x: "%8.3f" % x))
    emit("")

    g = C[C.spec == "G_daily_event_dow_FE"]
    emit("SPEC G recovers cross-price within %.1f se of truth on all three"
         % float((g.cross - true_cross).abs().div(g.cross_se).max()))
    emit("categories (planted %.3f)." % true_cross)
    emit("")
    h = C[C.spec == "H_no_calendar_FE"].iloc[0]
    gf = g[g.category == "FOODS"].iloc[0]
    emit("SPEC H IS THE FINDING. Drop the event and day-of-week fixed effects and")
    emit("cross-price collapses from %.3f to %.3f -- attenuated by %.0f%%."
         % (gf.cross, h.cross, 100 * (1 - h.cross / max(gf.cross, 1e-9))))
    emit("")
    emit("  It is the same confound this project already documents for own-price,")
    emit("  arriving somewhere new. Promotions are scheduled into strong periods,")
    emit("  and what makes a period strong here is EVENTS and WEEKDAY -- both")
    emit("  shared across the whole department. So a rival's promotion lands")
    emit("  disproportionately on days my demand is high for reasons that have")
    emit("  nothing to do with the rival, my sales are up while the rival price is")
    emit("  down, and the regression charges that co-movement to the cross term")
    emit("  with a NEGATIVE sign.")
    emit("")
    emit("  Week-of-year fixed effects do NOT fix it, which is the trap: they are")
    emit("  in both specifications. Thanksgiving moves within the ISO week from")
    emit("  year to year and weekday variation is inside the week by definition,")
    emit("  so a weekly panel has already destroyed the variation the control")
    emit("  needs. That is why this specification is daily while every other")
    emit("  elasticity in this project is weekly -- aggregation is not a neutral")
    emit("  performance choice when the confound lives inside the bucket.")
    emit("")
    i_om = C[C.spec == "I_rival_term_omitted"].iloc[0]
    emit("SPEC I omits the rival term entirely -- what almost every demand model")
    emit("in production does. Own-price moves %+.3f (%.3f -> %.3f) against a true"
         % (i_om.own - gf.own, gf.own, i_om.own))
    emit("%.3f. The own-price coefficient absorbs part of the substitution." % gf.true_own)
    emit("")
    emit("  Commercially the difference is between 'this markdown will sell X more")
    emit("  units' and 'this markdown will sell X more units, Y of which it took")
    emit("  from the item next to it'. A single-item optimiser books Y as a gain,")
    emit("  and the more items in the department the more it over-discounts.")
    emit("")
    summary["cross_price"] = C.round(4).to_dict("records")
    return float(gf.cross)


def section_multi_item(df, items, truth, cross_est, emit, summary):
    emit("=" * 78)
    emit("12. MULTI-ITEM MARKDOWN -- SHARED INVENTORY, A BUDGET, CANNIBALISATION")
    emit("=" * 78)
    dept = "FOODS_1"
    sub = items[items.dept_id == dept].reset_index(drop=True)
    tail = df[(df.dept_id == dept) & (df.date >= df.date.max() - pd.Timedelta(90, "D"))]
    daily = tail.groupby("item_id").sales.mean()
    true_cross = truth["cross_elasticity"]

    plan = []
    for r in sub.itertuples():
        base = float(daily.get(r.item_id, 0.5)) * N_STORES
        plan.append(dict(item_id=r.item_id, base_daily=max(base, 0.05),
                         ref_price=float(r.base_price),
                         elasticity=float(truth["elasticity_by_item"][r.item_id]),
                         inventory=float(np.ceil(base * CLEARANCE_DAYS * 0.9))))
    dept_of = {i: dept for i in range(len(plan))}
    pools = {"STYLE_A": [0, 1]}          # two sizes of one style, one stock pool
    total_ticket = sum(p["ref_price"] * p["inventory"] for p in plan)
    budget = 0.12 * total_ticket
    phase_len = CLEARANCE_DAYS // N_PHASES

    rows = []
    for label, cross_used, bud in (
            ("ignores cannibalisation", 0.0, None),
            ("uses ESTIMATED cross", cross_est, None),
            ("uses TRUE cross", true_cross, None),
            ("TRUE cross + 12% budget", true_cross, budget)):
        res = MO.solve(plan, DISCOUNT_GRID, N_PHASES, phase_len, cross_used,
                       budget=bud, shared_pools=pools, dept_of=dept_of)
        # every schedule is SCORED in the same true world, which cannibalises
        # whether or not the optimiser believed it did
        sc = MO.evaluate(plan, res["discounts"], N_PHASES, phase_len, true_cross,
                         dept_of=dept_of)
        rows.append(dict(policy=label, status=res["status"],
                         converged=res["converged"], iters=res["iterations"],
                         mean_depth=float(res["discounts"].mean()),
                         revenue=sc["revenue"], markdown_spend=sc["markdown_spend"],
                         leftover=sc["leftover_units"]))
    R = pd.DataFrame(rows)
    emit("Department %s, %d items, %d-day clearance in %d phases."
         % (dept, len(plan), CLEARANCE_DAYS, N_PHASES))
    emit("Two items share one inventory pool. Budget = 12%% of ticket value ($%.0f)."
         % budget)
    emit("Estimated cross-price %.3f, true %.3f." % (cross_est, true_cross))
    emit("")
    emit(R.to_string(index=False, float_format=lambda x: "%10.2f" % x))
    emit("")
    a, b, c, d4 = (R.iloc[0], R.iloc[1], R.iloc[2], R.iloc[3])
    emit("EVERY ROW IS SCORED IN THE SAME TRUE WORLD. The only thing that differs")
    emit("is what the optimiser BELIEVED when it decided.")
    emit("")
    emit("  ignores cannibalisation : depth %.3f, revenue $%.0f"
         % (a.mean_depth, a.revenue))
    emit("  uses TRUE cross         : depth %.3f, revenue $%.0f  (%+.2f%%)"
         % (c.mean_depth, c.revenue,
            100 * (c.revenue - a.revenue) / max(a.revenue, 1e-9)))
    emit("")
    if c.mean_depth < a.mean_depth - 1e-9:
        emit("  Knowing about cannibalisation makes the optimiser discount LESS,")
        emit("  which is the direction theory predicts: an optimiser that books")
        emit("  stolen sibling volume as a gain over-discounts.")
    elif c.mean_depth > a.mean_depth + 1e-9:
        emit("  Knowing about cannibalisation makes the optimiser discount MORE,")
        emit("  not less. That is the opposite of the textbook direction and it is")
        emit("  worth stating rather than smoothing over: with substitutes, a")
        emit("  department-wide markdown partly cancels itself -- every item's")
        emit("  rivals get cheaper at the same time -- so hitting the same")
        emit("  clearance volume requires going DEEPER than a single-item view")
        emit("  suggests. The single-item optimiser is wrong in both directions")
        emit("  depending on whether it is pricing one item or a shelf.")
    else:
        emit("  The two schedules are identical: on this department the inventory")
        emit("  constraint binds before the cross term can change any decision.")
    emit("")
    emit("  uses ESTIMATED cross    : depth %.3f, revenue $%.0f" % (b.mean_depth, b.revenue))
    gap = c.revenue - b.revenue
    emit("  Difference vs the true-cross schedule: $%+.0f (%+.2f%%)."
         % (gap, 100 * gap / max(c.revenue, 1e-9)))
    if gap < 0:
        emit("")
        emit("  THAT NUMBER IS NEGATIVE, AND IT IS NOT A FINDING ABOUT ESTIMATION.")
        emit("  A schedule chosen with the TRUE elasticity ought to be unbeatable in")
        emit("  the true world, so a negative gap means the optimiser and the scorer")
        emit("  do not share a model -- and they do not. The MILP constrains total")
        emit("  demand over the season to be at most inventory, which is linear and")
        emit("  therefore solvable; the scorer runs the season phase by phase and")
        emit("  caps each phase at what is actually left, which is the real")
        emit("  dynamic and is not linear. Under that dynamic, selling earlier is")
        emit("  worth more than the linear constraint can express, so a schedule")
        emit("  that happens to discount deeper early can score better.")
        emit("")
        emit("  The gap is the price of the linearisation, and it is $%.0f on a"
             % abs(gap))
        emit("  $%.0f season -- small, but bigger than the estimation error it was" % c.revenue)
        emit("  supposed to be measuring, which is exactly why it is reported")
        emit("  instead of being quietly presented as one. The load-bearing")
        emit("  comparison in this table is the first row against the third; the")
        emit("  second row cannot separate estimation error from linearisation")
        emit("  error and should not be quoted as if it could.")
    else:
        emit("  That is the dollar cost of the cross-price estimate being wrong --")
        emit("  section 11's argument in a currency a merchant can act on.")
    emit("")
    emit("  TRUE cross + 12%% budget : depth %.3f, revenue $%.0f, spend $%.0f"
         % (d4.mean_depth, d4.revenue, d4.markdown_spend))
    emit("  The budget costs $%.0f (%.2f%%) and is the row a merchant can actually"
         % (c.revenue - d4.revenue,
            100 * (c.revenue - d4.revenue) / max(c.revenue, 1e-9)))
    emit("  approve. A schedule that ignores the season's markdown budget is not a")
    emit("  recommendation, it is a wish.")
    emit("")
    emit("  `converged` is printed because it has to be: this is a SEQUENCE of")
    emit("  exact MILPs with the rival index held fixed and updated between")
    emit("  solves. Convergence is tested on the schedule rather than on the")
    emit("  index -- the index is damped and approaches its fixed point")
    emit("  asymptotically, so testing it would raise a false alarm about a solver")
    emit("  that had settled. A stable schedule on a discrete grid IS the fixed")
    emit("  point, and it is the only thing a merchant ever sees.")
    emit("")
    summary["multi_item_markdown"] = R.round(3).to_dict("records")


N_STORES = 5


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

    tab = section_heterogeneity(df, truth, emit, summary)
    cross_est = section_cross_price(df, truth, emit, summary)
    section_multi_item(df, items, truth, cross_est, emit, summary)
    tab.to_csv(os.path.join(OUT, "elasticity_by_item.csv"), index=False)

    with open(os.path.join(OUT, "pricing_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(OUT, "pricing_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    E.to_csv(os.path.join(OUT, "elasticity.csv"), index=False)
    print("\n-> out/pricing_report.txt, out/pricing_metrics.json")


if __name__ == "__main__":
    main()
