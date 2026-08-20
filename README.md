# ML-1 — Forecast-to-Price Decision System

**Roughly 50% of the spec.** The machinery the hiring-manager doc calls the
differentiator, plus the two biggest gaps the first pass named: **prediction
intervals and the replenishment handoff** (the spec's own question, previously
answered only in prose), and the **reconciliation survey** that had been one
method rather than a comparison.

Every number below was produced by running the code in this directory. Nothing
is quoted from a paper or a leaderboard.

```bash
python src/generate.py     # ~4s     build the panel
python run_forecast.py     # ~20min  6-fold backtest: point + 5 quantiles + 3 MinT variants
python report.py           # ~25s    the FVA tables
python run_pricing.py      # ~3min   elasticity + markdown
python -m pytest tests -q  # 26 tests
```

## The data

M5 is not downloadable in this offline environment, so `src/generate.py` builds a
panel with M5's *structure* and a **known data-generating process**:

| | |
|---|---|
| 60 items × 5 stores × 1,460 days | 438,000 rows, 300 bottom series |
| hierarchy | item → dept → category, store → state, 353 series over 6 levels |
| calendar | SNAP days per state, Thanksgiving/Black Friday/Christmas run-ups, Super Bowl, July 4 |
| zeros | 39.6% of observation-days |
| planted truth | per-category price elasticity, promo display lift |

Knowing the truth is the point. It turns the pricing half from an assertion into
a scored exercise, and it lets the endogeneity be *measured* instead of
disclaimed — promotions are deliberately scheduled into weeks demand is already
strong, exactly as a merchant would.

## What the backtest found

Six folds, 28-day horizon, rolling origin. No random split anywhere.

**WMAPE by level, evaluation folds** (`out/forecast_report.txt` §1):

| level | seasonal naive | croston SBA | GBM | FVA vs snaive |
|---|---|---|---|---|
| total | 0.279 | 0.299 | **0.109** | +0.171 |
| store | 0.336 | 0.328 | **0.143** | +0.193 |
| store×dept | 0.479 | 0.405 | **0.256** | +0.222 |
| store×item | 0.751 | 0.592 | **0.505** | +0.245 |

**Syntetos–Boylan classification** of the 300 bottom series: 147 intermittent,
122 erratic, 20 smooth, 11 lumpy. MAPE is not implemented anywhere in this repo
— on a panel that is 40% zeros it is not a weak metric, it is an undefined one.
The intermittent bucket is scored on MASE, where the GBM reads **1.021**: it is
*worse than saying "same day last week"* on those series, which is stated in the
table rather than averaged away.

### Three results I did not expect

**1. Per-series model selection loses to just deploying the GBM everywhere.**
The textbook answer to "your model beats the baseline overall but loses on 18% of
series" is *select per series*. Measured here, that answer is wrong:

| policy | WMAPE | bias |
|---|---|---|
| seasonal naive | 0.751 | −3.5% |
| GBM everywhere | **0.505** | +0.6% |
| per-series, WMAPE gate | 0.528 | −3.5% |
| per-series, WMAPE + bias guard | 0.561 | −3.1% |

Two mechanisms, both visible in the output. WMAPE *rewards a degenerate forecast*
on intermittent demand — on a mostly-zero series "last value" is usually zero, so
naive predicts all-zeros, scores WMAPE ≈ 1.0, and cannot be beaten downward by
anything that ever puts a unit on the wrong day. The gate duly hands 96 series to
naive, whose standalone bias is **−29%**: a replenishment system that under-orders
every slow mover it owns. And the gate is estimated on 3 folds × 28 days of a
series that sells a unit every third day, so the choice is mostly noise.

**2. MinT loses to bottom-up at every level here** — all three weightings of it,
as the second pass confirms. Bottom-up totals 0.062 WMAPE against MinT-shrink's
0.080 and MinT-OLS's 0.097. The aggregate of the bottom-level forecasts is better
than any directly-fitted aggregate model (errors cancel on the way up), so
blending the weaker aggregates back in costs accuracy. All are exactly coherent
(max violation 0.000 units); the incoherent base forecasts violate by 1,520 units.

**3. Naive log-log elasticity gets the *sign* wrong.** Six specifications scored
against planted truth:

| spec | FOODS | HOUSEHOLD | HOBBIES | mean abs bias |
|---|---|---|---|---|
| **truth** | −2.10 | −1.20 | −0.60 | — |
| A: naive log-log OLS | **+0.24** | **+0.88** | −0.32 | 1.57 |
| C: + series FE + calendar | −1.11 | −0.67 | −0.37 | 0.59 |
| E: Poisson GLM, no promo control | −3.63 | −2.81 | −1.84 | 1.46 |
| F: Poisson GLM + promo + FE + WOY | −2.15 | −1.41 | −0.60 | **0.09** |

Two different biases pulling opposite ways. Without fixed effects the regression
is identified off the price gap *between products* — comparing a $6 item to a $2
item and calling it elasticity. Adding FE fixes the sign but leaves heavy
attenuation, because `log1p(sales)` is not `log(sales)` and on a zero-heavy count
series the transform crushes the observations carrying the signal. A Poisson log
link fixes that; *without* the promo control it then overshoots, which is the
genuine endogeneity — promotions land in strong weeks and carry a display lift.

## The markdown decision, priced

The optimiser **decides** on an observational estimate and is **scored** at the
world's true elasticity, which is the real situation a pricing team is in.

| category | true e | oracle schedule | sell-through | cost of deciding on spec A | on spec F |
|---|---|---|---|---|---|
| FOODS | −2.10 | 30/30/40 | 96% vs 45% | **28.9%** of clearance revenue | 0.4% |
| HOUSEHOLD | −1.20 | 40/50/50 | 88% vs 45% | 6.6% | 0.5% |
| HOBBIES | −0.60 | 0/0/0 | 45% vs 45% | 0% | 0% |

Markdown depth falls out of elasticity without being told: the inelastic category
should not be marked down at all, because there you give away margin on units
that were going to sell anyway. And the *estimation method* is worth 28.9% of
clearance revenue on FOODS — that number is the business case for a price
experiment, in dollars.

Sensitivity at ±30% elasticity: FOODS and HOBBIES **hold**, HOUSEHOLD **flips**
(the recommended markdown becomes −10.7% in the low-elasticity world). That flip
is reported by the code, not by me; it is the honest answer that HOUSEHOLD cannot
be priced off this data.

Phasing beats the best flat discount by well under 1% throughout. Most of the
value is in marking down *at all*, and a project that headlines the phasing gain
has its emphasis backwards.

## Two modelling defects I hit and fixed

Kept because they are the evidence the simulator is doing arithmetic rather than
theatre:

- **The optimiser refused to discount anything.** Correct economics for what I
  had built: salvage at 15% of ticket beats the margin on an incremental unit for
  an inelastic item. Real end-of-season goods are jobbed out near zero — salvage
  is now 5% and stated as a merchant-supplied input.
- **Phased markdown could not beat a flat one.** Also correct: with a constant
  demand rate and a multiplicative price effect, a single price is *provably*
  optimal and phasing can only tie. Clearance markdowns are phased because
  end-of-season demand decays. The simulator now decays it (21-day half-life,
  stated as an assumption and sensitivity-tested), and phasing earns a real if
  small win.

Also fixed: the per-series section originally dropped series with no sales in the
selection window, flattering the policy by removing its hardest cases. The gate
now falls back to the incumbent and a test asserts no series is dropped.

## Tests (`pytest tests -q`, 26 passing)

The load-bearing one corrupts every sale at or after the forecast origin
(`×1000 + 777`), rebuilds the features, and asserts the horizon feature rows are
bit-identical. If any lag or rolling window reached forward, it would fail. The
rest cover WMAPE/MASE against hand computation, the SBA bias correction being
strictly below classic Croston, the Syntetos–Boylan quadrants, MinT output being
coherent from a deliberately incoherent start, and the optimiser refusing to mark
down inelastic demand.

## Second pass: what you actually hand replenishment

The spec asks it directly: *replenishment wants one number per item×store×day,
and the model produces a distribution — what do you hand them?* The first pass
answered in prose and said the code couldn't do it. Now it can: a separate
quantile GBM per service level at the bottom level.

| tau | coverage | mean forecast | implied cost ratio |
|---|---|---|---|
| 0.50 | 0.6130 | 4.44 | 1× |
| 0.75 | 0.7869 | 7.06 | 3× |
| 0.90 | **0.9090** | 9.88 | 9× |
| 0.95 | **0.9573** | 11.83 | 19× |
| 0.98 | **0.9798** | 14.16 | 49× |

**5 of 5 quantiles win their own pinball loss** — each forecast minimises the
loss for the tau it was fitted to, which is the diagonal that proves the set is
calibrated rather than merely ordered.

**Why pinball and not MAE:** MAE is minimised by the *median*, so scoring a P90
on MAE would rank it worse than the P50 by construction — measuring the wrong
thing and calling the right answer wrong. A test demonstrates exactly that
inversion.

**The P50 over-covers (0.613 vs 0.50)** and that is explained rather than
glossed: coverage counts `actual ≤ forecast`, and on a series that is zero 40% of
the time the median forecast is often *zero*, so every zero day counts as covered
because `0 ≤ 0`. It's a property of coverage as a diagnostic on discrete
zero-inflated data, not a miscalibrated model — the pinball diagonal confirms the
P50 is the best P50 available. The same statistic that is informative at tau=0.95
is nearly meaningless at 0.50 here.

**So what do you hand them?** Not the mean — ordering to the mean means being
short about half the time, and the two errors don't cost the same. You hand them
the quantile the service target implies, and the newsvendor result says which:

```
q* = Cu / (Cu + Co)
```

Read that column backwards and a service level stops being a policy and becomes a
claim someone has to defend: **a 95% service target asserts that understocking
costs 19× what overstocking costs.** That reframing is the useful half — it turns
a service-level argument into a cost argument.

The mean forecast is 4.44 units/day and the P95 is 11.83 — the gap is the safety
stock the service level buys, **166% more inventory** than ordering to the mean.

**Honest limit:** these are *daily* quantiles, and replenishment needs the
quantile of demand over the **lead time**, which is not the sum of daily quantiles
— summing them overstates, because the days don't all go wrong together.
Converting one to the other needs the dependence structure across days, which is
exactly what DATA-2's negative-binomial experiment independently found it was
missing. Neither project has it, and arriving at the same gap from two directions
is at least consistent.

## Second pass: reconciliation as a survey, not one method

| level | base (incoherent) | bottom-up | MinT OLS | MinT WLS | MinT shrink |
|---|---|---|---|---|---|
| total | 0.1087 | **0.0616** | 0.0971 | 0.0802 | 0.0802 |
| state | 0.1419 | **0.0817** | 0.1232 | 0.1048 | 0.1048 |
| store | 0.1427 | **0.1226** | 0.1560 | 0.1420 | 0.1420 |
| store×dept | 0.2562 | **0.2672** | 0.2851 | 0.2700 | 0.2701 |
| store×item | 0.5051 | **0.5051** | 0.5646 | 0.5068 | 0.5068 |

The three MinT variants differ **only** in the weight matrix they invert: OLS
treats every series' error as equally important (obviously false across a
hierarchy spanning a 300-unit total and a 0.2-unit item, and included to show how
false); WLS scales by each series' own residual variance; shrink also uses the
*correlation* between series — the thing that makes MinT better in principle and
hardest to estimate from 353 series and 180 residual days.

**Bottom-up wins at every level, and that's the honest result.** MinT *should*
beat it — it uses information bottom-up discards. The reason it doesn't is
visible in the `base` column: directly-fitted aggregate models are worse than
aggregating the bottom-level forecasts (total 0.11 vs 0.06) because errors cancel
on the way up. MinT blends those weaker aggregates back in, and blending in a
worse signal makes things worse. That is not a bug in MinT — it is MinT correctly
weighting information that happens not to be worth having on this hierarchy.

All four reconciliation methods are exactly coherent (max violation 0.000 units)
against the incoherent base's 1,520-unit violation.

## The other ~50% — what is NOT here

- **No serving artifact.** The spec asks for a planner-facing view (item×store →
  forecast, intervals, FVA, recommended markdown path). There is no UI and no API.
- **Quantiles are daily, not lead-time** (see above) — the conversion needs a
  cross-day dependence model neither this project nor DATA-2 has.
- **Quantiles are bottom-level only.** Running five extra models per level per
  fold would triple an already 20-minute backtest to produce numbers nobody
  orders from.
- **No global deep model** (the optional DeepAR/N-BEATS arm).
- **No middle-out reconciliation and no probabilistic reconciliation** — the
  quantiles are not reconciled across the hierarchy at all, so the P95s do not
  add up even though the point forecasts do.
- **The pooled-class selection gate** the report recommends (gate estimated per
  intermittency class rather than per series) is described and not built.
- **Elasticity is per category**, not per item or per segment, and there is no
  cross-price elasticity, so no cannibalisation in the markdown optimiser.
- **The markdown optimiser is single-item.** No shared inventory, no category
  budget constraint, no competitor response.
- LightGBM is not installed; `HistGradientBoostingRegressor` stands in (same
  histogram algorithm, Poisson loss available). statsmodels is not installed, so
  OLS with HC0 errors is written out in `run_pricing.py`.

**"What would you hand replenishment?"** — answered in code now, not prose: the
quantile the item's service target implies, with the calibration to back it and
the cost ratio that target is asserting. The prerequisite everyone skips still
holds and still matters: the point forecast is unbiased (+0.6%) *before* anyone
sizes a buffer on top of it, because safety stock sized on sigma does not cover a
drift.
