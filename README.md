# ML-1 — Forecast-to-Price Decision System

**This is not deployable.** It is the first ~20% of the spec: the machinery the
hiring-manager doc calls the differentiator, built and measured, with the
remaining 80% named at the bottom rather than left for a reader to discover.

Every number below was produced by running the code in this directory. Nothing
is quoted from a paper or a leaderboard.

```bash
python src/generate.py     # ~4s    build the panel
python run_forecast.py     # ~6min  6-fold rolling-origin backtest, all levels
python report.py           # ~20s   the FVA tables
python run_pricing.py      # ~3min  elasticity + markdown
python -m pytest tests -q  # 14 tests
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

**2. MinT loses to bottom-up at every level here.** Bottom-up totals 0.062 WMAPE
against MinT's 0.080. The aggregate of the bottom-level forecasts is better than
any directly-fitted aggregate model (errors cancel on the way up), so blending the
weaker aggregate base forecasts in costs accuracy. Both are exactly coherent
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

## Tests (`pytest tests -q`, 14 passing)

The load-bearing one corrupts every sale at or after the forecast origin
(`×1000 + 777`), rebuilds the features, and asserts the horizon feature rows are
bit-identical. If any lag or rolling window reached forward, it would fail. The
rest cover WMAPE/MASE against hand computation, the SBA bias correction being
strictly below classic Croston, the Syntetos–Boylan quadrants, MinT output being
coherent from a deliberately incoherent start, and the optimiser refusing to mark
down inelastic demand.

## The other 80% — what is NOT here

- **No serving artifact.** The spec asks for a planner-facing view (item×store →
  forecast, intervals, FVA, recommended markdown path). There is no UI and no API.
- **No prediction intervals**, so no P50/P90 service-level conversation and no
  quantile hand-off to replenishment. This is the largest single gap: the spec's
  "what do you hand replenishment" question is answered in prose below, not in code.
- **No global deep model** (the optional DeepAR/N-BEATS arm).
- **Reconciliation is one alternative, not a survey** — bottom-up and MinT(shrink);
  no OLS/WLS comparison run, no middle-out, no probabilistic reconciliation.
- **The pooled-class selection gate** the report recommends (gate estimated per
  intermittency class rather than per series) is described and not built.
- **Elasticity is per category**, not per item or per segment, and there is no
  cross-price elasticity, so no cannibalisation in the markdown optimiser.
- **The markdown optimiser is single-item.** No shared inventory, no category
  budget constraint, no competitor response.
- LightGBM is not installed; `HistGradientBoostingRegressor` stands in (same
  histogram algorithm, Poisson loss available). statsmodels is not installed, so
  OLS with HC0 errors is written out in `run_pricing.py`.

**If asked "what would you hand replenishment?"** — the mean is the wrong answer
and this repo cannot yet give the right one. Replenishment needs a quantile tied
to a target service level (P90 for an A item, nearer P50 for a C item), which
requires the intervals this build does not produce. What it *does* establish is
the prerequisite everyone skips: the point forecast is unbiased (+0.6%) before
anyone starts sizing a buffer on top of it, because safety stock sized on sigma
does not cover a drift.
