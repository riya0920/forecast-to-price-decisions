# ML-1 — Forecast-to-Price Decision System

**Complete against the spec.** Rolling-origin backtest with FVA discipline, a
deep arm, five reconciliation methods, calibrated quantiles converted to
**lead-time** demand, elasticity at three levels of granularity including
**cross-price**, a **multi-item MILP** markdown optimiser, and an **HTTP service**
a planner can actually call.

Every number below was produced by running the code in this directory. Nothing is
quoted from a paper or a leaderboard, and several of the headline results are
negative.

```bash
python src/generate.py       # ~8s     build the panel
python run_forecast.py       # ~18min  6-fold backtest: LightGBM + N-BEATS + 5 quantiles + 5 reconciliations
python report.py             # ~25s    the FVA tables
python run_advanced.py       # ~15s    pooled gate, lead-time demand, probabilistic reconciliation
python run_pricing.py        # ~4min   elasticity (category/item/segment/cross) + markdown MILP
uvicorn serve:app --port 8011   #       the planner service
python -m pytest tests -q    # 66 tests
```

300 item×store series, 1,460 days, 353 series across six hierarchy levels.

## No substituted dependencies

Both earlier passes carried a table of stand-ins. It is gone. **LightGBM**,
**statsmodels**, **PuLP**, **PyTorch** and **FastAPI** are the real libraries now,
and the honest report on the swap is that *the accuracy numbers barely moved*.
That is worth stating plainly: the substitutions were correctly described as
low-impact, and swapping them out confirmed rather than overturned that. Anyone
claiming a library swap bought them a large accuracy gain on tabular count data
should be asked what else changed at the same time.

What the real libraries did buy is **inference** — statsmodels gives standard
errors, and half the elasticity findings below are decided on whether a dispersion
exceeds its own standard error, which is a question you cannot ask of a point
estimate.

## What the backtest found

| level | seasonal naive | croston SBA | N-BEATS | LightGBM | FVA |
|---|---|---|---|---|---|
| total | 0.2620 | 0.3025 | — | **0.1034** | +0.1585 |
| store | 0.3063 | 0.3107 | — | **0.1211** | +0.1852 |
| store×dept | 0.4434 | 0.3772 | — | **0.2298** | +0.2136 |
| store×item | 0.7729 | 0.6093 | 0.5656 | **0.5291** | +0.2438 |

WMAPE, evaluation folds 4–6. MAPE is not in this repo: 32% of observation-days
are zero, so a per-observation percentage error divides by zero on a third of the
evaluation set.

### The deep arm loses, and its bias is the reason to care

N-BEATS at the bottom level: **0.5656 WMAPE against the GBM's 0.5291** — and
**−20.6% bias against the GBM's +0.25%**.

The WMAPE gap is the boring half. A model that is a fifth low on every order is
not slightly worse than an unbiased one, it is a different and worse decision, and
it fails for exactly the reason `naive` does: mean-scaled absolute error on
zero-heavy series *rewards forecasting low*, because a low forecast is right on
the many zero days and wrong only on the few that sell.

That connects two sections. The per-series gate hands N-BEATS **116 series** on a
plain WMAPE gate and only **69** once a bias guard is on. The gate was never
finding a better model for those series; it was finding the model that games the
metric hardest — and it took two different architectures to make that visible.

It also is not evidence that deep forecasting loses. 300 series is far below where
global deep models start paying, and *that* is the finding for anyone deciding
whether to staff the work.

## Three results I did not expect

- **Per-series model selection LOSES to one global model** (0.5451 vs 0.5291), and
  so does the bias-guarded version (0.5546). The textbook answer to "your model
  loses on 14% of series" is *select per series*; measured, that answer is wrong
  here.
- **Naive log-log elasticity gets the SIGN wrong** — +0.211 for FOODS against a
  truth of −2.100. Without fixed effects the regression is identified off the
  price gap *between* products. A Poisson GLM with fixed effects recovers −2.205.
- **Bottom-up beats all three MinT variants** at every level but one. MinT uses
  information bottom-up discards and *should* win; it doesn't, because the
  directly-fitted aggregates it blends back in are worse than the aggregated
  bottom level (total 0.1034 vs 0.0560).

## Reconciliation: five methods, including middle-out

| level | base (incoherent) | bottom-up | middle-out | MinT OLS | MinT WLS | MinT shrink |
|---|---|---|---|---|---|---|
| total | 0.1034 | **0.0560** | 0.0593 | 0.0886 | 0.0634 | 0.0634 |
| state | 0.1145 | **0.0698** | 0.0723 | 0.1032 | 0.0792 | 0.0792 |
| store | 0.1211 | **0.1050** | 0.1088 | 0.1352 | 0.1127 | 0.1127 |
| store×cat | 0.1859 | **0.1758** | 0.1758 | 0.1992 | 0.1811 | 0.1811 |
| store×dept | 0.2350 | 0.2350 | **0.2298** | 0.2520 | 0.2359 | 0.2359 |
| store×item | 0.5291 | 0.5291 | 0.5501 | 0.5619 | **0.5290** | 0.5290 |

All five are exactly coherent (max violation 0.000 units) against the incoherent
base's 1,101-unit violation.

**Middle-out wins at exactly one level — the one it forecasts directly** — and is
worst at the bottom, because item shares are held fixed across the horizon. That
assumption *is* the method and it is also its failure mode: a promotion changes
precisely those shares, so middle-out is at its weakest exactly when the forecast
matters most. Large retailers run it anyway because the middle level is where
series are smooth enough to model well; this table is the price of that
convenience, measured.

## Quantiles — and a defect this project shipped for two passes

**5 of 5 quantiles win their own pinball loss.** Coverage: 0.905 at τ=0.90, 0.948
at τ=0.95, 0.976 at τ=0.98.

Then the fan was checked for **crossing**, and it crosses on **0.96% of
item-days**: the fitted P90 lands *above* the fitted P95. Five independently
fitted quantile models have nothing coupling them, so nothing stops it. This is
not a numerical wart — an order system reading a crossed fan orders **more stock
for a lower service level**, inverting the meaning of the dial a planner is
turning.

Monotone rearrangement (sorting the fan at each point) removes it entirely and is
applied at the serving boundary. The theory says sorting is provably no worse
because the true quantile function is monotone; the table says **4 of 5 quantiles
improved and one got worse by 0.00044** against a loss of 0.680. That gap between
"provably no worse" and what a finite sample actually does is exactly the kind of
thing that gets quoted without the word *expected* in front of it, so it is
reported rather than rounded away.

## Lead-time demand — the conversion the last pass said it could not do

`P95(D₁+…+D_L) ≠ P95(D₁)+…+P95(D_L)`. Three answers, L=28, τ=0.95:

| method | assumes | value | vs bootstrap |
|---|---|---|---|
| sum of daily quantiles | perfect dependence | 289.9 | **1.75×** |
| block bootstrap of error paths | whatever the data has | 165.8 | — |
| iid normal, σ√L | independence | 158.4 | 0.96× |

**The two shortcuts err in opposite directions.** Which one is worse is decided by
the autocorrelation of the forecast errors, and here it is measured: **lag-1
ρ = +0.048**, essentially zero. So on *this* panel the iid formula is nearly right
and sum-of-quantiles is wildly wrong — but stating that as a general result would
be the easy mistake. With strongly autocorrelated errors the iid version is the
dangerous one, because it under-covers in exactly the clustered weeks that cause
the stockout.

The bootstrap's value is therefore not that it won by a lot. It is that **it did
not need to be told which regime it was in.** A planner choosing the iid formula
is making an assumption about autocorrelation whether they know it or not.

**This closes the DATA-2 join.** DATA-2 sized safety stock from historical mean
and sd, named forecast integration as its most valuable missing piece, and then
watched its negative-binomial experiment fail for precisely this reason — an iid
marginal, however well specified, cannot produce the right lead-time quantile.
This is the distribution it needed.

## Probabilistic reconciliation — why the P95s should *not* add up

400 joint bottom-level draws with the cross-series residual correlation preserved,
pushed through the summing matrix. Every sample path is coherent by construction.

| level | Σ children's P95 | own P95 | ratio |
|---|---|---|---|
| total | 83,060 | 45,363 | **1.83** |
| state | 83,060 | 46,648 | 1.78 |
| store | 83,060 | 49,423 | 1.68 |
| store×cat | 83,060 | 55,378 | 1.50 |
| store×dept | 83,060 | 59,636 | 1.39 |

The last README called it a gap that "the P95s do not add up even though the point
forecasts do". **They should not.** A quantile is not a linear functional of a
distribution; forcing the store P95 to equal the sum of its items' P95s asserts
that every item in the store has its bad week simultaneously. Adding up is a
property of *realisations*, which is why the fix is to reconcile sample paths and
read quantiles off the coherent set.

The ratio grows toward the top of the hierarchy: the more series you pool, the
more the independent part of their errors cancels. Aggregate safety stock sized by
summing item safety stocks is over-provisioned by that ratio — the argument for
holding buffer centrally rather than at the leaf, with a number on it.

## The pooled-class selection gate

The report recommended it and never built it. Per-series selection lost badly; the
diagnosis was that a champion chosen from three folds of one intermittent series
is chosen on noise. Pooling to the intermittency **class** makes each decision on
~1,350 rows instead of ~18.

| gate | decisions | WMAPE | bias |
|---|---|---|---|
| global | 1 | **0.1598** | +0.0025 |
| per class | 4 | 0.1603 | −0.0233 |
| per series | 300 | 0.2040 | −0.0353 |

**Pooling recovers almost the entire loss (0.2040 → 0.1603) and still does not
beat the global model.** That is the honest result and it is more useful than a
win would have been: it says the loss from per-series selection was *noise*, not a
missing signal, because averaging over 75× more data removed nearly all of it and
found nothing underneath.

The bias column is why this gate is not just the old one with bigger groups.
Candidates whose selection-window bias exceeds ±15% are rejected before accuracy
is considered — the first gate's failure was not inaccuracy, it was handing 96
series to a method running −29% bias.

## Elasticity, at three levels of granularity

### Per category — six specifications scored against planted truth

| spec | FOODS | HOUSEHOLD | HOBBIES |
|---|---|---|---|
| **TRUTH** | **−2.100** | **−1.200** | **−0.600** |
| A naive log-log | +0.211 | −0.398 | +0.055 |
| C promo + FE + calendar | −1.209 | −0.721 | −0.323 |
| E Poisson, no promo control | −3.763 | −2.209 | −1.719 |
| **F Poisson, full** | **−2.205** | **−1.027** | **−0.595** |

Spec A has the **wrong sign** on two of three categories. Spec E is biased *away*
from zero because promotions are scheduled into strong weeks and carry a display
lift; spec F controls both and lands close.

### Per item — is heterogeneity *estimable*, not just real?

The generator now draws each item's own elasticity around its category's, so this
is a question with a right answer instead of a fishing expedition. 60 items, one
Poisson GLM each:

| estimator | MAE | within-category MAE | within-category corr | sd of estimates |
|---|---|---|---|---|
| raw per item | 0.2356 | 0.2388 | 0.641 | 0.834 |
| **shrunk to category** | **0.1558** | **0.1484** | **0.739** | 0.732 |
| category mean only | 0.2182 | 0.1957 | 0.000 | 0.711 |

True sd of per-item elasticity: 0.707.

**Shrinkage wins**, so the heterogeneity is not merely real but recoverable. Two
things are worth reading carefully. The raw estimates have sd 0.834 against a true
0.707 — **1.18× the real dispersion**, so some of what a per-item table shows as
"this SKU is more elastic" is sampling noise a merchant would price on anyway. And
the *pooled* correlation with truth is 0.92 for every estimator including the
category mean, because category elasticities are −2.1/−1.2/−0.6 and almost all the
variance is between categories. The within-category column is the one that answers
"can I price this item differently from its neighbour on the same shelf" — and
there the category mean scores exactly 0.000 by construction.

### Cross-price — and the confound that eats it

| spec | own | cross | true cross |
|---|---|---|---|
| G daily, event + weekday FE (FOODS) | −2.237 | **0.423** | 0.350 |
| G daily, event + weekday FE (HOUSEHOLD) | −1.130 | **0.299** | 0.350 |
| G daily, event + weekday FE (HOBBIES) | −0.704 | **0.493** | 0.350 |
| H same, **no** calendar FE | −2.190 | **0.117** | 0.350 |
| I rival term omitted entirely | −2.181 | — | 0.350 |

Spec G recovers cross-price within 1.4 standard errors on all three categories.

**Drop the event and weekday fixed effects and it collapses to 0.117 — attenuated
by 72%.** It is the same confound this project already documents for own-price,
arriving somewhere new: promotions are scheduled into strong periods, what makes a
period strong is *events and weekday*, and those are shared across the whole
department. A rival's promotion lands disproportionately on days my demand is high
for reasons unrelated to the rival, and the regression charges that co-movement to
the cross term with a negative sign.

**Week-of-year fixed effects do not fix it** — they are in *both* specifications.
Thanksgiving moves within the ISO week from year to year and weekday variation is
inside the week by definition, so a weekly panel has already destroyed the
variation the control needs. That is why this one specification is daily while
every other elasticity here is weekly: **aggregation is not a neutral performance
choice when the confound lives inside the bucket.**

## Markdown: multi-item, shared inventory, a budget, cannibalisation

A department of 10 items, 42-day clearance in 3 phases, two items sharing one
stock pool, solved as a sequence of exact MILPs with the rival price index held
fixed and updated between solves. Every row is **scored in the same true world**;
only what the optimiser *believed* differs.

| policy | mean depth | revenue | markdown spend | leftover |
|---|---|---|---|---|
| ignores cannibalisation | 0.197 | $21,313 | $4,794 | 888 |
| uses estimated cross | 0.240 | $21,619 | $6,217 | 501 |
| uses **true** cross | 0.220 | $21,523 | $5,616 | 660 |
| true cross + 12% budget | 0.133 | $21,500 | **$3,598** | 1,273 |

**Modelling cannibalisation makes the optimiser discount MORE, not less** — the
opposite of the textbook direction, and worth stating rather than smoothing over.
With substitutes, a department-wide markdown partly cancels itself because every
item's rivals get cheaper at the same time, so reaching the same clearance volume
requires going *deeper* than a single-item view suggests. The single-item
optimiser is wrong in both directions depending on whether it is pricing one item
or a shelf.

**The estimated-cross row beats the true-cross row, and that is not a finding
about estimation.** A schedule chosen with the true elasticity should be
unbeatable in the true world, so a negative gap means the optimiser and the scorer
do not share a model — and they do not. The MILP constrains *total* season demand
to inventory, which is linear and therefore solvable; the scorer runs the season
phase by phase and caps each phase at what is left, which is the real dynamic and
is not linear. The $96 gap is the price of the linearisation, and it is larger
than the estimation error it was supposed to measure — which is exactly why it is
reported instead of being quietly presented as one. **The load-bearing comparison
in that table is row 1 against row 3.**

The cost of a bad elasticity estimate, priced: **28.98%** of FOODS clearance
revenue for a team using log-log OLS, against 0.52% for the Poisson spec.

## The planner service

`uvicorn serve:app --port 8011`

- `GET /forecast/{key}` — point forecast, the rearranged quantile fan, the
  intermittency class, and the FVA against seasonal naive. FVA travels *with* the
  forecast because an absolute WMAPE is not something a planner can act on.
- `POST /order` — the single number replenishment asked for, returned alongside
  `implied_cost_ratio`: a 95% target *is* the claim that understocking costs 19×
  overstocking. The response also carries the caveat that summed daily quantiles
  assume perfect cross-day dependence, and points at the measurement of how much
  that overstates.
- `POST /override` — a planner disagreeing, with an author and a **required**
  reason. Overrides are the largest source of forecast value *destruction* in real
  planning systems and the only way anyone finds out is by logging them and
  scoring them later. A 422 on a missing reason is the point, and a test asserts it.
- `GET /markdown/{key}` — the clearance path for one item, priced with that item's
  own elasticity, served from the same table the pricing report scores.

## Bugs this pass caught

- **The quantile fan crossed on 0.96% of item-days** and had done since the fan
  was built. Nothing in five independent fits couples them.
- **`SERVICE_LEVELS` was read one line before it was defined**, inside a `try`
  whose broad `except` swallowed the `NameError` into a silent `ok=False`. The
  service reported "artifacts missing" while the artifacts were fine.
- **The MILP fixed point was reported as non-converged for every policy**, because
  convergence was tested on the damped continuous rival index rather than on the
  discrete schedule. The solver had settled; the test had not noticed.
- **The report claimed neither this project nor DATA-2 had the cross-day
  dependence structure.** True when written, false once `run_advanced.py` existed,
  and the kind of stale cross-reference that survives three passes because nobody
  re-reads the other project's README.

## What is deliberately not here

- **No real dataset.** M5 is not downloadable here, and the generator is the point
  anyway: elasticity, cross-price and per-item heterogeneity are *scored* against
  planted truth, which no public dataset permits.
- **The generator is a model.** Its demand is a gamma-mixed Poisson with a
  multiplicative price effect, which is close to what the GBM assumes — so every
  method here does better than it would on real data.
- **Quantiles are bottom-level only.** Running five extra models per level per
  fold would triple the backtest to produce numbers nobody orders from.
- **No competitor response** in the markdown optimiser, and no network
  optimisation — it scores one department at a time with no view of the orders or
  inventory behind it.
- **The service has no auth, no rate limiting and no model registry.** It is a
  real HTTP surface over real artifacts, not a production deployment.
