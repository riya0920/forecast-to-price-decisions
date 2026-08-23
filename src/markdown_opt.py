"""Multi-item markdown optimisation: shared inventory, a budget, and cannibalisation.

WHAT THE SINGLE-ITEM VERSION COULD NOT SAY
------------------------------------------
The last pass optimised one item at a time by enumerating monotone schedules. It
was correct and it was answering an easier question than the one merchants ask,
because a clearance decision is never about one SKU:

* **Cannibalisation.** Marking down one item takes sales from its department
  siblings. A per-item optimiser books that stolen volume as a gain and never
  pays for it, so it over-discounts systematically -- and the more items in the
  department, the worse it gets.
* **A markdown budget.** Clearance dollars are a finance-approved number for the
  season. Without the constraint the optimiser will happily spend whatever it
  likes, and the schedule it returns is not one anyone can approve.
* **Shared inventory.** Sizes and colours of one style draw down one pool.

WHY MILP, AND WHERE IT IS HONEST ABOUT ITSELF
---------------------------------------------
Discount depth is a small grid (merchants price at 10/20/30/40/50, not at
17.3%), so the decision is genuinely combinatorial and a binary program is the
natural fit: one binary per (item, phase, depth), exactly one depth chosen per
item-phase, monotone non-increasing across phases.

The cannibalisation term is what stops this being a straight MILP -- each item's
demand depends on the OTHER items' chosen prices, which are decision variables,
so the objective is not linear. Rather than pretend otherwise, this solves a
**sequence of MILPs**: fix the rival price index at its current value, solve
exactly, recompute the index from the solution, repeat. That is a fixed-point
iteration, and it is reported as one -- including whether it converged, because
an alternating scheme on a non-convex problem can cycle, and a solver that
quietly returns iteration 20 of a cycle is worse than one that says it did not
settle.

WHAT THIS IS NOT
----------------
Not a competitor-response model, and not a global optimum of the true non-linear
problem. It is the exact optimum of the linearised problem at a fixed point.
"""
from __future__ import annotations

import numpy as np
import pulp

SALVAGE_FRACTION = 0.05
DECAY_HALFLIFE_DAYS = 21.0


def phase_demand(base_daily: float, elasticity: float, discount: float,
                 rival_log_rel: float, cross: float, phase_start: int,
                 phase_len: int, halflife: float = DECAY_HALFLIFE_DAYS) -> float:
    """Expected units for one item in one phase at one discount depth.

    Three multiplicative pieces, all of them stated rather than buried:
      own price   (1 - d) ** elasticity
      rivals      exp(cross * mean log relative rival price)
      decay       end-of-season demand fades, integrated over the phase
    """
    lam = np.log(2.0) / max(halflife, 1e-6)
    days = np.arange(phase_start, phase_start + phase_len)
    decay = float(np.exp(-lam * days).sum())
    own = (1.0 - discount) ** elasticity
    return base_daily * decay * own * np.exp(cross * rival_log_rel)


def solve_once(items: list[dict], grid: list[float], n_phases: int,
               phase_len: int, rival_log_rel: np.ndarray,
               cross: float, budget: float | None,
               shared_pools: dict | None = None,
               salvage_fraction: float = SALVAGE_FRACTION) -> dict:
    """One MILP with the rival index held fixed.

    items: dicts with item_id, base_daily, ref_price, elasticity, inventory,
           optional pool (a shared-inventory group id)
    rival_log_rel: (n_items x n_phases) log relative rival price, treated as data
    """
    prob = pulp.LpProblem("markdown", pulp.LpMaximize)
    n_i, n_p, n_d = len(items), n_phases, len(grid)

    x = {(i, t, d): pulp.LpVariable("x_%d_%d_%d" % (i, t, d), cat="Binary")
         for i in range(n_i) for t in range(n_p) for d in range(n_d)}

    demand = np.zeros((n_i, n_p, n_d))
    revenue = np.zeros((n_i, n_p, n_d))
    spend = np.zeros((n_i, n_p, n_d))
    for i, it in enumerate(items):
        for t in range(n_p):
            for d, disc in enumerate(grid):
                q = phase_demand(it["base_daily"], it["elasticity"], disc,
                                 float(rival_log_rel[i, t]), cross,
                                 t * phase_len, phase_len)
                demand[i, t, d] = q
                revenue[i, t, d] = q * it["ref_price"] * (1 - disc)
                spend[i, t, d] = q * it["ref_price"] * disc

    # one depth per item-phase
    for i in range(n_i):
        for t in range(n_p):
            prob += pulp.lpSum(x[i, t, d] for d in range(n_d)) == 1

    # monotone: a clearance price never goes back up. Not a modelling
    # convenience -- a retailer that raises a clearance price teaches customers
    # to buy immediately and destroys the option value of waiting, which is the
    # only thing a markdown ladder has to sell.
    for i in range(n_i):
        for t in range(n_p - 1):
            prob += (pulp.lpSum(grid[d] * x[i, t + 1, d] for d in range(n_d))
                     >= pulp.lpSum(grid[d] * x[i, t, d] for d in range(n_d)))

    # per-item inventory: cannot sell what is not there
    pools = shared_pools or {}
    pooled = set()
    for pool, members in pools.items():
        pooled.update(members)
    for i, it in enumerate(items):
        if i in pooled:
            continue
        prob += (pulp.lpSum(demand[i, t, d] * x[i, t, d]
                            for t in range(n_p) for d in range(n_d))
                 <= it["inventory"])

    # shared pools: sizes of one style draw down one stock
    for pool, members in pools.items():
        cap = sum(items[i]["inventory"] for i in members)
        prob += (pulp.lpSum(demand[i, t, d] * x[i, t, d]
                            for i in members for t in range(n_p)
                            for d in range(n_d)) <= cap)

    # the markdown budget: total discount dollars given away
    if budget is not None:
        prob += (pulp.lpSum(spend[i, t, d] * x[i, t, d]
                            for i in range(n_i) for t in range(n_p)
                            for d in range(n_d)) <= budget)

    # objective: revenue + salvage on whatever is left
    sold = {i: pulp.lpSum(demand[i, t, d] * x[i, t, d]
                          for t in range(n_p) for d in range(n_d))
            for i in range(n_i)}
    salvage = pulp.lpSum(
        salvage_fraction * items[i]["ref_price"] * (items[i]["inventory"] - sold[i])
        for i in range(n_i))
    prob += pulp.lpSum(revenue[i, t, d] * x[i, t, d]
                       for i in range(n_i) for t in range(n_p)
                       for d in range(n_d)) + salvage

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]

    chosen = np.zeros((n_i, n_p))
    for i in range(n_i):
        for t in range(n_p):
            for d in range(n_d):
                if x[i, t, d].value() and x[i, t, d].value() > 0.5:
                    chosen[i, t] = grid[d]
    return dict(status=status, discounts=chosen,
                objective=float(pulp.value(prob.objective) or 0.0))


def solve(items: list[dict], grid: list[float], n_phases: int, phase_len: int,
          cross: float, budget: float | None = None,
          shared_pools: dict | None = None, max_iter: int = 12,
          tol: float = 1e-4, dept_of=None) -> dict:
    """Fixed-point over the rival index. Reports whether it converged.

    `dept_of` maps item index -> department id; cannibalisation is scoped to the
    department because that is where the generator put it and where shoppers
    actually substitute.
    """
    n_i = len(items)
    dept_of = dept_of or {i: "ALL" for i in range(n_i)}
    rival = np.zeros((n_i, n_phases))
    history = []
    result = None
    converged = False
    prev_disc = None

    for _ in range(max_iter):
        result = solve_once(items, grid, n_phases, phase_len, rival, cross,
                            budget, shared_pools)
        disc = result["discounts"]

        # Convergence is tested on the DECISION, not on the rival index. The
        # index is a continuous quantity being averaged with damping, so it
        # approaches its fixed point asymptotically and would never hit a 1e-4
        # tolerance in a handful of iterations -- reporting "did not converge"
        # on that basis would be a false alarm about a solver that had in fact
        # settled. What matters is whether the schedule is still moving: the
        # discount grid is discrete, so a stable schedule is a genuine fixed
        # point of the best-response map and the only thing a merchant sees.
        if prev_disc is not None and np.array_equal(disc, prev_disc):
            converged = True
            history.append(0.0)
            break
        prev_disc = disc.copy()

        log_rel = np.log(np.clip(1.0 - disc, 1e-6, None))
        new = np.zeros_like(rival)
        for i in range(n_i):
            sibs = [j for j in range(n_i) if j != i and dept_of[j] == dept_of[i]]
            new[i] = log_rel[sibs].mean(axis=0) if sibs else 0.0
        history.append(float(np.abs(new - rival).max()))
        # damped update: an undamped alternating best-response on a non-convex
        # objective is the classic way to get a two-cycle that never settles
        rival = 0.5 * rival + 0.5 * new

    result = dict(result or {})
    result["converged"] = converged
    result["iterations"] = len(history)
    result["gap_history"] = history
    result["rival_log_rel"] = rival
    return result


def evaluate(items: list[dict], discounts: np.ndarray, n_phases: int,
             phase_len: int, cross: float, elasticity_key: str = "elasticity",
             salvage_fraction: float = SALVAGE_FRACTION,
             dept_of=None) -> dict:
    """Score a schedule under a given (possibly TRUE) set of elasticities.

    Separated from the solver on purpose: the optimiser decides with the
    ESTIMATED elasticity and is scored with the TRUE one, which is the only way
    to price the estimation error in dollars.
    """
    n_i = len(items)
    dept_of = dept_of or {i: "ALL" for i in range(n_i)}
    log_rel = np.log(np.clip(1.0 - discounts, 1e-6, None))
    rival = np.zeros_like(discounts)
    for i in range(n_i):
        sibs = [j for j in range(n_i) if j != i and dept_of[j] == dept_of[i]]
        rival[i] = log_rel[sibs].mean(axis=0) if sibs else 0.0

    revenue = 0.0
    units = 0.0
    spend = 0.0
    leftover = 0.0
    for i, it in enumerate(items):
        remaining = it["inventory"]
        for t in range(n_phases):
            q = phase_demand(it["base_daily"], it[elasticity_key],
                             float(discounts[i, t]), float(rival[i, t]), cross,
                             t * phase_len, phase_len)
            q = min(q, remaining)
            remaining -= q
            price = it["ref_price"] * (1 - discounts[i, t])
            revenue += q * price
            spend += q * it["ref_price"] * discounts[i, t]
            units += q
        leftover += remaining
        revenue += salvage_fraction * it["ref_price"] * remaining
    return dict(revenue=revenue, units=units, markdown_spend=spend,
                leftover_units=leftover)
