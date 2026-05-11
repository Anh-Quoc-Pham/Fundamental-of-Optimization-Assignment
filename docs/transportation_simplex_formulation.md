# Formulation: Inventory Transfer as a Transportation Problem
# Solved by the Transportation Simplex Method

## 1. Problem Mapping

| Transportation Problem | Inventory Transfer |
|---|---|
| Sources $i$ | Stores with **excess** inventory (supply) |
| Sinks $j$ | Stores with **shortage** inventory (demand) |
| Supply $s_i$ | `excess_units[i, p]` = stock − max_days × daily_sales |
| Demand $d_j$ | `needed_units[j, p]` = min_days × daily_sales − stock |
| Unit cost $c_{ij}$ | `transport_cost_matrix[i, j]` (VND per unit) |
| Decision variable $x_{ij}$ | Units of product $p$ transferred from store $i$ to store $j$ |

Each product $p$ is solved as an **independent** transportation problem.

---

## 2. Decision Variables

$$x_{ijp} \geq 0, \quad \forall\, i \in S_p,\; j \in D_p,\; p \in P$$

where:
- $S_p$ = set of stores with excess of product $p$
- $D_p$ = set of stores needing product $p$

Self-transfers ($i = j$) and routes with invalid costs are excluded.

---

## 3. Excess and Needed Quantities

**Days of Supply** (corrected formula):

$$\text{DoS}(i, p) = \frac{\text{current\_stock}(i,p)}{\bar{q}(i,p)}, \qquad \bar{q}(i,p) = \frac{\sum_t q_{ipt}}{T}$$

where $T = (\text{max\_date} - \text{min\_date}) + 1$ calendar days.

**Excess units** (units above `MAX_INVENTORY_DAYS` = 21 days):

$$e(i,p) = \max\!\left(0,\; \text{stock}(i,p) - \lfloor 21 \cdot \bar{q}(i,p) \rfloor\right)$$

**Needed units** (units to reach `MIN_INVENTORY_DAYS` = 7 days):

$$n(j,p) = \max\!\left(0,\; \lfloor 7 \cdot \bar{q}(j,p) \rfloor - \text{stock}(j,p)\right)$$

---

## 4. Transportation LP (per product $p$)

$$\min \sum_{i \in S_p} \sum_{j \in D_p} c_{ij}\, x_{ij}$$

subject to:

$$\sum_{j \in D_p} x_{ij} \leq e(i,p) \quad \forall\, i \in S_p \qquad \text{(supply)}$$

$$\sum_{i \in S_p} x_{ij} \leq n(j,p) \quad \forall\, j \in D_p \qquad \text{(demand)}$$

$$x_{ij} \geq 0, \quad x_{ii} = 0$$

The Transportation Simplex requires a **balanced** problem ($\sum s_i = \sum d_j$).

---

## 5. Balancing with Dummy Nodes

**Case 1 — Supply > Demand** (typical here: excess >> needed):

Add one **dummy demand node** $j^*$ with:

$$d_{j^*} = \sum_i e(i,p) - \sum_j n(j,p), \qquad c_{i,j^*} = 0 \;\forall\, i$$

Flows to $j^*$ represent surplus not transferred. They are excluded from the output.

**Case 2 — Demand > Supply**:

Add one **dummy supply node** $i^*$ with:

$$s_{i^*} = \sum_j n(j,p) - \sum_i e(i,p), \qquad c_{i^*,j} = M \;\forall\, j$$

where $M \gg \max c_{ij}$ is a Big-M penalty. Flows from $i^*$ represent **unmet demand** and are tracked as `unmet_units` in the output KPIs, but excluded from the transfer plan.

---

## 6. Transportation Simplex Algorithm

### Step 1 — Initial Basic Feasible Solution (Least-Cost Method)

Allocate to the cell with the **smallest unit cost** first, exhaust supply or demand, then repeat. This gives a better starting point than the Northwest Corner Rule.

A basis must contain exactly $m + n - 1$ basic cells (where $m$ = number of sources, $n$ = number of sinks). If fewer cells are allocated (degeneracy), **epsilon-cells** (zero-allocation basis cells) are added to maintain a connected basis.

### Step 2 — Compute Potentials $u_i$, $v_j$

For each basic cell $(i, j)$ the optimality condition of the Transportation Simplex is:

$$u_i + v_j = c_{ij} \quad \forall\,(i,j) \in \text{Basis}$$

Set $u_1 = 0$ (anchor), then propagate via BFS through basis cells to solve for all $u_i$ and $v_j$.

### Step 3 — Reduced Costs (Non-basic cells)

For each non-basic cell $(i, j)$:

$$r_{ij} = c_{ij} - u_i - v_j$$

Interpretation: $r_{ij}$ is the rate of change of the objective per unit introduced into cell $(i, j)$.

### Step 4 — Optimality Condition

If $r_{ij} \geq 0$ for **all** non-basic cells: current solution is **optimal** (stop).

This corresponds to the Dantzig criterion in the general Simplex: all reduced costs non-negative.

### Step 5 — Entering Cell

Select the non-basic cell with the **most negative** reduced cost:

$$(\hat{i}, \hat{j}) = \arg\min_{(i,j) \notin \text{Basis}} r_{ij}$$

### Step 6 — Closed Loop (Cycle)

When $(\hat{i}, \hat{j})$ enters the basis, a unique **closed loop** forms: a sequence of cells alternating along rows and columns, passing only through basic cells, starting and ending at $(\hat{i}, \hat{j})$.

This loop is the network equivalent of the pivot column in the general Simplex tableau.

### Step 7 — Sign Assignment

Label cells along the loop alternately $+$ and $-$, with $(\hat{i}, \hat{j})$ as $+$.

### Step 8 — Theta (Pivot Quantity)

$$\theta = \min\{x_{ij} : (i,j) \text{ has sign } -\}$$

This is the maximum quantity that can be shifted along the loop without violating non-negativity.

### Step 9 — Pivot

$$x_{ij} \leftarrow x_{ij} + \theta \quad \text{for } + \text{ cells}$$
$$x_{ij} \leftarrow x_{ij} - \theta \quad \text{for } - \text{ cells}$$

The $-$ cell that reaches zero leaves the basis. If multiple $-$ cells tie at $\theta$ (degeneracy), one is removed arbitrarily (tie-breaking by index for stability).

### Step 10 — Iterate

Return to Step 2. The algorithm terminates in a finite number of steps (under non-degeneracy; cycling is prevented by consistent tie-breaking).

---

## 7. Correspondence with General Simplex

| Transportation Simplex | General Simplex |
|---|---|
| $m + n - 1$ basic cells | Basis $B$ of $m$ basic variables |
| Potentials $u_i, v_j$ | Simplex multipliers $\mathbf{y} = c_B B^{-1}$ |
| Reduced cost $r_{ij} = c_{ij} - u_i - v_j$ | $\bar{c}_j = c_j - c_B B^{-1} A_j$ |
| Closed loop | Pivot column direction (eta vector) |
| Theta $\theta$ | Step length (ratio test) |
| Optimality: all $r_{ij} \geq 0$ | Optimality: all $\bar{c}_j \geq 0$ |
| Degeneracy: $\theta = 0$ | Degeneracy: degenerate pivot |

The Transportation Simplex exploits the **network structure** of the constraint matrix (totally unimodular) to avoid explicit matrix operations: pivoting reduces to simple loop arithmetic on the allocation table.

---

## 8. Implementation Notes (src/engine/transportation_simplex.py)

- **Class**: `TransportationSimplexOptimizer`
- **Initial BFS**: `_least_cost_bfs()` — Least-Cost Method
- **Potentials**: `_compute_potentials()` — BFS propagation
- **Loop finding**: `_find_loop()` — DFS on basis cells
- **Main loop**: `_solve_one_product()` — iterates Steps 2–9
- **Degeneracy**: epsilon-cells added when basis count < $m + n - 1$
- **Dummy nodes**: handled in `_solve_product()` before calling `_solve_one_product()`
- **Validation** (optional): `_validate_with_linprog()` compares objective with `scipy.optimize.linprog` (HiGHS LP solver)
