"""
Transportation Simplex Optimizer
=================================
Implements the Transportation Simplex Method — a specialisation of the
Simplex algorithm for the classical balanced transportation problem.

For each product p independently:
  - Sources  : stores with excess inventory  (supply = excess_units)
  - Sinks    : stores with needed inventory  (demand = needed_units)
  - Cost     : transport_cost_matrix[i, j]   (VND per unit)
  - Variable : x_ij = units transferred from source i to sink j

Steps:
  1. Balance supply/demand with a dummy node if needed.
  2. Initial Basic Feasible Solution via Least-Cost Method.
  3. Compute potentials (u_i, v_j) for basis cells.
  4. Compute reduced costs r_ij = c_ij - u_i - v_j.
  5. If all r_ij >= 0: optimal.  Else select most-negative cell as entering.
  6. Find closed loop (cycle) through basis cells.
  7. Determine theta = min allocation on '-' cells of the loop.
  8. Pivot: add/subtract theta along the loop, remove a '-' cell from basis.
  9. Repeat until optimal or max_iterations reached.

Relationship to general Simplex:
  - Basis       <-> m+n-1 basic cells
  - Reduced cost <-> simplex multiplier (cj - zj)
  - Pivot        <-> loop adjustment (Gauss-Jordan step on the network)
  - Optimality   <-> all reduced costs >= 0
"""

import sys
import time
from pathlib import Path

if __name__ == "__main__" or "src.engine" not in sys.modules:
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.utils.logger import get_optimization_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SENTINEL_DOS = 9999   # sentinel DoS for zero-sales items (see analyzer.py)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _least_cost_bfs(supply: np.ndarray, demand: np.ndarray,
                    cost: np.ndarray) -> np.ndarray:
    """
    Least-Cost Method to find an initial Basic Feasible Solution.

    Returns allocation matrix X with shape (m, n).
    The basis will have exactly m+n-1 allocated cells (degeneracy handled
    by adding epsilon allocations where needed).
    """
    m, n = len(supply), len(demand)
    X = np.zeros((m, n))
    sup = supply.copy().astype(float)
    dem = demand.copy().astype(float)

    # Build list of (cost, i, j) sorted ascending
    cells = sorted(
        [(cost[i, j], i, j) for i in range(m) for j in range(n)],
        key=lambda t: t[0]
    )

    allocated = set()
    for _, i, j in cells:
        if sup[i] <= 0 or dem[j] <= 0:
            continue
        amt = min(sup[i], dem[j])
        X[i, j] += amt
        sup[i] -= amt
        dem[j] -= amt
        allocated.add((i, j))

    # Degeneracy repair: need exactly m+n-1 basic cells
    basis_needed = m + n - 1
    if len(allocated) < basis_needed:
        eps = 0.0   # zero-allocation basis cells (epsilon technique)
        for i in range(m):
            for j in range(n):
                if (i, j) not in allocated and len(allocated) < basis_needed:
                    # Only add if it does not complete a cycle with existing basis
                    # (simple check: add greedily; _find_loop will detect issues)
                    X[i, j] = eps
                    allocated.add((i, j))
                    if len(allocated) == basis_needed:
                        break
            if len(allocated) == basis_needed:
                break

    return X


def _compute_potentials(basis_mask: np.ndarray,
                        cost: np.ndarray) -> tuple:
    """
    Compute row potentials u and column potentials v from the optimality
    condition:  u[i] + v[j] = cost[i,j]  for all basis cells (i,j).

    Sets u[0] = 0 and propagates via BFS over basis cells.
    Returns (u, v) as numpy arrays of shape (m,) and (n,).
    """
    m, n = cost.shape
    u = np.full(m, np.nan)
    v = np.full(n, np.nan)
    u[0] = 0.0

    changed = True
    while changed:
        changed = False
        for i in range(m):
            for j in range(n):
                if not basis_mask[i, j]:
                    continue
                if not np.isnan(u[i]) and np.isnan(v[j]):
                    v[j] = cost[i, j] - u[i]
                    changed = True
                elif not np.isnan(v[j]) and np.isnan(u[i]):
                    u[i] = cost[i, j] - v[j]
                    changed = True

    # If any potential still NaN (disconnected basis — rare), set to 0
    u = np.where(np.isnan(u), 0.0, u)
    v = np.where(np.isnan(v), 0.0, v)
    return u, v


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

def _find_loop(basis: set, enter_i: int, enter_j: int, m: int, n: int):
    from collections import defaultdict
    adj = defaultdict(list)
    for (i, j) in basis:
        adj[i].append(m + j)
        adj[m + j].append(i)
        
    start = enter_i
    target = m + enter_j
    
    queue = [[start]]
    visited = {start}
    path = None
    
    while queue:
        curr = queue.pop(0)
        u = curr[-1]
        if u == target:
            path = curr
            break
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                queue.append(curr + [v])
                
    if not path:
        return None
        
    loop = [(enter_i, enter_j)]
    for idx in range(len(path) - 1):
        u = path[idx]
        v = path[idx+1]
        if u < m:
            cell = (u, v - m)
        else:
            cell = (v, u - m)
        loop.append(cell)
        
    return loop

def _solve_one_product(supply: np.ndarray, demand: np.ndarray,
                       cost: np.ndarray,
                       max_iterations: int = 10000,
                       tolerance: float = 1e-9) -> dict:
    m, n = len(supply), len(demand)
    assert abs(supply.sum() - demand.sum()) < 1.0, "Supply/demand not balanced"

    X = np.zeros((m, n), dtype=float)
    sup = supply.copy().astype(float)
    dem = demand.copy().astype(float)

    cells = sorted(
        [(cost[i, j], i, j) for i in range(m) for j in range(n)],
        key=lambda t: t[0]
    )

    basis = set()
    uf = UnionFind(m + n)
    for _, i, j in cells:
        if sup[i] <= tolerance or dem[j] <= tolerance:
            continue
        amt = min(sup[i], dem[j])
        X[i, j] += amt
        sup[i] -= amt
        dem[j] -= amt
        basis.add((i, j))
        uf.union(i, m + j)

    # Repair basis to ensure exactly m+n-1 cells (spanning tree)
    target = m + n - 1
    if len(basis) < target:
        for _, i, j in cells:
            if (i, j) not in basis:
                if uf.union(i, m + j):
                    basis.add((i, j))
                if len(basis) == target:
                    break

    status = "suboptimal"
    itr = 0

    for itr in range(1, max_iterations + 1):
        basis_mask = np.zeros((m, n), dtype=bool)
        for (i, j) in basis:
            basis_mask[i, j] = True

        u, v = _compute_potentials(basis_mask, cost)

        best_r = -tolerance
        enter_i, enter_j = -1, -1
        # Tie-break: lexicographic smallest
        for i in range(m):
            for j in range(n):
                if basis_mask[i, j]:
                    continue
                r = cost[i, j] - u[i] - v[j]
                if r < best_r:
                    best_r = r
                    enter_i, enter_j = i, j

        if enter_i == -1:
            status = "optimal"
            break

        loop = _find_loop(basis, enter_i, enter_j, m, n)
        if not loop:
            # Should mathematically never happen if basis is a spanning tree
            cost = cost.copy()
            cost[enter_i, enter_j] += 1e15
            continue

        signs = [1 if k % 2 == 0 else -1 for k in range(len(loop))]
        minus_cells = [loop[k] for k in range(len(loop)) if signs[k] == -1]
        
        theta = min(X[i, j] for (i, j) in minus_cells)

        for k, (i, j) in enumerate(loop):
            X[i, j] += signs[k] * theta
            if X[i, j] < tolerance:
                X[i, j] = 0.0

        # Lexicographic tie-break for leaving cell
        leaving = min(
            ((i, j) for (i, j) in minus_cells if X[i, j] == 0.0),
            key=lambda cell: cell,
            default=None,
        )
        if leaving and leaving in basis:
            basis.remove(leaving)
        basis.add((enter_i, enter_j))

    else:
        status = "max_iterations"

    objective = float(np.sum(X * cost))
    return {"X": X, "status": status, "iterations": itr, "objective": objective, "basis": basis}



# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class TransportationSimplexOptimizer:
    """
    Inventory transfer optimizer using the Transportation Simplex Method.

    For each product independently, formulates and solves a balanced
    transportation LP using the Transportation Simplex algorithm — a
    specialised version of the Simplex method for transportation networks.

    Interface matches RuleBasedOptimizer and GeneticAlgorithmOptimizer.
    """

    def __init__(
        self,
        distance_matrix=None,
        transport_cost_matrix=None,
        max_iterations: int = 10000,
        tolerance: float = 1e-9,
    ):
        self.distance_matrix = distance_matrix
        self.transport_cost_matrix = transport_cost_matrix
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.transfer_plan: pd.DataFrame = pd.DataFrame()
        self.solver_stats: dict = {}
        self.logger_system = get_optimization_logger()

    # ------------------------------------------------------------------
    def load_matrices(self, distance_path: str, cost_path: str):
        """Load distance and transport cost matrices from CSV files."""
        self.distance_matrix = pd.read_csv(distance_path, index_col=0)
        self.transport_cost_matrix = pd.read_csv(cost_path, index_col=0)
        self.distance_matrix.index = self.distance_matrix.index.astype(int)
        self.distance_matrix.columns = self.distance_matrix.columns.astype(int)
        self.transport_cost_matrix.index = self.transport_cost_matrix.index.astype(int)
        self.transport_cost_matrix.columns = self.transport_cost_matrix.columns.astype(int)
        print("Transportation Simplex: distance and cost matrices loaded.")

    # ------------------------------------------------------------------
    def optimize(
        self,
        excess_inventory: pd.DataFrame,
        needed_inventory: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run Transportation Simplex for each product and return a transfer plan.

        Parameters
        ----------
        excess_inventory : DataFrame with columns store_id, product_id, excess_units
        needed_inventory : DataFrame with columns store_id, product_id, needed_units

        Returns
        -------
        DataFrame with columns:
            from_store_id, to_store_id, product_id, units,
            distance_km, transport_cost
        """
        start_time = time.time()
        params = {
            "excess_items": len(excess_inventory),
            "needed_items": len(needed_inventory),
            "algorithm": "Transportation Simplex",
        }
        self.logger_system.log_execution_start("transportation_simplex", params)
        print("\nStarting Transportation Simplex Optimization...")

        if excess_inventory.empty or needed_inventory.empty:
            print("No excess or needed inventory. No transfers needed.")
            self.transfer_plan = pd.DataFrame()
            return self.transfer_plan

        # Find products with both supply and demand
        excess_products = set(excess_inventory["product_id"].unique())
        needed_products = set(needed_inventory["product_id"].unique())
        valid_products = list(excess_products & needed_products)

        if not valid_products:
            print("No products with both excess and shortage.")
            self.transfer_plan = pd.DataFrame()
            return self.transfer_plan

        print(f"Found {len(valid_products)} products to optimise.")

        all_transfers = []
        per_product_stats = []

        for product_id in valid_products:
            result = self._solve_product(
                product_id, excess_inventory, needed_inventory
            )
            all_transfers.extend(result["transfers"])
            per_product_stats.append({
                "product_id": product_id,
                "status": result["status"],
                "iterations": result["iterations"],
                "objective": result["objective"],
                "covered_units": result["covered_units"],
                "unmet_units": result["unmet_units"],
            })

        self.transfer_plan = pd.DataFrame(all_transfers) if all_transfers else pd.DataFrame()
        self.solver_stats = {
            "per_product": per_product_stats,
            "total_products_solved": len(per_product_stats),
        }

        execution_time = time.time() - start_time

        if not self.transfer_plan.empty:
            total_units = self.transfer_plan["units"].sum()
            total_cost = self.transfer_plan["transport_cost"].sum()
            avg_cost = total_cost / total_units if total_units > 0 else 0
            total_unmet = sum(s["unmet_units"] for s in per_product_stats)

            print(f"\nTransportation Simplex Results:")
            print(f"   • Total transfers      : {len(self.transfer_plan)}")
            print(f"   • Total units moved    : {total_units:,}")
            print(f"   • Unmet units          : {total_unmet:,}")
            print(f"   • Total transport cost : {total_cost:,.0f} VND")
            print(f"   • Avg cost / unit      : {avg_cost:,.0f} VND")
            print(f"   • Execution time       : {execution_time:.2f} s")

            results = {
                "transfers_generated": len(self.transfer_plan),
                "total_units": int(total_units),
                "total_cost": float(total_cost),
                "avg_cost_per_unit": float(avg_cost),
                "unmet_units": int(total_unmet),
                "execution_time_seconds": round(execution_time, 3),
            }
        else:
            print("No transfers generated.")
            results = {"transfers_generated": 0}

        self.logger_system.log_execution_end(
            "transportation_simplex", execution_time, results
        )
        return self.transfer_plan

    # ------------------------------------------------------------------
    def _solve_product(
        self,
        product_id,
        excess_df: pd.DataFrame,
        needed_df: pd.DataFrame,
    ) -> dict:
        """
        Solve the Transportation Simplex for a single product.

        Returns dict with: transfers, status, iterations, objective,
                           covered_units, unmet_units.
        """
        # Extract sources and sinks for this product
        src_df = excess_df[excess_df["product_id"] == product_id].copy()
        snk_df = needed_df[needed_df["product_id"] == product_id].copy()

        src_ids = src_df["store_id"].tolist()
        snk_ids = snk_df["store_id"].tolist()
        supply_raw = src_df["excess_units"].values.astype(float)
        demand_raw = snk_df["needed_units"].values.astype(float)

        m, n = len(src_ids), len(snk_ids)

        # --- Build cost matrix, filter invalid routes ---
        cost_raw = np.full((m, n), np.nan)
        for ii, si in enumerate(src_ids):
            for jj, dj in enumerate(snk_ids):
                if si == dj:
                    continue  # no self-transfer
                if (
                    self.transport_cost_matrix is not None
                    and si in self.transport_cost_matrix.index
                    and dj in self.transport_cost_matrix.columns
                ):
                    c = float(self.transport_cost_matrix.loc[si, dj])
                    if not np.isnan(c) and c > 0:
                        cost_raw[ii, jj] = c

        # Check if at least one valid route exists
        if np.all(np.isnan(cost_raw)):
            print(f"  [WARN] Product {product_id}: no valid routes — skipping.")
            return {
                "transfers": [], "status": "no_routes", "iterations": 0,
                "objective": 0, "covered_units": 0,
                "unmet_units": int(demand_raw.sum()),
            }

        # Replace NaN routes with a large penalty (effectively block them)
        BIG_M = float(np.nanmax(cost_raw)) * (supply_raw.sum() + demand_raw.sum() + 1) + 1
        cost_working = np.where(np.isnan(cost_raw), BIG_M, cost_raw)

        total_supply = supply_raw.sum()
        total_demand = demand_raw.sum()

        # --- Balance with dummy node ---
        has_dummy_demand = False
        has_dummy_supply = False
        supply = supply_raw.copy()
        demand = demand_raw.copy()

        if total_supply > total_demand + self.tolerance:
            # Add dummy demand node: absorbs surplus, cost = 0
            dummy_col_cost = np.zeros((m, 1))
            cost_working = np.hstack([cost_working, dummy_col_cost])
            demand = np.append(demand, total_supply - total_demand)
            has_dummy_demand = True

        elif total_demand > total_supply + self.tolerance:
            # Add dummy supply node: represents unmet demand, cost = BIG_M
            dummy_row_cost = np.full((1, cost_working.shape[1]), BIG_M)
            cost_working = np.vstack([cost_working, dummy_row_cost])
            supply = np.append(supply, total_demand - total_supply)
            has_dummy_supply = True

        # --- Solve via Transportation Simplex ---
        sol = _solve_one_product(
            supply, demand, cost_working,
            self.max_iterations, self.tolerance
        )
        X = sol["X"]

        # --- Extract real transfers (exclude dummy rows/cols) ---
        # Use X allocations from TS directly.
        # Apply a demand-cap safety guard for any numerical overflow cases.
        remaining_demand = demand_raw.copy()
        transfers = []
        covered_units = 0
        unmet_units = 0

        for ii in range(m):
            for jj in range(n):
                amt = X[ii, jj]
                if amt < 0.5:
                    continue
                si = src_ids[ii]
                dj = snk_ids[jj]
                c_val = cost_raw[ii, jj]
                if np.isnan(c_val) or c_val <= 0:
                    continue  # skip big-M / blocked routes

                # Safety cap: never send more than originally needed
                amt_int = min(int(round(amt)), int(remaining_demand[jj]))
                if amt_int <= 0:
                    continue

                dist_km = 0.0
                if (
                    self.distance_matrix is not None
                    and si in self.distance_matrix.index
                    and dj in self.distance_matrix.columns
                ):
                    dist_km = float(self.distance_matrix.loc[si, dj])

                transport_cost = c_val * amt_int
                transfers.append({
                    "from_store_id": si,
                    "to_store_id": dj,
                    "product_id": product_id,
                    "units": amt_int,
                    "distance_km": dist_km,
                    "transport_cost": transport_cost,
                })
                covered_units += amt_int
                remaining_demand[jj] -= amt_int

        # Compute unmet units = demand not satisfied by real sources
        if has_dummy_supply:
            dummy_row = X[m, :n]   # last row, real columns
            unmet_units = int(round(dummy_row.sum()))
        unmet_units = max(unmet_units, int(demand_raw.sum()) - covered_units)

        return {
            "transfers": transfers,
            "status": sol["status"],
            "iterations": sol["iterations"],
            "objective": sol["objective"],
            "covered_units": covered_units,
            "unmet_units": max(0, unmet_units),
        }

    # ------------------------------------------------------------------
    def add_store_product_names(self, stores_df=None, products_df=None):
        """Add human-readable store/product names to the transfer plan."""
        if self.transfer_plan is None or self.transfer_plan.empty:
            return
        if stores_df is not None:
            store_map = stores_df.set_index("store_id")["store_name"].to_dict()
            self.transfer_plan["from_store"] = self.transfer_plan["from_store_id"].map(store_map)
            self.transfer_plan["to_store"] = self.transfer_plan["to_store_id"].map(store_map)
        if products_df is not None:
            prod_map = products_df.set_index("product_id")["product_name"].to_dict()
            self.transfer_plan["product"] = self.transfer_plan["product_id"].map(prod_map)

    # ------------------------------------------------------------------
    def _validate_with_linprog(
        self,
        excess_inventory: pd.DataFrame,
        needed_inventory: pd.DataFrame,
        tol_pct: float = 1.0,
    ):
        """
        VALIDATION ONLY — not the main solver.

        Uses scipy.optimize.linprog to solve the same problem and compares
        the objective value to the Transportation Simplex result.
        Emits a warning if the difference exceeds tol_pct percent.

        This function is optional and for academic verification only.
        """
        try:
            from scipy.optimize import linprog
        except ImportError:
            print("[Validation] scipy not available — skipping linprog check.")
            return

        if self.transfer_plan is None or self.transfer_plan.empty:
            return

        ts_cost = self.transfer_plan["transport_cost"].sum()

        # Build a single joint LP across all products (simplified)
        excess_products = set(excess_inventory["product_id"].unique())
        needed_products = set(needed_inventory["product_id"].unique())
        valid_products = list(excess_products & needed_products)

        lp_cost_total = 0.0
        for product_id in valid_products:
            src_df = excess_inventory[excess_inventory["product_id"] == product_id]
            snk_df = needed_inventory[needed_inventory["product_id"] == product_id]
            src_ids = src_df["store_id"].tolist()
            snk_ids = snk_df["store_id"].tolist()
            supply = src_df["excess_units"].values.astype(float)
            demand = snk_df["needed_units"].values.astype(float)
            m, n = len(src_ids), len(snk_ids)

            c_flat = []
            for si in src_ids:
                for dj in snk_ids:
                    if (si == dj
                            or si not in self.transport_cost_matrix.index
                            or dj not in self.transport_cost_matrix.columns):
                        c_flat.append(1e12)
                        continue
                    cv = float(self.transport_cost_matrix.loc[si, dj])
                    c_flat.append(cv if (not np.isnan(cv) and cv > 0) else 1e12)

            # Inequality constraints (≤) matching the actual problem:
            #   sum_j x_ij <= supply[i]   (source capacity)
            #   sum_i x_ij <= demand[j]   (demand cap)
            # EQUALITY for demand: sum_i x_ij = demand[j]  (force full coverage)
            A_ub, b_ub = [], []
            A_eq, b_eq = [], []

            # Supply upper-bound constraints
            for ii in range(m):
                row = [0.0] * (m * n)
                for jj in range(n):
                    row[ii * n + jj] = 1.0
                A_ub.append(row)
                b_ub.append(supply[ii])

            # Demand equality constraints (must satisfy all demand)
            for jj in range(n):
                row = [0.0] * (m * n)
                for ii in range(m):
                    row[ii * n + jj] = 1.0
                A_eq.append(row)
                b_eq.append(demand[jj])

            bounds = [(0, None)] * (m * n)
            res = linprog(
                c_flat,
                A_ub=A_ub if A_ub else None,
                b_ub=b_ub if b_ub else None,
                A_eq=A_eq if A_eq else None,
                b_eq=b_eq if b_eq else None,
                bounds=bounds,
                method="highs"
            )
            if res.success:
                lp_cost_total += res.fun

        diff_pct = abs(ts_cost - lp_cost_total) / max(lp_cost_total, 1) * 100
        print(f"\n[Validation] Transportation Simplex cost : {ts_cost:,.0f} VND")
        print(f"[Validation] scipy linprog (HiGHS LP)    : {lp_cost_total:,.0f} VND")
        print(f"[Validation] Difference                  : {diff_pct:.2f}%")
        if diff_pct > tol_pct:
            print(f"[Validation] WARNING: difference > {tol_pct}% — check Transportation Simplex.")
        else:
            print(f"[Validation] OK — Transportation Simplex matches LP reference within {tol_pct}%.")
