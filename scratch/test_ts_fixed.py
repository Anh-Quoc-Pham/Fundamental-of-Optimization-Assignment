import pandas as pd, numpy as np, sys
sys.path.insert(0,'d:/inventory_optimization')
from src.engine.transportation_simplex import _solve_one_product, UnionFind

def solve_fixed(supply, demand, cost, max_iterations=10000, tolerance=1e-9):
    m, n = len(supply), len(demand)
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
        if sup[i] <= tolerance or dem[j] <= tolerance: # FIXED HERE
            continue
        amt = min(sup[i], dem[j])
        X[i, j] += amt
        sup[i] -= amt
        dem[j] -= amt
        basis.add((i, j))
        uf.union(i, m + j)

    target = m + n - 1
    if len(basis) < target:
        for _, i, j in cells:
            if (i, j) not in basis:
                if uf.union(i, m + j):
                    basis.add((i, j))
                if len(basis) == target:
                    break
                    
    # test potentials computation
    from src.engine.transportation_simplex import _compute_potentials, _find_loop
    status = "suboptimal"
    itr = 0
    for itr in range(1, max_iterations + 1):
        basis_mask = np.zeros((m, n), dtype=bool)
        for (i, j) in basis:
            basis_mask[i, j] = True
        u, v = _compute_potentials(basis_mask, cost)
        
        best_r = -tolerance
        enter_i, enter_j = -1, -1
        for i in range(m):
            for j in range(n):
                if basis_mask[i, j]: continue
                r = cost[i, j] - u[i] - v[j]
                if r < best_r:
                    best_r = r
                    enter_i, enter_j = i, j
        if enter_i == -1:
            status = "optimal"
            break
            
        loop = _find_loop(basis, enter_i, enter_j, m, n)
        if not loop:
            print("LOOP FINDER FAILED!")
            break
            
        signs = [1 if k % 2 == 0 else -1 for k in range(len(loop))]
        minus_cells = [loop[k] for k in range(len(loop)) if signs[k] == -1]
        theta = min(X[i, j] for (i, j) in minus_cells)
        for k, (i, j) in enumerate(loop):
            X[i, j] += signs[k] * theta
            if X[i, j] < tolerance: X[i, j] = 0.0
            
        leaving = min(((i, j) for (i, j) in minus_cells if X[i, j] == 0.0), key=lambda cell: cell, default=None)
        if leaving and leaving in basis:
            basis.remove(leaving)
        basis.add((enter_i, enter_j))
    else:
        status = "max_iterations"
    return {"X": X, "status": status, "iterations": itr, "basis": basis}

excess = pd.read_csv('d:/inventory_optimization/results/excess_inventory.csv')
needed = pd.read_csv('d:/inventory_optimization/results/needed_inventory.csv')
cost_mat = pd.read_csv('d:/inventory_optimization/data/transport_cost_matrix.csv', index_col=0)
cost_mat.index = cost_mat.index.astype(int)
cost_mat.columns = cost_mat.columns.astype(int)

for p in [24, 19]:
    e = excess[excess.product_id==p]
    n = needed[needed.product_id==p]
    src_ids = e['store_id'].tolist()
    snk_ids = n['store_id'].tolist()
    supply = e['excess_units'].values.astype(float)
    demand = n['needed_units'].values.astype(float)
    m_r, n_r = len(src_ids), len(snk_ids)
    cost_raw = np.full((m_r, n_r), np.nan)
    for ii, si in enumerate(src_ids):
        for jj, dj in enumerate(snk_ids):
            if si != dj and si in cost_mat.index and dj in cost_mat.columns:
                c = float(cost_mat.loc[si, dj])
                if not np.isnan(c) and c > 0:
                    cost_raw[ii, jj] = c
    BIG_M = float(np.nanmax(cost_raw)) * (supply.sum() + demand.sum() + 1) + 1
    cost_w = np.hstack([np.where(np.isnan(cost_raw), BIG_M, cost_raw), np.zeros((m_r,1))])
    d_full = np.append(demand, supply.sum()-demand.sum())
    sol = solve_fixed(supply, d_full, cost_w, max_iterations=50)
    print('Product', p, 'status=', sol['status'], 'itr=', sol['iterations'])
    print('Final basis size:', len(sol['basis']), 'target:', m_r + len(d_full) - 1)
