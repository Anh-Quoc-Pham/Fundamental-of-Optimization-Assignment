import pandas as pd, numpy as np, sys
sys.path.insert(0,'d:/inventory_optimization')
from src.engine.transportation_simplex import _solve_one_product

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
    sol = _solve_one_product(supply, d_full, cost_w, max_iterations=50)
    print('Product', p, 'status=', sol['status'], 'itr=', sol['iterations'])
    print('Final basis size:', len(sol['basis']), 'target:', m_r + len(d_full) - 1)
