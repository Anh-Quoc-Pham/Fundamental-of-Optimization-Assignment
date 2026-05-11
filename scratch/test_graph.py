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

def _find_loop_graph(basis, enter_i, enter_j, m, n):
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
                
    if not path: return None
        
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

basis = {(0, 0), (0, 1), (1, 1), (1, 2)}
loop = _find_loop_graph(basis, 0, 2, 2, 3)
print("Loop cells:", loop)
