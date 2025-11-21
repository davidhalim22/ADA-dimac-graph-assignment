import heapq
import gzip
import time

class DisjointSet:
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}
    
    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]
    
    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        elif self.rank[xroot] > self.rank[yroot]:
            self.parent[yroot] = xroot
        else:
            self.parent[yroot] = xroot
            self.rank[xroot] += 1

def kruskal(graph):
    edges = []
    
    for u in graph:
        for w, v in graph[u]:
            edges.append((w, u, v))
    edges.sort()
    
    ds = DisjointSet(graph.keys())
    mst = []
    
    for w, u, v in edges:
        if ds.find(u) != ds.find(v):
            mst.append((u, v, w))
            ds.union(u, v)
    return mst

def prim(graph, start=0):
    mst = []
    visited = set()
    min_heap = [(0, start, -1)]
    
    while min_heap and len(visited) < len(graph):
        weight, u, from_vertex = heapq.heappop(min_heap)
        
        if u in visited:
            continue
        
        visited.add(u)
        
        if from_vertex != -1:
            mst.append((from_vertex, u, weight))
            
        for v, edge_weight in graph[u]:
            if v not in visited:
                heapq.heappush(min_heap, (edge_weight, v, u))
    
    return mst

def load_dimacs_gr(path):
    graph = {}

    with open(path, "r") as f:
        for line in f:
            if line.startswith("c") or line.startswith("p"):
                continue

            parts = line.split()
            if parts[0] == "a":
                u = int(parts[1])
                v = int(parts[2])
                w = float(parts[3])

                if u not in graph:
                    graph[u] = []
                if v not in graph:
                    graph[v] = []

                # Add undirected edges because Prim needs undirected graph
                graph[u].append((v, w))
                graph[v].append((w, u))

    return graph

def test_mst(graph, algorithm_func, start_node = None, runs = 1):
    results = []
    
    for i in range(runs):
        start_time = time.time()
        
        if start_node is not None:
            mst = algorithm_func(graph, start = start_node)
        else:
            mst = algorithm_func(graph)
        
        end_time = time.time()
        
        execution_time = end_time - start_time
        mst_weight = sum(edge[2] for edge in mst)
        mst_edges = len(mst)
        
        results.append({
            "time": execution_time,
            "weight": mst_weight,
            "edges": mst_edges,
            "mst": mst
        })
    
    # Check differences between runs
    differences = False
    if runs > 1:
        base = results[0]["mst"]
        for r in results[1:]:
            if r["mst"] != base:
                differences = True
                break
    
    return {
        "execution_times": [r["time"] for r in results],
        "average_time": sum(r["time"] for r in results) / runs,
        "mst_weight": results[0]["weight"],
        "num_edges": results[0]["edges"],
        "differences_between_runs": differences
    }

# graph = {
#     0: [(1, 2), (2, 4)],
#     1: [(0, 2), (3, 8)],
#     2: [(0, 4), (3, 7)],
#     3: [(1, 8), (2, 7)]
# }

# mst = prim(graph, 0)
# for u, v, weight in mst:
#     print(f"Edge: {u} -> {v}, Weight: {weight}")
    
    
    
san_francisco_graph = load_dimacs_gr("USA-road-d.BAY.gr")
new_york_graph = load_dimacs_gr("USA-road-d.NY.gr")
colorado_graph = load_dimacs_gr("USA-road-d.COL.gr")
florida_graph = load_dimacs_gr("USA-road-d.FLA.gr")

sf_prim = prim(san_francisco_graph, start=1)
ny_prim = prim(new_york_graph, start=1)
co_prim = prim(colorado_graph, start=1)
fl_prim = prim(florida_graph, start=1)

sf_kruskal = kruskal(san_francisco_graph)
ny_kruskal = kruskal(new_york_graph)
co_kruskal = kruskal(colorado_graph)
fl_kruskal = kruskal(florida_graph)

sf_prim_results = test_mst(san_francisco_graph, prim, start_node = 1, runs = 3)
ny_prim_results = test_mst(new_york_graph, prim, start_node = 1, runs = 3)
co_prim_results = test_mst(colorado_graph, prim, start_node = 1, runs = 3)
fl_prim_results = test_mst(florida_graph, prim, start_node = 1, runs = 3)

sf_kruskal_results = test_mst(san_francisco_graph, kruskal, runs = 3)
ny_kruskal_results = test_mst(new_york_graph, kruskal, runs = 3)
co_kruskal_results = test_mst(colorado_graph, kruskal, runs = 3)
fl_kruskal_results = test_mst(florida_graph, kruskal, runs = 3)




print(sf_prim_results)
# print(ny_prim_results)
# print(co_prim_results)
# print(fl_prim_results)

# print(sf_kruskal_results)
# print(ny_kruskal_results)
# print(co_kruskal_results)
# print(fl_kruskal_results)




# for u, v, w in mst:
#     print(f"{u} -> {v} (weight {w})")