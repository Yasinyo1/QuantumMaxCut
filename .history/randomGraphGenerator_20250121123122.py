import random
import numpy as np
import rustworkx as rx
def gen_graph(n, prob):
    rnd_graph = rx.PyGraph()
    rnd_graph.add_nodes_from(np.arange(0, n, 1))
    edges = []
    seen = set()

    for i in range(n - 1):
        edge = (i, i + 1, 1)
        edges.append(edge)
        seen.add((i, i + 1))
        seen.add((i + 1, i))

    for node_i in range(n):
        for node_j in range(n):
            if node_i == node_j or (node_i, node_j) in seen or (node_j, node_i) in seen:
                continue
            if random.uniform(0, 1) <= prob:
                edge = (node_i, node_j, 1)
                edges.append(edge)
                seen.add((node_i, node_j))
                seen.add((node_j, node_i))
    rnd_graph.add_edges_from(edges)
    return rnd_graph
