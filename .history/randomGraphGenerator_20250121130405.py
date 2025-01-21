import random
import numpy as np
import rustworkx as rx
import networkx as nx

def gen_graph(n, prob):
    rnd_graph = rx.PyGraph()
    rnd_graph.add_nodes_from(np.arange(0, n, 1))
    edges = []
    seen = set()

    # Add edges to create a linear graph
    for i in range(n - 1):
        edge = (i, i + 1, 1)
        edges.append(edge)
        seen.add((i, i + 1))
        seen.add((i + 1, i))

    # Add random edges based on probability
    for node_i in range(n):
        for node_j in range(n):
            if node_i == node_j or (node_i, node_j) in seen or (node_j, node_i) in seen:
                continue
            if random.uniform(0, 1) < prob:
                edge = (node_i, node_j, 1)
                edges.append(edge)
                seen.add((node_i, node_j))
                seen.add((node_j, node_i))

    # Add edges to the graph
    rnd_graph.add_edges_from(edges)
    filename = f"instances/{n}nodes_{prob}prob_{len(edges)}edges.dot"
    # Save the graph to a file
    with open(filename, 'w') as file:
        file.write(rnd_graph.to_dot())

    # Load the graph from the file
    graph_nx = nx.drawing.nx_pydot.read_dot(filename)
    graph = rx.networkx_converter(graph_nx)

    return graph
