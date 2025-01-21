import rustworkx as rx
import cvxpy as cp
import numpy as np


def run_classical_implementation(graph):
    """
    Solves a max-cut problem using a semidefinite programming approach on a PyGraph.

    Args:
        graph (rx.PyGraph): The input graph.

    Returns:
        tuple: A tuple containing the optimal value, partition vector, and evaluated max cut value.
    """

    # Extract the number of nodes
    n = graph.num_nodes()

    edges = [v for u, v in graph.edge_index_map().items()]

    # Define the semidefinite variable
    X = cp.Variable((n, n), symmetric=True)

    # Define the constraints
    constraints = [X >> 0]  # X is positive semidefinite
    constraints += [X[i, i] == 1 for i in range(n)]  # Diagonal elements are 1

    # Define the objective function
    objective = cp.Maximize(
        sum(0.5 * (1 - X[i, j]) * 1 for (i, j, _) in edges)
    )

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Check the solution matrix
    eigenvalues, eigenvectors = np.linalg.eigh(X.value)
    embedding_matrix = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))

    # Random hyperplane rounding
    random_vector = np.random.randn(embedding_matrix.shape[1])
    partition_vector = np.sign(embedding_matrix @ random_vector)

    # Evaluate the max cut value
    max_cut_value = 0
    for u, v, _ in edges:
        if partition_vector[u] != partition_vector[v]:  # Edges crossing the cut
            max_cut_value += 1

    return prob.value, partition_vector, max_cut_value
