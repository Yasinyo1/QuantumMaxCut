import numpy as np
import cvxpy as cp
import time 


def classical(n,edges):
    
    start = time.time()
    X = cp.Variable((n, n), symmetric=True)

    # Define the constraints
    constraints = [X >> 0]  # X is positive semidefinite
    constraints += [X[i, i] == 1 for i in range(n)]  # Diagonal elements are 1

    # Define the objective
    objective = cp.Maximize(
        sum(0.5 * (1 - X[i, j]) for (i, j,_) in edges)
    )

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Check the solution matrix
    print("Optimal value of X:")
    print(X.value)

    # Eigenvalue decomposition to get the embedding matrix
    eigenvalues, eigenvectors = np.linalg.eigh(X.value)
    embedding_matrix = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))

    # Random hyperplane rounding
    u = np.random.randn(n)  # Random vector
    x = np.sign(embedding_matrix @ u)  # Partition assignments
    end = time.time()

    # Print the results
    print("Partition vector (x):")
    print(x)
    print("Time taken: ", end-start)