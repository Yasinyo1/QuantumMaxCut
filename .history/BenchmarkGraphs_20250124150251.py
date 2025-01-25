from typing import Sequence
import ClassicalSolver
import networkx as nx
import numpy as np
import rustworkx as rx
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from scipy.optimize import minimize
from qiskit_ibm_runtime import SamplerV2 as Sampler

def load_graph(filename):
    graph_nx = nx.drawing.nx_pydot.read_dot('instances/' + filename)
    graph = rx.networkx_converter(graph_nx)
    return graph


def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list.

    This function does the inverse of `build_max_cut_graph`
    """
    pauli_list = []
    for edge in list(graph.edge_list()):
        paulis = ["I"] * len(graph)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

        weight = 1

        pauli_list.append(("".join(paulis)[::-1], weight))

    return pauli_list


def get_cost_homiltonian(max_cut_paulis):
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
    return cost_hamiltonian


def build_circuit(cost_hamiltonian):
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=1)
    circuit.measure_all()
    # Create pass manager for transpilation
    pm = generate_preset_pass_manager(optimization_level=3,
                                      backend=backend)

    candidate_circuit = pm.run(circuit)
    return candidate_circuit

def cost_func_estimator(params, ansatz, hamiltonian, estimator):

    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()[0]
    cost = results.data.evs

    return cost

# auxiliary functions to sample most likely bitstring
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

def evaluate_sample(x: Sequence[int], graph: rx.PyGraph) -> float:
    assert len(x) == len(
        list(graph.nodes())), "The length of x must coincide with the number of nodes in the graph."
    return sum(x[u] * (1 - x[v]) + x[v] * (1 - x[u]) for u, v in list(graph.edge_list()))

def setup_runtime(backendName = None):
    QiskitRuntimeService.save_account(channel="ibm_quantum", token="e165ad9cc84324915879e521d7f577612e34ea976aa17022223b985ebded78f42a595994d0142f1519098e538fbc35ea41c65897701e4082d38af557ae1ca1f7", overwrite=True, set_as_default=True)
    service = QiskitRuntimeService(channel='ibm_quantum')
    backend = service.backend(name=backendName) if backendName else service.least_busy(min_num_qubits=127)
    return backend

import time


def optimize(graph,instance_name):
    pass

def run_quantum_implementation():
    
    instances = [10,30,50,70,90,110]
    edges = [18,32,120,290,259,798,482,1558,823,2577,1255,3902]

    for i in range(len(instances)):
        name_lo = str(instances[i]) + "nodes_"+ str(0.1) + "prob_" + str(edges[2*i]) + str("edges")
        name_hi = str(instances[i]) + "nodes_"+ str(0.4) + "prob_" + str(edges[2*i + 1]) + str("edges")

        graph_lo = load_graph(f"{name_lo}.dot")
        graph_hi = load_graph(f"{name_hi}.dot")
        
        optimize(graph_lo,name_lo)
        optimize(graph_hi,name_hi)

    
    
    
    with open(f"results/{instance_name}.txt", "a") as result_file:
        result_file.write(f"----- Quantum Implementation Results for {instance_name} -----\n")

        initial_gamma = np.pi
        initial_beta = np.pi / 2
        init_params = [initial_gamma, initial_beta]
        max_cut_paulist = build_max_cut_paulis(graph)
        cost_hamiltonian = get_cost_homiltonian(max_cut_paulist)
        candidate_circuit = build_circuit(cost_hamiltonian)


        with Session(backend=backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = 10
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = "XY4"
            

            result = minimize(
                cost_func_estimator,
                init_params,
                args=(candidate_circuit, cost_hamiltonian, estimator),
                method="COBYLA",
                tol=1e-1
            )
            result_file.write(f"Optimization Session details: {session.details()}\n")
            result_file.write(f"Optimization runtime usage: {session.details().get('usage_time')}s\n")
            result_file.write(f"Optimization result: {result}\n")

        optimized_circuit = candidate_circuit.assign_parameters([ 2.867e+00  ,1.430e+00]) #(result.x)
        with Session(backend=backend) as session:
            sampler = Sampler(mode=session)
            sampler.options.default_shots = 10
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.num_randomizations = "auto"

            pub = (optimized_circuit,)
            job = sampler.run([pub], shots=10)
            counts_int = job.result()[0].data.meas.get_int_counts()

            final_distribution_int = {key: val / sum(counts_int.values()) for key, val in counts_int.items()}
            most_likely = max(final_distribution_int, key=final_distribution_int.get)
            most_likely_bitstring = to_bitstring(most_likely, len(graph))
            most_likely_bitstring.reverse()
            cut_value = evaluate_sample(most_likely_bitstring, graph)

            result_file.write(f"Quantum Max Cut runtime usage: {job.usage()}s\n")
            result_file.write(f"Final bitstring distribution: {final_distribution_int}\n")
            result_file.write(f"Most likely bitstring: {most_likely_bitstring}\n")
            result_file.write(f"The value of the cut is: {cut_value}\n")


def run_classical_implementation(graph, instance_name):
    with open(f"results/{instance_name}.txt", "a") as result_file:
        result_file.write(f"\n----- Classical Implementation Results for {instance_name} -----\n")

        start_time = time.time()
        optimal_value, partition_vector, max_cut_value = ClassicalSolver.run_classical_implementation(graph)
        runtime = time.time() - start_time

        result_file.write(f"Runtime: {runtime:.2f} seconds\n")
        result_file.write(f"Solution bitstring: {partition_vector}\n")
        result_file.write(f"The value of the cut is: {max_cut_value}\n")


def benchmark_graph(graph, instance_name):
    print(f"Running Quantum Maximum Cut algorithm on instance {instance_name}")
    run_quantum_implementation(graph, instance_name)

    print(f"Running Classical Maximum Cut algorithm on instance {instance_name}")
    run_classical_implementation(graph, instance_name)

backend = setup_runtime('ibm_kyiv')

run_quantum_implementation()

# instanceName = '70nodes_0.1prob_482edges'
# graph = load_graph(f"{instanceName}.dot")
# benchmark_graph(graph, instanceName)




