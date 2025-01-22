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
def setup_runtime():
    QiskitRuntimeService.save_account(channel="ibm_quantum", token="3819a09e3b88f9f4fb3ab15059854c1a460441c2bec015eec1e32282b85fce1815cf675efe943f0a0e29f8bb2820df3e7cebc3330c7e9b25c6c97f0c23c203ff", overwrite=True, set_as_default=True)
    service = QiskitRuntimeService(channel='ibm_quantum')
    backend = service.least_busy(min_num_qubits=127)
    return backend

import time

def run_quantum_implementation(graph, instance_name):
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
            estimator.options.default_shots = 1000
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = "XY4"

            result = minimize(
                cost_func_estimator,
                init_params,
                args=(candidate_circuit, cost_hamiltonian, estimator),
                method="COBYLA",
                tol=2e-1
            )
            result_file.write(f"Optimization Session details: {session.details()}\n")
            result_file.write(f"Optimization runtime usage: {session.details().get('usage_time')}s\n")
            result_file.write(f"Optimization result: {result}\n")

        optimized_circuit = candidate_circuit.assign_parameters(result.x)
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = 10000
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.num_randomizations = "auto"

        pub = (optimized_circuit,)
        job = sampler.run([pub], shots=int(1e4))
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

backend = setup_runtime()
instanceName = '50nodes_0.4prob_798edges'
graph = load_graph(f"{instanceName}.dot")
benchmark_graph(graph, instanceName)



