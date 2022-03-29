# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test TrotterQRTE. """

import unittest
from test.python.opflow import QiskitOpflowTestCase
from math import sqrt
from typing import List
from ddt import ddt, data, unpack
import numpy as np
from numpy.testing import assert_raises
from scipy.linalg import expm

from qiskit import BasicAer, QuantumCircuit
from qiskit.algorithms import EvolutionProblem
from qiskit.algorithms.evolvers.real.trotterization.trotter_qrte import (
    TrotterQRTE,
)
from qiskit.quantum_info import Statevector, SparsePauliOp, Pauli, PauliTable
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.circuit import Parameter
from qiskit.opflow import (
    X,
    Z,
    Zero,
    VectorStateFn,
    StateFn,
    I,
    Y,
    MatrixExpectation,
    SummedOp,
)
from qiskit.synthesis import SuzukiTrotter, QDrift


@ddt
class TestTrotterQRTE(QiskitOpflowTestCase):
    """TrotterQRTE tests."""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        backend_statevector = BasicAer.get_backend("statevector_simulator")
        backend_qasm = BasicAer.get_backend("qasm_simulator")  # TODO add to tests
        self.quantum_instance = QuantumInstance(
            backend=backend_statevector,
            shots=1,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        self.quantum_instance_qasm = QuantumInstance(
            backend=backend_qasm,
            shots=4000,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        self.backends_dict = {
            "qi_sv": self.quantum_instance,
            "qi_qasm": self.quantum_instance_qasm,
            "b_sv": backend_statevector,
            "b_qasm": backend_qasm,
            "None": None,
        }

        self.backends_names = ["qi_qasm", "b_sv", "None", "b_qasm", "qi_sv"]
        self.backends_names_not_none = ["qi_sv", "b_sv"]

    def calculate_counts(self, statevector: VectorStateFn) -> List[float]:

        primitive = statevector.primitive
        coeff1 = primitive.data[0]
        coeff2 = primitive.data[1]

        sampled1 = sqrt(coeff1 * np.conjugate(coeff1))
        sampled2 = sqrt(coeff2 * np.conjugate(coeff2))

        return [sampled1, sampled2]

    # TODO add valid param binding Hamiltonian
    @data(
        (None, expm(-1j * Z.to_matrix()) @ expm(-1j * X.to_matrix())),
        (
            SuzukiTrotter(),
            expm(-1j * X.to_matrix() * 0.5)
            @ expm(-1j * Z.to_matrix())
            @ expm(-1j * X.to_matrix() * 0.5),
        ),
    )
    @unpack
    def test_trotter_qrte_trotter_single_qubit(self, product_formula, expected_state_part):
        """Test for default TrotterQRTE on a single qubit."""
        operator = SummedOp([X, Z])
        initial_state = StateFn([1, 0])
        time = 1
        evolution_problem = EvolutionProblem(operator, time, initial_state)
        # Calculate the expected state
        expected_state = expected_state_part @ initial_state.to_matrix()

        for backend_name in self.backends_names:
            with self.subTest(msg=f"Test {backend_name} backend."):
                backend = self.backends_dict[backend_name]
                expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2,)))

                trotter_qrte = TrotterQRTE(
                    quantum_instance=backend, product_formula=product_formula
                )
                evolution_result_state = trotter_qrte.evolve(evolution_problem).evolved_state

                if backend_name in {"qi_qasm", "b_qasm"}:
                    evolution_result_state = evolution_result_state.to_matrix()[0]
                    decimal = 1
                    expected_evolved_state = self.calculate_counts(expected_evolved_state)
                    np.testing.assert_almost_equal(
                        evolution_result_state, expected_evolved_state, decimal=decimal
                    )
                else:
                    np.testing.assert_equal(evolution_result_state, expected_evolved_state)

    # def test_trotter_qrte_trotter_single_qubit_aux_ops(self):
    #     """Test for default TrotterQRTE on a single qubit with auxiliary operators."""
    #     operator = SummedOp([X, Z])
    #     # LieTrotter with 1 rep
    #     aux_ops = [X, Y]
    #     expectation = MatrixExpectation()
    #
    #     initial_state = Zero
    #     time = 3
    #     evolution_problem = EvolutionProblem(operator, time, initial_state, aux_ops)
    #     # Calculate the expected state
    #     expected_state = (
    #         expm(-time * 1j * Z.to_matrix())
    #         @ expm(-time * 1j * X.to_matrix())
    #         @ initial_state.to_matrix()
    #     )
    #     expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2,)))
    #     expected_aux_ops_evaluated = [(0.078073, 0.0), (0.268286, 0.0)]
    #
    #     for backend_name in self.backends_names_not_none:
    #         with self.subTest(msg=f"Test {backend_name} backend."):
    #             backend = self.backends_dict[backend_name]
    #             trotter_qrte = TrotterQRTE(quantum_instance=backend, expectation=expectation)
    #             evolution_result = trotter_qrte.evolve(evolution_problem)
    #
    #             np.testing.assert_equal(evolution_result.evolved_state, expected_evolved_state)
    #             np.testing.assert_array_almost_equal(
    #                 evolution_result.aux_ops_evaluated, expected_aux_ops_evaluated
    #             )
    #
    # @data(
    #     SummedOp([(X ^ Y), (Y ^ X)]),
    #     (Z ^ Z) + (Z ^ I) + (I ^ Z),
    #     Y ^ Y,
    #     SparsePauliOp(Pauli("XI")),
    #     SparsePauliOp(PauliTable.from_labels(["XX", "ZZ"])),
    # )
    # def test_trotter_qrte_trotter_two_qubits(self, operator):
    #     """Test for TrotterQRTE on two qubits with various types of a Hamiltonian."""
    #     # LieTrotter with 1 rep
    #     initial_state = StateFn([1, 0, 0, 0])
    #     # Calculate the expected state
    #     expected_state = initial_state.to_matrix()
    #     expected_state = expm(-1j * operator.to_matrix()) @ expected_state
    #     expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2, 2)))
    #
    #     evolution_problem = EvolutionProblem(operator, 1, initial_state)
    #
    #     for backend_name in self.backends_names:
    #         with self.subTest(msg=f"Test {backend_name} backend."):
    #             backend = self.backends_dict[backend_name]
    #             trotter_qrte = TrotterQRTE(quantum_instance=backend)
    #             evolution_result = trotter_qrte.evolve(evolution_problem)
    #             np.testing.assert_equal(evolution_result.evolved_state, expected_evolved_state)
    #
    # @data(Zero, QuantumCircuit(1).compose(Pauli("Z").to_instruction(), [0]))
    # def test_trotter_qrte_qdrift_fractional_time(self, initial_state):
    #     """Test for TrotterQRTE with QDrift."""
    #     operator = SummedOp([X, Z])
    #     time = 1
    #     evolution_problem = EvolutionProblem(operator, time, initial_state)
    #     sampled_ops = [Z, X, X, X, Z, Z, Z, Z]
    #     evo_time = 0.25
    #     # Calculate the expected state
    #     if isinstance(initial_state, QuantumCircuit):
    #         initial_state = StateFn(initial_state)
    #     expected_state = initial_state.to_matrix()
    #     for op in sampled_ops:
    #         expected_state = expm(-1j * op.to_matrix() * evo_time) @ expected_state
    #     expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2,)))
    #
    #     for backend_name in self.backends_names:
    #         with self.subTest(msg=f"Test {backend_name} backend."):
    #             algorithm_globals.random_seed = 0
    #             backend = self.backends_dict[backend_name]
    #             trotter_qrte = TrotterQRTE(quantum_instance=backend, product_formula=QDrift())
    #             evolution_result = trotter_qrte.evolve(evolution_problem)
    #             np.testing.assert_equal(evolution_result.evolved_state, expected_evolved_state)
    #
    # @data((Parameter("t"), {}), (None, {Parameter("t"): 2}))
    # @unpack
    # def test_trotter_qrte_trotter_errors(self, t_param, hamiltonian_value_dict):
    #     """Test TrotterQRTE with raising errors."""
    #     operator = X * Parameter("t") + Z
    #     initial_state = Zero
    #     time = 1
    #     for backend_name in self.backends_names:
    #         with self.subTest(msg=f"Test {backend_name} backend."):
    #             algorithm_globals.random_seed = 0
    #             backend = self.backends_dict[backend_name]
    #             trotter_qrte = TrotterQRTE(quantum_instance=backend)
    #             with assert_raises(ValueError):
    #                 evolution_problem = EvolutionProblem(
    #                     operator,
    #                     time,
    #                     initial_state,
    #                     t_param=t_param,
    #                     hamiltonian_value_dict=hamiltonian_value_dict,
    #                 )
    #                 _ = trotter_qrte.evolve(evolution_problem)


if __name__ == "__main__":
    unittest.main()
