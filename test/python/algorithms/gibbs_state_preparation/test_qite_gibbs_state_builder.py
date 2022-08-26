# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests QiteGibbsStateBuilder class."""
import unittest

import numpy as np

from qiskit import BasicAer
from qiskit.algorithms import VarQITE
from qiskit.algorithms.evolvers.variational import ImaginaryMcLachlanPrinciple
from qiskit.algorithms.gibbs_state_preparation.varqite_gibbs_state_builder import (
    VarQiteGibbsStateBuilder,
)
from qiskit.utils import QuantumInstance, algorithm_globals
from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import X


class TestQiteGibbsStateBuilder(QiskitAlgorithmsTestCase):
    """Tests QiteGibbsStateBuilder class."""

    def setUp(self):
        super().setUp()
        self.seed = 11
        np.random.seed(self.seed)
        backend_statevector = BasicAer.get_backend("statevector_simulator")
        backend_qasm = BasicAer.get_backend("qasm_simulator")
        self.quantum_instance = QuantumInstance(
            backend=backend_statevector,
            shots=1,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        self.quantum_instance_qasm = QuantumInstance(
            backend=backend_qasm,
            shots=40,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        self.backends_dict = {
            "qi_qasm": self.quantum_instance_qasm,
            "b_sv": backend_statevector,
            "qi_sv": self.quantum_instance,
        }

        self.backends_names = ["qi_qasm", "b_sv", "qi_sv"]

    def test_build(self):
        """Build test."""

        ansatz = EfficientSU2(2)
        init_param_values = np.zeros(len(ansatz.ordered_parameters))
        param_dict = dict(zip(ansatz.parameters, init_param_values))

        hamiltonian = X
        temperature = 12

        var_princip = ImaginaryMcLachlanPrinciple()

        # no backend given so matrix multiplication will be used
        qite_algorithm = VarQITE(ansatz, var_princip, init_param_values, num_timesteps=5)

        expected_parameter_values = [
            2.48553698605043e-17,
            -0.392553646634161,
            4.80865003568788e-17,
            6.61879494105776e-17,
            4.23399339414617e-17,
            -0.392553646634161,
            2.05198219009111e-17,
            8.40731618835306e-17,
            -2.73540840610989e-17,
            -0.392553646634161,
            3.02199171321979e-17,
            1.26270155363491e-16,
            3.6188537047339e-17,
            -0.392553646634161,
            3.57368105999006e-17,
            1.94753553277173e-16,
        ]

        for backend_name in self.backends_names:
            with self.subTest(msg=f"Test {backend_name} backend."):
                algorithm_globals.random_seed = self.seed
                backend = self.backends_dict[backend_name]

                gibbs_state_builder = VarQiteGibbsStateBuilder(qite_algorithm, backend)
                gibbs_state = gibbs_state_builder.build(hamiltonian, temperature, param_dict)
                parameter_values = gibbs_state.gibbs_state_function.evolved_state.data[0][0].params

                expected_aux_registers = {1}
                expected_aux_ops_evaluated = None
                expected_hamiltonian = hamiltonian
                expected_temperature = temperature

                np.testing.assert_equal(expected_aux_registers, gibbs_state.aux_registers)
                np.testing.assert_equal(
                    expected_aux_ops_evaluated, gibbs_state.gibbs_state_function.aux_ops_evaluated
                )

                self._assert_parameter_vals(expected_parameter_values, parameter_values)
                np.testing.assert_equal(expected_hamiltonian, gibbs_state.hamiltonian)
                np.testing.assert_equal(expected_temperature, gibbs_state.temperature)

    @staticmethod
    def _assert_parameter_vals(expected_parameter_values, parameter_values):
        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(
                float(parameter_value), expected_parameter_values[i], decimal=3
            )


if __name__ == "__main__":
    unittest.main()
