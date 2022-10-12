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

from qiskit.algorithms import VarQITE
from qiskit.algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
from qiskit.algorithms.gibbs_state_preparation.varqite_gibbs_state_builder import (
    VarQiteGibbsStateBuilder,
)
from qiskit.primitives import Sampler
from qiskit.quantum_info import Pauli
from qiskit.utils import algorithm_globals
from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit.circuit.library import EfficientSU2


class TestQiteGibbsStateBuilder(QiskitAlgorithmsTestCase):
    """Tests QiteGibbsStateBuilder class."""

    def setUp(self):
        super().setUp()
        self.seed = 11
        np.random.seed(self.seed)

        self.sampler = Sampler()
        self.sampler_shots = Sampler(options={"seed": self.seed, "shots": 40})

        self.samplers_dict = {"sampler": self.sampler, "sampler_shots": self.sampler_shots}

        self.sampler_names = ["sampler", "sampler_shots"]

    def test_build(self):
        """Build test."""

        ansatz = EfficientSU2(2)
        init_param_values = np.zeros(len(ansatz.ordered_parameters))
        param_dict = dict(zip(ansatz.parameters, init_param_values))

        hamiltonian = Pauli("X")
        temperature = 12

        var_princip = ImaginaryMcLachlanPrinciple()

        # no sampler given so matrix multiplication will be used
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

        for sampler_name in self.sampler_names:
            with self.subTest(msg=f"Test {sampler_name} sampler."):
                algorithm_globals.random_seed = self.seed
                sampler = self.samplers_dict[sampler_name]

                gibbs_state_builder = VarQiteGibbsStateBuilder(sampler, qite_algorithm)
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
