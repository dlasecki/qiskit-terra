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

"""Test Variational Quantum Imaginary Time Evolution algorithm."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import ddt, data
import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.algorithms.gradients import LinCombQFI, LinCombEstimatorGradient
from qiskit.primitives import Estimator
from qiskit.algorithms.time_evolvers.variational.var_qite import VarQITE
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.utils import algorithm_globals
from qiskit.algorithms import TimeEvolutionProblem
from qiskit.algorithms.time_evolvers.variational import (
    ImaginaryMcLachlanPrinciple,
)
from qiskit.circuit.library import EfficientSU2


@ddt
class TestVarQITE(QiskitAlgorithmsTestCase):
    """Test Variational Quantum Imaginary Time Evolution algorithm."""

    def setUp(self):
        super().setUp()
        self.seed = 11
        np.random.seed(self.seed)

    def test_run_d_1_with_aux_ops(self):
        """Test VarQITE for d = 1 and t = 1 with evaluating auxiliary operator and the Forward
        Euler solver.."""

        observable = SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        )
        aux_ops = [Pauli("XX"), Pauli("YZ")]
        d = 1
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(len(parameters)):
            init_param_values[i] = np.pi / 2
        init_param_values[0] = 1
        time = 1

        evolution_problem = TimeEvolutionProblem(observable, time, aux_operators=aux_ops)

        thetas_expected = [
            0.87984606025879,
            2.04681975664763,
            2.68980594039104,
            2.75915988512186,
            2.38796546567171,
            1.78144857115127,
            2.13109162826101,
            1.9259609596416,
        ]

        thetas_expected_shots = [
            0.866388421629894,
            2.00011242816716,
            2.66024902588707,
            2.74940256936693,
            2.35690200373631,
            1.76230783558349,
            2.04591301743054,
            1.9074429050726,
        ]

        with self.subTest(msg="Test exact backend."):
            algorithm_globals.random_seed = self.seed
            estimator = Estimator()
            qfi = LinCombQFI(estimator)
            gradient = LinCombEstimatorGradient(estimator)
            var_principle = ImaginaryMcLachlanPrinciple(qfi, gradient)

            var_qite = VarQITE(
                ansatz,
                var_principle,
                init_param_values,
                estimator,
                num_timesteps=25,
            )
            evolution_result = var_qite.evolve(evolution_problem)

            evolved_state = evolution_result.evolved_state
            aux_ops = evolution_result.aux_ops_evaluated

            parameter_values = evolved_state.data[0][0].params

            expected_aux_ops = (-0.2177982985749799, 0.2556790598588627)

            for i, parameter_value in enumerate(parameter_values):
                np.testing.assert_almost_equal(
                    float(parameter_value), thetas_expected[i], decimal=2
                )

            np.testing.assert_array_almost_equal(
                [result[0] for result in aux_ops], expected_aux_ops
            )

        with self.subTest(msg="Test shot-based backend."):
            algorithm_globals.random_seed = self.seed

            estimator = Estimator(options={"shots": 4096, "seed": self.seed})
            qfi = LinCombQFI(estimator)
            gradient = LinCombEstimatorGradient(estimator)
            var_principle = ImaginaryMcLachlanPrinciple(qfi, gradient)

            var_qite = VarQITE(
                ansatz,
                var_principle,
                init_param_values,
                estimator,
                num_timesteps=25,
            )
            evolution_result = var_qite.evolve(evolution_problem)

            evolved_state = evolution_result.evolved_state
            aux_ops = evolution_result.aux_ops_evaluated

            parameter_values = evolved_state.data[0][0].params

            expected_aux_ops = (-0.20541478709978211, 0.26583767857803875)

            for i, parameter_value in enumerate(parameter_values):
                np.testing.assert_almost_equal(
                    float(parameter_value), thetas_expected_shots[i], decimal=2
                )

            np.testing.assert_array_almost_equal(
                [result[0] for result in aux_ops], expected_aux_ops
            )

    def test_run_d_1_t_7(self):
        """Test VarQITE for d = 1 and t = 7 with RK45 ODE solver."""

        observable = SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        )

        d = 1
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(len(parameters)):
            init_param_values[i] = np.pi / 2
        init_param_values[0] = 1
        var_principle = ImaginaryMcLachlanPrinciple()

        time = 7
        var_qite = VarQITE(
            ansatz,
            var_principle,
            init_param_values,
            ode_solver="RK45",
            num_timesteps=25,
        )

        thetas_expected = [
            0.828917365718767,
            1.88481074798033,
            3.14111335991238,
            3.14125849601269,
            2.33768562678401,
            1.78670990729437,
            2.04214275514208,
            2.04009918594422,
        ]

        self._test_helper(observable, thetas_expected, time, var_qite, 2)

    @data(
        SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        ),
        PauliSumOp(
            SparsePauliOp.from_list(
                [
                    ("II", 0.2252),
                    ("ZZ", 0.5716),
                    ("IZ", 0.3435),
                    ("ZI", -0.4347),
                    ("YY", 0.091),
                    ("XX", 0.091),
                ]
            )
        ),
    )
    def test_run_d_2(self, observable):
        """Test VarQITE for d = 2 and t = 1 with RK45 ODE solver."""
        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(len(parameters)):
            init_param_values[i] = np.pi / 4

        var_principle = ImaginaryMcLachlanPrinciple()

        time = 1
        var_qite = VarQITE(
            ansatz,
            var_principle,
            init_param_values,
            ode_solver="RK45",
            num_timesteps=25,
        )

        thetas_expected = [
            1.29495364023786,
            1.08970061333559,
            0.667488228710748,
            0.500122687902944,
            1.4377736672043,
            1.22881086103085,
            0.729773048146251,
            1.01698854755226,
            0.050807780587492,
            0.294828474947149,
            0.839305697704923,
            0.663689581255428,
        ]

        self._test_helper(observable, thetas_expected, time, var_qite, 4)

    def _test_helper(self, observable, thetas_expected, time, var_qite, decimal):
        evolution_problem = TimeEvolutionProblem(observable, time)
        evolution_result = var_qite.evolve(evolution_problem)
        evolved_state = evolution_result.evolved_state

        parameter_values = evolved_state.data[0][0].params

        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(
                float(parameter_value), thetas_expected[i], decimal=decimal
            )


if __name__ == "__main__":
    unittest.main()
