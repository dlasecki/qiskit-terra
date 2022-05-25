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

"""Test solver of linear equations."""

import unittest
from functools import partial

from test.python.algorithms import QiskitAlgorithmsTestCase
from test.python.algorithms.evolvers.variational.solvers.expected_results.test_varqte_linear_solver_expected_1 import (
    expected_metric_res_1,
)
from ddt import ddt, data
import numpy as np

from qiskit import BasicAer
from qiskit.algorithms.evolvers.variational.variational_principles.imaginary_mc_lachlan_principle import (
    ImaginaryMcLachlanPrinciple,
)
from qiskit.algorithms.evolvers.variational.solvers.var_qte_linear_solver import (
    VarQTELinearSolver,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z, CircuitSampler


@ddt
class TestVarQTELinearSolver(QiskitAlgorithmsTestCase):
    """Test solver of linear equations."""

    @data(CircuitSampler(BasicAer.get_backend("statevector_simulator")), None)
    def test_solve_sle_no_backend(self, circuit_sampler):
        """Test SLE solver."""

        observable = SummedOp(
            [
                0.2252 * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        ).reduce()

        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = ansatz.ordered_parameters
        init_param_values = np.zeros(len(parameters))
        for i in range(ansatz.num_qubits):
            init_param_values[-(ansatz.num_qubits + i + 1)] = np.pi / 2

        param_dict = dict(zip(parameters, init_param_values))

        var_principle = ImaginaryMcLachlanPrinciple()

        metric_tensor = var_principle.calc_metric_tensor(ansatz, parameters)
        evolution_grad = var_principle.calc_evolution_grad(observable, ansatz, parameters)

        linear_solver = partial(np.linalg.lstsq, rcond=1e-2)
        linear_solver = VarQTELinearSolver(
            metric_tensor,
            evolution_grad,
            linear_solver,
            circuit_sampler,
        )

        nat_grad_res, metric_res, grad_res = linear_solver.solve_sle(param_dict)

        expected_nat_grad_res = [
            3.43500000e-01,
            -2.89800000e-01,
            2.43575264e-16,
            1.31792695e-16,
            -9.61200000e-01,
            -2.89800000e-01,
            1.27493709e-17,
            1.12587456e-16,
            3.43500000e-01,
            -2.89800000e-01,
            3.69914720e-17,
            1.95052083e-17,
        ]

        expected_grad_res = [
            (0.17174999999999926 - 0j),
            (-0.21735000000000085 + 0j),
            (4.114902862895087e-17 - 0j),
            (4.114902862895087e-17 - 0j),
            (-0.24030000000000012 + 0j),
            (-0.21735000000000085 + 0j),
            (4.114902862895087e-17 - 0j),
            (4.114902862895087e-17 - 0j),
            (0.17174999999999918 - 0j),
            (-0.21735000000000076 + 0j),
            (1.7789936190837538e-17 - 0j),
            (-8.319872568662832e-17 + 0j),
        ]

        np.testing.assert_array_almost_equal(nat_grad_res, expected_nat_grad_res, decimal=4)
        np.testing.assert_array_almost_equal(grad_res, expected_grad_res, decimal=4)
        np.testing.assert_array_almost_equal(metric_res, expected_metric_res_1, decimal=4)


if __name__ == "__main__":
    unittest.main()
