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

import unittest

import numpy as np

from qiskit.algorithms.quantum_time_evolution.variational.principles.real.implementations\
    .real_mc_lachlan_variational_principle import (
    RealMcLachlanVariationalPrinciple,
)
from qiskit import Aer
from qiskit.algorithms.quantum_time_evolution.variational.var_qrte import VarQrte
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import (
    SummedOp,
    X,
    Y,
    I,
    Z,
)
from test.python.algorithms import QiskitAlgorithmsTestCase


class TestVarQrte(QiskitAlgorithmsTestCase):
    pass
    # # TODO incorrect results
    # def test_run_d_1(self):
    #     observable = SummedOp(
    #         [
    #             0.2252 * (I ^ I),
    #             0.5716 * (Z ^ Z),
    #             0.3435 * (I ^ Z),
    #             -0.4347 * (Z ^ I),
    #             0.091 * (Y ^ Y),
    #             0.091 * (X ^ X),
    #         ]
    #     ).reduce()
    #
    #     d = 1
    #     ansatz = EfficientSU2(observable.num_qubits, reps=d)
    #
    #     parameters = ansatz.ordered_parameters
    #     init_param_values = np.zeros(len(ansatz.ordered_parameters))
    #     for i in range(len(ansatz.ordered_parameters)):
    #         init_param_values[i] = np.pi / 2
    #     init_param_values[0] = 1
    #     var_principle = RealMcLachlanVariationalPrinciple()
    #
    #     param_dict = dict(zip(parameters, init_param_values))
    #
    #     reg = None
    #     backend = Aer.get_backend("statevector_simulator")
    #
    #     var_qite = VarQrte(
    #         var_principle, regularization=reg, backend=backend, error_based_ode=False
    #     )
    #     time = 1
    #
    #     evolution_result = var_qite.evolve(
    #         observable,
    #         time,
    #         ansatz,  # ansatz is a state in this case
    #         hamiltonian_value_dict=param_dict,
    #     )
    #
    #     # values from the prototype
    #     thetas_expected = [
    #         0.372598111322136, 1.51900599789371, 2.80259002647779, 2.00206987650666,
    #         1.55578693792189, 2.68064238326861, 2.32464633347459, 1.10390724525525
    #     ]
    #
    #     parameter_values = evolution_result.data[0][0].params
    #     for i, parameter_value in enumerate(parameter_values):
    #         np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=3)


if __name__ == "__main__":
    unittest.main()
