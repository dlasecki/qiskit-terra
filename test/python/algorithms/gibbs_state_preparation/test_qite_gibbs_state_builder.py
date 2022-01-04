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
"""Tests QiteGibbsStateBuilder class."""
import unittest

from qiskit import Aer
from qiskit.quantum_info import Statevector
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np

from qiskit.algorithms.gibbs_state_preparation.qite_gibbs_state_builder import QiteGibbsStateBuilder
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import X


class TestQiteGibbsStateBuilder(QiskitAlgorithmsTestCase):
    """Tests QiteGibbsStateBuilder class."""

    # TODO finish when VarQite is available.
    # def test_build(self):
    #     """Build test."""
    #     qite_algorithm = None # TODO
    #     ansatz = EfficientSU2(2)
    #     init_param_values = np.zeros(len(ansatz.ordered_parameters))
    #     param_dict = dict(zip(ansatz.parameters, init_param_values))
    #     gibbs_state_builder = QiteGibbsStateBuilder(qite_algorithm, ansatz, param_dict)
    #
    #     hamiltonian = X
    #     temperature = 42
    #
    #     gibbs_state = gibbs_state_builder.build(hamiltonian, temperature)

    def test_build_n_mes(self):
        num_states = 2
        backend = Aer.get_backend("statevector_simulator")
        mes = QiteGibbsStateBuilder._build_n_mes(num_states, backend)
        expected_mes = Statevector([0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0.5 + 0.j, -0. + 0.j, -0. + 0.j,
                                    -0. + 0.j, -0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                    0.5 + 0.j, -0. + 0.j, -0. + 0.j, 0.5 + 0.j],
                                   dims=(2, 2, 2, 2))
        expected_num_qubits = 4

        np.testing.assert_almost_equal(np.asarray(mes), expected_mes)
        np.testing.assert_equal(mes.num_qubits, expected_num_qubits)


if __name__ == "__main__":
    unittest.main()
