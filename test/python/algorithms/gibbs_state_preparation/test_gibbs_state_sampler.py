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
"""Tests GibbsState class."""
import unittest

from ddt import ddt, unpack, data
import numpy as np
from numpy import array

from qiskit import QuantumCircuit
from qiskit.algorithms.gibbs_state_preparation.default_ansatz_builder import (
    build_ansatz,
    build_init_ansatz_params_vals,
)
from qiskit.algorithms.gradients import ParamShiftSamplerGradient
from qiskit.primitives import Sampler
from qiskit.quantum_info import Pauli, SparsePauliOp
from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit.circuit import Parameter
from qiskit.algorithms.gibbs_state_preparation.gibbs_state_sampler import GibbsStateSampler


# TODO test all backends and a quantum instance
@ddt
class TestGibbsStateSampler(QiskitAlgorithmsTestCase):
    """Tests GibbsState class."""

    def test_gibbs_state_init(self):
        """Initialization test."""
        gibbs_state_function = QuantumCircuit(1)
        hamiltonian = Pauli("X")
        temperature = 42
        sampler = Sampler()

        gibbs_state = GibbsStateSampler(sampler, gibbs_state_function, hamiltonian, temperature)

        np.testing.assert_equal(gibbs_state.hamiltonian, Pauli("X"))
        np.testing.assert_equal(gibbs_state.temperature, 42)

    def test_sample(self):
        """Tests if Gibbs state probabilities are sampled correctly.."""
        gibbs_state_function = QuantumCircuit(1)
        hamiltonian = SparsePauliOp.from_list(
            [
                ("ZZII", 0.3),
                ("ZIII", 0.2),
                ("IZII", 0.5),
            ]
        )
        temperature = 42

        seed = 170
        sampler = Sampler(options={"seed": seed, "shots": 1024})

        depth = 1
        num_qubits = 4

        aux_registers = set(range(2, 4))

        ansatz = build_ansatz(num_qubits, depth)
        param_values_init = build_init_ansatz_params_vals(num_qubits, depth)

        params_dict = dict(zip(ansatz.ordered_parameters, param_values_init))
        gibbs_state = GibbsStateSampler(
            sampler,
            gibbs_state_function,
            hamiltonian,
            temperature,
            ansatz,
            params_dict,
            aux_registers=aux_registers,
        )

        probs = gibbs_state.sample()

        # pre-primitives values
        # expected_probs = [0.222656, 0.25293, 0.25293, 0.271484]

        # TODO unverified values
        expected_probs = [0.23242188, 0.25585938, 0.26074219, 0.25097656]
        np.testing.assert_array_almost_equal(probs, expected_probs)

    @data([73, 9], [72, 8], [0, 0], [1, 1], [24, 0], [56, 0], [2, 2], [64, 16])
    @unpack
    def test_reduce_label(self, label, expected_label):
        """Tests if binary labels are reduced correctly by discarding aux registers."""
        gibbs_state_function = QuantumCircuit(1)
        hamiltonian = SparsePauliOp.from_list(
            [
                ("ZZIIIII", 0.3),
                ("ZIIIIII", 0.2),
                ("IZIIIII", 0.5),
            ]
        )

        temperature = 42

        sampler = Sampler(options={"shots": 1024})

        depth = 1
        num_qubits = 7

        aux_registers = set(range(3, 6))

        ansatz = build_ansatz(num_qubits, depth)
        param_values_init = build_init_ansatz_params_vals(num_qubits, depth)

        params_dict = dict(zip(ansatz.ordered_parameters, param_values_init))
        gibbs_state = GibbsStateSampler(
            sampler,
            gibbs_state_function,
            hamiltonian,
            temperature,
            ansatz,
            params_dict,
            aux_registers=aux_registers,
        )

        label = 73
        reduced_label = gibbs_state._reduce_label(label)
        expected_label = 9
        np.testing.assert_equal(reduced_label, expected_label)

    def test_calc_ansatz_gradients(self):
        """Tests if ansatz gradients are calculated correctly."""
        gibbs_state_function = QuantumCircuit(1)
        hamiltonian = SparsePauliOp.from_list(
            [
                ("ZZII", 0.3),
                ("ZIII", 0.2),
                ("IZII", 0.5),
            ]
        )
        temperature = 42

        seed = 170
        sampler = Sampler(options={"seed": seed, "shots": 1024})

        depth = 1
        num_qubits = 4

        aux_registers = set(range(2, 4))

        ansatz = build_ansatz(num_qubits, depth)
        param_values_init = build_init_ansatz_params_vals(num_qubits, depth)

        params_dict = dict(zip(ansatz.ordered_parameters, param_values_init))
        gibbs_state = GibbsStateSampler(
            sampler,
            gibbs_state_function,
            hamiltonian,
            temperature,
            ansatz,
            params_dict,
            aux_registers=aux_registers,
        )

        gradient_method = ParamShiftSamplerGradient(sampler)
        gradients = gibbs_state.calc_ansatz_gradients(gradient_method)

        # pre-primitives unverified gradients
        # expected_gradients = [
        #     [0.06372666, 0.0565455, 0.06128526, 0.06875253],
        #     [0.05608201, 0.06671929, 0.05841851, 0.0692656],
        #     [6.19888306e-05, 2.51054764e-04, 8.10623169e-05, 5.96046448e-06],
        #     [3.00645828e-04, 4.24385071e-05, 3.26633453e-05, 1.28269196e-04],
        #     [0.00029206, 0.00030899, 0.00011539, 0.00010514],
        #     [1.93119049e-05, 1.93119049e-05, 9.53674316e-07, 9.53674316e-05],
        #     [4.02927399e-05, 5.96046448e-06, 2.59637833e-04, 4.00781631e-04],
        #     [3.81469727e-06, 4.00781631e-04, 3.08990479e-04, 1.93119049e-05],
        #     [0.06008244, 0.06152725, 0.06496525, 0.06348038],
        #     [0.07159805, 0.05401993, 0.06152725, 0.06348038],
        #     [4.41074371e-04, 1.38282776e-05, 3.45706940e-05, 1.64270401e-04],
        #     [1.19209290e-05, 5.55515289e-05, 2.16722488e-04, 5.96046448e-05],
        #     [2.88486481e-05, 6.89029694e-05, 1.73807144e-04, 2.38418579e-07],
        #     [3.81469727e-06, 3.43322754e-05, 2.14576721e-04, 4.67300415e-05],
        #     [2.14576721e-06, 5.96046448e-06, 7.47680664e-04, 6.95228577e-04],
        #     [2.14576721e-04, 2.88486481e-05, 2.14576721e-06, 1.15394592e-04],
        # ]

        # TODO unverified values
        expected_gradients = [
            array([-0.25146484, 0.26660156, -0.24853516, 0.23339844]),
            array([0.23876953, -0.26367188, -0.23632812, 0.26123047]),
            array([0.00830078, -0.00292969, -0.00146484, -0.00390625]),
            array([0.00830078, -0.00292969, -0.00097656, -0.00439453]),
            array([-0.00097656, -0.00292969, 0.00097656, 0.00292969]),
            array([-0.00976562, 0.00244141, 0.00537109, 0.00195312]),
            array([0.00683594, -0.00634766, -0.01660156, 0.01611328]),
            array([0.01611328, -0.00146484, -0.00146484, -0.01318359]),
            array([-0.24072266, 0.25634766, -0.25927734, 0.24365234]),
            array([-0.24560547, -0.25439453, 0.25537109, 0.24462891]),
            array([0.00341797, -0.00244141, -0.00146484, 0.00048828]),
            array([0.00439453, -0.00097656, 0.00878906, -0.01220703]),
            array([-0.01123047, -0.00634766, 0.01269531, 0.00488281]),
            array([0.00048828, -0.01416016, 0.00878906, 0.00488281]),
            array([0.0, 0.01318359, -0.00976562, -0.00341797]),
            array([0.00927734, 0.01025391, -0.02197266, 0.00244141]),
        ]

        for ind, gradient in enumerate(expected_gradients):
            np.testing.assert_array_almost_equal(gradients[ind], gradient)

    def test_calc_ansatz_gradients_missing_ansatz(self):
        """Tests if an expected error is raised when an ansatz is missing when calculating
        ansatz gradients."""
        gibbs_state_function = QuantumCircuit(1)
        hamiltonian = SparsePauliOp.from_list(
            [
                ("ZZII", 0.3),
                ("ZIII", 0.2),
                ("IZII", 0.5),
            ]
        )
        temperature = 42

        sampler = Sampler(options={"shots": 1024})

        param_values_init = np.zeros(2)

        aux_registers = set(range(2, 4))

        params_dict = dict(zip([Parameter("a"), Parameter("b")], param_values_init))
        gibbs_state = GibbsStateSampler(
            sampler,
            gibbs_state_function,
            hamiltonian,
            temperature,
            ansatz_params_dict=params_dict,
            aux_registers=aux_registers,
        )

        gradient_method = "param_shift"
        np.testing.assert_raises(
            ValueError,
            gibbs_state.calc_ansatz_gradients,
            gradient_method,
        )


if __name__ == "__main__":
    unittest.main()
