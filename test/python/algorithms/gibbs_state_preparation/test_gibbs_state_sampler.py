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

from qiskit import QuantumCircuit
from qiskit.algorithms.gibbs_state_preparation.default_ansatz_builder import (
    build_ansatz,
    build_init_ansatz_params_vals,
)
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
        expected_probs = [0.222656, 0.25293, 0.25293, 0.271484]
        # np.testing.assert_array_almost_equal(probs, expected_probs)
        print(probs)

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

        gradient_method = "param_shift"
        gradients = gibbs_state.calc_ansatz_gradients(gradient_method)

        expected_gradients = [
            [0.06372666, 0.0565455, 0.06128526, 0.06875253],
            [0.05608201, 0.06671929, 0.05841851, 0.0692656],
            [6.19888306e-05, 2.51054764e-04, 8.10623169e-05, 5.96046448e-06],
            [3.00645828e-04, 4.24385071e-05, 3.26633453e-05, 1.28269196e-04],
            [0.00029206, 0.00030899, 0.00011539, 0.00010514],
            [1.93119049e-05, 1.93119049e-05, 9.53674316e-07, 9.53674316e-05],
            [4.02927399e-05, 5.96046448e-06, 2.59637833e-04, 4.00781631e-04],
            [3.81469727e-06, 4.00781631e-04, 3.08990479e-04, 1.93119049e-05],
            [0.06008244, 0.06152725, 0.06496525, 0.06348038],
            [0.07159805, 0.05401993, 0.06152725, 0.06348038],
            [4.41074371e-04, 1.38282776e-05, 3.45706940e-05, 1.64270401e-04],
            [1.19209290e-05, 5.55515289e-05, 2.16722488e-04, 5.96046448e-05],
            [2.88486481e-05, 6.89029694e-05, 1.73807144e-04, 2.38418579e-07],
            [3.81469727e-06, 3.43322754e-05, 2.14576721e-04, 4.67300415e-05],
            [2.14576721e-06, 5.96046448e-06, 7.47680664e-04, 6.95228577e-04],
            [2.14576721e-04, 2.88486481e-05, 2.14576721e-06, 1.15394592e-04],
        ]
        for ind, gradient in enumerate(expected_gradients):
            # np.testing.assert_array_almost_equal(gradients[ind], gradient)
            print(gradients[ind])

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
