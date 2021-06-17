# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# =============================================================================

""" Test Quantum Gradient Framework """

import unittest
from collections import OrderedDict
from test.python.opflow import QiskitOpflowTestCase
from itertools import product
import numpy as np
from ddt import ddt, data, idata, unpack

try:
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

from qiskit import QuantumCircuit, QuantumRegister, BasicAer
from qiskit.test import slow_test
from qiskit.utils import QuantumInstance
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.utils import algorithm_globals
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import CG
from qiskit.opflow import I, X, Y, Z, StateFn, CircuitStateFn, ListOp, CircuitSampler, TensoredOp
from qiskit.opflow.gradients import Gradient, NaturalGradient, Hessian
from qiskit.opflow.gradients.qfi import QFI
from qiskit.opflow.gradients.circuit_qfis import LinCombFull, OverlapBlockDiag, OverlapDiag
from qiskit.circuit import Parameter
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes


def _sample_most_likely(state_vector):
    """Compute the most likely binary string from a state vector.
    Args:
        state_vector (numpy.ndarray or dict): state vector or counts.
    Returns:
        numpy.ndarray: binary string as numpy.ndarray of ints.
    """
    if isinstance(state_vector, (OrderedDict, dict)):
        # get the binary string with the largest count
        binary_string = sorted(state_vector.items(), key=lambda kv: kv[1])[-1][0]
        x = np.asarray([int(y) for y in reversed(list(binary_string))])
        return x
    elif isinstance(state_vector, StateFn):
        binary_string = list(state_vector.sample().keys())[0]
        x = np.asarray([int(y) for y in reversed(list(binary_string))])
        return x
    else:
        n = int(np.log2(state_vector.shape[0]))
        k = np.argmax(np.abs(state_vector))
        x = np.zeros(n)
        for i in range(n):
            x[i] = k % 2
            k >>= 1
        return x


@ddt
class TestGradients(QiskitOpflowTestCase):
    """Test Qiskit Gradient Framework"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 50

    @data("lin_comb", "param_shift", "fin_diff")
    def test_gradient_p(self, method):
        """Test the state gradient for p
        |psi> = 1/sqrt(2)[[1, exp(ia)]]
        Tr(|psi><psi|Z) = 0
        Tr(|psi><psi|X) = cos(a)
        d<H>/da = - 0.5 sin(a)
        """
        ham = 0.5 * X - 1 * Z
        a = Parameter("a")
        params = a

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.p(a, q[0])
        op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)

        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4}, {a: 0}, {a: np.pi / 2}]
        correct_values = [-0.5 / np.sqrt(2), 0, -0.5]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                state_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_gradient_u(self, method):
        """Test the state gradient for U
        Tr(|psi><psi|Z) = - 0.5 sin(a)cos(c)
        Tr(|psi><psi|X) = cos^2(a/2) cos(b+c) - sin^2(a/2) cos(b-c)
        """

        ham = 0.5 * X - 1 * Z
        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.u(a, b, c, q[0])

        op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)
        params = [a, b, c]
        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4, b: 0, c: 0}, {a: np.pi / 4, b: np.pi / 4, c: np.pi / 4}]
        correct_values = [[0.3536, 0, 0], [0.3232, -0.42678, -0.92678]]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                state_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

        # Tr(|psi><psi|Z) = - 0.5 sin(a)cos(c)
        # Tr(|psi><psi|X) = cos^2(a/2) cos(b+c) - sin^2(a/2) cos(b-c)
        # dTr(|psi><psi|H)/da = 0.5(cos(2a)) + 0.5()

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.u(a, a, a, q[0])

        op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)
        params = [a]
        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4}, {a: np.pi / 2}]
        correct_values = [[-1.03033], [-1]]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                state_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_gradient_rxx(self, method):
        """Test the state gradient for XX rotation"""
        ham = TensoredOp([Z, X])
        a = Parameter("a")

        q = QuantumRegister(2)
        qc = QuantumCircuit(q)
        qc.h(q[0])
        qc.rxx(a, q[0], q[1])

        op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)
        params = [a]
        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4}, {a: np.pi / 2}]
        correct_values = [[-0.707], [-1.0]]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                state_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_gradient_ryy(self, method):
        """Test the state gradient for YY rotation"""
        alpha = Parameter("alpha")
        ham = TensoredOp([Y, alpha * Y])
        a = Parameter("a")

        q = QuantumRegister(2)
        qc = QuantumCircuit(q)
        qc.ryy(a, q[0], q[1])

        op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)
        state_grad = Gradient(grad_method=method).convert(operator=op, params=a)
        values_dict = [{a: np.pi / 8}, {a: np.pi}]
        correct_values = [[0], [0]]
        for i, value_dict in enumerate(values_dict):
            value_dict[alpha] = 1.0
            np.testing.assert_array_almost_equal(
                state_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_gradient_rzz(self, method):
        """Test the state gradient for ZZ rotation"""
        ham = Z ^ X
        a = Parameter("a")

        q = QuantumRegister(2)
        qc = QuantumCircuit(q)
        qc.h(q[0])
        qc.rzz(a, q[0], q[1])

        op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)
        params = [a]
        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4}, {a: np.pi / 2}]
        correct_values = [[-0.707], [-1.0]]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                state_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_gradient_rzx(self, method):
        """Test the state gradient for ZX rotation"""
        ham = Z ^ Z
        a = Parameter("a")

        q = QuantumRegister(2)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rzx(a, q[0], q[1])

        op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)
        params = [a]
        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 8}, {a: np.pi / 2}]
        correct_values = [[0.0], [0.0]]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                state_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_state_gradient1(self, method):
        """Test the state gradient

        Tr(|psi><psi|Z) = sin(a)sin(b)
        Tr(|psi><psi|X) = cos(a)
        d<H>/da = - 0.5 sin(a) - 1 cos(a)sin(b)
        d<H>/db = - 1 sin(a)cos(b)
        """

        ham = 0.5 * X - 1 * Z
        a = Parameter("a")
        b = Parameter("b")
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])
        op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)

        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [
            {a: np.pi / 4, b: np.pi},
            {params[0]: np.pi / 4, params[1]: np.pi / 4},
            {params[0]: np.pi / 2, params[1]: np.pi / 4},
        ]
        correct_values = [
            [-0.5 / np.sqrt(2), 1 / np.sqrt(2)],
            [-0.5 / np.sqrt(2) - 0.5, -1 / 2.0],
            [-0.5, -1 / np.sqrt(2)],
        ]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                state_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_state_gradient2(self, method):
        """Test the state gradient 2

        Tr(|psi><psi|Z) = sin(a)sin(a)
        Tr(|psi><psi|X) = cos(a)
        d<H>/da = - 0.5 sin(a) - 2 cos(a)sin(a)
        """
        ham = 0.5 * X - 1 * Z
        a = Parameter("a")
        # b = Parameter('b')
        params = [a]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(a, q[0])
        qc.rx(a, q[0])
        op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)

        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4}, {a: 0}, {a: np.pi / 2}]
        correct_values = [-1.353553, -0, -0.5]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                state_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_state_gradient3(self, method):
        """Test the state gradient 3

        Tr(|psi><psi|Z) = sin(a)sin(c(a)) = sin(a)sin(cos(a)+1)
        Tr(|psi><psi|X) = cos(a)
        d<H>/da = - 0.5 sin(a) - 1 cos(a)sin(cos(a)+1) + 1 sin^2(a)cos(cos(a)+1)
        """
        ham = 0.5 * X - 1 * Z
        a = Parameter("a")
        # b = Parameter('b')
        params = a
        c = np.cos(a) + 1
        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(a, q[0])
        qc.rx(c, q[0])
        op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)

        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4}, {a: 0}, {a: np.pi / 2}]
        correct_values = [-1.1220, -0.9093, 0.0403]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                state_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_state_gradient4(self, method):
        """Test the state gradient 4
        Tr(|psi><psi|ZX) = -cos(a)
        daTr(|psi><psi|ZX) = sin(a)
        """

        ham = X ^ Z
        a = Parameter("a")
        params = a

        q = QuantumRegister(2)
        qc = QuantumCircuit(q)
        qc.x(q[0])
        qc.h(q[1])
        qc.crz(a, q[0], q[1])
        op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)

        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4}, {a: 0}, {a: np.pi / 2}]
        correct_values = [1 / np.sqrt(2), 0, 1]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                state_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_state_gradient5(self, method):
        """Test the state gradient

        Tr(|psi><psi|Z) = sin(a0)sin(a1)
        Tr(|psi><psi|X) = cos(a0)
        d<H>/da0 = - 0.5 sin(a0) - 1 cos(a0)sin(a1)
        d<H>/da1 = - 1 sin(a0)cos(a1)
        """

        ham = 0.5 * X - 1 * Z
        a = ParameterVector("a", 2)
        params = a

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])
        op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)

        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [
            {a: [np.pi / 4, np.pi]},
            {a: [np.pi / 4, np.pi / 4]},
            {a: [np.pi / 2, np.pi / 4]},
        ]
        correct_values = [
            [-0.5 / np.sqrt(2), 1 / np.sqrt(2)],
            [-0.5 / np.sqrt(2) - 0.5, -1 / 2.0],
            [-0.5, -1 / np.sqrt(2)],
        ]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                state_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_state_hessian(self, method):
        """Test the state Hessian

        Tr(|psi><psi|Z) = sin(a)sin(b)
        Tr(|psi><psi|X) = cos(a)
        d^2<H>/da^2 = - 0.5 cos(a) + 1 sin(a)sin(b)
        d^2<H>/dbda = - 1 cos(a)cos(b)
        d^2<H>/dbda = - 1 cos(a)cos(b)
        d^2<H>/db^2 = + 1 sin(a)sin(b)
        """

        ham = 0.5 * X - 1 * Z
        params = ParameterVector("a", 2)

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])

        op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)
        state_hess = Hessian(hess_method=method).convert(operator=op)

        values_dict = [
            {params[0]: np.pi / 4, params[1]: np.pi},
            {params[0]: np.pi / 4, params[1]: np.pi / 4},
        ]
        correct_values = [
            [[-0.5 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 0]],
            [[-0.5 / np.sqrt(2) + 0.5, -1 / 2.0], [-1 / 2.0, 0.5]],
        ]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                state_hess.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @unittest.skipIf(not _HAS_JAX, "Skipping test due to missing jax module.")
    @data("lin_comb", "param_shift", "fin_diff")
    def test_state_hessian_custom_combo_fn(self, method):
        """Test the state Hessian with on an operator which includes
            a user-defined combo_fn.

        Tr(|psi><psi|Z) = sin(a)sin(b)
        Tr(|psi><psi|X) = cos(a)
        d^2<H>/da^2 = - 0.5 cos(a) + 1 sin(a)sin(b)
        d^2<H>/dbda = - 1 cos(a)cos(b)
        d^2<H>/dbda = - 1 cos(a)cos(b)
        d^2<H>/db^2 = + 1 sin(a)sin(b)
        """

        ham = 0.5 * X - 1 * Z
        a = Parameter("a")
        b = Parameter("b")
        params = [(a, a), (a, b), (b, b)]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(a, q[0])
        qc.rx(b, q[0])

        op = ListOp(
            [~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)],
            combo_fn=lambda x: x[0] ** 3 + 4 * x[0],
        )
        state_hess = Hessian(hess_method=method).convert(operator=op, params=params)

        values_dict = [
            {a: np.pi / 4, b: np.pi},
            {a: np.pi / 4, b: np.pi / 4},
            {a: np.pi / 2, b: np.pi / 4},
        ]

        correct_values = [
            [-1.28163104, 2.56326208, 1.06066017],
            [-0.04495626, -2.40716991, 1.8125],
            [2.82842712, -1.5, 1.76776695],
        ]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                state_hess.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_prob_grad(self, method):
        """Test the probability gradient

        dp0/da = cos(a)sin(b) / 2
        dp1/da = - cos(a)sin(b) / 2
        dp0/db = sin(a)cos(b) / 2
        dp1/db = - sin(a)cos(b) / 2
        """

        a = Parameter("a")
        b = Parameter("b")
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])

        op = CircuitStateFn(primitive=qc, coeff=1.0)

        prob_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [
            {a: np.pi / 4, b: 0},
            {params[0]: np.pi / 4, params[1]: np.pi / 4},
            {params[0]: np.pi / 2, params[1]: np.pi},
        ]
        correct_values = [
            [[0, 0], [1 / (2 * np.sqrt(2)), -1 / (2 * np.sqrt(2))]],
            [[1 / 4, -1 / 4], [1 / 4, -1 / 4]],
            [[0, 0], [-1 / 2, 1 / 2]],
        ]
        for i, value_dict in enumerate(values_dict):
            for j, prob_grad_result in enumerate(prob_grad.assign_parameters(value_dict).eval()):
                np.testing.assert_array_almost_equal(
                    prob_grad_result, correct_values[i][j], decimal=1
                )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_prob_hess(self, method):
        """Test the probability Hessian using linear combination of unitaries method

        d^2p0/da^2 = - sin(a)sin(b) / 2
        d^2p1/da^2 =  sin(a)sin(b) / 2
        d^2p0/dadb = cos(a)cos(b) / 2
        d^2p1/dadb = - cos(a)cos(b) / 2
        """

        a = Parameter("a")
        b = Parameter("b")
        params = [(a, a), (a, b)]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(a, q[0])
        qc.rx(b, q[0])

        op = CircuitStateFn(primitive=qc, coeff=1.0)

        prob_hess = Hessian(hess_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4, b: 0}, {a: np.pi / 4, b: np.pi / 4}, {a: np.pi / 2, b: np.pi}]
        correct_values = [
            [[0, 0], [1 / (2 * np.sqrt(2)), -1 / (2 * np.sqrt(2))]],
            [[-1 / 4, 1 / 4], [1 / 4, -1 / 4]],
            [[0, 0], [0, 0]],
        ]
        for i, value_dict in enumerate(values_dict):
            for j, prob_hess_result in enumerate(prob_hess.assign_parameters(value_dict).eval()):
                np.testing.assert_array_almost_equal(
                    prob_hess_result, correct_values[i][j], decimal=1
                )

    @idata(
        product(
            ["lin_comb", "param_shift", "fin_diff"],
            [None, "lasso", "ridge", "perturb_diag", "perturb_diag_elements"],
        )
    )
    @unpack
    def test_natural_gradient(self, method, regularization):
        """Test the natural gradient"""
        try:
            for params in (ParameterVector("a", 2), [Parameter("a"), Parameter("b")]):
                ham = 0.5 * X - 1 * Z

                q = QuantumRegister(1)
                qc = QuantumCircuit(q)
                qc.h(q)
                qc.rz(params[0], q[0])
                qc.rx(params[1], q[0])

                op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)
                nat_grad = NaturalGradient(
                    grad_method=method, regularization=regularization
                ).convert(operator=op)
                values_dict = [{params[0]: np.pi / 4, params[1]: np.pi / 2}]

                # reference values obtained by classically computing the natural gradients
                correct_values = [[-3.26, 1.63]] if regularization == "ridge" else [[-4.24, 0]]

                for i, value_dict in enumerate(values_dict):
                    np.testing.assert_array_almost_equal(
                        nat_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
                    )
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_natural_gradient2(self):
        """Test the natural gradient 2"""
        with self.assertRaises(TypeError):
            _ = NaturalGradient().convert(None, None)

    @idata(
        zip(
            ["lin_comb_full", "overlap_block_diag", "overlap_diag"],
            [LinCombFull, OverlapBlockDiag, OverlapDiag],
        )
    )
    @unpack
    def test_natural_gradient3(self, qfi_method, circuit_qfi):
        """Test the natural gradient 3"""
        nat_grad = NaturalGradient(qfi_method=qfi_method)
        self.assertIsInstance(nat_grad.qfi_method, circuit_qfi)

    @idata(
        product(
            ["lin_comb", "param_shift", "fin_diff"],
            ["lin_comb_full", "overlap_block_diag", "overlap_diag"],
            [None, "ridge", "perturb_diag", "perturb_diag_elements"],
        )
    )
    @unpack
    def test_natural_gradient4(self, grad_method, qfi_method, regularization):
        """Test the natural gradient 4"""

        # Avoid regularization = lasso intentionally because it does not converge
        try:
            ham = 0.5 * X - 1 * Z
            a = Parameter("a")
            params = a

            q = QuantumRegister(1)
            qc = QuantumCircuit(q)
            qc.h(q)
            qc.rz(a, q[0])

            op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)
            nat_grad = NaturalGradient(
                grad_method=grad_method, qfi_method=qfi_method, regularization=regularization
            ).convert(operator=op, params=params)
            values_dict = [{a: np.pi / 4}]
            correct_values = [[0.0]] if regularization == "ridge" else [[-1.41421342]]
            for i, value_dict in enumerate(values_dict):
                np.testing.assert_array_almost_equal(
                    nat_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=3
                )
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    @unittest.skipIf(not _HAS_JAX, "Skipping test due to missing jax module.")
    @idata(product(["lin_comb", "param_shift", "fin_diff"], [True, False]))
    @unpack
    def test_jax_chain_rule(self, method: str, autograd: bool):
        """Test the chain rule functionality using Jax

        d<H>/d<X> = 2<X>
        d<H>/d<Z> = - sin(<Z>)
        <Z> = Tr(|psi><psi|Z) = sin(a)sin(b)
        <X> = Tr(|psi><psi|X) = cos(a)
        d<H>/da = d<H>/d<X> d<X>/da + d<H>/d<Z> d<Z>/da = - 2 cos(a)sin(a)
                    - sin(sin(a)sin(b)) * cos(a)sin(b)
        d<H>/db = d<H>/d<X> d<X>/db + d<H>/d<Z> d<Z>/db = - sin(sin(a)sin(b)) * sin(a)cos(b)
        """

        a = Parameter("a")
        b = Parameter("b")
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])

        def combo_fn(x):
            return jnp.power(x[0], 2) + jnp.cos(x[1])

        def grad_combo_fn(x):
            return np.array([2 * x[0], -np.sin(x[1])])

        op = ListOp(
            [
                ~StateFn(X) @ CircuitStateFn(primitive=qc, coeff=1.0),
                ~StateFn(Z) @ CircuitStateFn(primitive=qc, coeff=1.0),
            ],
            combo_fn=combo_fn,
            grad_combo_fn=None if autograd else grad_combo_fn,
        )

        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [
            {a: np.pi / 4, b: np.pi},
            {params[0]: np.pi / 4, params[1]: np.pi / 4},
            {params[0]: np.pi / 2, params[1]: np.pi / 4},
        ]
        correct_values = [[-1.0, 0.0], [-1.2397, -0.2397], [0, -0.45936]]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                state_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_grad_combo_fn_chain_rule(self, method):
        """Test the chain rule for a custom gradient combo function."""
        np.random.seed(2)

        def combo_fn(x):
            amplitudes = x[0].primitive.data
            pdf = np.multiply(amplitudes, np.conj(amplitudes))
            return np.sum(np.log(pdf)) / (-len(amplitudes))

        def grad_combo_fn(x):
            amplitudes = x[0].primitive.data
            pdf = np.multiply(amplitudes, np.conj(amplitudes))
            grad = []
            for prob in pdf:
                grad += [-1 / prob]
            return grad

        qc = RealAmplitudes(2, reps=1)
        grad_op = ListOp([StateFn(qc)], combo_fn=combo_fn, grad_combo_fn=grad_combo_fn)
        grad = Gradient(grad_method=method).convert(grad_op, qc.ordered_parameters)
        value_dict = dict(zip(qc.ordered_parameters, np.random.rand(len(qc.ordered_parameters))))
        correct_values = [
            [(-0.16666259133549044 + 0j)],
            [(-7.244949702732864 + 0j)],
            [(-2.979791752749964 + 0j)],
            [(-5.310186078432614 + 0j)],
        ]
        np.testing.assert_array_almost_equal(
            grad.assign_parameters(value_dict).eval(), correct_values
        )

    def test_grad_combo_fn_chain_rule_nat_grad(self):
        """Test the chain rule for a custom gradient combo function."""
        np.random.seed(2)

        def combo_fn(x):
            amplitudes = x[0].primitive.data
            pdf = np.multiply(amplitudes, np.conj(amplitudes))
            return np.sum(np.log(pdf)) / (-len(amplitudes))

        def grad_combo_fn(x):
            amplitudes = x[0].primitive.data
            pdf = np.multiply(amplitudes, np.conj(amplitudes))
            grad = []
            for prob in pdf:
                grad += [-1 / prob]
            return grad

        try:
            qc = RealAmplitudes(2, reps=1)
            grad_op = ListOp([StateFn(qc)], combo_fn=combo_fn, grad_combo_fn=grad_combo_fn)
            grad = NaturalGradient(grad_method="lin_comb", regularization="ridge").convert(
                grad_op, qc.ordered_parameters
            )
            value_dict = dict(
                zip(qc.ordered_parameters, np.random.rand(len(qc.ordered_parameters)))
            )
            correct_values = [[0.20777236], [-18.92560338], [-15.89005475], [-10.44002031]]
            np.testing.assert_array_almost_equal(
                grad.assign_parameters(value_dict).eval(), correct_values, decimal=3
            )
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    @data("lin_comb", "param_shift", "fin_diff")
    def test_operator_coefficient_gradient(self, method):
        """Test the operator coefficient gradient

        Tr( | psi > < psi | Z) = sin(a)sin(b)
        Tr( | psi > < psi | X) = cos(a)
        """
        a = Parameter("a")
        b = Parameter("b")
        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(a, q[0])
        qc.rx(b, q[0])

        coeff_0 = Parameter("c_0")
        coeff_1 = Parameter("c_1")
        ham = coeff_0 * X + coeff_1 * Z
        op = StateFn(ham, is_measurement=True) @ CircuitStateFn(primitive=qc, coeff=1.0)
        gradient_coeffs = [coeff_0, coeff_1]
        coeff_grad = Gradient(grad_method=method).convert(op, gradient_coeffs)
        values_dict = [
            {coeff_0: 0.5, coeff_1: -1, a: np.pi / 4, b: np.pi},
            {coeff_0: 0.5, coeff_1: -1, a: np.pi / 4, b: np.pi / 4},
        ]
        correct_values = [[1 / np.sqrt(2), 0], [1 / np.sqrt(2), 1 / 2]]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                coeff_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_operator_coefficient_hessian(self, method):
        """Test the operator coefficient hessian

        <Z> = Tr( | psi > < psi | Z) = sin(a)sin(b)
        <X> = Tr( | psi > < psi | X) = cos(a)
        d<H>/dc_0 = 2 * c_0 * <X> + c_1 * <Z>
        d<H>/dc_1 = c_0 * <Z>
        d^2<H>/dc_0^2 = 2 * <X>
        d^2<H>/dc_0dc_1 = <Z>
        d^2<H>/dc_1dc_0 = <Z>
        d^2<H>/dc_1^2 = 0
        """
        a = Parameter("a")
        b = Parameter("b")
        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(a, q[0])
        qc.rx(b, q[0])

        coeff_0 = Parameter("c_0")
        coeff_1 = Parameter("c_1")
        ham = coeff_0 * coeff_0 * X + coeff_1 * coeff_0 * Z
        op = StateFn(ham, is_measurement=True) @ CircuitStateFn(primitive=qc, coeff=1.0)
        gradient_coeffs = [(coeff_0, coeff_0), (coeff_0, coeff_1), (coeff_1, coeff_1)]
        coeff_grad = Hessian(hess_method=method).convert(op, gradient_coeffs)
        values_dict = [
            {coeff_0: 0.5, coeff_1: -1, a: np.pi / 4, b: np.pi},
            {coeff_0: 0.5, coeff_1: -1, a: np.pi / 4, b: np.pi / 4},
        ]

        correct_values = [[2 / np.sqrt(2), 0, 0], [2 / np.sqrt(2), 1 / 2, 0]]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                coeff_grad.assign_parameters(value_dict).eval(), correct_values[i], decimal=1
            )

    @data("lin_comb", "param_shift", "fin_diff")
    def test_circuit_sampler(self, method):
        """Test the gradient with circuit sampler

        Tr(|psi><psi|Z) = sin(a)sin(b)
        Tr(|psi><psi|X) = cos(a)
        d<H>/da = - 0.5 sin(a) - 1 cos(a)sin(b)
        d<H>/db = - 1 sin(a)cos(b)
        """

        ham = 0.5 * X - 1 * Z
        a = Parameter("a")
        b = Parameter("b")
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])
        op = ~StateFn(ham) @ CircuitStateFn(primitive=qc, coeff=1.0)

        shots = 8000
        if method == "fin_diff":
            np.random.seed(8)
            state_grad = Gradient(grad_method=method, epsilon=shots ** (-1 / 6.0)).convert(
                operator=op
            )
        else:
            state_grad = Gradient(grad_method=method).convert(operator=op)
        values_dict = [
            {a: np.pi / 4, b: np.pi},
            {params[0]: np.pi / 4, params[1]: np.pi / 4},
            {params[0]: np.pi / 2, params[1]: np.pi / 4},
        ]
        correct_values = [
            [-0.5 / np.sqrt(2), 1 / np.sqrt(2)],
            [-0.5 / np.sqrt(2) - 0.5, -1 / 2.0],
            [-0.5, -1 / np.sqrt(2)],
        ]

        backend = BasicAer.get_backend("qasm_simulator")
        q_instance = QuantumInstance(backend=backend, shots=shots)

        for i, value_dict in enumerate(values_dict):
            sampler = CircuitSampler(backend=q_instance).convert(
                state_grad, params={k: [v] for k, v in value_dict.items()}
            )
            np.testing.assert_array_almost_equal(sampler.eval()[0], correct_values[i], decimal=1)

    @data("lin_comb", "param_shift", "fin_diff")
    def test_circuit_sampler2(self, method):
        """Test the probability gradient with the circuit sampler

        dp0/da = cos(a)sin(b) / 2
        dp1/da = - cos(a)sin(b) / 2
        dp0/db = sin(a)cos(b) / 2
        dp1/db = - sin(a)cos(b) / 2
        """

        a = Parameter("a")
        b = Parameter("b")
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])

        op = CircuitStateFn(primitive=qc, coeff=1.0)

        shots = 8000
        if method == "fin_diff":
            np.random.seed(8)
            prob_grad = Gradient(grad_method=method, epsilon=shots ** (-1 / 6.0)).convert(
                operator=op, params=params
            )
        else:
            prob_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [
            {a: [np.pi / 4], b: [0]},
            {params[0]: [np.pi / 4], params[1]: [np.pi / 4]},
            {params[0]: [np.pi / 2], params[1]: [np.pi]},
        ]
        correct_values = [
            [[0, 0], [1 / (2 * np.sqrt(2)), -1 / (2 * np.sqrt(2))]],
            [[1 / 4, -1 / 4], [1 / 4, -1 / 4]],
            [[0, 0], [-1 / 2, 1 / 2]],
        ]

        backend = BasicAer.get_backend("qasm_simulator")
        q_instance = QuantumInstance(backend=backend, shots=shots)

        for i, value_dict in enumerate(values_dict):
            sampler = CircuitSampler(backend=q_instance).convert(prob_grad, params=value_dict)
            result = sampler.eval()[0]
            self.assertTrue(np.allclose(result[0].toarray(), correct_values[i][0], atol=0.1))
            self.assertTrue(np.allclose(result[1].toarray(), correct_values[i][1], atol=0.1))

    @idata(["statevector_simulator", "qasm_simulator"])
    def test_gradient_wrapper(self, backend_type):
        """Test the gradient wrapper for probability gradients
        dp0/da = cos(a)sin(b) / 2
        dp1/da = - cos(a)sin(b) / 2
        dp0/db = sin(a)cos(b) / 2
        dp1/db = - sin(a)cos(b) / 2
        """
        method = "param_shift"
        a = Parameter("a")
        b = Parameter("b")
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])

        op = CircuitStateFn(primitive=qc, coeff=1.0)

        shots = 8000
        backend = BasicAer.get_backend(backend_type)
        q_instance = QuantumInstance(backend=backend, shots=shots)
        if method == "fin_diff":
            np.random.seed(8)
            prob_grad = Gradient(grad_method=method, epsilon=shots ** (-1 / 6.0)).gradient_wrapper(
                operator=op, bind_params=params, backend=q_instance
            )
        else:
            prob_grad = Gradient(grad_method=method).gradient_wrapper(
                operator=op, bind_params=params, backend=q_instance
            )
        values = [[np.pi / 4, 0], [np.pi / 4, np.pi / 4], [np.pi / 2, np.pi]]
        correct_values = [
            [[0, 0], [1 / (2 * np.sqrt(2)), -1 / (2 * np.sqrt(2))]],
            [[1 / 4, -1 / 4], [1 / 4, -1 / 4]],
            [[0, 0], [-1 / 2, 1 / 2]],
        ]
        for i, value in enumerate(values):
            result = prob_grad(value)
            if backend_type == "qasm_simulator":  # sparse result
                result = [result[0].toarray(), result[1].toarray()]

            self.assertTrue(np.allclose(result[0], correct_values[i][0], atol=0.1))
            self.assertTrue(np.allclose(result[1], correct_values[i][1], atol=0.1))

    @slow_test
    def test_vqe1(self):
        """Test VQE with gradient"""

        method = "lin_comb"
        backend = "qasm_simulator"
        q_instance = QuantumInstance(
            BasicAer.get_backend(backend), seed_simulator=79, seed_transpiler=2
        )
        # Define the Hamiltonian
        h2_hamiltonian = (
            -1.05 * (I ^ I) + 0.39 * (I ^ Z) - 0.39 * (Z ^ I) - 0.01 * (Z ^ Z) + 0.18 * (X ^ X)
        )
        h2_energy = -1.85727503

        # Define the Ansatz
        wavefunction = QuantumCircuit(2)
        params = ParameterVector("theta", length=8)
        itr = iter(params)
        wavefunction.ry(next(itr), 0)
        wavefunction.ry(next(itr), 1)
        wavefunction.rz(next(itr), 0)
        wavefunction.rz(next(itr), 1)
        wavefunction.cx(0, 1)
        wavefunction.ry(next(itr), 0)
        wavefunction.ry(next(itr), 1)
        wavefunction.rz(next(itr), 0)
        wavefunction.rz(next(itr), 1)

        # Conjugate Gradient algorithm
        optimizer = CG(maxiter=10)

        grad = Gradient(grad_method=method)

        # Gradient callable
        vqe = VQE(
            ansatz=wavefunction, optimizer=optimizer, gradient=grad, quantum_instance=q_instance
        )

        result = vqe.compute_minimum_eigenvalue(operator=h2_hamiltonian)
        np.testing.assert_almost_equal(result.optimal_value, h2_energy, decimal=0)

    @slow_test
    def test_vqe2(self):
        """Test VQE with natural gradient"""

        method = 'lin_comb'
        backend = 'qasm_simulator'
        q_instance = QuantumInstance(BasicAer.get_backend(backend), seed_simulator=79,
                                     seed_transpiler=2)
        # Define the Hamiltonian
        h2_hamiltonian = -1.05 * (I ^ I) + 0.39 * (I ^ Z) - 0.39 * (Z ^ I) \
                         - 0.01 * (Z ^ Z) + 0.18 * (X ^ X)
        h2_energy = -1.85727503

        # Define the Ansatz
        wavefunction = QuantumCircuit(2)
        params = ParameterVector('theta', length=8)
        itr = iter(params)
        wavefunction.ry(next(itr), 0)
        wavefunction.ry(next(itr), 1)
        wavefunction.rz(next(itr), 0)
        wavefunction.rz(next(itr), 1)
        wavefunction.cx(0, 1)
        wavefunction.ry(next(itr), 0)
        wavefunction.ry(next(itr), 1)
        wavefunction.rz(next(itr), 0)
        wavefunction.rz(next(itr), 1)

        # Conjugate Gradient algorithm
        optimizer = CG(maxiter=10)

        grad = NaturalGradient(grad_method=method)

        # Gradient callable
        vqe = VQE(var_form=wavefunction,
                  optimizer=optimizer,
                  gradient=grad,
                  quantum_instance=q_instance)

        result = vqe.compute_minimum_eigenvalue(operator=h2_hamiltonian)
        np.testing.assert_almost_equal(result.optimal_value, h2_energy, decimal=0)

    def test_qaoa(self):
        """ QAOA test
        qubit_op is computed using max_cut.get_operator(w)
        for
            w = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
            ])

        """

        seed = 2
        np.random.seed(2)
        p = 1
        m = (I ^ I ^ I ^ X) + (I ^ I ^ X ^ I) + (I ^ X ^ I ^ I) + (X ^ I ^ I ^ I)
        solution = {'0101', '1010'}

        backend = BasicAer.get_backend('statevector_simulator')
        optimizer = CG(maxiter=10)
        qubit_op = PauliSumOp(SparsePauliOp
                              ([[False, False, False, False, True, True, False, False],
                                [False, False, False, False, False, True, True, False],
                                [False, False, False, False, True, False, False, True],
                                [False, False, False, False, False, False, True, True]],
                               coeffs=[0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j]), coeff=1.0)

        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)
        qaoa = QAOA(optimizer, p, mixer=m, gradient=NaturalGradient(),
                    quantum_instance=quantum_instance)

        result = qaoa.compute_minimum_eigenvalue(qubit_op)
        x = _sample_most_likely(result.eigenstate)
        graph_solution = 1 - x
        self.assertIn(''.join([str(int(i)) for i in graph_solution]), solution)

    def test_qfi_overlap_works_with_bound_parameters(self):
        """Test all QFI methods work if the circuit contains a gate with bound parameters."""

        x = Parameter("x")
        circuit = QuantumCircuit(1)
        circuit.ry(np.pi / 4, 0)
        circuit.rx(x, 0)
        state = StateFn(circuit)

        methods = ["lin_comb_full", "overlap_diag", "overlap_block_diag"]
        reference = 0.5

        for method in methods:
            with self.subTest(method):
                qfi = QFI(method)
                value = np.real(qfi.convert(state, [x]).bind_parameters({x: 0.12}).eval())
                self.assertAlmostEqual(value[0][0], reference)


@ddt
class TestParameterGradients(QiskitOpflowTestCase):
    """Test taking the gradient of parameter expressions."""

    def test_grad(self):
        """Test taking the gradient of parameter expressions."""
        x, y = Parameter("x"), Parameter("y")
        with self.subTest("linear"):
            expr = 2 * x + y

            grad = expr.gradient(x)
            self.assertEqual(grad, 2)

            grad = expr.gradient(y)
            self.assertEqual(grad, 1)

        with self.subTest("polynomial"):
            expr = x * x * x - x * y + y * y

            grad = expr.gradient(x)
            self.assertEqual(grad, 3 * x * x - y)

            grad = expr.gradient(y)
            self.assertEqual(grad, -1 * x + 2 * y)

    def test_converted_to_float_if_bound(self):
        """Test the gradient is a float when no free symbols are left."""
        x = Parameter("x")
        expr = 2 * x + 1
        grad = expr.gradient(x)
        self.assertIsInstance(grad, float)


@ddt
class TestQFI(QiskitOpflowTestCase):
    """Tests for the quantum Fisher information."""

    @data("lin_comb_full", "overlap_block_diag", "overlap_diag")
    def test_qfi_simple(self, method):
        """Test if the quantum fisher information calculation is correct for a simple test case.

        QFI = [[1, 0], [0, 1]] - [[0, 0], [0, cos^2(a)]]
        """
        # create the circuit
        a, b = Parameter("a"), Parameter("b")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(a, 0)
        qc.rx(b, 0)

        # convert the circuit to a QFI object
        op = CircuitStateFn(qc)
        qfi = QFI(qfi_method=method).convert(operator=op)

        # test for different values
        values_dict = [{a: np.pi / 4, b: 0.1}, {a: np.pi, b: 0.1}, {a: np.pi / 2, b: 0.1}]
        correct_values = [[[1, 0], [0, 0.5]], [[1, 0], [0, 0]], [[1, 0], [0, 1]]]

        for i, value_dict in enumerate(values_dict):
            actual = qfi.assign_parameters(value_dict).eval()
            np.testing.assert_array_almost_equal(actual, correct_values[i], decimal=1)

    def test_qfi_maxcut(self):
        """Test the QFI for a simple MaxCut problem.

        This is interesting because it contains the same parameters in different gates.
        """
        # create maxcut circuit for the hamiltonian
        # H = (I ^ I ^ Z ^ Z) + (I ^ Z ^ I ^ Z) + (Z ^ I ^ I ^ Z) + (I ^ Z ^ Z ^ I)

        x = ParameterVector("x", 2)
        ansatz = QuantumCircuit(4)

        # initial hadamard layer
        ansatz.h(ansatz.qubits)

        # e^{iZZ} layers
        def expiz(qubit0, qubit1):
            ansatz.cx(qubit0, qubit1)
            ansatz.rz(2 * x[0], qubit1)
            ansatz.cx(qubit0, qubit1)

        expiz(2, 1)
        expiz(3, 0)
        expiz(2, 0)
        expiz(1, 0)

        # mixer layer with RX gates
        for i in range(ansatz.num_qubits):
            ansatz.rx(2 * x[1], i)

        point = {x[0]: 0.4, x[1]: 0.69}

        # reference computed via finite difference
        reference = np.array([[16.0, -5.551], [-5.551, 18.497]])

        # QFI from gradient framework
        qfi = QFI().convert(CircuitStateFn(ansatz), params=x[:])
        actual = np.array(qfi.bind_parameters(point).eval()).real
        np.testing.assert_array_almost_equal(actual, reference, decimal=3)

    def test_qfi_circuit_shared_params(self):
        """Test the QFI circuits for parameters shared across some gates."""
        # create the test circuit
        x = Parameter("x")
        circuit = QuantumCircuit(1)
        circuit.rx(x, 0)
        circuit.rx(x, 0)

        # construct the QFI circuits used in the evaluation

        circuit1 = QuantumCircuit(2)
        circuit1.h(1)
        circuit1.x(1)
        circuit1.cx(1, 0)
        circuit1.x(1)
        circuit1.cx(1, 0)
        # circuit1.rx(x, 0)  # trimmed
        # circuit1.rx(x, 0)  # trimmed
        circuit1.h(1)

        circuit2 = QuantumCircuit(2)
        circuit2.h(1)
        circuit2.x(1)
        circuit2.cx(1, 0)
        circuit2.x(1)
        circuit2.rx(x, 0)
        circuit2.cx(1, 0)
        # circuit2.rx(x, 0)  # trimmed
        circuit2.h(1)

        circuit3 = QuantumCircuit(2)
        circuit3.h(1)
        circuit3.cx(1, 0)
        circuit3.x(1)
        circuit3.rx(x, 0)
        circuit3.cx(1, 0)
        # circuit3.rx(x, 0)  # trimmed
        circuit3.x(1)
        circuit3.h(1)

        circuit4 = QuantumCircuit(2)
        circuit4.h(1)
        circuit4.rx(x, 0)
        circuit4.x(1)
        circuit4.cx(1, 0)
        circuit4.x(1)
        circuit4.cx(1, 0)
        # circuit4.rx(x, 0)  # trimmed
        circuit4.h(1)

        # this naming and adding of register is required bc circuit's are only equal if the
        # register have the same names
        circuit5 = QuantumCircuit(2)
        circuit5.h(1)
        circuit5.sdg(1)
        circuit5.cx(1, 0)
        # circuit5.rx(x, 0)  # trimmed
        circuit5.h(1)

        circuit6 = QuantumCircuit(2)
        circuit6.h(1)
        circuit6.sdg(1)
        circuit6.rx(x, 0)
        circuit6.cx(1, 0)
        circuit6.h(1)

        # compare
        qfi = QFI().convert(StateFn(circuit), params=[x])

        circuit_sets = (
            [circuit1, circuit2, circuit3, circuit4],
            [circuit5, circuit6],
            [circuit5, circuit6],
        )
        list_ops = (
            qfi.oplist[0].oplist[0].oplist[:-1],
            qfi.oplist[0].oplist[0].oplist[-1].oplist[0].oplist,
            qfi.oplist[0].oplist[0].oplist[-1].oplist[1].oplist,
        )

        # compose both on the same circuit such that the comparison works
        base = QuantumCircuit(2)

        for i, (circuit_set, list_op) in enumerate(zip(circuit_sets, list_ops)):
            for j, (reference, composed_op) in enumerate(zip(circuit_set, list_op)):
                with self.subTest(f"set {i} circuit {j}"):
                    self.assertEqual(
                        base.compose(composed_op[1].primitive), base.compose(reference)
                    )

    def test_overlap_qfi_bound_parameters(self):
        """Test the overlap QFI works on a circuit with multi-parameter bound gates."""
        x = Parameter("x")
        circuit = QuantumCircuit(1)
        circuit.u(1, 2, 3, 0)
        circuit.rx(x, 0)

        qfi = QFI("overlap_diag").convert(StateFn(circuit), [x])
        value = qfi.bind_parameters({x: 1}).eval()[0][0]
        ref = 0.87737713
        self.assertAlmostEqual(value, ref)

    def test_overlap_qfi_raises_on_multiparam(self):
        """Test the overlap QFI raises an appropriate error on multi-param unbound gates."""
        x = ParameterVector("x", 2)
        circuit = QuantumCircuit(1)
        circuit.u(x[0], x[1], 2, 0)

        with self.assertRaises(NotImplementedError):
            _ = QFI("overlap_diag").convert(StateFn(circuit), [x])

    def test_overlap_qfi_raises_on_unsupported_gate(self):
        """Test the overlap QFI raises an appropriate error on multi-param unbound gates."""
        x = Parameter("x")
        circuit = QuantumCircuit(1)
        circuit.p(x, 0)

        with self.assertRaises(NotImplementedError):
            _ = QFI("overlap_diag").convert(StateFn(circuit), [x])


if __name__ == "__main__":
    unittest.main()
