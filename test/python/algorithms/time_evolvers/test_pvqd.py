# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for PVQD."""
import unittest
from functools import partial

from ddt import ddt, data, unpack
import numpy as np

from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import Sampler, Estimator
from qiskit.quantum_info import Pauli
from qiskit.test import QiskitTestCase
from qiskit import QiskitError
from qiskit.circuit import QuantumCircuit, Parameter, Gate
from qiskit.algorithms.evolvers import EvolutionProblem
from qiskit.algorithms.time_evolvers.pvqd import PVQD
from qiskit.algorithms.optimizers import L_BFGS_B, GradientDescent, SPSA, OptimizerResult
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import X, Z, I


# pylint: disable=unused-argument, invalid-name
def gradient_supplied(fun, x0, jac, info):
    """A mock optimizer that checks whether the gradient is supported or not."""
    result = OptimizerResult()
    result.x = x0
    result.fun = 0
    info["has_gradient"] = jac is not None

    return result


class WhatAmI(Gate):
    """A custom opaque gate that can be inverted but not decomposed."""

    def __init__(self, angle):
        super().__init__(name="whatami", num_qubits=2, params=[angle])

    def inverse(self):
        return WhatAmI(-self.params[0])


@ddt
class TestPVQD(QiskitTestCase):
    """Tests for the pVQD algorithm."""

    def setUp(self):
        super().setUp()
        self.hamiltonian = 0.1 * (Z ^ Z) + (I ^ X) + (X ^ I)
        self.observable = Pauli("ZZ")
        self.ansatz = EfficientSU2(2, reps=1)
        self.initial_parameters = np.zeros(self.ansatz.num_parameters)

    @data(
        ("ising", True, 2),
        ("pauli", False, None),
    )
    @unpack
    def test_pvqd(self, hamiltonian_type, gradient, num_timesteps):
        """Test a simple evolution."""
        time = 0.02

        if hamiltonian_type == "ising":
            hamiltonian = self.hamiltonian
        else:  # hamiltonian_type == "pauli":
            hamiltonian = Pauli("XX")

        # parse input arguments
        if gradient:
            optimizer = GradientDescent(maxiter=1)
        else:
            optimizer = L_BFGS_B(maxiter=1)

        sampler = Sampler()
        estimator = Estimator()
        fidelity_primitive = ComputeUncompute(sampler)

        # run pVQD keeping track of the energy and the magnetization
        pvqd = PVQD(
            self.ansatz,
            self.initial_parameters,
            fidelity_primitive,
            estimator,
            num_timesteps=num_timesteps,
            optimizer=optimizer,
        )
        problem = EvolutionProblem(hamiltonian, time, aux_operators=[hamiltonian, self.observable])
        result = pvqd.evolve(problem)

        self.assertTrue(len(result.fidelities) == 3)
        self.assertTrue(np.all(result.times == [0.0, 0.01, 0.02]))
        self.assertTrue(np.asarray(result.observables).shape == (3, 2))
        num_parameters = self.ansatz.num_parameters
        self.assertTrue(
            len(result.parameters) == 3
            and np.all([len(params) == num_parameters for params in result.parameters])
        )

    def test_step(self):
        """Test calling the step method directly."""
        sampler = Sampler()
        estimator = Estimator()
        fidelity_primitive = ComputeUncompute(sampler)
        pvqd = PVQD(
            self.ansatz,
            self.initial_parameters,
            fidelity_primitive,
            estimator,
            optimizer=L_BFGS_B(maxiter=100),
        )

        # perform optimization for a timestep of 0, then the optimal parameters are the current
        # ones and the fidelity is 1
        theta_next, fidelity = pvqd.step(
            self.hamiltonian.to_matrix_op(),
            self.ansatz,
            self.initial_parameters,
            dt=0.0,
            initial_guess=np.zeros_like(self.initial_parameters),
        )

        self.assertTrue(np.allclose(theta_next, self.initial_parameters))
        self.assertAlmostEqual(fidelity, 1)

    def test_get_loss(self):
        """Test getting the loss function directly."""

        sampler = Sampler()
        estimator = Estimator()
        fidelity_primitive = ComputeUncompute(sampler)

        pvqd = PVQD(
            self.ansatz,
            self.initial_parameters,
            fidelity_primitive,
            estimator,
            use_parameter_shift=False,
        )

        theta = np.ones(self.ansatz.num_parameters)
        loss, gradient = pvqd.get_loss(
            self.hamiltonian, self.ansatz, dt=0.0, current_parameters=theta
        )

        displacement = np.arange(self.ansatz.num_parameters)

        with self.subTest(msg="check gradient is None"):
            self.assertIsNone(gradient)

        with self.subTest(msg="check loss works"):
            self.assertGreater(loss(displacement), 0)
            self.assertAlmostEqual(loss(np.zeros_like(theta)), 0)

    def test_invalid_num_timestep(self):
        """Test raises if the num_timestep is not positive."""
        sampler = Sampler()
        estimator = Estimator()
        fidelity_primitive = ComputeUncompute(sampler)
        pvqd = PVQD(
            self.ansatz,
            self.initial_parameters,
            fidelity_primitive,
            estimator,
            num_timesteps=0,
            optimizer=L_BFGS_B(),
        )
        problem = EvolutionProblem(
            self.hamiltonian, time=0.01, aux_operators=[self.hamiltonian, self.observable]
        )

        with self.assertRaises(ValueError):
            _ = pvqd.evolve(problem)

    def test_initial_guess_and_observables(self):
        """Test doing no optimizations stays at initial guess."""
        initial_guess = np.zeros(self.ansatz.num_parameters)
        sampler = Sampler()
        estimator = Estimator()
        fidelity_primitive = ComputeUncompute(sampler)

        pvqd = PVQD(
            self.ansatz,
            self.initial_parameters,
            fidelity_primitive,
            estimator,
            num_timesteps=10,
            optimizer=SPSA(maxiter=0, learning_rate=0.1, perturbation=0.01),
            initial_guess=initial_guess,
        )
        problem = EvolutionProblem(
            self.hamiltonian, time=0.1, aux_operators=[self.hamiltonian, self.observable]
        )

        result = pvqd.evolve(problem)

        observables = result.aux_ops_evaluated
        self.assertEqual(observables[0], 0.1)  # expected energy
        self.assertEqual(observables[1], 1)  # expected magnetization

    def test_zero_parameters(self):
        """Test passing an ansatz with zero parameters raises an error."""
        problem = EvolutionProblem(self.hamiltonian, time=0.02)
        sampler = Sampler()
        fidelity_primitive = ComputeUncompute(sampler)

        pvqd = PVQD(
            QuantumCircuit(2),
            np.array([]),
            fidelity_primitive,
            optimizer=SPSA(maxiter=10, learning_rate=0.1, perturbation=0.01),
        )

        with self.assertRaises(QiskitError):
            _ = pvqd.evolve(problem)

    def test_initial_state_raises(self):
        """Test passing an initial state raises an error for now."""
        initial_state = QuantumCircuit(2)
        initial_state.x(0)

        problem = EvolutionProblem(
            self.hamiltonian,
            time=0.02,
            initial_state=initial_state,
        )

        sampler = Sampler()
        fidelity_primitive = ComputeUncompute(sampler)

        pvqd = PVQD(
            self.ansatz,
            self.initial_parameters,
            fidelity_primitive,
            optimizer=SPSA(maxiter=0, learning_rate=0.1, perturbation=0.01),
        )

        with self.assertRaises(NotImplementedError):
            _ = pvqd.evolve(problem)


class TestPVQDUtils(QiskitTestCase):
    """Test some utility functions for PVQD."""

    def setUp(self):
        super().setUp()
        self.hamiltonian = 0.1 * (Z ^ Z) + (I ^ X) + (X ^ I)
        self.ansatz = EfficientSU2(2, reps=1)

    def test_gradient_supported(self):
        """Test the gradient support is correctly determined."""
        # gradient supported here
        wrapped = EfficientSU2(2)  # a circuit wrapped into a big instruction
        plain = wrapped.decompose()  # a plain circuit with already supported instructions

        # gradients not supported on the following circuits
        x = Parameter("x")
        duplicated = QuantumCircuit(2)
        duplicated.rx(x, 0)
        duplicated.rx(x, 1)

        needs_chainrule = QuantumCircuit(2)
        needs_chainrule.rx(2 * x, 0)

        custom_gate = WhatAmI(x)
        unsupported = QuantumCircuit(2)
        unsupported.append(custom_gate, [0, 1])

        tests = [
            (wrapped, True),  # tuple: (circuit, gradient support)
            (plain, True),
            (duplicated, False),
            (needs_chainrule, False),
            (unsupported, False),
        ]

        # used to store the info if a gradient callable is passed into the
        # optimizer of not
        info = {"has_gradient": None}
        optimizer = partial(gradient_supplied, info=info)

        sampler = Sampler()
        estimator = Estimator()
        fidelity_primitive = ComputeUncompute(sampler)

        pvqd = PVQD(
            ansatz=None,
            initial_parameters=np.array([]),
            fidelity_primitive=fidelity_primitive,
            estimator=estimator,
            optimizer=optimizer,
        )
        problem = EvolutionProblem(self.hamiltonian, time=0.01)
        for circuit, expected_support in tests:
            with self.subTest(circuit=circuit, expected_support=expected_support):
                pvqd.ansatz = circuit
                pvqd.initial_parameters = np.zeros(circuit.num_parameters)
                _ = pvqd.evolve(problem)
                self.assertEqual(info["has_gradient"], expected_support)


if __name__ == "__main__":
    unittest.main()
