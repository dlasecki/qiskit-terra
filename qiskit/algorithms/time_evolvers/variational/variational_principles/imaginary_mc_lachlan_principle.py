# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class for an Imaginary McLachlan's Variational Principle."""
from __future__ import annotations

import warnings

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import Estimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.algorithms.gradients import BaseEstimatorGradient, BaseQGT, QFI, DerivativeType

from .imaginary_variational_principle import (
    ImaginaryVariationalPrinciple,
)


class ImaginaryMcLachlanPrinciple(ImaginaryVariationalPrinciple):
    """Class for an Imaginary McLachlan's Variational Principle. It aims to minimize the distance
    between both sides of the Wick-rotated Schrödinger equation with a quantum state given as a
    parametrized trial state. The principle leads to a system of linear equations handled by a
    linear solver. The imaginary variant means that we consider imaginary time dynamics.
    """

    def __init__(
        self,
        qgt: BaseQGT | None = None,
        gradient: BaseEstimatorGradient | None = None,
    ) -> None:
        """
        Args:
            qfi: Instance of a the GQT class used to compute the QFI. If ``None`` provided, ``LinCombQGT``
                is used.
            gradient: Instance of a class used to compute the state gradient. If ``None`` provided,
                ``LinCombEstimatorGradient`` is used.
        """

        self._validate_grad_settings(gradient)
        # pylint: disable=cyclic-import
        from qiskit.algorithms.gradients import LinCombQGT, LinCombEstimatorGradient

        if gradient is not None and gradient._estimator is not None and qgt is None:
            estimator = gradient._estimator
            qgt = LinCombQGT(estimator)
        elif qgt is None and gradient is None:
            estimator = Estimator()
            qgt = LinCombQGT(estimator)
            gradient = LinCombEstimatorGradient(estimator)

        super().__init__(QFI(qgt), gradient)

    def evolution_gradient(
        self,
        hamiltonian: BaseOperator,
        ansatz: QuantumCircuit,
        param_values: list[float],
        gradient_params: list[Parameter] | None = None,
    ) -> np.ndarray:
        """
        Calculates an evolution gradient according to the rules of this variational principle.

        Args:
            hamiltonian: Operator used for Variational Quantum Time Evolution.
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            param_values: Values of parameters to be bound.
            gradient_params: List of parameters with respect to which gradients should be computed.
                If ``None`` given, gradients w.r.t. all parameters will be computed.

        Returns:
            An evolution gradient.

        Raises:
            AlgorithmError: If a gradient job fails.
        """
        # pylint: disable=cyclic-import
        from qiskit.algorithms import AlgorithmError

        try:
            evolution_grad_lse_rhs = (
                self.gradient.run([ansatz], [hamiltonian], [param_values], [gradient_params])
                .result()
                .gradients[0]
            )

        except Exception as exc:
            raise AlgorithmError("The gradient primitive job failed!") from exc

        return -0.5 * evolution_grad_lse_rhs

    @staticmethod
    def _validate_grad_settings(gradient):
        if (
            gradient is not None
            and hasattr(gradient, "_derivative_type")
            and gradient._derivative_type != DerivativeType.REAL
        ):
            warnings.warn(
                "A gradient instance with a setting for calculating imaginary part of "
                "the gradient was provided. This variational principle requires the"
                "real part. The setting to real was changed automatically."
            )
            gradient._derivative_type = DerivativeType.REAL
