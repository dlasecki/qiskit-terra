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

"""Class for a Real Time Dependent Variational Principle."""

from typing import Union, List

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import Y, OperatorBase, ListOp, StateFn
from ..calculators import (
    qfi_calculator,
    evolution_grad_calculator,
)
from .real_variational_principle import (
    RealVariationalPrinciple,
)


class RealTimeDependentVariationalPrinciple(RealVariationalPrinciple):
    """Class for a Real Time Dependent Variational Principle. It works by evaluating the Lagrangian
    corresponding the given system at a parametrized trial state and applying the Euler-Lagrange
    equation. The principle leads to a system of linear equations handled by the
    `~qiskit.algorithms.evolvers.variational.solvers.VarQTELinearSolver` class. The real variant
    means that we consider real time dynamics.
    """

    def calc_metric_tensor(
        self,
        ansatz: QuantumCircuit,
        parameters: List[Parameter],
    ) -> ListOp:
        """
        Calculates a metric tensor according to the rules of this variational principle.

        Args:
            ansatz: Quantum state in the form of a parametrized quantum circuit to be used for
                calculating a metric tensor.
            parameters: Parameters with respect to which gradients should be computed.

        Returns:
            Transformed metric tensor.
        """
        raw_metric_tensor_imag = qfi_calculator.calculate(
            ansatz, parameters, self._qfi_method, basis=-Y
        )

        return raw_metric_tensor_imag * 0.25

    def calc_evolution_grad(
        self,
        hamiltonian: OperatorBase,
        ansatz: Union[StateFn, QuantumCircuit],
        parameters: List[Parameter],
    ) -> OperatorBase:
        """
        Calculates an evolution gradient according to the rules of this variational principle.

        Args:
            hamiltonian: Hamiltonian for which an evolution gradient should be calculated.
            ansatz: Quantum state in the form of a parametrized quantum circuit to be used for
                calculating an evolution gradient.
            parameters: Parameters with respect to which gradients should be computed.

        Returns:
            Transformed evolution gradient.
        """
        raw_evolution_grad_real = evolution_grad_calculator.calculate(
            hamiltonian, ansatz, parameters, self._grad_method
        )

        return raw_evolution_grad_real * 0.5
