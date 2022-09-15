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

"""An algorithm to implement a Trotterization real time-evolution."""

from __future__ import annotations

import warnings

from qiskit.algorithms.observables_evaluator import estimate_observables

from qiskit import QuantumCircuit
from qiskit.algorithms.time_evolvers.time_evolution_problem import TimeEvolutionProblem
from qiskit.algorithms.time_evolvers.time_evolution_result import TimeEvolutionResult
from qiskit.algorithms.time_evolvers.real_time_evolver import RealTimeEvolver
from qiskit.opflow import (
    PauliSumOp,
)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info import Pauli
from qiskit.synthesis import ProductFormula, LieTrotter


class TrotterQRTE(RealTimeEvolver):
    """Quantum Real Time Evolution using Trotterization.
    Type of Trotterization is defined by a ProductFormula provided.

    Attributes:
        product_formula: A Lie-Trotter-Suzuki product formula. The default is the Lie-Trotter
            first order product formula with a single repetition.

    Examples:

        .. code-block:: python

            from qiskit.opflow import X, Z, Zero
            from qiskit.algorithms.time_evolvers import TimeEvolutionProblem, TrotterQRTE

            operator = X + Z
            initial_state = Zero
            time = 1
            evolution_problem = TimeEvolutionProblem(operator, 1, initial_state)
            # LieTrotter with 1 rep
            trotter_qrte = TrotterQRTE()
            evolved_state = trotter_qrte.evolve(evolution_problem).evolved_state
    """

    def __init__(
        self,
        product_formula: ProductFormula | None = None,
        estimator: BaseEstimator | None = None,
    ) -> None:
        """
        Args:
            product_formula: A Lie-Trotter-Suzuki product formula. The default is the Lie-Trotter
                first order product formula with a single repetition.
            estimator: An estimator primitive used for calculating expectation values of
                EvolutionProblem.aux_operators.
        """
        if product_formula is None:
            product_formula = LieTrotter()
        self.product_formula = product_formula
        self._estimator = estimator

    @property
    def estimator(self) -> BaseEstimator | None:
        """
        Returns an estimator.
        """
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimator) -> None:
        """
        Sets an estimator.
        """
        self._estimator = estimator

    @classmethod
    def supports_aux_operators(cls) -> bool:
        """
        Whether computing the expectation value of auxiliary operators is supported.

        Returns:
            True if ``aux_operators`` expectations in the EvolutionProblem can be evaluated, False
                otherwise.
        """
        return True

    def evolve(self, evolution_problem: TimeEvolutionProblem) -> TimeEvolutionResult:
        """
        Evolves a quantum state for a given time using the Trotterization method
        based on a product formula provided. The result is provided in the form of a quantum
        circuit. If auxiliary operators are included in the ``evolution_problem``, they are
        evaluated on an evolved state using an estimator primitive provided.

        .. note::
            Time-dependent Hamiltonians are not yet supported.

        Args:
            evolution_problem: Instance defining evolution problem. For the included Hamiltonian,
                ``Pauli`` or ``PauliSumOp`` are supported by TrotterQRTE.

        Returns:
            Evolution result that includes an evolved state as a quantum circuit and, optionally,
            auxiliary operators evaluated for a resulting state on an estimator primitive.

        Raises:
            ValueError: If ``t_param`` is not set to ``None`` in the EvolutionProblem (feature not
                currently supported).
            ValueError: If the ``initial_state`` is not provided in the EvolutionProblem.
            ValueError: If an unsupported Hamiltonian type is provided.
        """
        evolution_problem.validate_params()
        if evolution_problem.t_param is not None:
            raise ValueError(
                "TrotterQRTE does not accept a time dependent hamiltonian,"
                "``t_param`` from the EvolutionProblem should be set to None."
            )

        if evolution_problem.aux_operators is not None and self.estimator is None:
            warnings.warn(
                "The evolution problem contained aux_operators but no estimator was provided. "
                "The algorithm continues without calculating these quantities. "
            )
        hamiltonian = evolution_problem.hamiltonian
        if not isinstance(hamiltonian, (Pauli, PauliSumOp)):
            raise ValueError(
                f"TrotterQRTE only accepts Pauli | PauliSumOp, {type(hamiltonian)} provided."
            )
        # the evolution gate
        evolution_gate = PauliEvolutionGate(hamiltonian, evolution_problem.time, synthesis=self.product_formula)

        if evolution_problem.initial_state is not None:
            initial_state = evolution_problem.initial_state
            evolved_state = QuantumCircuit(initial_state.num_qubits)
            evolved_state.append(initial_state, evolved_state.qubits)
            evolved_state.append(evolution_gate, evolved_state.qubits)

        else:
            raise ValueError("``initial_state`` must be provided in the EvolutionProblem.")

        evaluated_aux_ops = None
        if evolution_problem.aux_operators is not None:
            evaluated_aux_ops = estimate_observables(
                self.estimator,
                evolved_state,
                evolution_problem.aux_operators,
                evolution_problem.truncation_threshold,
            )

        return TimeEvolutionResult(evolved_state, evaluated_aux_ops)
