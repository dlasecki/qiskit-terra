# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Result object for varQTE."""
from __future__ import annotations

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.algorithms.list_or_dict import ListOrDict
from ..time_evolution_result import TimeEvolutionResult


class VarQTEResult(TimeEvolutionResult):
    """The result object for the variational quantum time evolution algorithms.

    Attributes:
        evolved_state (QuantumCircuit|Statevector): An evolved quantum state.
        aux_ops_evaluated (ListOrDict[tuple[complex, complex]] | None): Optional list of
            observables for which expected values on an evolved state are calculated. These values
            are in fact tuples formatted as (mean, standard deviation).
        observables (ListOrDict[tuple[np.ndarray, np.ndarray]] | None): Optional list of
            observables for which expected on an evolved state are calculated at each timestep.
            These values are in fact lists of tuples formatted as (mean, standard deviation).
        times (np.array | None): Optional list of times at which each observable has been evaluated.
        optimal_parameters (np.array | None): Optimal parameter values after optimization.

    """

    def __init__(
        self,
        evolved_state: QuantumCircuit | Statevector,
        aux_ops_evaluated: ListOrDict[tuple[complex, complex]] | None = None,
        optimal_parameters: np.ndarray | None = None,
    ):
        """
        Args:
            evolved_state: An evolved quantum state.
            aux_ops_evaluated: Optional list of observables for which expected values on an evolved
                state are calculated. These values are in fact tuples formatted as (mean, standard
                deviation).
            optimal_parameters: Optimal parameter values after optimization.
        """

        super().__init__(evolved_state, aux_ops_evaluated)
        self.optimal_parameters = optimal_parameters
