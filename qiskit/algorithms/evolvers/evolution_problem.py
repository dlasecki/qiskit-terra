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

"""Evolution problem class."""

from typing import Union, Optional, Dict

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase, StateFn
from ..list_or_dict import ListOrDict


class EvolutionProblem:
    """Evolution problem class.

    This class is the input to time evolution algorithms and must contain information on the total
    evolution time, a quantum state to be evolved and under which Hamiltonian the state is evolved.
    """

    def __init__(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: Optional[Union[StateFn, QuantumCircuit]] = None,
        aux_operators: Optional[ListOrDict[OperatorBase]] = None,
        truncation_threshold: float = 1e-12,
        t_param: Optional[Parameter] = None,
    ):
        """
        Args:
            hamiltonian: The Hamiltonian under which to evolve the system. It cannot be parametrized
                unless it is a time parameter specified in ``t_param``.
            time: Total time of evolution.
            initial_state: The quantum state to be evolved for methods like Trotterization.
                For variational time evolutions, where the evolution happens in an ansatz,
                this argument is not required.
            aux_operators: Optional list of auxiliary operators to be evaluated with the
                evolved ``initial_state`` and their expectation values returned.
            truncation_threshold: Defines a threshold under which values can be assumed to be 0.
                Used when ``aux_operators`` is provided.
            t_param: Time parameter in case of a time-dependent Hamiltonian. This
                free parameter must be within the ``hamiltonian``.

        Raises:
            ValueError: If non-positive time of evolution is provided.
        """

        self.t_param = t_param
        self.hamiltonian = hamiltonian
        self.time = time
        self.initial_state = initial_state
        self.aux_operators = aux_operators
        self.truncation_threshold = truncation_threshold

    @property
    def time(self) -> float:
        """Returns time."""
        return self._time

    @time.setter
    def time(self, time: float) -> None:
        """
        Sets time and validates it.

        Raises:
            ValueError: If time is not positive.
        """
        if time <= 0:
            raise ValueError(f"Evolution time must be > 0 but was {time}.")
        self._time = time

    def validate_params(self) -> None:
        """
        Checks if Hamiltonian does not include parameters unless it is a time parameter specified
        in ``t_param``.

        Raises:
            ValueError: If Hamiltonian contains illegal parameters.
        """
        if isinstance(self.hamiltonian, OperatorBase):
            hamiltonian_params = self.hamiltonian.parameters
            if len(hamiltonian_params) > 1:
                raise ValueError(
                    f"Unbound parameters detected in the Hamiltonian: {hamiltonian_params}"
                )
            if len(hamiltonian_params) == 1 and list(hamiltonian_params)[0] != self.t_param:
                raise ValueError(
                    "Time parameter in the Hamiltonian does not match the ``t_param`` provided."
                )
