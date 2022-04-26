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
        initial_state: Union[StateFn, QuantumCircuit],
        aux_operators: Optional[ListOrDict[OperatorBase]] = None,
        truncation_threshold: float = 1e-12,
        t_param: Optional[Parameter] = None,
        hamiltonian_value_dict: Optional[Dict[Parameter, complex]] = None,
    ):
        """
        Args:
            hamiltonian: The Hamiltonian under which to evolve the system.
            time: Total time of evolution.
            initial_state: Quantum state to be evolved.
            aux_operators: Optional list of auxiliary operators to be evaluated with the
                evolved ``initial_state`` and their expectation values returned.
            truncation_threshold: Defines a threshold under which values can be assumed to be 0.
                Used when ``aux_operators`` is provided.
            t_param: Time parameter in case of a time-dependent Hamiltonian. This
                free parameter must be within the ``hamiltonian``.
            hamiltonian_value_dict: If the Hamiltonian contains free parameters, this
                dictionary maps all these parameters to values.

        Raises:
            ValueError: If non-positive time of evolution is provided.
            ValueError: If no ``initial_state`` is provided.
            ValueError: If not all parameter values are provided.
        """

        self.t_param = t_param
        self.hamiltonian_value_dict = hamiltonian_value_dict
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
        Checks if all parameters present in the Hamiltonian are also present in the dictionary
        that maps them to values.

        Raises:
            ValueError: If Hamiltonian parameters cannot be bound with data provided.
        """
        if isinstance(self.hamiltonian, OperatorBase):
            t_param_set = set()
            if self.t_param is not None:
                t_param_set.add(self.t_param)
            hamiltonian_dict_param_set = set()
            if self.hamiltonian_value_dict is not None:
                hamiltonian_dict_param_set = hamiltonian_dict_param_set.union(
                    set(self.hamiltonian_value_dict.keys())
                )
            params_set = t_param_set.union(hamiltonian_dict_param_set)
            hamiltonian_param_set = set(self.hamiltonian.parameters)

            if hamiltonian_param_set != params_set:
                raise ValueError(
                    f"Provided parameters {params_set} do not match Hamiltonian parameters "
                    f"{hamiltonian_param_set}."
                )
