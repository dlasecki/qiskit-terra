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
from qiskit.algorithms.eigen_solvers.eigen_solver import ListOrDict
from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase, StateFn


class EvolutionProblem:
    """Evolution problem class."""

    def __init__(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: Union[StateFn, QuantumCircuit],
        aux_operators: Optional[ListOrDict[OperatorBase]] = None,
        t_param: Optional[Parameter] = None,
        hamiltonian_value_dict: Optional[Dict[Parameter, Union[complex]]] = None,
    ):
        """
        Args:
            hamiltonian: The Hamiltonian under which to evolve the system.
            time: Total time of evolution.
            initial_state: Quantum state to be evolved.
            aux_operators: Optional list of auxiliary operators to be evaluated with the
                evolved initial_state and their expectation values returned.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            hamiltonian_value_dict: Dictionary that maps all parameters in a Hamiltonian to certain
                values, including the t_param.
        """

        self.hamiltonian = hamiltonian
        self.time = time
        self.initial_state = initial_state
        self.aux_operators = aux_operators
        self.t_param = t_param
        self.hamiltonian_value_dict = hamiltonian_value_dict
