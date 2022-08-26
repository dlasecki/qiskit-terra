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
"""Class for building Gibbs States using Quantum Imaginary Time Evolution algorithms."""
from typing import Dict, Union, Optional

from qiskit.algorithms import VarQITE, EvolutionProblem
from qiskit.algorithms.gibbs_state_preparation.default_ansatz_builder import (
    build_ansatz,
    build_init_ansatz_params_vals,
)
from qiskit.algorithms.gibbs_state_preparation.gibbs_state_sampler import GibbsStateSampler
from qiskit.algorithms.gibbs_state_preparation.gibbs_state_builder import GibbsStateBuilder
from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase, I, CircuitSampler
from qiskit.providers import Backend
from qiskit.quantum_info import Statevector
from qiskit.utils import QuantumInstance
from qiskit.utils.backend_utils import is_aer_provider


class VarQiteGibbsStateBuilder(GibbsStateBuilder):
    """Class for building Gibbs States using Variational Quantum Imaginary Time Evolution
    algorithms."""

    def __init__(
        self,
        qite_algorithm: VarQITE,
        quantum_instance: Optional[Union[Backend, QuantumInstance]] = None,
    ):
        """
        Args:
            qite_algorithm: Variational Quantum Imaginary Time Evolution algorithm to be used for
                Gibbs State preparation.
            quantum_instance: A quantum instance to evaluate the circuits.

        Raises:
            ValueError: if an ansatz is defined on an odd number of qubits.

        """
        self._qite_algorithm = qite_algorithm
        self._ansatz = None
        self._ansatz_init_params_dict = None
        self._quantum_instance = None
        self._circuit_sampler = None
        if quantum_instance is not None:
            self.quantum_instance = quantum_instance

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Returns quantum instance."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Union[QuantumInstance, Backend]) -> None:
        """Sets quantum_instance"""
        if not isinstance(quantum_instance, QuantumInstance):
            quantum_instance = QuantumInstance(quantum_instance)

        self._quantum_instance = quantum_instance
        self._circuit_sampler = CircuitSampler(
            quantum_instance, param_qobj=is_aer_provider(quantum_instance.backend)
        )

    def _evaluate_initial_ansatz(self) -> Statevector:
        """Binds initial parameters values to an ansatz and returns the result as a state vector."""
        maximally_entangled_states = self._ansatz.assign_parameters(self._ansatz_init_params_dict)
        return Statevector(maximally_entangled_states)

    def build(
        self,
        problem_hamiltonian: OperatorBase,
        temperature: float,
        problem_hamiltonian_param_dict: Optional[Dict[Parameter, Union[complex, float]]] = None,
    ) -> GibbsStateSampler:
        """
        Creates a Gibbs state from given parameters.

        Args:
            problem_hamiltonian: Hamiltonian that defines a desired Gibbs state.
            temperature: Temperature of a desired Gibbs state.
            problem_hamiltonian_param_dict: If a problem Hamiltonian is parametrized, a dictionary
                that maps all of its parameters to certain values.

        Returns:
            GibbsState object that includes a relevant quantum state functions as well as
            metadata.
        """
        if self._ansatz is None:  # set default ansatz
            self._set_default_ansatz(problem_hamiltonian)

        time = 1 / (2 * self.BOLTZMANN_CONSTANT * temperature)

        param_dict = {**self._ansatz_init_params_dict, **problem_hamiltonian_param_dict}
        extended_hamiltonian = self._extend_hamiltonian_to_aux_registers(problem_hamiltonian)
        evolution_problem = EvolutionProblem(
            hamiltonian=extended_hamiltonian, time=time, param_value_dict=param_dict
        )
        gibbs_state_function = self._qite_algorithm.evolve(evolution_problem)

        aux_registers = set(range(int(self._ansatz.num_qubits / 2), int(self._ansatz.num_qubits)))

        return GibbsStateSampler(
            gibbs_state_function=gibbs_state_function,
            hamiltonian=problem_hamiltonian,
            temperature=temperature,
            ansatz=self._ansatz,
            ansatz_params_dict=None,
            aux_registers=aux_registers,
            quantum_instance=self._quantum_instance,
        )

    def _extend_hamiltonian_to_aux_registers(self, hamiltonian: OperatorBase):
        """This class operates of a purified Gibbs state which includes auxiliary registers not
        present in the original Hamiltonian. This method creates an extended Hamiltonian in the
        bigger space that includes identity operators on auxiliary registers.

        Args:
            hamiltonian: Hamiltonian to be extended to a bigger space.

        Returns:
            An extended Hamiltonian in a bigger space, with identity operators on auxiliary
            registers.
        """
        num_qubits = self._ansatz.num_qubits
        if hamiltonian.num_qubits != num_qubits / 2:
            raise ValueError("Mismatch between number of qubits in a Hamiltonian and in an ansatz.")

        return hamiltonian ^ (I * (num_qubits / 2))

    def _set_default_ansatz(self, problem_hamiltonian: OperatorBase) -> None:
        """
        Sets a default ansatz with default parameters for a Gibbs state preparation.

        Args:
            problem_hamiltonian: Hamiltonian that defines a desired Gibbs state.
        """
        num_qubits = problem_hamiltonian.num_qubits
        depth = 1
        self._ansatz = build_ansatz(2 * num_qubits, depth)
        ansatz_init_params_vals = build_init_ansatz_params_vals(2 * num_qubits, depth)
        self._ansatz_init_params_dict = dict(
            zip(self._ansatz.ordered_parameters, ansatz_init_params_vals)
        )
