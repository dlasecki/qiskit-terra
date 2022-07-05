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
"""Class representing a quantum state of a Gibbs State along with metadata and gradient
calculation methods."""
from typing import Optional, Union, Dict, Set

import numpy as np
from numpy.typing import NDArray

from qiskit.circuit import Parameter
from qiskit.opflow import StateFn, OperatorBase, Gradient, CircuitStateFn, CircuitSampler
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
from qiskit.utils.backend_utils import is_aer_provider


class GibbsStateSampler:
    """Class representing a quantum state of a Gibbs State along with metadata and gradient
    calculation methods."""

    def __init__(
        self,
        gibbs_state_function: StateFn,
        hamiltonian: OperatorBase,
        temperature: float,
        ansatz: Optional[OperatorBase] = None,
        ansatz_params_dict: Optional[Dict[Parameter, Union[complex, float]]] = None,
        aux_registers: Optional[Set[int]] = None,
        quantum_instance: Optional[Union[Backend, QuantumInstance]] = None,
    ):
        """
        Args:
            gibbs_state_function: Quantum state function of a Gibbs state.
            hamiltonian: Hamiltonian used to build a Gibbs state.
            temperature: Temperature used to build a Gibbs state.
            ansatz: Ansatz that gave rise to a Gibbs state.
            ansatz_params_dict: Dictionary that maps ansatz parameters to values optimal for a
                Gibbs state.
            aux_registers: Set of indices (0-indexed) of registers in an ansatz that are auxiliary,
                i.e. they do not contain a Gibbs state. E.g. in VarQiteGibbsStateBuilder
                the second half or registers is auxiliary.
            quantum_instance: A quantum instance to evaluate the circuits.
        """
        self.gibbs_state_function = gibbs_state_function
        self.hamiltonian = hamiltonian
        self.temperature = temperature
        self.ansatz = ansatz
        self.ansatz_params_dict = ansatz_params_dict
        self.aux_registers = aux_registers
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

    def eval_gibbs_state_matrix(self):
        """Evaluates a Gibbs state matrix on a given backend. Note that this process is generally
        not efficient and should not be used in production settings."""
        pass

    def sample(self) -> NDArray[Union[complex, float]]:  # calc p_qbm
        """
        Samples probabilities from a Gibbs state.

        Returns:
            An array of samples probabilities.
        """
        operator = CircuitStateFn(self.ansatz)
        sampler = self._circuit_sampler.convert(operator, self.ansatz_params_dict)
        amplitudes_with_aux_regs = sampler.eval().primitive
        probs = self._discard_aux_registers(amplitudes_with_aux_regs)
        return probs

    def calc_ansatz_gradients(
        self,
        gradient_method: str = "param_shift",
    ) -> NDArray[NDArray[Union[complex, float]]]:
        """
        Calculates gradients of a Gibbs state w.r.t. desired gradient_params that parametrize the
        Gibbs state.

        Args:
            gradient_method: A desired gradient method chosen from the Qiskit Gradient Framework.

        Returns:
            Calculated gradients with respect to each parameter indicated in gradient_params
            with bound parameter values.

        Raises:
            ValueError: If ansatz and ansatz_params_dict are not both provided.
        """
        if not self.ansatz or not self.ansatz_params_dict:
            raise ValueError(
                "Both ansatz and ansatz_params_dict must be present in the class to compute "
                "gradients."
            )

        operator = CircuitStateFn(self.ansatz)

        gradient_amplitudes_with_aux_regs = Gradient(grad_method=gradient_method).gradient_wrapper(
            operator, self.ansatz.ordered_parameters, backend=self.quantum_instance
        )
        # Get the values for the gradient of the sampling probabilities w.r.t. the Ansatz parameters
        gradient_amplitudes_with_aux_regs = gradient_amplitudes_with_aux_regs(
            self.ansatz_params_dict.values()
        )
        # TODO gradients of amplitudes or probabilities?
        state_grad = self._discard_aux_registers_gradients(gradient_amplitudes_with_aux_regs)
        return state_grad

    def _discard_aux_registers(
        self, amplitudes_with_aux_regs: NDArray[Union[complex, float]]
    ) -> NDArray[Union[complex, float]]:
        """
        Accepts an object with complex amplitudes sampled from a state with auxiliary
        registers and processes bit strings of qubit labels. For the default choice of an ansatz
        in the GibbsStateBuilder, this method gets rid of the second half of qubits that
        correspond to an auxiliary system. Then, it aggregates complex amplitudes and returns the
        vector of probabilities. Indices of returned probability vector correspond to labels of a
        reduced qubit state.

        Args:
            amplitudes_with_aux_regs: An array of amplitudes sampled from a Gibbs state circuit
                that includes auxiliary registers and their measurement outcomes.

        Returns:
            An array of probability samples from a Gibbs state only (excluding auxiliary registers).

        Raises:
            ValueError: If a provided number of qubits for an ansatz is not even.
        """
        kept_num_qubits = self.ansatz.num_qubits - len(self.aux_registers)

        amplitudes = amplitudes_with_aux_regs.data
        amplitudes_qubit_labels_ints = amplitudes_with_aux_regs.indices
        reduced_qubits_amplitudes = np.zeros(pow(2, kept_num_qubits))

        for qubit_label_int, amplitude in zip(amplitudes_qubit_labels_ints, amplitudes):
            reduced_label = self._reduce_label(qubit_label_int)
            reduced_qubits_amplitudes[reduced_label] += np.conj(amplitude) * amplitude

        return reduced_qubits_amplitudes

    def _reduce_label(self, label: int) -> int:
        """Accepts an integer label that represents a measurement outcome and discards auxiliary
        registers in the label.

        Args:
            label: An integer label that represents a measurement outcome.

        Returns:
            A reduced label after discarding indices of auxiliary quantum registers.
        """
        cnt = len(bin(label)) - 2
        cnt2 = 0
        reduced_label_bits = []
        while cnt:
            bit = label & 1
            label = label >> 1
            if cnt2 not in self.aux_registers:
                reduced_label_bits.append(bit)
            cnt -= 1
            cnt2 += 1
        reduced_label = 0
        for bit in reduced_label_bits[::-1]:
            reduced_label = (reduced_label << 1) | bit
        return reduced_label

    def _discard_aux_registers_gradients(
        self, amplitudes_with_aux_regs: NDArray[Union[complex, float]]
    ) -> NDArray[NDArray[Union[complex, float]]]:
        """
        Accepts an object with complex amplitude gradients sampled from a state with auxiliary
        registers and processes bit strings of qubit labels. For the default choice of an
        ansatz in the GibbsStateBuilder, this method gets rid of the second half of qubits that
        correspond to an auxiliary system. Then, it aggregates complex amplitudes gradients and
        returns the vector of probability gradients. Indices of returned probability gradients
        vector correspond to labels of a reduced qubit state.

        Args:
            amplitudes_with_aux_regs: An array of amplitudes gradients sampled and calculated from
                a Gibbs state circuit that includes auxiliary registers and their measurement
                outcomes.

        Returns:
            An array of probability gradients from a Gibbs state only (excluding auxiliary
            registers).
        """
        reduced_qubits_amplitudes = np.zeros(len(amplitudes_with_aux_regs), dtype=object)
        for ind, amplitude_data in enumerate(amplitudes_with_aux_regs):
            res = self._discard_aux_registers(amplitude_data)
            reduced_qubits_amplitudes[ind] = res

        return reduced_qubits_amplitudes
