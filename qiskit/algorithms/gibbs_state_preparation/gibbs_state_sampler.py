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
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from qiskit import QuantumCircuit
from qiskit.algorithms import AlgorithmError
from qiskit.algorithms.gradients import BaseSamplerGradient, ParamShiftSamplerGradient
from qiskit.circuit import Parameter
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseSampler
from qiskit.quantum_info.operators.base_operator import BaseOperator


class GibbsStateSampler:
    """Class representing a quantum state of a Gibbs State along with metadata and gradient
    calculation methods."""

    def __init__(
        self,
        sampler: BaseSampler,
        gibbs_state_function: QuantumCircuit,
        hamiltonian: BaseOperator | PauliSumOp,
        temperature: float,
        ansatz: QuantumCircuit | None = None,
        ansatz_params_dict: dict[Parameter, complex | float] | None = None,
        aux_registers: set[int] | None = None,
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
        """
        self.sampler = sampler
        self.gibbs_state_function = gibbs_state_function
        self.hamiltonian = hamiltonian
        self.temperature = temperature
        self.ansatz = ansatz
        self.ansatz_params_dict = ansatz_params_dict
        self.aux_registers = aux_registers

    def sample(self) -> NDArray[complex | float]:  # calc p_qbm
        """
        Samples probabilities from a Gibbs state.

        Returns:
            An array of samples probabilities.
        """
        # TODO what if None?
        try:
            amplitudes_with_aux_regs = self.sampler.run(
                [self.ansatz], [list(self.ansatz_params_dict.values())]
            ).result()[0]
        except AlgorithmError as exc:
            raise AlgorithmError("Estimator job failed.") from exc

        probs = self._discard_aux_registers(amplitudes_with_aux_regs)
        return probs

    def calc_ansatz_gradients(
        self,
        gradient: BaseSamplerGradient | None = None,
    ) -> NDArray[NDArray[complex | float]]:
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
        if gradient is None:
            gradient = ParamShiftSamplerGradient(self.sampler)
        if not self.ansatz or not self.ansatz_params_dict:
            raise ValueError(
                "Both ansatz and ansatz_params_dict must be present in the class to compute "
                "gradients."
            )
        try:
            gradient_amplitudes_with_aux_regs = gradient.run(
                [self.ansatz], [list(self.ansatz_params_dict.values())]
            ).result()[0]
        except AlgorithmError as exc:
            raise AlgorithmError("Estimator job failed.") from exc

        # TODO gradients of amplitudes or probabilities?
        state_grad = self._discard_aux_registers_gradients(gradient_amplitudes_with_aux_regs)
        return state_grad

    def _discard_aux_registers(
        self, amplitudes_with_aux_regs: NDArray[complex | float]
    ) -> NDArray[complex | float]:
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
        self, amplitudes_with_aux_regs: NDArray[complex | float]
    ) -> NDArray[NDArray[complex | float]]:
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
