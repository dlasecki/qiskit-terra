# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An algorithm to implement a Trotterization real time-evolution."""

from typing import Union, Dict, Optional

from qiskit.algorithms import EvolutionProblem, EvolutionResult, RealEvolver, eval_observables
from qiskit.circuit import Parameter
from qiskit.opflow import (
    OperatorBase,
    SummedOp,
    PauliOp,
    CircuitOp,
    ExpectationBase,
    CircuitSampler,
)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.providers import Backend, BaseBackend
from qiskit.synthesis import ProductFormula, LieTrotter
from qiskit.utils import QuantumInstance, algorithm_globals
from .trotter_ops_validator import is_op_bound, validate_hamiltonian_form


class TrotterQRTE(RealEvolver):
    """Class for performing Quantum Real Time Evolution using Trotterization.
    Type of Trotterization is defined by a ProductFormula provided.

    Examples:

        .. jupyter-execute::

            from qiskit.opflow import X, Y, Zero
            from qiskit.algorithms import EvolutionProblem, TrotterQRTE
            from qiskit import BasicAer
            from qiskit.utils import QuantumInstance

            operator = X + Z
            initial_state = Zero
            time = 1
            evolution_problem = EvolutionProblem(operator, 1, initial_state)
            # LieTrotter with 1 rep
            backend = BasicAer.get_backend("statevector_simulator")
            quantum_instance = QuantumInstance(backend=backend)
            trotter_qrte = TrotterQRTE(quantum_instance=quantum_instance)
            evolved_state = trotter_qrte.evolve(evolution_problem).evolved_state
    """

    def __init__(
        self,
        product_formula: Optional[ProductFormula] = None,
        expectation: Optional[ExpectationBase] = None,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ) -> None:
        """
        Args:
            product_formula: A Lie-Trotter-Suzuki product formula. The default is the Lie-Trotter
                first order product formula with a single repetition.
            expectation: An instance of ExpectationBase which defines a method for calculating
                expectation values of EvolutionProblem.aux_operators.
            quantum_instance: A quantum instance used for calculations.
        """
        if product_formula is None:
            product_formula = LieTrotter()
        self._product_formula = product_formula
        self._quantum_instance = quantum_instance
        self._expectation = expectation

        self._circuit_sampler = CircuitSampler(quantum_instance)

    @property
    def product_formula(self) -> ProductFormula:
        """Returns a product formula used in the algorithm."""
        return self._product_formula

    @property
    def quantum_instance(self) -> Union[QuantumInstance, BaseBackend, Backend]:
        """Returns a quantum instance used in the algorithm."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Union[QuantumInstance, BaseBackend, Backend]):
        """Sets a quantum instance used in the algorithm."""
        self._quantum_instance = quantum_instance

    @property
    def expectation(self) -> ExpectationBase:
        """Returns an expectation used in the algorithm."""
        return self._expectation

    @classmethod
    def supports_aux_operators(cls) -> bool:
        """
        Whether computing the expectation value of auxiliary operators is supported.

        Returns:
            True if `aux_operators` expectations in the EvolutionProblem can be evaluated, False
                otherwise.
        """
        return True

    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        """
        Evolves a quantum state for a given time using the Trotterization method
        based on a product formula provided.

        .. note::
            Time-dependent Hamiltonians are not yet supported.

        Args:
            evolution_problem: Instance defining evolution problem. For the included Hamiltonian,
                only SummedOp, PauliOp are supported by TrotterQrte.

        Returns:
            Evolution result that includes an evolved state.

        Raises:
            ValueError: If `t_param` is not set to None in the EvolutionProblem (feature not
                currently supported).
            ValueError: If the `initial_state` is not provided in the EvolutionProblem.
        """
        if evolution_problem.t_param is not None:
            raise ValueError(
                "TrotterQrte does not accept a time dependent hamiltonian,"
                "`t_param` from the EvolutionProblem should be set to None."
            )

        if evolution_problem.aux_operators is not None and (
            self._quantum_instance is None or self._expectation is None
        ):
            raise ValueError(
                "aux_operators where provided for evaluations but no `expectation` or "
                "`quantum_instance` was provided."
            )
        validate_hamiltonian_form(evolution_problem.hamiltonian)
        hamiltonian = self._try_binding_params(
            evolution_problem.hamiltonian, evolution_problem.hamiltonian_value_dict
        )
        # the evolution gate
        evolution_gate = CircuitOp(
            PauliEvolutionGate(hamiltonian, evolution_problem.time, synthesis=self._product_formula)
        )

        if evolution_problem.initial_state is not None:
            quantum_state = evolution_gate @ evolution_problem.initial_state
            evolved_state = self._circuit_sampler.convert(quantum_state).eval()

        else:
            raise ValueError("`initial_state` must be provided in the EvolutionProblem.")

        evaluated_aux_ops = None
        if evolution_problem.aux_operators is not None:
            evaluated_aux_ops = eval_observables(
                self._quantum_instance,
                quantum_state.primitive,
                evolution_problem.aux_operators,
                self._expectation,
                algorithm_globals.numerical_tolerance_at_0,
            )

        return EvolutionResult(evolved_state.eval(), evaluated_aux_ops)

    @staticmethod
    def _try_binding_params(
        hamiltonian: Union[SummedOp, PauliOp],
        hamiltonian_value_dict: Dict[Parameter, Union[float, complex]],
    ) -> Union[SummedOp, PauliOp, OperatorBase]:
        """
        Tries binding parameters in a Hamiltonian.

        Args:
            hamiltonian: The Hamiltonian of that defines an evolution. Only SummedOp, PauliOp are
                supported by TrotterQrte.
            hamiltonian_value_dict: Dictionary that maps all parameters in a Hamiltonian to
                certain values.

        Returns:
            Bound Hamiltonian.

        Raises:
            ValueError: If a Hamiltonian is not of an expected type.
        """
        # PauliSumOp does not allow parametrized coefficients but after binding the parameters
        # we need to convert it into a PauliSumOp for the PauliEvolutionGate.
        if isinstance(hamiltonian, SummedOp):
            op_list = []
            for op in hamiltonian.oplist:
                if hamiltonian_value_dict is not None:
                    op_bound = op.bind_parameters(hamiltonian_value_dict)
                else:
                    op_bound = op
                is_op_bound(op_bound)
                op_list.append(op_bound)
            return sum(op_list)
        elif isinstance(hamiltonian, PauliOp):  # in case there is only a single summand
            if hamiltonian_value_dict is not None:
                op_bound = hamiltonian.bind_parameters(hamiltonian_value_dict)
            else:
                op_bound = hamiltonian

            is_op_bound(op_bound)
            return op_bound
