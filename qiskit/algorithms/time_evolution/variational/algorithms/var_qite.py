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

"""Variational Quantum Imaginary Time Evolution algorithm."""

from typing import Optional, Union, Dict, List

from scipy.integrate import OdeSolver

from qiskit.algorithms.time_evolution.evolution_result import EvolutionResult
from qiskit.algorithms.time_evolution.imaginary.qite import Qite

from qiskit.algorithms.time_evolution.variational.error_calculators.gradient_errors\
    .imaginary_error_calculator import (
    ImaginaryErrorCalculator,
)
from qiskit.algorithms.time_evolution.variational.solvers.ode.error_based_ode_function_generator \
    import \
    ErrorBasedOdeFunctionGenerator
from qiskit.algorithms.time_evolution.variational.variational_principles.imaginary\
    .imaginary_variational_principle import (
    ImaginaryVariationalPrinciple,
)
from qiskit.algorithms.time_evolution.variational.solvers.ode.abstract_ode_function_generator \
    import (
    AbstractOdeFunctionGenerator,
)
from qiskit.circuit import Parameter
from qiskit.opflow import (
    OperatorBase,
    Gradient,
    StateFn,
)
from qiskit.algorithms.time_evolution.variational.algorithms.var_qte import VarQte
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class VarQite(Qite, VarQte):
    """Variational Quantum Imaginary Time Evolution algorithm."""

    def __init__(
        self,
        variational_principle: ImaginaryVariationalPrinciple,
        ode_function_generator: AbstractOdeFunctionGenerator,
        regularization: Optional[str] = None,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
        ode_solver_callable: OdeSolver = "RK45",
        allowed_imaginary_part: float = 1e-7,
        allowed_num_instability_error: float = 1e-7,
    ):
        r"""
        Args:
            variational_principle: Variational Principle to be used.
            regularization: Use the following regularization with a least square method to solve the
                            underlying system of linear equations.
                            Can be either None or ``'ridge'`` or ``'lasso'`` or ``'perturb_diag'``
                            ``'ridge'`` and ``'lasso'`` use an automatic optimal parameter search
                            If regularization is None but the metric is ill-conditioned or singular
                            then a least square solver is used without regularization.
            backend: Backend used to evaluate the quantum circuit outputs

                             If True use the argument that minimizes the error.
            ode_solver_callable: ODE solver callable that follows a SciPy OdeSolver interface.
            allowed_imaginary_part: Allowed value of an imaginary part that can be neglected if no
                                    imaginary part is expected.
            allowed_num_instability_error: The amount of negative value that is allowed to be
                                           rounded up to 0 for quantities that are expected to be
                                           non-negative.
        """
        super().__init__(
            variational_principle,
            ode_function_generator,
            regularization,
            backend,
            ode_solver_callable,
            allowed_imaginary_part,
            allowed_num_instability_error,
        )

    def evolve(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: Optional[StateFn] = None,
        observable: Optional[OperatorBase] = None,
        t_param: Optional[Parameter] = None,
        hamiltonian_value_dict: Optional[Dict[Parameter, Union[float, complex]]] = None,
    ) -> EvolutionResult:
        """
        Apply Variational Quantum Imaginary Time Evolution (VarQITE) w.r.t. the given
        operator.
        Args:
            hamiltonian:
                ⟨ψ(ω)|H(t, theta)|ψ(ω)〉
                Operator used vor Variational Quantum Imaginary Time Evolution (VarQITE)
                The coefficient of the operator (operator.coeff) determines the evolution
                time.
                The operator may be given either as a composed op consisting of a Hermitian
                observable and a CircuitStateFn or a ListOp of a CircuitStateFn with a
                ComboFn.
                The latter case enables the evaluation of a Quantum Natural Gradient.
            time: Total time of evolution.
            initial_state: Quantum state to be evolved.
            observable: Observable to be evolved. Not supported by VarQite.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            hamiltonian_value_dict: Dictionary that maps all parameters in a Hamiltonian to
                                    certain values, including the t_param. If no state parameters
                                    are provided, they are generated randomly.
        Returns:
            StateFn (parameters are bound) which represents an approximation to the
            respective time evolution.
        """
        init_state_param_dict = self._create_init_state_param_dict(
            hamiltonian_value_dict, list(initial_state.parameters)
        )
        init_state_parameters = list(init_state_param_dict.keys())  # TODO
        self._variational_principle._lazy_init(hamiltonian, initial_state, init_state_parameters)
        self.bind_initial_state(StateFn(initial_state), init_state_param_dict)
        self._init_ham_objects()

        error_calculator = None
        if isinstance(self._ode_function_generator, ErrorBasedOdeFunctionGenerator):
            error_calculator = ImaginaryErrorCalculator(
                self._hamiltonian_squared,
                self._operator,
                self._h_squared_circ_sampler,
                self._operator_circ_sampler,
                self._backend,
                self._allowed_imaginary_part,
                self._allowed_num_instability_error,
            )

        evolved_object = super()._evolve_helper(
            init_state_param_dict,
            hamiltonian,
            time,
            t_param,
            error_calculator,
            initial_state,
            observable,
        )

        return EvolutionResult(evolved_object)

    def gradient(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: StateFn,
        gradient_object: Gradient,
        observable: Optional[OperatorBase] = None,
        t_param: Optional[Parameter] = None,
        hamiltonian_value_dict: Optional[Dict[Parameter, Union[float, complex]]] = None,
        gradient_params: Optional[List[Parameter]] = None,
    ):
        """Performs Variational Quantum Imaginary Time Evolution of gradient expressions."""
        raise NotImplementedError()
