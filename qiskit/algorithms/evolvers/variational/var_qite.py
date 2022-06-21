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
from typing import Optional, Union

from scipy.integrate import OdeSolver

from qiskit.opflow import ExpectationBase
from qiskit.algorithms.evolvers.imaginary_evolver import ImaginaryEvolver
from qiskit.utils import QuantumInstance
from .solvers.ode.ode_function_factory import OdeFunctionFactory
from .variational_principles import ImaginaryVariationalPrinciple
from .var_qte import VarQTE


class VarQITE(VarQTE, ImaginaryEvolver):
    """Variational Quantum Imaginary Time Evolution algorithm.

    .. code-block::python

        from qiskit.algorithms import EvolutionProblem
        from qiskit.algorithms import VarQITE
        from qiskit import BasicAer
        from qiskit.circuit.library import EfficientSU2
        from qiskit.opflow import SummedOp, I, Z, Y, X
        from qiskit.algorithms.evolvers.variational.variational_principles.
        imaginary_mc_lachlan_principle import (
            ImaginaryMcLachlanPrinciple,
        )
        from qiskit.algorithms.evolvers.variational.solvers.ode.ode_function_factory import (
            OdeFunctionFactory,
            OdeFunctionType,
        )
        from qiskit.algorithms import EvolutionProblem
        import numpy as np

        observable = SummedOp(
            [
                0.2252 * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        ).reduce()
        d = 1
        ansatz = EfficientSU2(observable.num_qubits, reps=d)
        parameters = ansatz.parameters
        init_param_values = np.zeros(len(ansatz.parameters))
        for i in range(len(ansatz.ordered_parameters)):
            init_param_values[i] = np.pi / 2
        param_dict = dict(zip(parameters, init_param_values))
        var_principle = ImaginaryMcLachlanPrinciple()
        backend = BasicAer.get_backend("statevector_simulator")
        time = 1
        evolution_problem = EvolutionProblem(observable, time, ansatz, param_value_dict=param_dict)
        ode_function = OdeFunctionFactory(OdeFunctionType.STANDARD_ODE)
        var_qite = VarQITE(var_principle, ode_function, quantum_instance=backend)
        evolution_result = var_qite.evolve(evolution_problem)
    """

    def __init__(
        self,
        variational_principle: ImaginaryVariationalPrinciple,
        ode_solver: Union[OdeSolver, str] = "RK45",
        ode_function_factory: Optional[OdeFunctionFactory] = None,
        expectation: Optional[ExpectationBase] = None,
        imag_part_tol: float = 1e-7,
        num_instability_tol: float = 1e-7,
        quantum_instance: Optional[QuantumInstance] = None,
    ) -> None:
        r"""
        Args:
            variational_principle: Variational Principle to be used.
            ode_solver: ODE solver callable that implements a SciPy ``OdeSolver`` interface or a
                string indicating a valid method offered by SciPy.
            ode_function_factory: Factory for the ODE function. If ``None`` provided, an instance
                with default settings is created.
            expectation: An instance of ``ExpectationBase`` which defines a method for calculating
                a metric tensor and an evolution gradient and, if provided, expectation values of
                ``EvolutionProblem.aux_operators``.
            imag_part_tol: Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.
            num_instability_tol: The amount of negative value that is allowed to be
                rounded up to 0 for quantities that are expected to be non-negative.
            quantum_instance: Backend used to evaluate the quantum circuit outputs. If ``None``
                provided, everything will be evaluated based on NumPy matrix multiplication
                (which might be slow for larger numbers of qubits).
        """
        super().__init__(
            variational_principle,
            ode_solver,
            ode_function_factory,
            expectation,
            imag_part_tol,
            num_instability_tol,
            quantum_instance,
        )
