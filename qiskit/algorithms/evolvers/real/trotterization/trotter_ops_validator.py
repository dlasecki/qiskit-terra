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

"""Set of method for validating input to TrotterQrte algorithm."""

import numbers
from typing import Union

from qiskit.circuit import Parameter, ParameterExpression
from qiskit.opflow import (
    SummedOp,
    PauliOp,
)


def is_op_bound(operator: Union[SummedOp, PauliOp]) -> None:
    """Checks if an operator provided has all parameters bound.

    Args:
        operator: Operator to be checked.

    Raises:
        ValueError: If an operator has unbound parameters.
    """
    if len(operator.parameters) != 0:
        raise ValueError(
            f"Did not manage to bind all parameters in the Hamiltonian, "
            f"these parameters encountered: {operator.parameters}."
        )


def validate_hamiltonian_form(hamiltonian: Union[SummedOp, PauliOp]):
    """Validates that a Hamiltonian is of a correct type and with expected dependence on
    parameters.

    Args:
        hamiltonian: Hamiltonian to be validated.

    Raises:
        ValueError: if an invalid Hamiltonian is provided.
    """
    value_error = ValueError(
        "Hamiltonian term has a coefficient that is not a linear function of a "
        "single parameter. It is not supported."
    )
    if isinstance(hamiltonian, SummedOp):
        if isinstance(hamiltonian.coeff, ParameterExpression):
            raise ValueError(
                f"The coefficient multiplying the whole Hamiltonian cannot be a "
                f"ParameterExpression. The following coefficient was detected: {hamiltonian.coeff}."
            )
        for op in hamiltonian.oplist:
            if not _is_pauli_lin_single_param(op):
                raise value_error
    elif isinstance(hamiltonian, PauliOp):
        if not _is_pauli_lin_single_param(hamiltonian):
            raise value_error
    else:
        raise ValueError(
            f"Hamiltonian not a SummedOp/PauliOp which are the only options supported. The "
            f"following type detected instead: {type(hamiltonian)}."
        )


def _is_pauli_lin_single_param(operator: PauliOp) -> bool:
    """Checks if an operator provided is linear w.r.t. one and only one parameter.

    Args:
        operator: Operator to be checked.

    Returns:
        True or False depending on whether an operator is linear in a single param and only contains
        a single param.

    Raises:
        ValueError: If an operator contains more than 1 parameter.
    """
    if not isinstance(operator, PauliOp):
        raise ValueError(f"Only PauliOp expected. {type(operator)} provided")
    if _is_operator_parametrized(operator):
        return True
    if len(operator.coeff.parameters) > 1:
        raise ValueError(
            "Term of a Hamiltonian has a coefficient that depends on several "
            "parameters. Only dependence on a single parameter is allowed."
        )
    gradient = _operator_derivative(operator)
    return isinstance(gradient, numbers.Number)


def _operator_derivative(
    operator: PauliOp,
) -> Union[numbers.Number, Parameter, ParameterExpression]:
    """
    Calculates the gradient of an operator coefficient.

    Args:
        operator: PauliOp.

    Returns:
          Gradient of a PauliOp coefficient.
    """
    single_parameter_expression = operator.coeff
    parameter = list(single_parameter_expression.parameters)[0]
    gradient = single_parameter_expression.gradient(parameter)
    return gradient


def _is_operator_parametrized(operator: PauliOp) -> bool:
    """
    Checks if an operator is parametrized.

    Args:
        operator: PauliOp.

    Returns:
        Boolean flag indicating whether the PauliOp is parametrized or not.
    """
    return (
        not isinstance(operator.coeff, Parameter)
        if not isinstance(operator.coeff, ParameterExpression)
        else len(operator.coeff.parameters) == 0
    )
