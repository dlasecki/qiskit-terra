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
"""This package contains Trotterization-based Quantum Real Time Evolution algorithm.
It is compliant with the new Quantum Time Evolution Framework and makes use of `ProductFormula` and
`PauliEvolutionGate` implementations.
The evolution with gradients assumes that a Hamiltonian is a linear combination of `PauliOp` objects
w.r.t. given parameters. It case of a single summand, it might be a `PauliOp`."""

from qiskit.algorithms.evolvers.real.trotterization.trotter_qrte import (
    TrotterQRTE,
)

__all__ = ["TrotterQRTE"]
