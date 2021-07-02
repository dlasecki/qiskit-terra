import numpy as np
import time

import os
os.environ['QISKIT_IN_PARALLEL'] = 'False'

from scipy.integrate import Radau, ode, solve_ivp, RK45, RK23
# from qiskit.working_files.varQTE.implicit_euler import BDF, backward_euler_fsolve



from qiskit import Aer, QuantumCircuit

from qiskit.circuit import Parameter, ParameterVector

from qiskit.circuit.library import EfficientSU2, RealAmplitudes

from qiskit.opflow.evolutions.varqtes.varqrte import VarQRTE
from qiskit.opflow.evolutions.varqtes.varqite import VarQITE
from qiskit.opflow.evolutions.varqte import ForwardEuler

from qiskit.opflow import StateFn, SummedOp
from qiskit.opflow import Z, I, Y, X
np.random.seed = 11

# Evolution time
t = 1
#num time steps
nts = 100
depths = [1]


# Define the expectation value given the Hamiltonian as observable and the state generated by the
#  Ansatz

ode_solvers = [ForwardEuler,  RK45]
ode_solvers_names = ['ForwardEuler', 'RK45']
# ode_solvers = [ForwardEuler]
# ode_solvers_names = ['ForwardEuler']
# ode_solvers = [ BDF]
# ode_solvers_names = ['RK45', 'BDF']

# ode_solvers = [BDF]
# ode_solvers_names = ['BDF']

# ode_solvers = [ RK45]
# ode_solvers_names = ['RK45']
# regs = ['ridge', 'perturb_diag', None]
# reg_names = ['ridge', 'perturb_diag', 'lstsq']
# regs = ['perturb_diag', None]
# reg_names = ['perturb_diag', 'None']
regs = [None]
reg_names = ['lstsq']
error_based_odes = [False, True]
error_based_ode_names = ['nat_grad', 'error']
# error_based_odes = [False]
# error_based_ode_names = ['nat_grad']
for l, error_based_ode in enumerate(error_based_odes):
    for k, ode_solver in enumerate(ode_solvers):
        for d in depths:
            for j, reg in enumerate(regs):
                print(ode_solvers_names[k])
                print(reg_names[j])
                # Define the Hamiltonian for the simulation
                # observable = (Y ^ Y)
                # observable = SummedOp([(Z ^ X), 0.8 * (Y ^ Y)]).reduce()
                observable = SummedOp([-0.25 * (I ^ X ^ I), -0.25 * (X ^ I ^ I),
                                       -0.25 * (I ^ I ^ X), 0.5 * (Z ^ Z ^ I),
                                         0.5 * (I ^ Z ^ Z)]).reduce()
                # observable = (Y ^ I)
                # observable = SummedOp([(Z ^ X), 3. * (Y ^ Y), (Z ^ X), (I ^ Z), (Z ^ I)]).reduce()
                # Define Ansatz
                ansatz = EfficientSU2(observable.num_qubits, reps=d)
                parameters = ansatz.ordered_parameters
                init_param_values = np.zeros(len(parameters))
                init_param_values[-3] = 0.69551298
                init_param_values[-2] = 0.59433922
                init_param_values[-1] = 1.16511845
                print(init_param_values)

                # Now we stack the observable and the quantum state together.
                # The evolution time needs to be added as a coefficient to the operator
                op = ~StateFn(observable) @ StateFn(ansatz)
                op = t * op

                print('number time steps', nts)
                print('depth ', d)
                print('---------------------------------------------------------------------')
                ising_dir = '../output_ising_transverse'

                varqite_snapshot_dir = os.path.join(ising_dir,
                                                    'imag',
                                                    reg_names[j],
                                                    ode_solvers_names[k] + error_based_ode_names[l])
                t0 = time.time()
                varqite = VarQITE(parameters=parameters,
                                  grad_method='lin_comb',
                                  init_parameter_values=init_param_values,
                                  num_time_steps=nts,
                                  ode_solver=ode_solver,
                                  backend=Aer.get_backend('statevector_simulator'),
                                  regularization=reg,
                                  error_based_ode=error_based_ode,
                                  snapshot_dir=varqite_snapshot_dir
                                  # snapshot_dir=os.path.join('..', 'test')
                                  )
                # varqite._operator = op
                approx_time_evolved_state_imag = varqite.convert(op)
                varqite_error_bounds = varqite.error_bound(varqite_snapshot_dir)
                np.save(os.path.join(varqite_snapshot_dir, 'error_bounds.npy'),
                        varqite_error_bounds)

                print('run time', (time.time() - t0) / 60)
                varqite.plot_results([varqite_snapshot_dir], [os.path.join(varqite_snapshot_dir,
                                                                           'error_bounds.npy')])

