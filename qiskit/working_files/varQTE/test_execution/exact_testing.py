import numpy as np
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['QISKIT_IN_PARALLEL'] = 'True'


from scipy.integrate import Radau, ode, solve_ivp, RK45, RK23
from qiskit.working_files.varQTE.implicit_euler import BDF, backward_euler_fsolve



from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Parameter

from qiskit.opflow.evolutions.varqtes.varqrte import VarQRTE
from qiskit.opflow.evolutions.varqtes.varqite import VarQITE
from qiskit.opflow.evolutions.varqte import ForwardEuler

from qiskit.opflow import StateFn, SummedOp
from qiskit.opflow import Z, I, Y, X
np.random.seed = 11

# Evolution time
t = 1

num_time_steps = [60]
depths = [1]


# Define the expectation value given the Hamiltonian as observable and the state generated by the
#  Ansatz

ode_solvers = [ForwardEuler, RK45]
ode_solvers_names = ['ForwardEuler', 'RK45']
regs = ['ridge', 'perturb_diag', None]
reg_names = ['ridge', 'perturb_diag', 'lstsq']
for nts in num_time_steps:
    for k, ode_solver in enumerate(ode_solvers):
        for d in depths:
            for j, reg in enumerate(regs):
                print(ode_solvers_names[k])
                print(reg_names[j])
                # Define the Hamiltonian for the simulation
                observable = 0.5 * Z
                # Define Ansatz
                parameters = Parameter('p')
                ansatz = QuantumCircuit(1)
                ansatz.rz(parameters, 0)
                # Define a set of initial parameters
                init_param_values = [np.pi/3]

                # Now we stack the observable and the quantum state together.
                # The evolution time needs to be added as a coefficient to the operator
                op = ~StateFn(observable) @ StateFn(ansatz)
                op = t * op

                print('number time steps', nts)
                print('depth ', d)
                print('---------------------------------------------------------------------')
                varqrte_snapshot_dir = os.path.join('..', 'output_exact_test_qasm', 'real',
                                                    str(nts),
                                                    reg_names[j],
                                                    ode_solvers_names[k] + 'nat_grad')
                t0 = time.time()
                varqrte = VarQRTE(parameters=[parameters],
                                grad_method='lin_comb',
                                init_parameter_values=init_param_values,
                                num_time_steps=nts,
                                ode_solver=ode_solver,
                                backend=Aer.get_backend('qasm_simulator'),
                                regularization=reg,
                                error_based_ode=False,
                                snapshot_dir=varqrte_snapshot_dir
                                # snapshot_dir=os.path.join('..', 'test')
                                )
                approx_time_evolved_state_real = varqrte.convert(op)
                varqrte_error_bounds = varqrte.error_bound(varqrte_snapshot_dir)
                np.save(os.path.join(varqrte_snapshot_dir, 'error_bounds.npy'),
                        varqrte_error_bounds)
                print('run time', (time.time()-t0)/60)
                varqrte.plot_results([varqrte_snapshot_dir], [os.path.join(varqrte_snapshot_dir,
                                                                            'error_bounds.npy')])
