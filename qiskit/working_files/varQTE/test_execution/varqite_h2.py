import numpy as np
import os
import time


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['QISKIT_IN_PARALLEL'] = 'True'


from scipy.integrate import Radau, ode, solve_ivp, RK45, RK23
from qiskit.working_files.varQTE.implicit_euler import BDF, backward_euler_fsolve



from qiskit import Aer

from qiskit.circuit.library import EfficientSU2, RealAmplitudes

from qiskit.opflow.evolutions.varqtes.varqrte import VarQRTE
from qiskit.opflow.evolutions.varqtes.varqite import VarQITE
from qiskit.opflow.evolutions.varqte import ForwardEuler

from qiskit.opflow import StateFn, SummedOp
from qiskit.opflow import Z, I, Y, X
np.random.seed = 11

# Evolution time
t = 5


#number of time steps
nts = 500
depths = [1]


# Define the expectation value given the Hamiltonian as observable and the state generated by the
#  Ansatz

ode_solvers = [ForwardEuler,  RK45]
ode_solvers_names = ['ForwardEuler', 'RK45']
# ode_solvers = [ RK45, BDF]
# ode_solvers_names = ['RK45', 'BDF']

# ode_solvers = [BDF]
# ode_solvers_names = ['BDF']

# ode_solvers = [ RK23]
# ode_solvers_names = ['RK23']
regs = ['ridge', 'perturb_diag']
reg_names = ['ridge', 'perturb_diag']
# regs = [ None]
# reg_names = ['lstsq']
# for nts in num_time_steps:
# nts = num_time_steps[1]
error_based_odes = [False, True]
error_based_ode_names = ['nat_grad', 'error']
for l, error_based_ode in enumerate(error_based_odes):
    for k, ode_solver in enumerate(ode_solvers):
        for d in depths:
            for j, reg in enumerate(regs):
                print(ode_solvers_names[k])
                print(reg_names[j])
                # Define the Hamiltonian for the simulation
                # observable = (Y ^ Y)
                # observable = SummedOp([(Z ^ X), 0.8 * (Y ^ Y)]).reduce()
                observable = SummedOp([0.2252 * (I ^ I), 0.5716 * (Z ^ Z), 0.3435 * (I ^ Z),
                                       -0.4347 * (Z ^ I), 0.091 * (Y ^ Y),
                                       0.091 * (X ^ X)]).reduce()
                # observable = (Y ^ I)
                # observable = SummedOp([(Z ^ X), 3. * (Y ^ Y), (Z ^ X), (I ^ Z), (Z ^ I)]).reduce()
                # Define Ansatz
                # ansatz = RealAmplitudes(observable.num_qubits, reps=d)
                ansatz = EfficientSU2(observable.num_qubits, reps=d)

                # Define a set of initial parameters
                parameters = ansatz.ordered_parameters
                init_param_values = np.zeros(len(ansatz.ordered_parameters))
                for i in range(ansatz.num_qubits):
                    init_param_values[-(ansatz.num_qubits + i + 1)] = np.pi / 2
                # initial_point = [np.pi/3, -np.pi/3, np.pi/2., np.pi/3.]
                # initial_point = np.zeros(len(parameters))
                # for i in range(ansatz.num_qubits):
                #     initial_point[-(ansatz.num_qubits + i + 1)] = np.pi / 2

                # initial_point = [np.pi/3, -np.pi/3, np.pi/2., np.pi / 5, np.pi/4, -np.pi/7,
                # np.pi/8., np.pi / 9]
                # for i in range(ansatz.num_qubits):
                #     initial_point[-(i + 1)] = np.pi / 2
                # print(initial_point)

                # Now we stack the observable and the quantum state together.
                # The evolution time needs to be added as a coefficient to the operator
                op = ~StateFn(observable) @ StateFn(ansatz)
                op = t * op

                print('number time steps', nts)
                print('depth ', d)
                print('---------------------------------------------------------------------')

                t0 = time.time()
                varqite_snapshot_dir = os.path.join('..', 'output_h2', 'imag',
                                                    reg_names[j],
                                                    ode_solvers_names[k] + error_based_ode_names[l])

                varqite = VarQITE(parameters=parameters, grad_method='lin_comb',
                                  init_parameter_values=init_param_values,
                                  num_time_steps=nts,
                                  ode_solver=ode_solver,
                                  backend=Aer.get_backend('statevector_simulator'),
                                  regularization=reg,
                                  error_based_ode=error_based_ode,
                                  snapshot_dir=varqite_snapshot_dir)

                varqite._operator = op
                # approx_time_evolved_state_imag = varqite.convert(op)
                varqite_error_bounds = varqite.error_bound(
                    varqite_snapshot_dir, imag_reverse_bound=False,
                    H=observable.to_matrix(massive=True))
                np.save(os.path.join(varqite_snapshot_dir, 'error_bounds.npy'),
                        varqite_error_bounds)
                # np.save(os.path.join(varqite_snapshot_dir, 'reverse_error_bounds.npy'),
                #         varqite_reverse_error_bounds)
                # dir_fast = '../output/imag/10/ridge/RK45error'
                # varqite.print_results([dir_fast], [os.path.join(dir_fast,
                #                                                'error_bounds.npy')])
                varqite.plot_results([varqite_snapshot_dir], [os.path.join(varqite_snapshot_dir,
                                                              'error_bounds.npy')])
                                     # ,
                                     #  [os.path.join(varqite_snapshot_dir,
                                     #                'reverse_error_bounds.npy')]
                                     #  )

                print('run time', (time.time()-t0)/60)
                """
                print('---------------------------------------------------------------------')
                varqrte_snapshot_dir = os.path.join('..', 'output_h2', 'real',
                                                    reg_names[j],
                                                    ode_solvers_names[k] + error_based_ode_names[l])
                t0 = time.time()
                varqrte = VarQRTE(parameters=parameters,
                                grad_method='lin_comb',
                                init_parameter_values=init_param_values,
                                num_time_steps=nts,
                                ode_solver=ode_solver,
                                backend=Aer.get_backend('statevector_simulator'),
                                regularization=reg,
                                error_based_ode=error_based_ode,
                                snapshot_dir=varqrte_snapshot_dir
                                # snapshot_dir=os.path.join('..', 'test')
                                )
                approx_time_evolved_state_real = varqrte.convert(op)
                varqrte_error_bounds = varqrte.error_bound(varqrte_snapshot_dir)
                np.save(os.path.join(varqrte_snapshot_dir, 'error_bounds.npy'),
                        varqrte_error_bounds)
                                                    # snapshot_dir=str(nts)+'/'+str(d)).convert(op)

                print('run time', (time.time()-t0)/60)
                varqrte.plot_results([varqrte_snapshot_dir], [os.path.join(varqrte_snapshot_dir,
                                                                            'error_bounds.npy')])
                """
