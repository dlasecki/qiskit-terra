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

from qiskit.opflow.evolutions.varqte import VarQTE

from qiskit.opflow.evolutions.varqte import ForwardEuler

from qiskit.opflow import StateFn, SummedOp
from qiskit.opflow import Z, I, Y, X
np.random.seed = 11


# Define the expectation value given the Hamiltonian as observable and the state generated by the
#  Ansatz

ode_solvers = [ForwardEuler, RK45]
ode_solvers_names = ['ForwardEuler', 'RK45']
regs = ['ridge', 'perturb_diag', None]
reg_names = ['ridge', 'perturb_diag', 'lstsq']
# ode_solvers = [ForwardEuler]
# ode_solvers_names = ['ForwardEuler']
# regs = [ None]
# reg_names = ['lstsq']
# output_dirs = ['illustrative']
output_dirs = ['h2', 'illustrative', 'illustrative_reverse', 'transverse_ising',
               'MaxCut/output_maxcut_superposition', 'MaxCut/output_maxcut_pretrained']
output_dir = '/Users/ouf/Box/ChristaPhDFolder/Papers/VarQTE_Error/output'
# output_dir = '/Users/ouf/Box/ChristaPhDFolder/Papers/VarQTE_Error/output/'
# output_dirs = ['MaxCut/output_maxcut_superposition/imag/25_depth_5']
for output in output_dirs:
    for k, ode_solver in enumerate(ode_solvers):
        for j, reg in enumerate(regs):
            print(ode_solvers_names[k])
            print(reg_names[j])

            # print('---------------------------------------------------------------------')
            # try:
            #     varqite_snapshot_dir = os.path.join(output_dir, output, 'imag',
            #                                         reg_names[j],
            #                                         ode_solvers_names[k] + 'nat_grad')
            #     # varqite_snapshot_dir = os.path.join(output_dir, output,
            #     #                                     reg_names[j],
            #     #                                     ode_solvers_names[k] + 'nat_grad')
            #
            #     VarQTE.plot_results([varqite_snapshot_dir], [os.path.join(varqite_snapshot_dir,
            #                                                   'error_bounds.npy')],
            #                           [os.path.join(varqite_snapshot_dir,
            #                                         'reverse_error_bounds.npy')]
            #                           )
            #     varqite_snapshot_dir = os.path.join(output_dir, output, 'imag',
            #                                         reg_names[j],
            #                                         ode_solvers_names[k] + 'error')
            #     # varqite_snapshot_dir = os.path.join(output_dir, output,
            #     #                                     reg_names[j],
            #     #                                     ode_solvers_names[k] + 'error')
            #     VarQTE.plot_results([varqite_snapshot_dir], [os.path.join(varqite_snapshot_dir,
            #                                                 'error_bounds.npy')],
            #                           [os.path.join(varqite_snapshot_dir,
            #                                         'reverse_error_bounds.npy')]
            #                           )
            # except Exception:
            #     pass
            #
            # print('run time', (time.time()-t0)/60)
            print('---------------------------------------------------------------------')
            try:
                varqrte_snapshot_dir = os.path.join(output_dir, output, 'real',
                                                    reg_names[j],
                                                    ode_solvers_names[k] + 'nat_grad')

                VarQTE.plot_results([varqrte_snapshot_dir],
                                    [os.path.join(varqrte_snapshot_dir,
                                                  'error_bounds.npy')])
                varqrte_snapshot_dir = os.path.join(output_dir, output, 'real',
                                                    reg_names[j],
                                                    ode_solvers_names[k] + 'error')

                VarQTE.plot_results([varqrte_snapshot_dir],
                                    [os.path.join(varqrte_snapshot_dir,
                                                  'error_bounds.npy')])
            except Exception:
                pass
