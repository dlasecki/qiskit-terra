from jax import grad, jit
import jax.numpy as jnp

import numpy as np
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.aqua.operators import StateFn, MatrixOp, CircuitOp, X, I, PauliOp, Z, Y
from qiskit.aqua.operators.gradients import NaturalGradient

np.random.seed = 2

H = (Z ^ X)
H = H.to_matrix()

ansatz = RealAmplitudes(2, reps=1, entanglement='full')
print(ansatz)
# init_params = np.random.rand(len(ansatz.parameters))
init_params = [jnp.pi/3, -jnp.pi/3, jnp.pi/2., np.pi / 3]
# init_params = np.zeros(len(ansatz.ordered_parameters))
# for i in range(ansatz.num_qubits):
#     init_params[-(i+1)] = np.pi / 2
# qc_state = StateFn(ansatz).bind_parameters(dict(zip(ansatz.ordered_parameters, init_params)))
# print(np.round(CircuitOp(ansatz).bind_parameters(dict(zip(ansatz.ordered_parameters,
#                                                   init_params))).to_matrix(),3))
# print(qc_state)
# qc_state = qc_state.eval()

def ry(theta):
    return jnp.array([[jnp.cos(theta/2.), -1*jnp.sin(theta/2)], [jnp.sin(theta/2),
                                                                 jnp.cos(theta/2)]])

def ryry(alpha, beta):
    return jnp.kron(ry(alpha), ry(beta))

cx = jnp.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
# cx = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
# cx = jnp.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
i = jnp.eye(2)
y = jnp.array([[0, -1j], [1j, 0]])
iy = -0.5j * jnp.kron(i, y)
yi = -0.5j * jnp.kron(y, i)

init = jnp.array([1, 0, 0, 0])

# TODO get the state for all entries in the vector and compute the gradient independently then

def state_fn(params):
    # print(np.round(jnp.matmul(ryry(params[2], params[3]), jnp.matmul(cx, ryry(params[0],
    #                                                                          params[1]))), 3))
    vec = jnp.dot(ryry(params[3], params[2]), jnp.dot(cx, jnp.dot(ryry(params[1], params[0]),
                                                                   init)))
    return vec

def grad_fn(params):
    # num_params x dim vec
    g = []
    g.append(jnp.dot(ryry(params[3], params[2]), jnp.dot(cx, jnp.dot(iy, jnp.dot(ryry(params[1],
                                                                                      params[0]),
                                                                   init)))))
    g.append(jnp.dot(ryry(params[3], params[2]), jnp.dot(cx, jnp.dot(yi, jnp.dot(ryry(params[1],
                                                                                      params[0]),
                                                                                 init)))))
    g.append(jnp.dot(iy, jnp.dot(ryry(params[3], params[2]), jnp.dot(cx, jnp.dot(ryry(params[1],params[0]),
                                                                                 init)))))
    g.append(jnp.dot(yi, jnp.dot(ryry(params[3], params[2]),
                                 jnp.dot(cx, jnp.dot(ryry(params[1], params[0]),
                                                     init)))))
    return g



# # stack together
def state0(params):
    vec = jnp.dot(ryry(params[3], params[2]), jnp.dot(cx, jnp.dot(ryry(params[1], params[0]),
                                                                   init)))
    return vec[0]

def state1(params):
    vec = jnp.dot(ryry(params[3], params[2]), jnp.dot(cx, jnp.dot(ryry(params[1], params[0]),
                                                                   init)))
    return vec[1]

def state2(params):
    vec = jnp.dot(ryry(params[3], params[2]), jnp.dot(cx, jnp.dot(ryry(params[1], params[0]),
                                                                   init)))
    return vec[2]

def state3(params):
    vec = jnp.dot(ryry(params[3], params[2]), jnp.dot(cx, jnp.dot(ryry(params[1], params[0]),
                                                                   init)))
    return vec[3]


# TODO Check transpose for gradient!!
def A(vec, gradient):
    vec = np.reshape(vec, (len(vec), 1))
    a = np.real(np.matmul(np.transpose(np.conj(gradient)), gradient))
    a_temp = np.matmul(np.transpose(np.conj(vec)), gradient)
    a = np.subtract(a, np.real(np.matmul(np.transpose(np.conj(a_temp)), a_temp)))
    return a

# TODO Check transpose for gradient!!
def C(vec, gradient, h):
    vec = np.reshape(vec, (1, len(vec)))
    c = np.real(np.matmul(np.conj(vec), np.matmul(h, gradient)))
    return (-1)*c[0]


def dt_params(a, c, regularization=None):
    if regularization:
        # If a regularization method is chosen then use a regularized solver to
        # construct the natural gradient.
        nat_grad = NaturalGradient._regularized_sle_solver(
            a, c, regularization=regularization)
    else:
        try:
            # Try to solve the system of linear equations Ax = C.
            nat_grad = np.linalg.solve(a, c)
        except np.linalg.LinAlgError:  # singular matrix
            nat_grad = np.linalg.lstsq(a, c)[0]
    return np.real(nat_grad)


params = init_params
time = 1
time_steps = 3
error = 0
for j in range(time_steps):
# print(qc_state)
    print('state', state_fn(params))
    # print(np.array_equal(qc_state))

    # print(jit(grad(state3))(init_params))
    # # def grad(params):
    grad0 = grad(state0)
    grad1 = grad(state1)
    grad2 = grad(state2)
    grad3 = grad(state3)
    # dim vec x num_params
    gradient = [grad0(params), grad1(params), grad2(params), grad3(params)]
    # gradient = [grad0(params), grad1(params), grad2(params)]
    gradient = np.round([[float(item) for item in g] for g in gradient], 3).astype(
        np.complex)
    # print('Gradient jax', gradient)
    grad_ = grad_fn(params)
    print('Gradient', np.round([[float(item) for item in g] for g in grad_], 3).astype(
        np.complex))

    state = state_fn(params)
    state = np.round([float(s) for s in state], 3).astype(np.complex)

    metric = A(state, gradient)
    c_grad = C(state, gradient, H)

    print('Metric',np.round(metric, 3))
    print('C', np.round(c_grad, 3))

    dt_weights = dt_params(metric, c_grad, 'perturb_diag')
    print('dtweights', np.round(dt_weights, 3))

    # dt_state = np.dot(grad, dt_weights)
    dt_state = np.dot(np.transpose(dt_weights), gradient)
    print('dt_state', np.round(dt_state, 3))

    dtdt = np.dot(np.transpose(np.conj(dt_state)), dt_state)
    print('dt dt', dtdt)
    # print('dt dt', np.dot(np.transpose(dt_state), dt_state) )

    h_squared = np.matmul(np.conj(np.transpose(state)), np.matmul(np.matmul(H, H), state))
    print('h^2', h_squared)

    # print('2 im', 2 * np.imag(np.matmul(np.conj(np.transpose(dt_state)), np.matmul(H, state))))
    energy = np.matmul(np.conj(np.transpose(state)), np.matmul(H, state))
    print('Energy^2', energy ** 2)

    # overlap = 2*np.real(np.dot(np.transpose(np.conj(dt_state)), state))
    # print('Energy*overlap', energy * overlap)

    dt = 2*np.real(np.matmul(np.conj(dt_state), np.matmul(H, state)))
    print('dt <H>', dt)

    # et = dtdt + h_squared - energy ** 2 + dt - energy * overlap
    et = dtdt + h_squared - energy ** 2 + dt
    error += time/time_steps * np.sqrt(et) * (1 + 2 * time/time_steps*np.linalg.norm(H))**(
             time-j*time/time_steps)

    print('e at time t', np.round(et, 6))
    print('error bound at time t', np.round(error, 3))
    params += time/time_steps * np.reshape(dt_weights, np.shape(params))

