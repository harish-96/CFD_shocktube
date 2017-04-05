import numpy as np
# import matplotlib.animation as animation
import matplotlib.pyplot as plt

GRIDPTS = 100
gamma = 1.4
cp = gamma * 287 / (gamma - 1)
cv = 287 / (gamma - 1)
p_0 = 101325
p_a = 84000
t_0 = 300


def init_1d(gridpts, timesteps, timeDomain=1, spaceDomain=1,
            ic=lambda x: [np.ones_like(x) for i in range(3)],):
    x = np.linspace(0, spaceDomain, gridpts)
    E = np.zeros((3, gridpts, timesteps))
    Q = np.zeros((3, gridpts, timesteps))
    Q[:, :, 0] = ic(x)
    Q[0, 0, 0] = p_0 / 287 / t_0
    Q[1, 0, 0] = 0
    Q[2, 0, 0] = Q[0, 0, 0] * (cv * t_0)
    E[:, :, 0] = compute_E(Q[:, :, 0])
    dtdx = timeDomain / timesteps / (spaceDomain / gridpts)
    return E, Q, dtdx


def compute_E(q_step):
    e_step = np.ones_like(q_step)
    e_step[0] = q_step[1]
    p = (gamma - 1) * (q_step[2] - q_step[1]**2 / 2 / q_step[0])
    e_step[1] = q_step[1]**2 / q_step[0] + p
    e_step[2] = (q_step[2] + p) * q_step[1] / q_step[0]
    return e_step


def FTCS_step(Q, E, dtdx, bc, t_step):
    #  bc=lambda t: np.ones((2, 3))):
    Q[:, 1:-1, t_step + 1] = Q[:, 1:-1, t_step] - dtdx / 2 * \
        (E[:, 2:, t_step] - E[:, :-2, t_step])  # + 0.01 * \
        # (Q[:, :-2, t_step] - 2 * Q[:, 1:-1, t_step] + Q[:, 2:, t_step])
    left_bc, right_bc = bc(t_step + 1, Q[:, :, t_step])
    Q[:, 0, t_step + 1] = left_bc
    Q[:, -1, t_step + 1] = right_bc
    E[:, :, t_step + 1] = compute_E(Q[:, :, t_step + 1])
    return Q


# P_ambient=84000, P0=101325, T0=300, T_ambient=300,
# initial_conditions=exit_conditions
def ic(x):
    p_a = 84000
    t_a = 300
    cv = 287 / (gamma - 1)
    q1 = p_a / 287 / t_a  # density
    q2 = 0  # \rho*u
    q3 = q1 * cv * t_a
    return np.array([[q1, q2, q3] for i in x]).T


def bc(t, q_prev):
    p_0 = 101325
    t_0 = 300
    p_r = 84000
    cv = 287 / (gamma - 1)
    u_l = q_prev[1, 1] / q_prev[0, 1]
    t_0 = 300
    t_l = t_0 - u_l**2 / 2 / (gamma * cv)
    p_l = p_0 * (t_l / t_0)**(gamma / (gamma - 1))
    q1_l = p_l / 287 / t_l
    q2_l = q1_l * u_l
    q3_l = q1_l * (cv * t_l + u_l**2 / 2)

    u_r = q_prev[1, -2] / q_prev[0, -2]
    t_r = t_0 - u_r**2 / 2 / (gamma * cv)
    q1_r = p_r / 287 / t_r
    q2_r = q1_r * u_r
    q3_r = q1_r * (cv * t_r + u_r**2 / 2)

    q_r = [q1_r, q2_r, q3_r]
    q_l = [q1_l, q2_l, q3_l]
    return [q_l, q_r]


E, Q, dtdx = init_1d(100, 1000000, ic=ic)
print(dtdx)

FTCS_step(Q, E, dtdx, bc, 0)
for i in range(1, 9):
    q_prev = Q[:, :, i - 1]
    plt.plot(Q[0, :, i - 1])
    plt.show()
    FTCS_step(Q, E, dtdx, bc, i)
    rho_l = Q[0, 0, i]
    t_0 = 300
    cv = 287 / (gamma - 1)
    u_l = q_prev[1, 1] / q_prev[0, 1]
    t_l = t_0 - u_l**2 / 2 / (gamma * cv)
    p_l = rho_l * 287 * t_l
    p_0 = p_l * (t_0 / t_l)**(gamma / (gamma - 1))
    print('ul', u_l)
