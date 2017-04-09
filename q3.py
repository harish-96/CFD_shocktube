from solver import *
from schemes import *


# P_ambient=84000, P0=101325, T0=300, T_ambient=300,
# initial_conditions=exit_conditions
gamma = 1.4
R = 287
cv = R / (gamma - 1)

pi_l = 1
rhoi_l = 1
ui_l = 0
ti_l = pi_l / rhoi_l / R

pi_r = 0.1
rhoi_r = 0.125
ui_r = 0
ti_r = pi_r / rhoi_r / R


def ic(x):
    q1_r = pi_r / 287 / ti_r  # density
    q2_r = 0  # \rho*u
    q3_r = q1_r * cv * ti_r

    q1_l = pi_l / 287 / ti_l  # density
    q2_l = 0  # \rho*u
    q3_l = q1_l * cv * ti_l

    x_r = list(filter(lambda var: var > 0, x))
    x_l = list(filter(lambda var: var <= 0, x))
    ic_r = np.array([[q1_r, q2_r, q3_r] for i in x_r])
    ic_l = np.array([[q1_l, q2_l, q3_l] for i in x_l])
    return np.concatenate((ic_l, ic_r)).T


def bc(t, q_step):
    # u_l = q_step[1, 1] / q_step[0, 1]
    qt_step = np.ones_like(q_step)
    qt_step[0] = q_step[0]
    qt_step[1] = q_step[1] / q_step[0]
    qt_step[2] = (gamma - 1) * (q_step[2] - q_step[1]**2 / 2 / q_step[0])

    u_l = -q_step[1, 1] / q_step[0, 1]
    # u_l = 0
    t_l = qt_step[2, 1] / R / qt_step[0, 1]
    p_l = qt_step[2, 1]
    q1_l = p_l / 287 / t_l
    q2_l = q1_l * u_l
    q3_l = q1_l * (cv * t_l + u_l**2 / 2)

    u_r = -q_step[1, -2] / q_step[0, -2]
    # u_r = 0
    t_r = qt_step[2, -2] / R / qt_step[0, -2]
    p_r = qt_step[2, -2]
    q1_r = p_r / 287 / t_r
    q2_r = q1_r * u_r
    q3_r = q1_r * (cv * t_r + u_r**2 / 2)

    q_r = [q1_r, q2_r, q3_r]
    q_l = [q1_l, q2_l, q3_l]
    return [q_l, q_r]


scheme = FTCS
shock_tube = solver(gridpts=500, dtdx=0.05, scheme=scheme,
                    ic=ic, bc=bc, spaceDomain=[-.5, .5], tmax=5)
print(shock_tube.grid.dtdx)
dt = shock_tube.grid.t[1] - shock_tube.grid.t[0]

shock_tube.animate(int(shock_tube.grid.timesteps),
                   art_viscosity=[0.007, 0.0007],
                   save=True, filename='q3_anim', fps=60)
