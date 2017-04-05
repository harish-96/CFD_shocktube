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
    u_l = q_step[1, 1] / q_step[0, 1]
    t_l = ti_l - u_l**2 / 2 / (gamma * cv)
    p_l = pi_l * (t_l / ti_l)**(gamma / (gamma - 1))
    q1_l = p_l / 287 / t_l
    q2_l = q1_l * u_l
    q3_l = q1_l * (cv * t_l + u_l**2 / 2)

    u_r = q_step[1, -2] / q_step[0, -2]
    t_r = ti_r - u_r**2 / 2 / (gamma * cv)
    p_r = pi_r * (t_r / ti_r)**(gamma / (gamma - 1))
    q1_r = p_r / 287 / t_r
    q2_r = q1_r * u_r
    q3_r = q1_r * (cv * t_r + u_r**2 / 2)

    q_r = [q1_r, q2_r, q3_r]
    q_l = [q1_l, q2_l, q3_l]
    return [q_l, q_r]


scheme = FTCS
shock_tube = solver(gridpts=100, dtdx=0.005, scheme=scheme,
                    ic=ic, bc=bc, spaceDomain=[-0.5, 0.5], tmax=5)
print(shock_tube.grid.dtdx)

shock_tube.animate(shock_tube.grid.timesteps,
                   art_viscosity=[50, 5], save=False)
