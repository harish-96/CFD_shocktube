from solver import *
from schemes import *


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


def bc(t, q_step):
    p_0 = 101325
    t_0 = 300
    p_r = 84000
    cv = 287 / (gamma - 1)
    u_l = q_step[1, 1] / q_step[0, 1]
    t_0 = 300
    t_l = t_0 - u_l**2 / 2 / (gamma * cv)
    p_l = p_0 * (t_l / t_0)**(gamma / (gamma - 1))
    q1_l = p_l / 287 / t_l
    q2_l = q1_l * u_l
    q3_l = q1_l * (cv * t_l + u_l**2 / 2)

    u_r = q_step[1, -2] / q_step[0, -2]
    t_r = t_0 - u_r**2 / 2 / (gamma * cv)
    q1_r = p_r / 287 / t_r
    q2_r = q1_r * u_r
    q3_r = q1_r * (cv * t_r + u_r**2 / 2)

    q_r = [q1_r, q2_r, q3_r]
    q_l = [q1_l, q2_l, q3_l]
    return [q_l, q_r]


scheme = Lax_Fred
shock_tube = solver(gridpts=50, dtdx=0.0002, scheme=scheme,
                    ic=ic, bc=bc, spaceDomain=100, timeDomain=50)
print(shock_tube.grid.dtdx)

shock_tube.animate(shock_tube.grid.timesteps)
# anim.save('blah.mp4', fps=200, extra_args=['-vcodec', 'libx264'])
# for i in range(10000 - 1):
#     shock_tube.FTCS_step(bc, i)
#     if i % 100 == 0:
#         shock_tube.grid.plot_step(i)
