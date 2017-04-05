import numpy as np
import matplotlib.pyplot as plt
from numba import jit

gamma = 1.4
cp = gamma * 287 / (gamma - 1)
cv = 287 / (gamma - 1)


class mesh:

    def __init__(self, gridpts, dtdx, tmax, spaceDomain):
        self.x = np.linspace(spaceDomain[0], spaceDomain[1], gridpts)
        self.t = np.arange(
            0, tmax, dtdx * (spaceDomain[-1] - spaceDomain[0]) / gridpts)
        self.timesteps = len(self.t)
        self.E = np.zeros((3, gridpts, self.timesteps))
        self.Q = np.zeros((3, gridpts, self.timesteps))
        self.dtdx = dtdx

    @jit
    def compute_E(self, t):
        q_step = self.Q[:, :, t]
        e_step = np.ones_like(q_step)
        e_step[0] = q_step[1].copy()
        p = (gamma - 1) * (q_step[2] - q_step[1]**2 / 2 / q_step[0])
        e_step[1] = q_step[1]**2 / q_step[0] + p
        e_step[2] = (q_step[2] + p) * q_step[1] / q_step[0]
        return e_step

    @jit
    def compute_Qt(self, t):
        q_step = self.Q[:, :, t]
        qt_step = np.ones_like(q_step)
        qt_step[0] = q_step[0]
        qt_step[1] = q_step[1] / q_step[0]
        qt_step[2] = (gamma - 1) * (q_step[2] - q_step[1]**2 / 2 / q_step[0])
        return qt_step

    # @jit
    # def compute_X(self, t):
    #     qt = self.compute_Qt(t)
    #     u = qt[1]
    #     c = np.sqrt(gamma * qt[2] / qt[0])
    #     row3 = [0.5 * u**2, c**2 / (gamma - 1) + 0.5 * u **
    #             2 + c * u, c**2 / (gamma - 1) + 0.5 * u**2 - c * u]
    #     X = np.array([[1, 1, 1], [u, u + c, u - c], row3])
    #     return X

    # @jit
    # def compute_Xinv(self, t):
    #     X = self.compute_X(t)
    #     return np.linalg.inv(X)

    # def plot_step(self, t_step):
    #     qt = self.compute_Qt(t_step)
    #     fig, axarr = plt.subplots(3, sharex=True)
    #     axarr[0].plot(self.x, qt[0])
    #     axarr[1].plot(self.x, qt[1])
    #     axarr[2].plot(self.x, qt[2])
    #     plt.show()
