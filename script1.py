import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import scipy.constants as sc
import pandas as pd
from scipy.interpolate import interp1d

cr.style_sheet()

rad_to_deg = lambda rad: rad * 180 / np.pi
deg_to_rad = lambda deg: deg * np.pi / 180

layer_info = pd.read_excel("NEID_mirrors.xlsx", sheet_name="Layer Info")
SiO2 = pd.read_excel("NEID_mirrors.xlsx", sheet_name="SiO2")
Nb2O5 = pd.read_excel("NEID_mirrors.xlsx", sheet_name="Nb2O5")


def term(n, n0, theta):
    return np.sqrt(n ** 2 - n0 ** 2 * np.sin(theta) ** 2) / n


# input arguments
n1 = SiO2.values[:, 1]
n2 = Nb2O5.values[:, 1]
wl = SiO2.values[:, 0]
n1 = n1[wl > Nb2O5.values[:, 0].min()]
wl = wl[wl > Nb2O5.values[:, 0].min()]
n2_gridded = interp1d(Nb2O5.values[:, 0], Nb2O5.values[:, 1], kind='cubic',
                      bounds_error=True)
n2 = n2_gridded(wl)
pol = "s"
d = layer_info.values[:, 1].astype(float)
n0 = 1
theta = deg_to_rad(0)
lamda = wl

# function ____________________________________________________________________

d_1 = d[::2]
d_2 = d[1::2]
gamma_1 = np.ones((len(d_1), len(n1))) * np.c_[d_1]
gamma_2 = np.ones((len(d_2), len(n1))) * np.c_[d_2]

gamma_1 *= term(n1, n0, theta) * n1 * (2 * np.pi / lamda)
gamma_2 *= term(n2, n0, theta) * n2 * (2 * np.pi / lamda)

gamma = np.zeros((len(d), len(n1)))
gamma[::2] = gamma_1
gamma[1::2] = gamma_2

phase_delay = np.ones((2, 2, len(d), len(wl)), dtype=complex)
phase_delay[0, 1] = 0
phase_delay[1, 0] = 0
phase_delay[0, 0] = np.exp(-1j * gamma)
phase_delay[1, 1] = np.exp(1j * gamma)

if pol == "s":
    # first layer to second layer
    r_1 = (term(n1, n0, theta) * n1 - n2 * term(n2, n0, theta)) / \
          (n1 * term(n1, n0, theta) + n2 * term(n2, n0, theta))

    t_1 = 2 * n1 * term(n1, n0, theta) / \
          (n1 * term(n1, n0, theta) + n2 * term(n2, n0, theta))

    # second layer to first layer
    r_2 = (term(n2, n0, theta) * n2 - n1 * term(n1, n0, theta)) / \
          (n2 * term(n2, n0, theta) + n1 * term(n1, n0, theta))

    t_2 = 2 * n2 * term(n2, n0, theta) / \
          (n2 * term(n2, n0, theta) + n1 * term(n1, n0, theta))

if pol == "p":
    # first layer to second layer
    r_1 = (n2 * term(n1, n0, theta) - n1 * term(n2, n0, theta)) / \
          (n2 * term(n1, n0, theta) + n1 * term(n2, n0, theta))

    # second layer to first layer
    t_1 = 2 * n1 * term(n1, n0, theta) / \
          (n2 * term(n1, n0, theta) + n1 * term(n2, n0, theta))

    r_2 = (n1 * term(n2, n0, theta) - n2 * term(n1, n0, theta)) / \
          (n1 * term(n2, n0, theta) + n2 * term(n1, n0, theta))

    t_2 = 2 * n2 * term(n2, n0, theta) / \
          (n1 * term(n2, n0, theta) + n2 * term(n1, n0, theta))

t_matrix = np.zeros((len(d), len(wl)))
r_matrix = np.zeros((len(d), len(wl)))
t_matrix[::2] = t_1
t_matrix[1::2] = t_2
r_matrix[::2] = r_1
r_matrix[1::2] = r_2
fresnel = np.zeros((2, 2, len(d), len(wl)))
fresnel[0, 0] = 1 / t_matrix
fresnel[0, 1] = r_matrix / t_matrix
fresnel[1, 0] = r_matrix / t_matrix
fresnel[1, 1] = 1 / t_matrix

phase_delay = phase_delay.T
fresnel = fresnel.T

# missing hitting stack
m = phase_delay[:, 0].copy()
for n in range(fresnel.shape[1] - 1):
    m = fresnel[:, n] @ m  # hitting layer
    m = phase_delay[:, n + 1] @ m  # propagating through layer

# missing lass interface (n2 - > glass)
R = abs(m[:, 1, 0] / m[:, 0, 0]) ** 2
plt.plot(wl, R)
