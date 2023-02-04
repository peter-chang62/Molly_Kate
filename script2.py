import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

cr.style_sheet()

rad_to_deg = lambda rad: rad * 180 / np.pi
deg_to_rad = lambda deg: deg * np.pi / 180


def _term(n, n0, theta):
    return np.sqrt(n ** 2 - n0 ** 2 * np.sin(theta) ** 2) / n


def gamma(n, n0, d, theta, wl):
    return _term(n, n0, theta) * n * d * (2 * np.pi / wl)


def f_coeff(n1, n2, n0, theta, pol):
    if pol == "s":
        r = (n1 * _term(n1, n0, theta) - n2 * _term(n2, n0, theta)) / \
            (n1 * _term(n1, n0, theta) + n2 * _term(n2, n0, theta))

        t = 2 * n1 * _term(n1, n0, theta) / \
            (n1 * _term(n1, n0, theta) + n2 * _term(n2, n0, theta))

    elif pol == "p":
        r = (n2 * _term(n1, n0, theta) - n1 * _term(n2, n0, theta)) / \
            (n2 * _term(n1, n0, theta) + n1 * _term(n2, n0, theta))

        t = 2 * n1 * _term(n1, n0, theta) / \
            (n2 * _term(n1, n0, theta) + n1 * _term(n2, n0, theta))

    else:
        raise ValueError("polarization (pol) should be s or p")

    return r, t


def p_matrix(n, n0, d, theta, wl):
    g = gamma(n, n0, d, theta, wl)
    if g.size == 1:
        return np.array([[np.exp(-1j * g), 0],
                         [0, np.exp(1j * g)]])
    else:
        mat = np.zeros((len(g), 2, 2), dtype=complex)
        mat[:, 0, 0] = np.exp(-1j * g)
        mat[:, 0, 1] = 0
        mat[:, 1, 0] = 0
        mat[:, 1, 1] = np.exp(1j * g)
        return mat


def f_matrix(n1, n2, n0, theta, pol):
    r, t = f_coeff(n1, n2, n0, theta, pol)
    if r.size == 1:
        return np.array([[1, r],
                         [r, 1]]) / t
    else:
        mat = np.zeros((len(r), 2, 2))
        mat[:, 0, 0] = 1 / t
        mat[:, 0, 1] = r / t
        mat[:, 1, 0] = r / t
        mat[:, 1, 1] = 1 / t
        return mat


# __________________ loading index of refraction and stack data _______________
layer_info = pd.read_excel("NEID_mirrors.xlsx", sheet_name="Layer Info")
SiO2 = pd.read_excel("NEID_mirrors.xlsx", sheet_name="SiO2")
Nb2O5 = pd.read_excel("NEID_mirrors.xlsx", sheet_name="Nb2O5")

# input arguments
n1 = SiO2.values[:, 1]  # n1 (SiO2)
n2 = Nb2O5.values[:, 1]  # n2 (Nb2O5)
wl = SiO2.values[:, 0]  # wavelength axis

# match the wavelength axis
n1 = n1[wl > Nb2O5.values[:, 0].min()]
wl = wl[wl > Nb2O5.values[:, 0].min()]
n2_gridded = interp1d(Nb2O5.values[:, 0], Nb2O5.values[:, 1], kind='cubic',
                      bounds_error=True)
n2 = n2_gridded(wl)

pol = "s"  # polarization
d = layer_info.values[:, 1].astype(float)  # stack thickness data
n0 = 1  # n0
theta = deg_to_rad(0)  # incident angle

# analysis ____________________________________________________________________
m = f_matrix(1, n1, n0, theta, pol)  # hitting the stack

n = [n1, n2]
ind = 0
for i in tqdm(range(len(d))):
    p = p_matrix(n[ind], n0, d[i], theta, wl)  # propagate through layer
    f = f_matrix(n[ind], n[not ind], n0, theta, pol)  # enter next layer
    m = p @ m
    m = f @ m

    ind = not ind

plt.plot(wl, abs(m[:, 1, 0] / m[:, 0, 0]) ** 2)
