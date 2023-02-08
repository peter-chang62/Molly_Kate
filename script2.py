import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
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
L = 7.498 * (10 ** 6) # spacer length

# analysis ____________________________________________________________________
m = f_matrix(1, n1, n0, theta, pol)  # hitting the stack

n = [n1, n2]
ind = 0
for i in tqdm(range(len(d))):
    p = p_matrix(n[ind], n0, d[i], theta, wl)  # propagate through layer
    if i == len(d) - 1:
        f = f_matrix(n[ind], 1.48, n0, theta, pol)  # enter last layer
    else:
        f = f_matrix(n[ind], n[not ind], n0, theta, pol)  # enter next layer
    m = p @ m
    m = f @ m

    ind = not ind

R = abs(m[:, 1, 0] / m[:, 0, 0]) ** 2
#plt.plot(wl, R, '.')
#plt.xlim(350, 950)
#plt.ylim(.925, .95)

def phase_calculation(ed, en, er):
    d1 = d + ed[:] + er[:]
    n1test = n + en

    m = f_matrix(1, n1, n0, theta, pol)  # hitting the stack

    ind = 0
    for i in tqdm(range(len(d1))):
        p = p_matrix(n[ind], n0, d1[i], theta, wl)  # propagate through layer
        if i == len(d1) - 1:
            f = f_matrix(n[ind], 1.48, n0, theta, pol)  # enter last layer
        else:
            f = f_matrix(n[ind], n[not ind], n0, theta, pol)  # enter next layer
        m = p @ m
        m = f @ m

        ind = not ind

    r = m[:, 1, 0] / m[:, 0, 0]
    phasewrapped = np.arctan(r.imag / r.real)

    unwrapped = np.unwrap(phasewrapped)
    phi = interp1d(wl, unwrapped, kind='cubic', bounds_error=True)

    modefunction = lambda l: abs(2 * np.pi * m - ((4 * np.pi * L) / l) - 2 * phi(l))
    modes = np.array([])
    modeloc = np.array([])

    for i in tqdm(range(38000, 15000, -1)):
        m = i
        y = float(fsolve(modefunction, 500))
        if y > 500 and y < 950:
            modes = np.append(modes, y)
            modeloc = np.append(modeloc, i)

    modedata = [modeloc, modes]

    return modedata

def layer_relaxation(dchange, nchange, loc):
    nochange_d = np.zeros((len(d)))
    nochange_n = np.zeros((2, len(n[0])))
    original_modes = phase_calculation(nochange_d, nochange_n, nochange_d)

    change_d = np.zeros((len(d)))
    change_d[loc - 1] = dchange

    new_modes = phase_calculation(change_d, nochange_n, nochange_d)

    if len(original_modes[0]) == len(new_modes[0]):
        mode_shift = [original_modes[1], new_modes[1] - original_modes[1]]
    else:
        mode_shift = 0

    while original_modes[0][0] != new_modes[0][0]:
         if new_modes[0][0] < original_modes[0][0]:
            original_modes = original_modes[:][1:]
         elif new_modes[0][0] > original_modes[0][0]:
            new_modes = new_modes[:][1:]

    while original_modes[0][-1] != new_modes[0][-1]:
         if new_modes[0][-1] < original_modes[0][-1]:
             new_modes = new_modes[:][:-1]
         if new_modes[0][-1] > original_modes[0][-1]:
             original_modes = original_modes[:][:-1]

    return mode_shift

#x = np.linspace(300, 2500)
#plt.plot(x, modefunction(x))
test2 = layer_relaxation(10 ** (-2), 0, 1)
plt.plot(test2[0], test2[1], '.')