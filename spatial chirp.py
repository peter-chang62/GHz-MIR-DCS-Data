import numpy as np
import matplotlib.pyplot as plt
from pynlo_peter import Fiber_PPLN_NLSE as fpn
import scipy.constants as sc
import clipboard_and_style_sheet
from scipy.interpolate import interp1d


# clipboard_and_style_sheet.style_sheet()


class MFD:
    def __init__(self):
        mfd_data = np.genfromtxt("SM_INF3_MFD.txt")
        self.wl_um = mfd_data[:, 0]
        self.mfd = mfd_data[:, 1]

        self.grid = interp1d(self.wl_um, self.mfd, 'cubic', bounds_error=True)


def abcd(a, b, c, d, q):
    return (a * q + b) / (c * q + d)


def waist(q, wl):
    return np.sqrt(1 / (np.imag(-1 / q) * np.pi / wl))


def q_focus(w, wl):
    return (-1j * wl / (np.pi * w ** 2)) ** -1


def lens(q, f):
    return abcd(1, 0, -1 / f, 1, q)


def medium(q, d, n=1.):
    return abcd(1, d / n, 0, 1, q)


def n_ppln(wl_um):
    wl_nm = wl_um * 1e3
    return fpn.DengSellmeier(23.).n(wl_nm)


def abcd_to_oap_input(wl, l_free_space_m):
    """
    OAP -> PPLN -> free space
    f OAP = 2 inches
    Length PPLN = 1 mm
    """
    wo_origin = 15.0e-6
    length_ppln = 1.0e-3
    f = sc.inch * 2

    q = q_focus(wo_origin, wl)
    q = medium(q, length_ppln / 2, n_ppln(wl * 1e6))
    q = medium(q, f - length_ppln / 2)
    q = lens(q, f)
    q = medium(q, l_free_space_m)
    return q


def waist_at_oap_collimator(wl):
    wo_origin = 15.0e-6
    length_ppln = 1.0e-3
    f = sc.inch * 2

    q = q_focus(wo_origin, wl)
    q = medium(q, length_ppln / 2, n_ppln(wl * 1e6))
    q = medium(q, f - length_ppln / 2)
    return waist(q, wl)


def waist_at_oap_input(wl, l_free_space_m):
    q = abcd_to_oap_input(wl, l_free_space_m)
    return waist(q, wl)


def waist_at_fiber_input(wl, f_mm, l_free_space_m):
    f = f_mm * 1e-3

    q = abcd_to_oap_input(wl, l_free_space_m)
    q = lens(q, f)
    q = medium(q, f)
    return waist(q, wl)


def waist_at_fiber_output(wl, f_mm):
    f = f_mm * 1e-3
    mfd = MFD().grid(wl * 1e6) * 1e-6
    w = mfd / 2

    q = q_focus(w, wl)
    q = medium(q, f)
    return waist(q, wl)


# %%
wl = np.linspace(3e-6, 5e-6, 5000)
l_freespace_m = np.linspace(.3, 1., 100)

D_oap = np.zeros((len(l_freespace_m), len(wl)))
D_fiber = np.zeros((len(l_freespace_m), len(wl)))
for n, l in enumerate(l_freespace_m):
    D_oap[n] = waist_at_oap_input(wl, l) * 2
    D_fiber[n] = waist_at_fiber_input(wl, 7., l) * 2

# %%
y = l_freespace_m * 100
x = wl * 1e6
plt.figure()
plt.pcolormesh(x, y, D_oap * 1e3, cmap='jet')
plt.colorbar()
plt.xlabel("$\mathrm{\lambda \mu m}$")
plt.ylabel("distance (cm)")
plt.title("Diameter at fiber input coupling OAP (mm)")

plt.figure()
plt.pcolormesh(x, y, D_fiber * 1e6, cmap='jet')
plt.colorbar()
plt.xlabel("$\mathrm{\lambda \mu m}$")
plt.ylabel("distance (cm)")
plt.title("Diameter at fiber input coupling OAP ($\mathrm{\mu m}$)")

plt.figure()
plt.title("Fiber stated MFD specs")
plt.plot(MFD().wl_um, MFD().mfd)
plt.xlabel("$\mathrm{\lambda \mu m}$")
plt.ylabel("MFD $\mathrm{\mu m}$")

plt.figure()
plt.title("Waist after PPLN OAP Collimator")
plt.plot(wl * 1e6, waist_at_oap_collimator(wl) * 1e3)
plt.xlabel("$\mathrm{\lambda \mu m}$")
plt.ylabel("mm")
