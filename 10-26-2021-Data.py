"""
I think we finally fixed:
    1.  aliasing issues (figured out what nyquist bandwidths are allowed
    2.  external clocking issue (may have been what was impeding us all along when we could not get our interferograms
        to stop walking
"""

import ProcessingFunctions as pf
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import nyquist_bandwidths as nyquist
import scipy.constants as sc
import time

clipboard_and_style_sheet.style_sheet()


def plot_section(arr, ppifg, npts):
    for dat in arr:
        plt.plot(dat[ppifg // 2 - npts:ppifg // 2 + npts])


def pad(arr, npts):
    return np.pad(arr, ([0, 0], [npts, npts]), constant_values=0 + 0j)


t1 = time.time()
# %% Retrieve the data
folder = r"G:\.shortcut-targets-by-id\1cPwz25CLF5JBH9c_yF0vSr5p3Bl_1-nM\MIR GHz DSC\211026/"
data = np.fromfile(folder + "ifg_new_computer.txt")
ppifg = pf.find_npts(data)[0]
data = data[ppifg // 2:]
N_ifgs = len(data) // ppifg
data = data[:N_ifgs * ppifg]
data.resize(N_ifgs, ppifg)

# %% limit amount to average
# n_avg = 100
# data = data[:n_avg]

# %% calculate shifts needed
n_zoom = 100
zoom = data[:, ppifg // 2 - n_zoom // 2: ppifg // 2 + n_zoom // 2 + 1]
ft_zoom_fftshift = np.fft.fft(np.fft.ifftshift(zoom, axes=1), axis=1)
ft_zoom_fftshift = np.fft.ifftshift(pad(np.fft.fftshift(ft_zoom_fftshift, axes=1), 2 ** 11), axes=1)
ref = np.zeros(ft_zoom_fftshift.shape, dtype=np.complex128)
ref[:] = ft_zoom_fftshift[0]
ft_cross_corr_fftshift = np.conj(ref) * ft_zoom_fftshift
cross_corr = np.fft.fftshift(np.fft.ifft(ft_cross_corr_fftshift, axis=1), axes=1).real
ind = np.argmax(cross_corr, axis=1) - len(cross_corr[0]) // 2
shifts = ind * len(zoom[0]) / len(cross_corr[0])

# %% FT the data and shift the data
ft_data_fftshift = np.fft.fft(np.fft.ifftshift(data, axes=1), axis=1)

# %% FT shift the data
fft_freq = np.fft.fftfreq(len(ft_data_fftshift[0]))
phase = np.zeros(ft_data_fftshift.shape, dtype=np.complex128)
phase[:] = fft_freq
phase *= 1j * shifts[:, np.newaxis] * 2 * np.pi
phase = np.exp(phase)
ft_data_fftshift *= phase

# %% final fft
ft_avg = np.mean(np.fft.fftshift(ft_data_fftshift), axis=0)

# %% got fr and dfr_guess from counter
fr = 1.01e9 - 9991829.1
dfr_guess = 23055.3
dfr = nyquist.calc_dfr_for_ppifg(fr, dfr_guess, ppifg)
dnu = nyquist.bandwidth(fr, dfr)
nu_start = dnu * 3
nu_end = dnu * 4
freq_axis = np.linspace(nu_start, nu_end, len(fft_freq) // 2)
wl_axis = (sc.c / freq_axis) * 1e6

# fig, ax = plt.subplots(1, 1, figsize=np.array([19.2, 9.63]))
fig, ax = plt.subplots(1, 1)
ax.plot(wl_axis, ft_avg[:len(ft_avg) // 2].__abs__())
ax.set_xlim(3.5, 4.5)
ax.set_ylim(0, 0.8)
ax.set_xlabel("$\\mathrm{\\mu m}$")
# plt.savefig("10-26-2021-Data.png")

t2 = time.time()
print(t2 - t1, "s")

# %% checking to see if shift correction was successful
shift_corr_data = np.fft.fftshift(np.fft.ifft(ft_data_fftshift, axis=1), axes=1).real

plt.figure()
plot_section(shift_corr_data, ppifg, 50)
avg = np.mean(shift_corr_data, axis=0)
plt.plot(avg[ppifg // 2 - 50: ppifg // 2 + 50], color='k')

# %% allowed nyquist windows
# to prevent naming conflicts, I have it in a function
# I eyeballed this from the FFT above
wls = np.array([4.4, 3.5])  # um
nus = sc.c / (wls * 1e-6)
dfr_ = np.linspace(50, fr ** 2 / (2 * np.diff(nus)), 5000).flatten()
dnu_ = nyquist.bandwidth(fr, dfr_)
ind = nyquist.return_allowed_indices_dnu(dnu_, *nus)

fig, ax = plt.subplots(1, 1)
x = np.zeros(len(dfr_))
ax.plot(dfr_, x)
ax.plot(dfr_[ind], x[ind], 'o')
[ax.axvline(i, color='r', linestyle='--') for i in nyquist.find_allowed_dfr(*nus, fr).flatten()]
