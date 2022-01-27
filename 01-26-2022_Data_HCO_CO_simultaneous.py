import clipboard_and_style_sheet
import numpy as np
import matplotlib.pyplot as plt
import ProcessingFunctions as pf
import nyquist_bandwidths as nq
import mkl_fft
import time


# clipboard_and_style_sheet.style_sheet()


# %%
# path = r'G:\.shortcut-targets-by-id\1cPwz25CLF5JBH9c_yF0vSr5p3Bl_1-nM\MIR GHz DSC\220126/'
# data = np.fromfile(path + "H2CO_filter_60972x65484.bin", '<h')
#
# # %%
# ppifg = 60972 // 2
# Nifgs = 65484
#
# # %%
# data = data[ppifg // 2:-ppifg // 2]
# data.resize(Nifgs - 1, ppifg)
#
#
# # %%
# def computeshifts():
#     zoom = data[:, ppifg // 2 - 200:ppifg // 2 + 200]
#     ftzoom = np.fft.ifftshift(mkl_fft.fft(np.fft.ifftshift(zoom, axes=1), axis=1), axes=1)
#
#     # getting rid of f0
#     ftzoom[:, 300:] = 0.0
#     ftzoom[:, :100] = 0.0
#
#     corrW = ftzoom * ftzoom[0].conj()
#     corrW = np.pad(corrW, ([0, 0], [2 ** 10, 2 ** 10]), constant_values=0.0)
#     corr = np.fft.ifftshift(mkl_fft.ifft(np.fft.ifftshift(corrW, axes=1), axis=1), axes=1)
#     inds = np.argmax(corr, axis=1) - len(corr[0]) // 2
#     shifts = inds * len(zoom[0]) / len(corr[0])
#     return shifts
#
#
# shifts = computeshifts()
#
# # %%
#
# # way too large to be handled all at one time in memory (32 GB), so use a for loop instead
# freq = np.fft.fftfreq(len(data[0]))
# for n, i in enumerate(data):
#     data[n] = np.fft.fftshift(mkl_fft.ifft(mkl_fft.fft(np.fft.ifftshift(i))
#                                            * np.exp(1j * 2 * np.pi * freq * shifts[n]))).real
#     if n % 100 == 0:
#         print('{n}/{N}'.format(n=n, N=Nifgs))
#
# # %%
# data.tofile(path + f"H2CO_filter_{Nifgs}x{ppifg}_phase_corrected.bin")
#
# # %%
# avg = np.mean(data, axis=0)
# ftavg = np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(avg)))

# %% after phase correction

def fft(x, axis=None):
    """
    calculates the 1D fft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if axis is None:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def ifft(x, axis=None):
    """
    calculates the 1D ifft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if axis is None:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


# %%
path = r'G:\.shortcut-targets-by-id\1cPwz25CLF5JBH9c_yF0vSr5p3Bl_1-nM\MIR GHz DSC\220126/'
dataH2CO = np.fromfile(path + "H2CO_filter_65484x30486_phase_corrected.bin", '<h')
dataH2CO.resize(65484, 30486)

# %%
dataCO = np.fromfile(path + "CO_filter_65484x30486_phase_corrected.bin", '<h')
dataCO.resize(65484, 30486)

# %%
Nifgs = 65484
ppifg = 30486

# %%
avgH2CO = np.mean(dataH2CO, axis=0)
ftavgH2CO = fft(avgH2CO)

# %%
avgCO = np.mean(dataCO, axis=0)
ftavgCO = fft(avgCO)

# %% apodization
appodH2CO = dataH2CO[:, ppifg // 2 - 200: ppifg // 2 + 200]

# %% apodization
appodCO = dataCO[:, ppifg // 2 - 200: ppifg // 2 + 200]

# %%
freq_full = np.fft.fftshift(np.fft.fftfreq(len(avgH2CO), 1e-9)) * 1e-6
freq_appod = np.fft.fftshift(np.fft.fftfreq(len(appodH2CO[0]), 1e-9)) * 1e-6

# %%
fig, ax = plt.subplots(1, 1)
ax.plot(freq_full, fft(dataH2CO[0]).__abs__(), label='single shot full')
ax.plot(freq_appod, fft(appodH2CO[0]).__abs__(), label='single shot appodized')
ax.plot(freq_full, ftavgH2CO.__abs__(), label=f'{Nifgs} ifgs averaged')
ax.legend(loc='best')
ax.set_xlim(75, 325)
ax.set_ylim(0, 50e3)
ax.set_xlabel("MHz")
fig.suptitle("$\mathrm{H_2CO}$")

# %%
fig, ax = plt.subplots(1, 1)
ax.plot(freq_full, fft(dataCO[0]).__abs__(), label='single shot full')
ax.plot(freq_appod, fft(appodCO[0]).__abs__(), label='single shot appodized')
ax.plot(freq_full, ftavgCO.__abs__(), label=f'{Nifgs} ifgs averaged')
ax.legend(loc='best')
ax.set_xlim(0, 500)
ax.set_ylim(0, 40e3)
ax.set_xlabel("MHz")
fig.suptitle("$\mathrm{CO}$")

# %%
fig, ax = plt.subplots(1, 1)
ax.plot(freq_full, ftavgH2CO.__abs__())
ax.set_xlim(75, 325)
ax.set_ylim(0, 15e3)
ax.set_xlabel("MHz")
fig.suptitle("$\mathrm{H_2CO}$")

# %%
fig, ax = plt.subplots(1, 1)
ax.plot(freq_full, ftavgCO.__abs__())
ax.set_xlim(0, 500)
ax.set_ylim(0, 30e3)
ax.set_xlabel("MHz")
fig.suptitle("$\mathrm{CO}$")

# %%
T = np.arange(-ppifg // 2, ppifg // 2)
fig, ax = plt.subplots(1, 2, figsize=[9.21, 4.78])
[i.plot(T, avgH2CO) for i in ax]
ax[0].set_xlim(-500, 500)
ax[1].set_xlim(-100, 100)
[i.set_xlabel("time (ns)") for i in ax]
fig.suptitle("$\mathrm{H_2CO}$")

# %%
T = np.arange(-ppifg // 2, ppifg // 2)
fig, ax = plt.subplots(1, 2, figsize=[9.21, 4.78])
[i.plot(T, avgCO) for i in ax]
ax[0].set_xlim(-500, 500)
ax[1].set_xlim(-100, 100)
[i.set_xlabel("time (ns)") for i in ax]
fig.suptitle("$\mathrm{CO}$")
