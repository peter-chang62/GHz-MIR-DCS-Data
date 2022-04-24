import clipboard_and_style_sheet
import numpy as np
import matplotlib.pyplot as plt
import ProcessingFunctions as pf
import nyquist_bandwidths as nq
import mkl_fft
import time
import scipy.constants as sc
import scipy.signal.windows as sw
import scipy.signal as si
import os
import scipy.interpolate as spi


# clipboard_and_style_sheet.style_sheet()

def normalize(vec):
    return vec / np.max(abs(vec))


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
path = r'C:/Users/fastdaq/Desktop/Data_04232022/'
data = np.fromfile(path + 'H2CO_test_for_alias_28272x17508.bin', '<h')
ppifg = 17508
Nifgs = 28272
center = ppifg // 2
data = data[ppifg // 2:-ppifg // 2]
data = data.reshape(Nifgs - 1, ppifg)

# %%
zoom = data[:, ppifg // 2 - 50:ppifg // 2 + 51]
ftzoom = fft(zoom)

# getting rid of f0 for H2CO (comment out this block for the CO one)
# ftzoom[:, 300:] = 0.0
# ftzoom[:, :100] = 0.0
# zoom = ifft(ftzoom).real
# zoom = zoom[:, 50:-50]  # some ringing on the edge due to the brick wall filter
# ftzoom = fft(zoom)

# %%
corrW = ftzoom * ftzoom[0].conj()
corrW = np.pad(corrW, ([0, 0], [2 ** 10, 2 ** 10]), constant_values=0.0)
corr = ifft(corrW)
inds = np.argmax(corr, axis=1) - len(corr[0]) // 2
shifts = inds * len(zoom[0]) / len(corr[0])

# %%
# plt.figure()
# plt.plot(corr[0].real)

# %%
# # way too large to be handled all at one time in memory (32 GB), so use a for loop instead
freq = np.fft.fftfreq(len(data[0]))
for n, i in enumerate(data):
    data[n] = np.fft.fftshift(mkl_fft.ifft(mkl_fft.fft(np.fft.ifftshift(i))
                                           * np.exp(1j * 2 * np.pi * freq * shifts[n]))).real
    if n % 100 == 0:
        print('{n}/{N}'.format(n=n, N=Nifgs))

avg = np.mean(data, axis=0)
ftavg = fft(avg)

# %%
freq_full = np.fft.fftshift(np.fft.fftfreq(ppifg, 1e-9)) * 1e-6

fr, dfr = 1010e6 - 9998056, 57108
N_nyq_short = 10
# N_nyq_long = 5
dnu = nq.bandwidth(fr, dfr)
# window_long = np.array([dnu * (N_nyq_long - 1), dnu * N_nyq_long])
window_short = np.array([dnu * (N_nyq_short - 1), dnu * N_nyq_short])
# freq_axis_long = np.linspace(*window_long, len(freq_full) // 2)
freq_axis_short = np.linspace(*window_short, len(freq_full) // 2)
# wl_axis_long = sc.c * 1e6 / freq_axis_long
wl_axis_short = sc.c * 1e6 / freq_axis_short

# %%
plt.figure()
plt.plot(wl_axis_short, ftavg[:center].__abs__())
