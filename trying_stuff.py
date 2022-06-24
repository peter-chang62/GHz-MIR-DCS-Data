import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc
import digital_phase_correction as dpc
import scipy.optimize as spo
import scipy.signal as ss


def apply_t0_shift(pdiff, freq, fft):
    # the polynomial fits the spectral phase in radians,
    # so the factor of 2 pi is already there
    fft[:] *= np.exp(1j * freq * pdiff[:, 0][:, np.newaxis])


def apply_phi0_shift(pdiff, hbt):
    # the polynomial fits the spectral phase in radians,
    # so the factor of 2 pi is already there
    hbt[:] *= np.exp(1j * pdiff[:, 1][:, np.newaxis])


def apply_t0_and_phi0_shift(pdiff, data):
    freq = np.fft.fftshift(np.fft.fftfreq(len(data[0])))
    fft = pc.fft(data, 1)
    apply_t0_shift(pdiff, freq, fft)
    td = pc.ifft(fft, 1).real

    hbt = ss.hilbert(td)
    apply_phi0_shift(pdiff, hbt)
    hbt = hbt.real

    data[:] = hbt.real


# %% ___________________________________________________________________________________________________________________
# ppifg = 198850
# center = ppifg // 2
# data = np.fromfile(r'D:\ShockTubeData\static cell/cell_with_mixture_and_4_5_filter_at200_4998x198850.bin', '<h')
# data, _ = pc.adjust_data_and_reshape(data, ppifg)
# data = data - np.c_[np.mean(data, 1)]

ppifg = 198850
center = ppifg // 2
data = np.fromfile(r'D:\ShockTubeData\static cell/cell_with_mixture_10030x198850.bin', '<h')
data, _ = pc.adjust_data_and_reshape(data, ppifg)
data = data - np.c_[np.mean(data, 1)]

# %% ___________________________________________________________________________________________________________________
N_apod = 400
zoom = data[:, center - N_apod // 2: center + N_apod // 2]
fft = pc.fft(zoom, 1)
phase = np.unwrap(np.arctan2(fft.imag, fft.real))

# ll_freq, ul_freq = 0.325, 0.345  # 4.5 um
ll_freq, ul_freq = 0.325, 0.487  # no filter
# f0 = (ul_freq - ll_freq) / 2 + ll_freq
f0 = 0
freq = np.fft.fftshift(np.fft.fftfreq(len(fft[0])))
ll, ul = np.argmin(abs(freq - ll_freq)), np.argmin(abs(freq - ul_freq))
p = np.polyfit(freq[ll:ul] - f0, phase.T[ll:ul], 1)  # column order for fitting
p_linear = p[-2:].T  # back to row order
p_linear[:, 1] -= p_linear[:, 0] * f0
pdiff_linear = p_linear[0] - p_linear

# %% ___________________________________________________________________________________________________________________
h = 0
step = 250
N_analyze = 1000
while h < N_analyze:
    apply_t0_and_phi0_shift(pdiff_linear[h: h + step], data[h: h + step])
    h += step
    print(N_analyze - h)

[plt.plot(i[center - 500:center + 500]) for i in data[:1000]]
