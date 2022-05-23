import matplotlib.pyplot as plt
import numpy as np
import clipboard_and_style_sheet
import mkl_fft
import phase_correction as pc
import scipy.integrate as si
import scipy.signal as ss

# %%
path_h2co = r'D:\ShockTubeData\04242022_Data\Surf_27\card2/'
ppifg = 17507
data = np.fromfile(r'D:\ShockTubeData\04242022_Data\Vacuum_Background/card2_114204x17507.bin', '<h')
data, _ = pc.adjust_data_and_reshape(data, ppifg)

ll_freq = 0.0597
ul_freq = 0.20

# %%
center = ppifg // 2
zoom = data[:, center - 100:center + 100]
zoom = (zoom.T - np.mean(zoom, 1)).T

fft = pc.fft(zoom, 1)
freq = np.fft.fftshift(np.fft.fftfreq(len(fft[0])))

phase = np.unwrap(np.arctan2(fft.imag, fft.real))
phase = phase.T  # column order for polynomial fitting
ll, ul = np.argmin(abs(freq - ll_freq)), np.argmin(abs(freq - ul_freq))
p = np.polyfit(freq[ll:ul], phase[ll:ul], 1).T
pdiff = p[0] - p

# %%
h = 0
step = 250
freq = np.fft.fftfreq(len(data[0]))
while h < len(data):
    sec = data[h: h + step]

    fft = mkl_fft.fft(np.fft.ifftshift(sec, axes=1), axis=1)
    fft *= np.exp(1j * freq * pdiff[h: h + step, 0][:, np.newaxis])
    fft = np.fft.fftshift(mkl_fft.ifft(fft, axis=1), axes=1).real

    hilbert = ss.hilbert(fft, axis=1)
    hilbert *= np.exp(1j * pdiff[h: h + step, 1][:, np.newaxis])

    sec = hilbert.real
    data[h: h + step] = sec
    h += step

    print(len(data) - h)
