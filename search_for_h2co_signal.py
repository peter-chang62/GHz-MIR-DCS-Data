import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc
from scipy.signal import windows as wd
import mkl_fft

clipboard_and_style_sheet.style_sheet()

# %%
dat1 = np.fromfile(r'D:\ShockTubeData\04242022_Data\Surf_27\PHASE_CORRECTED_DATA/H2CO_499x50x17507.bin')
dat2 = np.fromfile(r'D:\ShockTubeData\04242022_Data\Surf_28\PHASE_CORRECTED_DATA/H2CO_299x50x17507.bin')
dat1.resize((499, 50, 17507))
dat2.resize((299, 50, 17507))
dat = np.vstack([dat1, dat2])

# %%
zero = dat[:, 0, :]
zero = np.mean(zero, 0)
fft_zero = pc.fft(zero)

# %%
ref = np.fromfile('h2co_vacuum_background.bin')
fft_ref = pc.fft(ref)

# %%
freq = np.fft.fftshift(np.fft.fftfreq(len(fft_ref), 1e-9)) * 1e-6
plt.plot(freq, fft_ref.__abs__(), label='background')
plt.plot(freq, fft_zero.__abs__(), label='data')
plt.xlabel("MHz")
plt.legend(loc='best')
