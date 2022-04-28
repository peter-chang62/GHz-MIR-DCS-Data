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
h2co = np.vstack([dat1, dat2])

# %%
t0_h2co = h2co[:, 0, :]
t0_h2co = np.mean(t0_h2co, 0)
fft_t0_h2co = pc.fft(t0_h2co)

# %%
ref_h2co = np.fromfile('h2co_vacuum_background.bin')
fft_ref_h2co = pc.fft(ref_h2co)

# %%
freq = np.fft.fftshift(np.fft.fftfreq(len(fft_ref_h2co), 1e-9)) * 1e-6
plt.figure()
plt.plot(freq, fft_ref_h2co.__abs__(), label='background')
plt.plot(freq, fft_t0_h2co.__abs__(), label='data')
plt.xlabel("MHz")
plt.legend(loc='best')
plt.xlim(0, 320)
plt.ylim(0, 8000)

# %%
dat1 = np.fromfile(r'D:\ShockTubeData\04242022_Data\Surf_27\PHASE_CORRECTED_DATA/CO_499x50x17507.bin')
dat2 = np.fromfile(r'D:\ShockTubeData\04242022_Data\Surf_28\PHASE_CORRECTED_DATA/CO_299x50x17507.bin')
dat1.resize((499, 50, 17507))
dat2.resize((299, 50, 17507))
co = np.vstack([dat1, dat2])
ref_co = np.fromfile('co_vacuum_background.bin')

# %%
t0_co = co[:, 0, :]
t0_co = np.mean(t0_co, 0)
fft_t0_co = pc.fft(t0_co)
fft_ref_co = pc.fft(ref_co)

# %%
freq = np.fft.fftshift(np.fft.fftfreq(len(fft_ref_co), 1e-9)) * 1e-6
plt.figure()
plt.plot(freq, fft_ref_co.__abs__(), label='background')
plt.plot(freq, fft_t0_co.__abs__(), label='data')
plt.xlabel("MHz")
plt.legend(loc='best')
plt.xlim(100, 300)
plt.ylim(0, 1.217e4)

# %%
# plt.ioff()
# for i in range(co.shape[1]):
#     t0_co = co[:, i, :]
#     t0_co = np.mean(t0_co, 0)
#     fft_t0_co = pc.fft(t0_co)
#     fft_ref_co = pc.fft(ref_co)
#
#     freq = np.fft.fftshift(np.fft.fftfreq(len(fft_ref_co), 1e-9)) * 1e-6
#     plt.figure(figsize=np.array([10.81, 7.63]))
#     plt.plot(freq, fft_ref_co.__abs__(), label='background')
#     plt.plot(freq, fft_t0_co.__abs__(), label='data')
#     plt.xlabel("MHz")
#     plt.legend(loc='best')
#     plt.xlim(100, 300)
#     plt.ylim(0, 1.217e4)
#     plt.title(i)
#     plt.savefig('temp_fig/' + f'{i}.png')
#     print(i)
#
# plt.ion()
