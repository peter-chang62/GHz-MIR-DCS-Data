import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import numpy as np
import mkl_fft
import phase_correction as pc
from scipy.signal import windows as wd
import scipy.integrate as si
import scipy.signal as ss

# %%
h2co = np.fromfile(r'D:\ShockTubeData\04242022_Data\Vacuum_Background/card2_114204x17507.bin', '<h')
N_ifgs = 114204
ppifg = 17507
center = ppifg // 2
N_zoom = 24
h2co, ind = pc.adjust_data_and_reshape(h2co, ppifg)

# %%
sig1 = h2co[0][center - 100:center + 101]
sig2 = h2co[4500][center - 100:center + 101]

# %%
fft1 = pc.fft(sig1)
fft2 = pc.fft(sig2)
p1 = np.unwrap(np.arctan2(fft1.imag, fft1.real))
p2 = np.unwrap(np.arctan2(fft2.imag, fft2.real))
pdiff = p1 - p2

# %%
freq = np.fft.fftshift(np.fft.fftfreq(len(fft1)))
fft1 = fft1[len(freq) // 2:]
fft2 = fft2[len(freq) // 2:]
p1 = p1[len(freq) // 2:]
p2 = p2[len(freq) // 2:]
pdiff = pdiff[len(freq) // 2:]
freq = freq[len(freq) // 2:]

# %%
num1 = si.simps(freq * fft1.__abs__())
denom1 = si.simps(fft1.__abs__())
fc1 = num1 / denom1

num2 = si.simps(freq * fft2.__abs__())
denom2 = si.simps(fft2.__abs__())
fc2 = num2 / denom2

# %%
ll, ul = 15, 40

# %%
pdiff_sec = pdiff[ll:ul]
z = np.polyfit(freq[ll:ul], pdiff[ll:ul], 1)
p = np.poly1d(z)

# %%
Ucor1 = sig2.copy()
Ucor1 = np.fft.fftshift(mkl_fft.ifft(mkl_fft.fft(np.fft.ifftshift(Ucor1)) *
                                     np.exp(1j * np.fft.fftfreq(len(Ucor1)) * z[0])))

# %%
Scor1 = ss.hilbert(Ucor1.real)
# Ucor2 = np.real(Scor1 * np.exp(1j * 2 * np.pi * (fc1 - fc2)))
Ucor2 = Scor1.real

# %%
fft1 = pc.fft(sig1)
fft2 = pc.fft(Ucor2)
p1 = np.unwrap(np.arctan2(fft1.imag, fft1.real))
p2 = np.unwrap(np.arctan2(fft2.imag, fft2.real))
pdiff = p1 - p2

# %%
freq = np.fft.fftshift(np.fft.fftfreq(len(fft1)))
fft1 = fft1[len(freq) // 2:]
fft2 = fft2[len(freq) // 2:]
p1 = p1[len(freq) // 2:]
p2 = p2[len(freq) // 2:]
pdiff = pdiff[len(freq) // 2:]
freq = freq[len(freq) // 2:]

# %%
pdiff_sec = pdiff[ll:ul]
z = np.polyfit(freq[ll:ul], pdiff[ll:ul], 1)
p = np.poly1d(z)

# %%
Scor2 = ss.hilbert(Ucor2)
Ucor3 = (Scor2 * np.exp(1j * z[1])).real

# %%
plt.plot(sig1)
plt.plot(Ucor3)
plt.plot(sig2)
