import ProcessingFunctions as pf
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import nyquist_bandwidths as nyquist
import scipy.constants as sc
import time
import mkl_fft as mkl


def shift_vec(vec, shift):
    fftfreq = np.fft.fftfreq(len(vec))
    ft = mkl.fft(np.fft.ifftshift(vec))
    ft *= np.exp(1j * fftfreq * 2 * np.pi * shift)
    return np.fft.fftshift(mkl.ifft(ft)).real


# %%
# Data = np.fromfile("Data/03-01-2022/H2CO_filter_9996x399456.bin", 'h')
Data = np.fromfile("Data/03-01-2022/CO_filter_9996x399456.bin", 'h')
ppifg = 399456 // 2
Data = Data[ppifg // 2: - ppifg // 2]
Data.resize(9996 - 1, 399456 // 2)

# %% Calculate shifts for H2CO
# zoom = Data[:, ppifg // 2 - 500: ppifg // 2 + 500 + 1]
# zoom = zoom - np.mean(zoom, axis=1)[:, np.newaxis]
# ftzoom = np.fft.fftshift(mkl.fft(np.fft.ifftshift(zoom, axes=1), axis=1), axes=1)
# ftzoom[:, 150:300] = 0
# ftzoom[:, 700:850] = 0
# ftzoom = np.pad(ftzoom, ([0, 0], [2 ** 11, 2 ** 11]), constant_values=0 + 0j)
# cross = np.zeros(ftzoom.shape, dtype=np.complex128)
# cross[:] = ftzoom[0]
# cross *= ftzoom.conj()
# cross = np.fft.fftshift(mkl.ifft(np.fft.ifftshift(cross, axes=1), axis=1), axes=1)
# ind = np.argmax(cross, axis=1) - cross.shape[1] // 2
# shift = ind * zoom.shape[1] / cross.shape[1]

# %% Calculate shifts for CO
zoom = Data[:, ppifg // 2 - 500: ppifg // 2 + 500 + 1]
zoom = zoom - np.mean(zoom, axis=1)[:, np.newaxis]
ftzoom = np.fft.fftshift(mkl.fft(np.fft.ifftshift(zoom, axes=1), axis=1), axes=1)
ftzoom = np.pad(ftzoom, ([0, 0], [2 ** 11, 2 ** 11]), constant_values=0 + 0j)
cross = np.zeros(ftzoom.shape, dtype=np.complex128)
cross[:] = ftzoom[0]
cross *= ftzoom.conj()
cross = np.fft.fftshift(mkl.ifft(np.fft.ifftshift(cross, axes=1), axis=1), axes=1)
ind = np.argmax(cross, axis=1) - cross.shape[1] // 2
shift = ind * zoom.shape[1] / cross.shape[1]

N = len(Data)
for n in range(1, N):
    Data[n] = shift_vec(Data[n], - shift[n])

    if n % 10 == 0:
        print(n)


# %% save the data
# Data.tofile("Data/03-01-2022/H2CO_filter_phasecorr_{Nifgs}x{PPIFG}.bin".format(Nifgs=9996 - 1, PPIFG=399456 // 2))
Data.tofile("Data/03-01-2022/CO_filter_phasecorr_{Nifgs}x{PPIFG}.bin".format(Nifgs=9996 - 1, PPIFG=399456 // 2))


# %% loosk okay!
zoomnew = Data[:, ppifg // 2 - 500: ppifg // 2 + 500 + 1]
avg = np.mean(Data, 0)
ftavg = np.fft.fftshift(mkl.fft(np.fft.ifftshift(avg)))

# %%
plt.figure()
[plt.plot(i) for i in zoomnew[:100]]

plt.figure()
plt.plot(avg)

plt.figure()
plt.plot(ftavg.__abs__())
