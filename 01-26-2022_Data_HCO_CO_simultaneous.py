import numpy as np
import matplotlib.pyplot as plt
import ProcessingFunctions as pf
import nyquist_bandwidths as nq
import mkl_fft
import time

# %%
path = r'G:\.shortcut-targets-by-id\1cPwz25CLF5JBH9c_yF0vSr5p3Bl_1-nM\MIR GHz DSC\220126/'
data = np.fromfile(path + "H2CO_filter_60972x65484.bin", '<h')

# %%
ppifg = 60972 // 2
Nifgs = 65484

# %%
data = data[ppifg // 2:-ppifg // 2]
data.resize(Nifgs - 1, ppifg)


# %%
def computeshifts():
    zoom = data[:, ppifg // 2 - 200:ppifg // 2 + 200]
    ftzoom = np.fft.ifftshift(mkl_fft.fft(np.fft.ifftshift(zoom, axes=1), axis=1), axes=1)

    # getting rid of f0
    ftzoom[:, 300:] = 0.0
    ftzoom[:, :100] = 0.0

    corrW = ftzoom * ftzoom[0].conj()
    corrW = np.pad(corrW, ([0, 0], [2 ** 10, 2 ** 10]), constant_values=0.0)
    corr = np.fft.ifftshift(mkl_fft.ifft(np.fft.ifftshift(corrW, axes=1), axis=1), axes=1)
    inds = np.argmax(corr, axis=1) - len(corr[0]) // 2
    shifts = inds * len(zoom[0]) / len(corr[0])
    return shifts


shifts = computeshifts()

# %%

# way too large to be handled all at one time in memory (32 GB), so use a for loop instead
freq = np.fft.fftfreq(len(data[0]))
for n, i in enumerate(data):
    data[n] = np.fft.fftshift(mkl_fft.ifft(mkl_fft.fft(np.fft.ifftshift(i))
                                           * np.exp(1j * 2 * np.pi * freq * shifts[n]))).real
    if n % 100 == 0:
        print('{n}/{N}'.format(n=n, N=Nifgs))

# %%
data.tofile(path + f"H2CO_filter_{Nifgs}x{ppifg}_phase_corrected.bin")

# %%
avg = np.mean(data, axis=0)
ftavg = np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(avg)))
