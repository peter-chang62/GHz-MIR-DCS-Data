import scipy.constants as sc
import nyquist_bandwidths as nq
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import mkl_fft


def normalize(vec):
    return vec / np.max(abs(vec))


# %%
dataCO = np.fromfile("Data/03-01-2022/CO_filter_phasecorr_9995x199728.bin", 'h')
dataH2CO = np.fromfile("Data/03-01-2022/H2CO_filter_phasecorr_9995x199728.bin", 'h')
dataH2CO.resize(9995, 199728)
dataCO.resize(9995, 199728)

# %%
avgH2CO = np.mean(dataH2CO, axis=0)
ftH2CO = np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(avgH2CO)))

avgCO = np.mean(dataCO, axis=0)
ftCO = np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(avgCO)))

# %% filter f0 for CO
ftCO[38000:96000] = 0.0
ftCO[106000:162000] = 0.0
avgCO = np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(ftCO))).real

# %% filter f0 for H2CO
ftH2CO[101000:170000] = 0
ftH2CO[25000:98000] = 0
avgH2CO = np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(ftH2CO))).real

# %%
plt.figure()
plt.plot(avgH2CO)
plt.title("average H2CO")

plt.figure()
plt.plot(avgCO)
plt.title("average CO")

# %% Obtaining the frequency axis, at 5kHz we are in the first Nyquist window
fr = 1e9  # I forgot to record it, oh well...
dfr = nq.calc_dfr_for_ppifg(fr, dfr_guess=5e3, ppifg=199728)
dnu = nq.bandwidth(fr, dfr)
freq = np.linspace(0, dnu, len(ftCO) // 2)

# %%
startind = int(60e3)
plt.figure()
plt.title("frequency")
plt.plot(freq[startind:] * 1e-12, normalize(ftCO.__abs__()[len(ftCO) // 2:][startind:]))
plt.plot(freq[startind:] * 1e-12, normalize(ftH2CO.__abs__()[len(ftCO) // 2:][startind:]))
plt.xlabel("Frequency (THz)")

# %%
wl = sc.c * 1e6 / freq[startind:]
plt.figure()
plt.title("wavelength")
plt.plot(wl, normalize(ftCO.__abs__()[len(ftCO) // 2:][startind:]))
plt.plot(wl, normalize(ftH2CO.__abs__()[len(ftCO) // 2:][startind:]))
plt.xlabel("Wavelength ($\mathrm{\mu m}$)")

# %% CO nyquist window
viCO = 63e12
vfCO = 69.5e12
viH2CO = 80.7e12
vfH2CO = 89.2e12

dfrCO = nq.find_allowed_dfr(viCO, vfCO, fr)
dfrH2CO = nq.find_allowed_dfr(viH2CO, vfH2CO, fr)

# %%
plt.figure()
[plt.plot(i, [0, 0], 'C0') for i in dfrCO]
[[plt.axvline(i, color='C0') for i in e] for e in dfrCO]
[plt.plot(i, [1, 1], 'C1') for i in dfrH2CO]
[[plt.axvline(i, color='C1') for i in e] for e in dfrH2CO]
