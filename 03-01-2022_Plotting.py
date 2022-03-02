import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import mkl_fft

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
plt.title("average H2CO")
plt.plot(avgH2CO)

plt.figure()
plt.title("average H2CO")
plt.plot(ftH2CO.__abs__())

plt.figure()
plt.title("average CO")
plt.plot(avgCO)

plt.figure()
plt.title("average CO")
plt.plot(ftCO.__abs__())
