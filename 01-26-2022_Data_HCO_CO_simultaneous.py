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


def delete_files_in_image_directory():
    # first delete these files
    del_files = [i.path for i in os.scandir("TempImages")]
    [os.remove(i) for i in del_files]


def create_mp4(fps, name):
    command = "ffmpeg -r " + \
              str(fps) + \
              " -f image2 -s 1920x1080 -y -i TempImages/%d.png " \
              "-vcodec libx264 -crf 25  -pix_fmt yuv420p " + \
              name
    os.system(command)


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
# path = r'G:\.shortcut-targets-by-id\1cPwz25CLF5JBH9c_yF0vSr5p3Bl_1-nM\MIR GHz DSC\220126/'
# data = np.fromfile(path + "H2CO_filter_60972x65484.bin", '<h')
#
# # %%
# ppifg = 60972 // 2
# Nifgs = 65484
#
# # %%
# plotsec = lambda vec: plt.plot(vec[ppifg // 2 - 100: ppifg // 2 + 100])
#
# # %%
# data = data[ppifg // 2:-ppifg // 2]
# data.resize(Nifgs - 1, ppifg)
#
#
# # %%
# def computeshifts():
#     zoom = data[:, ppifg // 2 - 200:ppifg // 2 + 200]
#     ftzoom = fft(zoom)
#
#     # getting rid of f0 for H2CO (comment out this block for the CO one)
#     ftzoom[:, 300:] = 0.0
#     ftzoom[:, :100] = 0.0
#     zoom = ifft(ftzoom).real
#     zoom = zoom[:, 50:-50]  # some ringing on the edge due to the brick wall filter
#     ftzoom = fft(zoom)
#
#     corrW = ftzoom * ftzoom[0].conj()
#     corrW = np.pad(corrW, ([0, 0], [2 ** 10, 2 ** 10]), constant_values=0.0)
#     corr = ifft(corrW)
#     inds = np.argmax(corr, axis=1) - len(corr[0]) // 2
#     shifts = inds * len(zoom[0]) / len(corr[0])
#     return shifts
#
#
# shifts = computeshifts()
#
# # %%
#
# # way too large to be handled all at one time in memory (32 GB), so use a for loop instead
# freq = np.fft.fftfreq(len(data[0]))
# for n, i in enumerate(data):
#     data[n] = np.fft.fftshift(mkl_fft.ifft(mkl_fft.fft(np.fft.ifftshift(i))
#                                            * np.exp(1j * 2 * np.pi * freq * shifts[n]))).real
#     if n % 100 == 0:
#         print('{n}/{N}'.format(n=n, N=Nifgs))
#
# # %%
# data.tofile(path + f"H2CO_filter_{Nifgs}x{ppifg}_phase_corrected.bin")
#
# # %%
# avg = np.mean(data, axis=0)
# ftavg = np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(avg)))

# %% after phase correction
path = r'G:\.shortcut-targets-by-id\1cPwz25CLF5JBH9c_yF0vSr5p3Bl_1-nM\MIR GHz DSC\220126/'
# path = r"C:\Users\fastdaq\Documents\Data\01-26-2022/"
# dataH2CO = np.fromfile(path + "H2CO_filter_65483x30486_phase_corrected.bin", '<h')
# dataH2CO.resize(65483, 30486)

# %%
dataCO = np.fromfile(path + "CO_filter_65483x30486_phase_corrected.bin", '<h')
dataCO.resize(65483, 30486)

# %%
Nifgs = 65484
ppifg = 30486

# %%
# avgH2CO = np.mean(dataH2CO, axis=0)
# ftavgH2CO = fft(avgH2CO)

# %%
avgCO = np.mean(dataCO, axis=0)
ftavgCO = fft(avgCO)

# %% apodization
# appodH2CO = dataH2CO[:, ppifg // 2 - 200: ppifg // 2 + 200]

# %% apodization
# appodCO = dataCO[:, ppifg // 2 - 200: ppifg // 2 + 200]

# %%
freq_full = np.fft.fftshift(np.fft.fftfreq(ppifg, 1e-9)) * 1e-6
# freq_appod = np.fft.fftshift(np.fft.fftfreq(len(appodH2CO[0]), 1e-9)) * 1e-6

# %%
# fig, ax = plt.subplots(1, 1)
# ax.plot(freq_full, fft(dataH2CO[0]).__abs__(), label='single shot full')
# ax.plot(freq_appod, fft(appodH2CO[0]).__abs__(), label='single shot appodized')
# ax.plot(freq_full, ftavgH2CO.__abs__(), label=f'{Nifgs} ifgs averaged')
# ax.legend(loc='best')
# ax.set_xlim(75, 325)
# ax.set_ylim(0, 50e3)
# ax.set_xlabel("MHz")
# fig.suptitle("$\mathrm{H_2CO}$")
#
# # %%
# fig, ax = plt.subplots(1, 1)
# ax.plot(freq_full, fft(dataCO[0]).__abs__(), label='single shot full')
# ax.plot(freq_appod, fft(appodCO[0]).__abs__(), label='single shot appodized')
# ax.plot(freq_full, ftavgCO.__abs__(), label=f'{Nifgs} ifgs averaged')
# ax.legend(loc='best')
# ax.set_xlim(0, 500)
# ax.set_ylim(0, 40e3)
# ax.set_xlabel("MHz")
# fig.suptitle("$\mathrm{CO}$")
#
# # %%
# fig, ax = plt.subplots(1, 1)
# ax.plot(freq_full, ftavgH2CO.__abs__())
# ax.set_xlim(75, 325)
# ax.set_ylim(0, 15e3)
# ax.set_xlabel("MHz")
# fig.suptitle("$\mathrm{H_2CO}$")
#
# # %%
# fig, ax = plt.subplots(1, 1)
# ax.plot(freq_full, ftavgCO.__abs__())
# ax.set_xlim(0, 500)
# ax.set_ylim(0, 30e3)
# ax.set_xlabel("MHz")
# fig.suptitle("$\mathrm{CO}$")
#
# # %%
# T = np.arange(-ppifg // 2, ppifg // 2)
# fig, ax = plt.subplots(1, 2, figsize=[9.21, 4.78])
# [i.plot(T, avgH2CO) for i in ax]
# ax[0].set_xlim(-500, 500)
# ax[1].set_xlim(-100, 100)
# [i.set_xlabel("time (ns)") for i in ax]
# fig.suptitle("$\mathrm{H_2CO}$")
#
# # %%
# T = np.arange(-ppifg // 2, ppifg // 2)
# fig, ax = plt.subplots(1, 2, figsize=[9.21, 4.78])
# [i.plot(T, avgCO) for i in ax]
# ax[0].set_xlim(-500, 500)
# ax[1].set_xlim(-100, 100)
# [i.set_xlabel("time (ns)") for i in ax]
# fig.suptitle("$\mathrm{CO}$")
#
# %%
fr = 1e9  # approximate
dfr = 1 / ((1 / fr) * ppifg)

# wl_lim_short = [3.4, 3.7]
# nu_lim_short = sc.c / (np.array(wl_lim_short)[::-1] * 1e-6)
# wl_lim_long = [4.3, 4.8]
# nu_lim_long = sc.c / (np.array(wl_lim_long)[::-1] * 1e-6)
#
# N_nyq_short = 6
# N_nyq_long = 5
# dnu = nq.bandwidth(fr, dfr)
# window_long = np.array([dnu * (N_nyq_long - 1), dnu * N_nyq_long])
# window_short = np.array([dnu * (N_nyq_short - 1), dnu * N_nyq_short])
# freq_axis_long = np.linspace(*window_long, len(freq_full) // 2)
# freq_axis_short = np.linspace(*window_short, len(freq_full) // 2)
# wl_axis_long = sc.c * 1e6 / freq_axis_long
# wl_axis_short = sc.c * 1e6 / freq_axis_short
#
# # %%
# fig, ax = plt.subplots(1, 1)
# ax.plot(wl_axis_long, ftavgCO.__abs__()[len(freq_full) // 2:])
# ax.set_xlim(3.94, 4.89)
# ax.set_ylim(0, 30e3)
# ax.set_xlabel("wavelength $\mathrm{\mu m}$")
# fig.suptitle("$\mathrm{CO}$")
#
# # %%
# fig, ax = plt.subplots(1, 1)
# ax.plot(wl_axis_short, ftavgH2CO.__abs__()[len(freq_full) // 2:])
# ax.set_xlim(3.3, 3.9)
# ax.set_ylim(0, 15e3)
# ax.set_xlabel("wavelength $\mathrm{\mu m}$")
# fig.suptitle("$\mathrm{H_2CO}$")

# %% phase noise analysis
# window = sw.tukey(100)
# pad = (ppifg - len(window)) // 2
# window = np.pad(window, (pad, pad), constant_values=0.)
#
# fig, ax = plt.subplots(1, 1)
# for n in range(150):
#     ft = fft(np.roll(window, n) * avgCO)
#     ax.clear()
#     ax.plot(freq_full[ppifg // 2:][50:], ft.__abs__()[ppifg // 2:][50:])
#     ax.set_title(str(n))
#     # plt.savefig(f"TempImages/{n}.png")
#     plt.pause(.001)

# %% signal level changes
# fig, ax = plt.subplots(1, 1)
# for i, n in enumerate(range(100, int(ppifg), 100)):
#     window = sw.tukey(n)
#     pad = (ppifg - len(window)) // 2
#     window = np.pad(window, (pad, pad), constant_values=0.)
#     ft = fft(dataCO[0] * window)
#
#     ax.clear()
#     ax.plot(freq_full[ppifg // 2:][50:], ft.__abs__()[ppifg // 2:][50:])
#     ax.set_title(str(n))
#     # plt.savefig(f"TempImages/{i}.png")
#     plt.pause(.001)

# %% averaged arrays
dataCO_avg = np.zeros(dataCO.shape, dtype='float64')
dataCO_avg[0] = dataCO[0]
for n in range(1, len(dataCO)):
    dataCO_avg[n] = (dataCO[n] / (n + 1)) + (dataCO_avg[n - 1] * (n / (n + 1)))

print("done")

# %%
# dataCO_avg.tofile(path + "CO_filter_65483x30486_phase_corrected_AVERAGE.bin")

# %%
# dataH2CO_avg = np.zeros(dataH2CO.shape, dtype='float64')
# dataH2CO_avg[0] = dataH2CO[0]
# for n in range(1, len(dataH2CO)):
#     dataH2CO_avg[n] = (dataH2CO[n] / (n + 1)) + (dataH2CO_avg[n - 1] * (n / (n + 1)))
#
# print("done")

# %%
# dataH2CO_avg.tofile(path + "H2CO_filter_65483x30486_phase_corrected_AVERAGE.bin")

# %% noise analysis for H2CO
# ll_ind, ul_ind = 19986, 21775
# ll_ind, ul_ind = 19986 + 750, 19986 + 1000
# amp = ftavgH2CO.__abs__()
# s = 2000000
# spl = spi.UnivariateSpline(freq_full[ll_ind: ul_ind], amp[ll_ind: ul_ind], s=3e5)
# bckgnd = spl(freq_full[ll_ind:ul_ind])

# double checking spline results
# plt.figure()
# plt.plot(amp[ll_ind:ul_ind])
# plt.plot(bckgnd)

# %%
# noise = np.zeros(Nifgs - 1)
# mean = np.mean(amp[ll_ind:ul_ind])
#
# fig, ax = plt.subplots(1, 1)
# h = 0
# delete_files_in_image_directory()
#
# for n, i in enumerate(dataH2CO_avg):
#     amp = fft(i).__abs__()[ll_ind:ul_ind]
#     amp += mean - np.mean(amp)
#
#     noise[n] = np.std(- np.log10(amp / bckgnd))
#
#     if n % 100 == 0:
#         ax.clear()
#         ax.plot(freq_full[ll_ind:ul_ind], amp)
#         ax.plot(freq_full[ll_ind:ul_ind], bckgnd)
#         ax.set_xlabel("DCS frequency (MHz)")
#         ax.set_title(n)
#         plt.savefig(f'TempImages/{h}.png')
#         plt.pause(.001)
#
#         h += 1

# %%
# plt.figure()
# plt.loglog((np.arange(Nifgs - 1) + 1) * (1 / dfr), noise)
# plt.xlabel("time (s)")
# plt.ylabel("absorbance noise for $\mathrm{H_2CO}$")

# %% noise analysis for CO
amp = ftavgCO.__abs__()
ll_ind, ul_ind = 20540, 20540 + 250
spl = spi.UnivariateSpline(freq_full[ll_ind:ul_ind], amp[ll_ind:ul_ind], s=7e4)
bckgnd = spl(freq_full[ll_ind:ul_ind])

# plt.plot(amp[ll_ind:ul_ind])
# plt.plot(bckgnd)

# %%
noise = np.zeros(Nifgs - 1)
mean = np.mean(amp[ll_ind:ul_ind])

# fig, ax = plt.subplots(1, 1)
h = 0
# delete_files_in_image_directory()

for n, i in enumerate(dataCO_avg):
    amp = fft(i).__abs__()[ll_ind:ul_ind]
    amp += mean - np.mean(amp)

    noise[n] = np.std(- np.log10(amp / bckgnd))

    if n % 100 == 0:
        print(n)

        # ax.clear()
        # ax.plot(freq_full[ll_ind:ul_ind], amp)
        # ax.plot(freq_full[ll_ind:ul_ind], bckgnd)
        # ax.set_xlabel("DCS frequency (MHz)")
        # ax.set_title(n)
        # plt.savefig(f'TempImages/{h}.png')
        # plt.pause(.001)
        #
        # h += 1

# %%
clipboard_and_style_sheet.style_sheet()
plt.figure()
plt.loglog((np.arange(Nifgs - 1) + 1) * (1 / dfr), noise)
plt.xlabel("time (s)")
plt.ylabel("absorbance noise for $\mathrm{CO}$")
