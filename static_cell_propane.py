import numpy as np
import matplotlib.pyplot as plt
import phase_correction as pc
import scipy.signal as si
import scipy.constants as sc
import clipboard_and_style_sheet
import digital_phase_correction as dpc

clipboard_and_style_sheet.style_sheet()

# %% ___________________________________________________________________________________________________________________
"""Vacuum Background """
# ppifg = 198850
# center = ppifg // 2
# data = np.fromfile(r'D:\ShockTubeData\static cell/no_cell_and_4_5_filter_at200_4998x198850.bin', '<h')
# data, _ = pc.adjust_data_and_reshape(data, ppifg)

# ppifg = 199346
# center = ppifg // 2
# data = np.fromfile(r'D:\ShockTubeData\static cell/with_cell_vacuum_bckgnd_10030x199346.bin', '<h')
# data, _ = pc.adjust_data_and_reshape(data, ppifg)

# %% ___________________________________________________________________________________________________________________
"""Mixture Data """
ppifg = 198850
center = ppifg // 2
data = np.fromfile(r'D:\ShockTubeData\static cell/cell_with_mixture_and_4_5_filter_at200_4998x198850.bin', '<h')
data, _ = pc.adjust_data_and_reshape(data, ppifg)

# ppifg = 198850
# center = ppifg // 2
# data = np.fromfile(r'D:\ShockTubeData\static cell/cell_with_mixture_10030x198850.bin', '<h')
# data, _ = pc.adjust_data_and_reshape(data, ppifg)

# %% ___________________________________________________________________________________________________________________
# ll_freq, ul_freq = 0.425, 0.452  # 3.5 um
# ll_freq, ul_freq = 0.325, 0.345  # 4.5 um
ll_freq, ul_freq = 0.325, 0.487  # no filter
p = dpc.get_pdiff(data, ll_freq, ul_freq, Nzoom=400)
h = 0
step = 250
while h < len(data):
    dpc.apply_t0_and_phi0_shift(p[h: h + step], data[h: h + step])

    h += step
    print(len(data) - h)

# %%
avg = np.mean(data, 0)
fft = pc.fft(avg)

# %% ___________________________________________________________________________________________________________________
fr = 1010e6 - 10008148.6
freq = np.arange(0, center) * fr
wl = sc.c * 1e6 / freq

# %% ___________________________________________________________________________________________________________________
# ll, ul = np.argmin(abs(wl - 3.6)), np.argmin(abs(wl - 3.3))
# ll, ul = np.argmin(abs(wl - 4.8)), np.argmin(abs(wl - 4.3))
ll, ul = np.argmin(abs(wl - 5.0)), np.argmin(abs(wl - 3.0))

# %% ___________________________________________________________________________________________________________________
fig = plt.figure()
plt.plot(wl[ll:ul], fft[center:][ll:ul].__abs__())
plt.xlabel("wavelength ($\mathrm{\mu m}$)")
plt.ylabel("a.u.")

# # %% ___________________________________________________________________________________________________________________
"""Looking at phase corrected data """


def get_freq_wl_ll_ul(ppifg):
    center = ppifg // 2
    fr = 1010e6 - 10008148.6
    freq = np.arange(0, center) * fr
    wl = sc.c * 1e6 / freq
    ll, ul = np.argmin(abs(wl - 5)), np.argmin(abs(wl - 3))
    return freq, wl, ll, ul


# %% ___________________________________________________________________________________________________________________
data_ = np.fromfile(r'D:\ShockTubedata\static cell\PHASE_CORRECTED/cell_with_mixture_averaged.bin')
bckgnd = np.fromfile(r'D:\ShockTubedata\static cell\PHASE_CORRECTED/with_cell_vacuum_bckgnd_averaged.bin')

freq1, wl1, ll1, ul1 = get_freq_wl_ll_ul(len(data_))
freq2, wl2, ll2, ul2 = get_freq_wl_ll_ul(len(bckgnd))

# %% ___________________________________________________________________________________________________________________
fft_data_ = pc.fft(data_)[len(data_) // 2:].__abs__()
fft_bckgnd = pc.fft(bckgnd)[len(bckgnd) // 2:].__abs__()

# %% ___________________________________________________________________________________________________________________
fig = plt.figure()
plt.plot(wl1[ll1:ul1], fft_data_[ll1:ul1], label='data_')
plt.plot(wl2[ll2:ul2], fft_bckgnd[ll2:ul2], label='background')
plt.legend(loc='best')
plt.xlabel("wavelength ($\mathrm{\mu m}$)")
plt.ylabel("a.u.")

# %% ___________________________________________________________________________________________________________________
fig = plt.figure()
plt.plot(wl[ll:ul], -np.log(fft_data_[ll:ul] / fft_bckgnd[ll:ul]))
plt.xlabel("wavelength ($\mathrm{\mu m}$)")
plt.ylabel("a.u.")
