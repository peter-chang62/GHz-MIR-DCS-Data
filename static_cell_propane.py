import numpy as np
import matplotlib.pyplot as plt
import phase_correction as pc
import scipy.signal as si
import scipy.constants as sc
import clipboard_and_style_sheet
import digital_phase_correction as dpc

clipboard_and_style_sheet.style_sheet()

# %%
"""Vacuum Background """
# ppifg = 198850
# center = ppifg // 2
# data = np.fromfile(r'D:\ShockTubeData\static cell/no_cell_and_3_5_filter_at200_4998x198850.bin', '<h')
# data, _ = pc.adjust_data_and_reshape(data, ppifg)

# %%
"""Mixture Data """
ppifg = 198850
center = ppifg // 2
data = np.fromfile(r'D:\ShockTubeData\static cell/cell_with_mixture_and_3_5_filter_at200_4998x198850.bin', '<h')
data, _ = pc.adjust_data_and_reshape(data, ppifg)

# %%
ll_freq, ul_freq = 0.425, 0.452
p = dpc.get_pdiff(data, ppifg, ll_freq, ul_freq, Nzoom=400)
h = 0
step = 250
while h < len(data):
    dpc.apply_t0_and_phi0_shift(p[h: h + step], data[h: h + step])

    h += step
    print(len(data) - h)

# %%
avg = np.mean(data, 0)
fft = pc.fft(avg)

# %%
fr = 1010e6 - 10008148.6
freq = np.arange(0, center) * fr
wl = sc.c * 1e6 / freq

# %%
ll, ul = np.argmin(abs(wl - 3.6)), np.argmin(abs(wl - 3.3))

# %%
fig = plt.figure()
plt.plot(wl[ll:ul], fft[center:][ll:ul].__abs__())
plt.xlabel("wavelength ($\mathrm{\mu m}$)")
plt.ylabel("a.u.")

# %%
"""Looking at phase corrected data """

ppifg = 198850
center = ppifg // 2
fr = 1010e6 - 10008148.6
freq = np.arange(0, center) * fr
wl = sc.c * 1e6 / freq
ll, ul = np.argmin(abs(wl - 3.6)), np.argmin(abs(wl - 3.3))

data = np.fromfile(r'D:\ShockTubeData\static cell\PHASE_CORRECTED/cell_with_mixture_and_3_5_filter_at200_4997x198850'
                   r'.bin', '<h')
bckgnd = np.fromfile(r'D:\ShockTubeData\static cell\PHASE_CORRECTED/no_cell_and_3_5_filter_at200_4997x198850.bin',
                     '<h')
data.resize((4997, 198850))
bckgnd.resize((4997, 198850))

# %%
data = np.mean(data, 0)
bckgnd = np.mean(bckgnd, 0)

# %%
fft_data = pc.fft(data)[center:].__abs__()
fft_bckgnd = pc.fft(bckgnd)[center:].__abs__()

# %%
fig = plt.figure()
plt.plot(wl[ll:ul], fft_data[ll:ul], label='data')
plt.plot(wl[ll:ul], fft_bckgnd[ll:ul], label='background')
plt.legend(loc='best')
plt.xlabel("wavelength ($\mathrm{\mu m}$)")
plt.ylabel("a.u.")

# %%
fig = plt.figure()
plt.plot(wl[ll:ul], -np.log(fft_data[ll:ul] / fft_bckgnd[ll:ul]))
plt.xlabel("wavelength ($\mathrm{\mu m}$)")
plt.ylabel("a.u.")
