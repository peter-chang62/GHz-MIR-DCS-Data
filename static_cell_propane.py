import numpy as np
import matplotlib.pyplot as plt
import phase_correction as pc
import scipy.signal as si
import scipy.constants as sc
import clipboard_and_style_sheet

# %%
clipboard_and_style_sheet.style_sheet()

# %%
# data = np.fromfile(r'D:\ShockTubeData\static cell/no_cell_and_3_5_filter_at200_4998x198850.bin', '<h')

# %%
ppifg = 198850
center = ppifg // 2

# %%
# data, _ = pc.adjust_data_and_reshape(data, ppifg)

# %%
# h = 0
# step = 500
# while h < len(data):
#     sec = data[h: h + step]
#     sec = np.vstack([data[0], sec])
#     sec, shifts = pc.Phase_Correct(sec, ppifg, 125, False)
#
#     data[h: h + step] = sec[1:]
#
#     h += step
#     print(h)

# %%
# avg = np.mean(data, 0)
# fft = pc.fft(avg)

# %%
fr = 1010e6 - 10008148.6
freq = np.arange(0, center) * fr
wl = sc.c * 1e6 / freq

# %%
ll, ul = np.argmin(abs(wl - 3.6)), np.argmin(abs(wl - 3.3))

# %%
# fig = plt.figure()
# plt.plot(wl[ll:ul], fft[center:][ll:ul].__abs__())
# plt.xlabel("wavelength ($\mathrm{\mu m}$)")
# plt.ylabel("a.u.")

# %%
"""Looking at phase corrected data """
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
plt.plot(wl[ll:ul], -np.log10(fft_data[ll:ul] / fft_bckgnd[ll:ul]))
plt.xlabel("wavelength ($\mathrm{\mu m}$)")
plt.ylabel("a.u.")
