import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc
from scipy.signal import windows as wd
import mkl_fft

clipboard_and_style_sheet.style_sheet()

"""Phase Correct H2CO Data"""

# %%
# h2co = np.fromfile(r'D:\ShockTubeData\04242022_Data\after_data/long_shock_card2_228408x17507.bin', '<h')
# ppifg = 17507
# center = ppifg // 2
# N_zoom = 24
# h2co, ind = pc.adjust_data_and_reshape(h2co, ppifg)

# %%
# h = 0
# ref = h2co[0]
# AVG = []
# N_stop = len(h2co)
# step = 50
# while h < N_stop:
#     section = h2co[h: h + step]
#     section, shift = pc.Phase_Correct(section, ppifg, N_zoom, False)
#     AVG.append(np.mean(section, 0))
#
#     h2co[h: h + step] = section
#     h += step
#
#     print(N_stop - h)
#
# AVG = np.array(AVG)

# %%
# avg = np.copy(AVG)
# ref = avg[0]
# SGN = np.zeros(len(avg))
# SHIFT = np.zeros(len(avg))
# for n, i in enumerate(avg):
#     arr1 = np.vstack([ref, i])
#     arr2 = np.vstack([ref, -i])
#
#     ifg1, shift1 = pc.Phase_Correct(arr1, ppifg, N_zoom, False)
#     ifg2, shift2 = pc.Phase_Correct(arr2, ppifg, N_zoom, False)
#     ifg1 = ifg1[1]
#     ifg2 = ifg2[1]
#     shift1 = shift1[1]
#     shift2 = shift2[1]
#
#     window = N_zoom // 2
#     diff1 = np.mean(abs(ref[center - window:center + window] - ifg1[center - window: center + window]))
#     diff2 = np.mean(abs(ref[center - window:center + window] - ifg2[center - window: center + window]))
#
#     if diff1 < diff2:
#         avg[n] = ifg1
#         SGN[n] = 1
#         SHIFT[n] = shift1
#     else:
#         avg[n] = ifg2
#         SGN[n] = -1
#         SHIFT[n] = shift2
#
#     print(len(avg) - n)
#
# avg = (avg.T - np.mean(avg, 1)).T

# %% important that these variables are passed from the earlier block
# h = 0
# n = 0
# while h < N_stop:
#     section = h2co[h: h + step]
#     section *= int(SGN[n])
#     section = pc.shift_2d(section, np.repeat(SHIFT[n], len(section)))
#
#     h2co[h: h + step] = section
#
#     h += step
#     n += 1
#     print(N_stop - h)

"""CO Data"""

# %%
# co = np.fromfile(r'D:\ShockTubeData\04242022_Data\after_data/long_shock_card1_228408x17507.bin', '<h')
# ppifg = 17507
# center = ppifg // 2
# N_zoom = 50
# co, ind = pc.adjust_data_and_reshape(co, ppifg)

# %%
# h = 0
# ref = co[0]
# N_stop = len(co)
# step = 1000
# while h < N_stop:
#     section = np.vstack([ref, co[h: h + step]])
#     section, shift = pc.Phase_Correct(section, ppifg, N_zoom, False)
#     section = section[1:]
#
#     co[h: h + step] = section
#     h += step
#
#     print(N_stop - h)

"""Looking at phase corrected results """

# %%
data = np.fromfile(r'D:\ShockTubeData\04242022_Data\after_data\PHASE_CORRECTED_DATA\long_shock/H2CO_228407x17507.bin',
                   '<h')
# data = np.fromfile(r'D:\ShockTubeData\04242022_Data\after_data\PHASE_CORRECTED_DATA\long_shock/CO_228407x17507.bin',
#                    '<h')
data.resize((228407, 17507))
ppifg = 17507
center = ppifg // 2

# %%
h = 0
step = 500
freq = np.fft.fftshift(np.fft.fftfreq(len(data[0]), 1e-9)) * 1e-6
n = 0
plt.ioff()
N_stop = len(data[::step])
# ll, ul = np.argmin(abs(freq - 80)), np.argmin(abs(freq - 320))
ll, ul = np.argmin(abs(freq - 5.2)), np.argmin(abs(freq - 270))
while n < N_stop:
    fig = plt.figure(figsize=np.array([9.94, 6.94]))
    avg = np.mean(data[h:h + step], 0)
    fft = pc.fft(avg)
    fft = fft[ll:ul]
    fft = pc.normalize(fft)
    plt.plot(freq[ll:ul], fft.__abs__())
    plt.title(f'{n * h}')
    plt.savefig(f'temp_fig/{n}.png')
    plt.close()

    h += step
    n += 1
    print(N_stop - n)

plt.ion()
