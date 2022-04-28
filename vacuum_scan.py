import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc
from scipy.signal import windows as wd
import mkl_fft

clipboard_and_style_sheet.style_sheet()

# %%
h2co = np.fromfile(r'D:\ShockTubeData\04242022_Data\Vacuum_Background/card2_114204x17507.bin', '<h')
co = np.fromfile(r'D:\ShockTubeData\04242022_Data\Vacuum_Background/card1_114204x17507.bin', '<h')
N_ifgs = 114204
ppifg = 17507
center = ppifg // 2

# %%
h2co, ind = pc.adjust_data_and_reshape(h2co, ppifg)

# %%
ll, ul = 1120 + ppifg // 2, 3600 + ppifg // 2

# %%
h = 0
done = False
ref = h2co[0]
AVG = []
# N_stop = int(1e4)
N_stop = N_ifgs
step = 50
while h < N_stop:
    section = h2co[h: h + step]
    section, shift = pc.Phase_Correct(section, ppifg, 25, False)
    # section, shift, sgn = pc.fix_sign_and_phase_correct(section, ll, ul, ppifg, 25, False, 10)
    AVG.append(np.mean(section, 0))

    h2co[h: h + step] = section
    h += step

    if h % 500 == 0:
        print(N_stop - h)

AVG = np.array(AVG)

# %%
avg = np.copy(AVG)
ref = avg[0]
SGN = np.zeros(len(avg))
for n, i in enumerate(avg):
    arr1 = np.vstack([ref, i])
    arr2 = np.vstack([ref, -i])

    ifg1 = pc.Phase_Correct(arr1, ppifg, 25, False)[0][1]
    ifg2 = pc.Phase_Correct(arr2, ppifg, 25, False)[0][1]

    diff1 = np.mean(abs(ref[center - 50:center + 50] - ifg1[center - 50: center + 50]))
    diff2 = np.mean(abs(ref[center - 50:center + 50] - ifg2[center - 50: center + 50]))

    if diff1 < diff2:
        avg[n] = ifg1
        SGN[n] = 1
    else:
        avg[n] = ifg2
        SGN[n] = -1

    if n % 100 == 0:
        print(len(avg) - n)

# %%
avg = (avg.T - np.mean(avg, 1)).T

# %% looks good I think :)
plt.figure()
[plt.plot(i[center - 50:center + 50]) for i in avg]

# %%
final = np.mean(avg, 0)
fft = pc.fft(final)

# %%
ref = np.fromfile('h2co_vacuum_background.bin')
data = np.fromfile('surf_27_and_28_post_shocks_20ifgs_798_shocks.bin')
fft_ref = pc.fft(ref)
fft_data = pc.fft(data)
freq = np.fft.fftshift(np.fft.fftfreq(len(fft_ref), 1e-9)) * 1e-6

plt.figure()
plt.plot(freq, fft_ref.__abs__(), label='background')
plt.plot(freq, fft_data.__abs__(), label='data')
plt.legend(loc='best')
