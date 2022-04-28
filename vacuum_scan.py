import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc
from scipy.signal import windows as wd
import mkl_fft

clipboard_and_style_sheet.style_sheet()

# %%
h2co = np.fromfile(r'D:\ShockTubeData\04242022_Data\Vacuum_Background/card2_114204x17507.bin', '<h')
N_ifgs = 114204
ppifg = 17507
center = ppifg // 2
h2co = h2co[center:- (ppifg - center)]
h2co = h2co.reshape((N_ifgs - 1, ppifg))

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

# %%
AVG = np.array(AVG)
AVG2 = pc.Phase_Correct(AVG, ppifg, 25, True)[0]

# %%
h = 0
step = 25
while h < len(AVG2):
    section = AVG2[h: h + step]
    section, _, _ = pc.fix_sign_and_phase_correct(section, ll, ul, ppifg, 25, False, 10)
    AVG2[h: h + step] = section
    h += step
    print(h)

# %%
[plt.plot(i[center - 25:center + 25]) for i in AVG2[-25:]]
