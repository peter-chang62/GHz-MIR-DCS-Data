import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import digital_phase_correction as dpc
import phase_correction as pc

clipboard_and_style_sheet.style_sheet()

ppifg = 17507
center = ppifg // 2

# %%____________________________________________________________________________________________________________________
path = r'D:\ShockTubeData\04242022_Data/'
ind = np.hstack([
    np.load(path + 'Surf_13/PHASE_CORRECTED_DATA/ind_minus_indi.npy'),
    np.load(path + 'Surf_14/PHASE_CORRECTED_DATA/ind_minus_indi.npy'),
    np.load(path + 'Surf_15/PHASE_CORRECTED_DATA/ind_minus_indi.npy'),
    # np.load(path + 'Surf_16/PHASE_CORRECTED_DATA/ind_minus_indi.npy'),
    # np.load(path + 'Surf_17/PHASE_CORRECTED_DATA/ind_minus_indi.npy'),
    # np.load(path + 'Surf_22/PHASE_CORRECTED_DATA/ind_minus_indi.npy'),
])

# %%____________________________________________________________________________________________________________________
Nbins = 4
step = center // Nbins
IND = []
for n in range(-Nbins, Nbins):
    IND.append(np.logical_and(ind > step * n, ind < step * (n + 1)).nonzero()[0])

# %%____________________________________________________________________________________________________________________
# data = np.vstack([
#     # np.load(path + 'Surf_13/PHASE_CORRECTED_DATA/CO_499x70x17507.npy'),
#     # np.load(path + 'Surf_14/PHASE_CORRECTED_DATA/CO_499x70x17507.npy'),
#     # np.load(path + 'Surf_15/PHASE_CORRECTED_DATA/CO_499x70x17507.npy'),
#     np.load(path + 'Surf_16/PHASE_CORRECTED_DATA/CO_499x70x17507.npy'),
#     np.load(path + 'Surf_17/PHASE_CORRECTED_DATA/CO_210x70x17507.npy'),
#     np.load(path + 'Surf_22/PHASE_CORRECTED_DATA/CO_299x70x17507.npy'),
# ])

# %%____________________________________________________________________________________________________________________
data = np.vstack([
    np.load(path + 'Surf_13/PHASE_CORRECTED_DATA/H2CO_499x70x17507.npy'),
    np.load(path + 'Surf_14/PHASE_CORRECTED_DATA/H2CO_499x70x17507.npy'),
    np.load(path + 'Surf_15/PHASE_CORRECTED_DATA/H2CO_499x70x17507.npy'),
    # np.load(path + 'Surf_16/PHASE_CORRECTED_DATA/H2CO_499x70x17507.npy'),
    # np.load(path + 'Surf_17/PHASE_CORRECTED_DATA/H2CO_210x70x17507.npy'),
    # np.load(path + 'Surf_22/PHASE_CORRECTED_DATA/H2CO_299x70x17507.npy'),
])

# %%____________________________________________________________________________________________________________________
AVG = np.zeros((len(IND), 70, ppifg))
for n, i in enumerate(IND):
    AVG[n] = np.mean(data[i], axis=0)

# %%____________________________________________________________________________________________________________________
# just double-checking, nice!
# fig, ax = plt.subplots(1, 1)
# ll, ul = ppifg * 19, ppifg * 21
# for n, i in enumerate(IND[1]):
#     ax.clear()
#     ax.plot(data[i].flatten()[ll:ul])
#     ax.set_title(n)
#     plt.pause(.01)

# %%____________________________________________________________________________________________________________________
# save results
with open("temp/h2co_surfs_13_14_15_8timebins.npy", 'wb') as f:
    np.save(f, AVG)

# %%____________________________________________________________________________________________________________________

"""Didn't have enough memory to process all at once, so saved to separate files. Now averaging them!"""

one = np.load('temp/h2co_surfs_13_14_15_8timebins.npy')
two = np.load('temp/h2co_surfs_16_17_22_8timebins.npy')
avg_tgthr = (one + two) / 2

with open("temp/h2co_8timebins.npy", 'wb') as f:
    np.save(f, avg_tgthr)


def plot(n):
    x = np.hstack([avg_tgthr[n, 19], avg_tgthr[n, 20]])
    plt.plot(x)
