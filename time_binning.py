import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import digital_phase_correction as dpc
import phase_correction as pc

clipboard_and_style_sheet.style_sheet()

ppifg = 17507
center = ppifg // 2

"""If ind ~ 0 then the shock occurred right at the edge of the interferogram (end of one and start of the next 
interferogram). I'll call these two interferograms ifg_a and ifg_b If ind ~ +- center then the shock occurred very 
close to a centerburst. If ind ~ - center then the shock occured close to the centerburst of ifg_b, and if ind ~ + 
center then the shock occurred close to ifg_a """

# %%
ind = np.hstack([
    np.fromfile('ind_minus_indi_surf27.bin'),
    np.fromfile('ind_minus_indi_surf28.bin')
])

# %%
Nbins = 4
step = center // Nbins
IND = []
for n in range(-Nbins, Nbins):
    IND.append(np.logical_and(ind > step * n, ind < step * (n + 1)).nonzero()[0])

"""There are roughly 100 interferograms in each bin if split into 8 bins! """

# %%
# data = np.hstack([
#     np.fromfile(r'D:\ShockTubeData\04242022_Data\Surf_27\PHASE_CORRECTED_DATA/CO_499x70x17507.bin'),
#     np.fromfile(r'D:\ShockTubeData\04242022_Data\Surf_28\PHASE_CORRECTED_DATA/CO_299x70x17507.bin')
# ])
data = np.hstack([
    np.fromfile(r'D:\ShockTubeData\04242022_Data\Surf_27\PHASE_CORRECTED_DATA/H2CO_499x70x17507.bin'),
    np.fromfile(r'D:\ShockTubeData\04242022_Data\Surf_28\PHASE_CORRECTED_DATA/H2CO_299x70x17507.bin')
])

data.resize((499 + 299, 70, 17507))

AVG = np.zeros((len(IND), 70, ppifg))
for n, i in enumerate(IND):
    AVG[n] = np.mean(data[i], axis=0)

# %%
"""from here 19 is the closest pre-shock, and 20 is the closest post shock, with time delay ordered from shortest to 
longest in AVG """

# this is illustrated below
# plt.figure()
# plt.plot(data[np.argmax(ind)].flatten())
# plt.axvline(ppifg * 20, color='r')
# plt.title("longest time delay")
# plt.xlim(ppifg * 20 - 1 * ppifg, ppifg * 20 + 1 * ppifg)
#
# plt.figure()
# plt.plot(data[np.argmin(ind)].flatten())
# plt.axvline(ppifg * 20, color='r')
# plt.title("shortest time delay")
# plt.xlim(ppifg * 20 - 1 * ppifg, ppifg * 20 + 1 * ppifg)

# %%
"""from the above plots, you kind of don't want the shock to blur into the interferogram, or else the "longest" delay 
may actually be the shortest one and vice versa """

# IND = IND[1:-1]
# AVG = AVG[1:-1]

# %%
# should be better now
# plt.figure()
# plt.plot(data[np.argmin(abs(ind - max(ind[IND[-1]])))].flatten())
# plt.axvline(ppifg * 20, color='r')
# plt.title("longest time delay")
# plt.xlim(ppifg * 20 - 1 * ppifg, ppifg * 20 + 1 * ppifg)
#
# plt.figure()
# plt.plot(data[np.argmin(abs(ind - min(ind[IND[0]])))].flatten())
# plt.axvline(ppifg * 20, color='r')
# plt.title("shortest time delay")
# plt.xlim(ppifg * 20 - 1 * ppifg, ppifg * 20 + 1 * ppifg)

# %% just double checking, nice!
fig, ax = plt.subplots(1, 1)
ll, ul = ppifg * 19, ppifg * 21
for i in IND[7]:
    ax.clear()
    ax.plot(data[i].flatten()[ll:ul])
    plt.pause(.01)

# %% save results (make sure to comment out when done!)
with open("h2co_surf27_and_28_6timebins.npy", 'wb') as f:
    np.save(f, AVG)
