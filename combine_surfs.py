import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc

clipboard_and_style_sheet.style_sheet()

"""Combining surf 27 and surf 28, DONE!"""

# %%
surf_27 = np.fromfile(r'D:\ShockTubeData\04242022_Data\Surf_27\PHASE_CORRECTED_DATA/H2CO_499x50x17507.bin')
surf_28 = np.fromfile(r'D:\ShockTubeData\04242022_Data\Surf_28\PHASE_CORRECTED_DATA/H2CO_299x50x17507.bin')

surf_27.resize((499, 50, 17507))
surf_28.resize((299, 50, 17507))

# %%
ppifg = 17507
center = ppifg // 2

# %% the way you've stacked them, you have to fix surf_28
# avg_27 = np.mean(surf_27, (0, 1))
# avg_28 = np.mean(surf_28, (0, 1))
# array = np.vstack([avg_27, avg_28])
# array, shifts = pc.Phase_Correct(array, ppifg, 25, True)
#
# # %%
# for n, i in enumerate(surf_28):
#     i = pc.shift_2d(i, np.repeat(shifts[1], 50))
#     surf_28[n] = i
#     print(n)

# %% is there H2CO absorption signal?
clos_27 = surf_27[:, :20, :]
clos_28 = surf_28[:, :20, :]
avg_27 = np.mean(clos_27, (0, 1))
avg_28 = np.mean(clos_28, (0, 1))
avg = np.mean(np.vstack([avg_27, avg_28]), 0)
fft = pc.fft(avg)

# %%
frep = 1010e6 - 9998051.1
freq = np.fft.fftshift(np.fft.fftfreq(ppifg, 1 / frep))

# %%
plt.figure()
plt.plot(freq * 1e-6, fft.__abs__())
