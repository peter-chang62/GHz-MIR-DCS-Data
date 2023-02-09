"""April 24, 2022: surfs 27 and 28 and corresponding vacuum background """

import sys

sys.path.append("include/")
import numpy as np
import clipboard_and_style_sheet as cr
import phase_correction as pc
import digital_phase_correction as dpc
import os
import matplotlib.pyplot as plt
import mkl_fft
import scipy.signal as ss

# %% get data paths for surf 27
path_surf27 = r"D:\ShockTubeData\04242022_Data\Surf_27/"
path_co_surf27 = path_surf27 + "card1/"
path_h2co_surf27 = path_surf27 + "card2/"

names_co_surf27 = [i.name for i in os.scandir(path_co_surf27)]
names_h2co_surf27 = [i.name for i in os.scandir(path_h2co_surf27)]

key = lambda s: float(s.split("LoopCount_")[1].split("_Datetime")[0])
names_co_surf27.sort(key=key)
names_h2co_surf27.sort(key=key)

names_co_surf27 = [path_co_surf27 + i for i in names_co_surf27]
names_h2co_surf27 = [path_h2co_surf27 + i for i in names_h2co_surf27]

# %% get data paths for surf 28
path_surf28 = r"D:\ShockTubeData\04242022_Data\Surf_28/"
path_co_surf28 = path_surf28 + "card1/"
path_h2co_surf28 = path_surf28 + "card2/"

names_co_surf28 = [i.name for i in os.scandir(path_co_surf28)]
names_h2co_surf28 = [i.name for i in os.scandir(path_h2co_surf28)]

key = lambda s: float(s.split("LoopCount_")[1].split("_Datetime")[0])
names_co_surf28.sort(key=key)
names_h2co_surf28.sort(key=key)

names_co_surf28 = [path_co_surf28 + i for i in names_co_surf28]
names_h2co_surf28 = [path_h2co_surf28 + i for i in names_h2co_surf28]

# %% all data paths
names_co = names_co_surf27 + names_co_surf28
names_h2co = names_h2co_surf27 + names_h2co_surf28

# %% save paths
save_path = r"D:\ShockTubeData\04242022_Data\Surf_27\PHASE_CORRECTED_DATA_SURFS_27_AND_28/"
save_path_co = save_path + "co_card1/"
save_path_h2co = save_path + "h2co_card2/"

# %% set ppifg
ppifg = 17507
center = ppifg // 2

f_MHz = np.fft.fftshift(np.fft.fftfreq(ppifg, d=1e-9)) * 1e-6
freq_nounit = np.fft.fftshift(np.fft.fftfreq(ppifg))
ll_freq_co, ul_freq_co = 0.1548, 0.2443
ll_freq_h2co, ul_freq_h2co = 0.0791, 0.1686

# %% analysis
# CO = []
# H2CO = []
# IND_SHOCK = []

# N_shocks = len(names_co)
# # N_shocks = 2
# for i in range(N_shocks):
#     co = np.fromfile(names_co[i], '<h')[:-64]
#     co = co / co.max()

#     # load data and reshape
#     ind = np.argmax(co[:ppifg])
#     co = co[ind:]
#     N1 = len(co) // ppifg
#     co = co[:N1 * ppifg]
#     co = co[center:-center]
#     N2 = len(co) // ppifg
#     co = co[:N2 * ppifg]
#     co = np.reshape(co, (N2, ppifg))

#     if i == 0:
#         co_global_reference = co[0].copy()

#     # locate shock
#     ft = pc.fft(co, axis=1)
#     ft_filtered = ft.copy()
#     for n, m in enumerate(ft_filtered):
#         ft_filtered[n] = np.where(abs(f_MHz) < 5, m, 0)
#     bckgnd = pc.ifft(ft_filtered, axis=1).real.flatten()
#     bckgnd_flipped = bckgnd * -1  # get the reflected shock
#     ind_shock = np.argmax(bckgnd_flipped)

#     # phase correction
#     if i > 0:
#         co_pdiff = np.vstack([co_global_reference, co])
#         pdiff = dpc.get_pdiff(co_pdiff, ll_freq_co, ul_freq_co, 200)
#         pdiff = pdiff[1:]
#     else:
#         pdiff = dpc.get_pdiff(co, ll_freq_co, ul_freq_co, 200)

#     dpc.apply_t0_shift(pdiff, freq_nounit, ft)
#     td = pc.ifft(ft, axis=1).real
#     hbt = ss.hilbert(td)
#     dpc.apply_phi0_shift(pdiff, hbt)
#     hbt = hbt.real

#     # save data
#     IND_SHOCK.append(ind_shock)
#     # CO.append(hbt)
#     np.save(save_path_co + f"{i}.npy", hbt)

#     # repeat for h2co but use co shock location and data truncation
#     h2co = np.fromfile(names_h2co[i], '<h')[:-64]
#     h2co = h2co / h2co.max()

#     h2co = h2co[ind:]
#     h2co = h2co[:N1 * ppifg]
#     h2co = h2co[center:-center]
#     h2co = h2co[:N2 * ppifg]
#     h2co = np.reshape(h2co, (N2, ppifg))

#     if i == 0:
#         h2co_global_reference = h2co[0].copy()

#     ll_freq_h2co, ul_freq_h2co = 0.0791, 0.1686
#     if i > 0:
#         h2co_pdiff = np.vstack([h2co_global_reference, h2co])
#         pdiff = dpc.get_pdiff(h2co_pdiff, ll_freq_h2co, ul_freq_h2co, 200)
#         pdiff = pdiff[1:]
#     else:
#         pdiff = dpc.get_pdiff(h2co, ll_freq_h2co, ul_freq_h2co, 200)
#     dpc.apply_t0_and_phi0_shift(pdiff, h2co)

#     # H2CO.append(h2co)
#     np.save(save_path_h2co + f"{i}.npy", h2co)

#     print(N_shocks - i - 1)

# np.save(save_path + "IND_SHOCKS.npy", IND_SHOCK)

# %% average shocks together
# path_corrected = r"D:\ShockTubeData\04242022_Data\Surf_27\PHASE_CORRECTED_DATA_SURFS_27_AND_28/"
# path_co_corrected = path_corrected + "co_card1/"
# path_h2co_corrected = path_corrected + "h2co_card2/"
# names_co_corrected = [i.name for i in os.scandir(path_co_corrected)]
# names_h2co_corrected = [i.name for i in os.scandir(path_h2co_corrected)]

# key = lambda s: float(s.split(".npy")[0])
# names_co_corrected.sort(key=key)
# names_h2co_corrected.sort(key=key)
# names_co_corrected = [path_co_corrected + i for i in names_co_corrected]
# names_h2co_corrected = [path_h2co_corrected + i for i in names_h2co_corrected]

# ind_shocks = np.load(path_corrected + "IND_SHOCKS.npy")
# ppifg_shock_ref = ind_shocks.min() // ppifg

# for n, i in enumerate(ind_shocks):
#     ppifg_shock = i // ppifg
#     ppifg_diff = ppifg_shock_ref - ppifg_shock

#     if n == 0:
#         co = np.load(names_co_corrected[n])
#         if ppifg_diff < 0:
#             co = co[abs(ppifg_diff):]
#         elif ppifg_diff > 0:
#             co = co[:ppifg_diff]

#         # ___________________________________________________________________
#         h2co = np.load(names_h2co_corrected[n])
#         if ppifg_diff < 0:
#             h2co = h2co[abs(ppifg_diff):]
#         elif ppifg_diff > 0:
#             h2co = h2co[:ppifg_diff]

#     else:
#         x = np.load(names_co_corrected[n])
#         if ppifg_diff < 0:
#             x = x[abs(ppifg_diff):]
#         elif ppifg_diff > 0:
#             assert ppifg_diff < 0, "this shouldn't happen"
#             x = x[:ppifg_diff]

#         if len(co) > len(x):
#             co = co[:len(x)] + x
#         elif len(co) < len(x):
#             co += x[:len(co)]
#         else:
#             co += x

#         # ___________________________________________________________________
#         y = np.load(names_h2co_corrected[n])
#         if ppifg_diff < 0:
#             y = y[abs(ppifg_diff):]
#         elif ppifg_diff > 0:
#             assert ppifg_diff < 0, "this shouldn't happen"
#             y = y[:ppifg_diff]

#         if len(h2co) > len(y):
#             h2co = h2co[:len(y)] + y
#         elif len(h2co) < len(y):
#             h2co += y[:len(h2co)]
#         else:
#             h2co += y

#     print(len(names_co_corrected) - n - 1)

# np.save(path_corrected + "co_averaged_surfs_27_and_28.npy", co)
# np.save(path_corrected + "h2co_averaged_surfs_27_and_28.npy", h2co)

# %% vacuum background co
# path_vacuum_bckgnd = r"D:\ShockTubeData\04242022_Data\Vacuum_Background/"
# co_vacuum_bckgnd = np.fromfile(path_vacuum_bckgnd + "card1_114204x17507.bin", '<h')
# co_vacuum_bckgnd = co_vacuum_bckgnd / co_vacuum_bckgnd.max()

# co_vacuum_bckgnd = co_vacuum_bckgnd[center:-center]
# co_vacuum_bckgnd = co_vacuum_bckgnd[:len(co_vacuum_bckgnd) // ppifg * ppifg]
# co_vacuum_bckgnd = np.reshape(co_vacuum_bckgnd, (114204 - 1, ppifg))

# pdiff_co = dpc.get_pdiff(co_vacuum_bckgnd, ll_freq_co, ul_freq_co, 200)
# h = 0
# step = 250
# while h < len(co_vacuum_bckgnd):
#     dpc.apply_t0_and_phi0_shift(pdiff_co[h: h + step], co_vacuum_bckgnd[h: h + step])
#     h += step
#     print(len(co_vacuum_bckgnd) - h)

# avg_co_vacuum_bckgnd = np.sum(co_vacuum_bckgnd, axis=0)
# np.save(path_vacuum_bckgnd + "PHASE_CORRECTED/co_vacuum_bckgnd_avg.npy", avg_co_vacuum_bckgnd)

# %% vacuum background h2co
# path_vacuum_bckgnd = r"D:\ShockTubeData\04242022_Data\Vacuum_Background/"
# h2co_vacuum_bckgnd = np.fromfile(path_vacuum_bckgnd + "card2_114204x17507.bin", '<h')
# h2co_vacuum_bckgnd = h2co_vacuum_bckgnd / h2co_vacuum_bckgnd.max()

# h2co_vacuum_bckgnd = h2co_vacuum_bckgnd[center:-center]
# h2co_vacuum_bckgnd = h2co_vacuum_bckgnd[:len(h2co_vacuum_bckgnd) // ppifg * ppifg]
# h2co_vacuum_bckgnd = np.reshape(h2co_vacuum_bckgnd, (114204 - 1, ppifg))

# pdiff_h2co = dpc.get_pdiff(h2co_vacuum_bckgnd, ll_freq_h2co, ul_freq_h2co, 200)
# h = 0
# step = 250
# while h < len(h2co_vacuum_bckgnd):
#     dpc.apply_t0_and_phi0_shift(pdiff_h2co[h: h + step], h2co_vacuum_bckgnd[h: h + step])
#     h += step
#     print(len(h2co_vacuum_bckgnd) - h)

# avg_h2co_vacuum_bckgnd = np.sum(h2co_vacuum_bckgnd, axis=0)
# np.save(path_vacuum_bckgnd + "PHASE_CORRECTED/h2co_vacuum_bckgnd_avg.npy", avg_h2co_vacuum_bckgnd)
