"""June 30, 2022: battalions (28, 31) """

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

key = lambda s: float(s.split("LoopCount_")[1].split("_Datetime")[0])
ppifg = 17506
center = ppifg // 2

f_MHz = np.fft.fftshift(np.fft.fftfreq(ppifg, d=1e-9)) * 1e-6
freq_nounit = np.fft.fftshift(np.fft.fftfreq(ppifg))

ll_freq_co, ul_freq_co = 0.1549, 0.2211
ll_freq_h2co, ul_freq_h2co = 0.0791, 0.1686

# %% load path names
path_batt_28 = r"D:\DATA_MATT_PATRICK_TRIP_2\06-30-2022\Battalion_28/"
path_batt_28_co = path_batt_28 + "card1/"
path_batt_28_h2co = path_batt_28 + "card2/"

path_batt_31 = r"D:\DATA_MATT_PATRICK_TRIP_2\06-30-2022\Battalion_31/"
path_batt_31_co = path_batt_31 + "card1/"
path_batt_31_h2co = path_batt_31 + "card2/"

# load co (card 1) path names
names_co_batt_28 = [i.name for i in os.scandir(path_batt_28_co)]
names_co_batt_31 = [i.name for i in os.scandir(path_batt_31_co)]
names_co_batt_28.sort(key=key)
names_co_batt_31.sort(key=key)

names_co_batt_28 = [path_batt_28_co + i for i in names_co_batt_28]
names_co_batt_31 = [path_batt_31_co + i for i in names_co_batt_31]

# load co (card 2) path names
names_h2co_batt_28 = [i.name for i in os.scandir(path_batt_28_h2co)]
names_h2co_batt_31 = [i.name for i in os.scandir(path_batt_31_h2co)]
names_h2co_batt_28.sort(key=key)
names_h2co_batt_31.sort(key=key)

names_h2co_batt_28 = [path_batt_28_h2co + i for i in names_h2co_batt_28]
names_h2co_batt_31 = [path_batt_31_h2co + i for i in names_h2co_batt_31]

names_co = names_co_batt_28 + names_co_batt_31
names_h2co = names_h2co_batt_28 + names_h2co_batt_31

# %% save paths
save_path = (
    r"D:\DATA_MATT_PATRICK_TRIP_2\06-30-2022\Battalion_28"
    r"\PHASE_CORRECTED_BATT_28_AND_31/"
)
save_path_co = save_path + "co_card1/"
save_path_h2co = save_path + "h2co_card2/"

# %% analysis
"""
different from the previous scripts, to get the incident shock now, I need to
find the incident one first and then tell it to search for the shock after
that. I think this is because the shocks here were a little weaker than the
shocks analyzed in the previous scripts
"""

# IND_SHOCK = []

# N_shocks = len(names_co)
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
#     ind_incident = np.argmax(bckgnd)
#     bckgnd_flipped = bckgnd * -1  # get the reflected shock
#     ind_shock = np.argmax(bckgnd_flipped[ind_incident:]) + ind_incident

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

#     if i > 0:
#         h2co_pdiff = np.vstack([h2co_global_reference, h2co])
#         pdiff = dpc.get_pdiff(h2co_pdiff, ll_freq_h2co, ul_freq_h2co, 200)
#         pdiff = pdiff[1:]
#     else:
#         pdiff = dpc.get_pdiff(h2co, ll_freq_h2co, ul_freq_h2co, 200)
#     dpc.apply_t0_and_phi0_shift(pdiff, h2co)

#     np.save(save_path_h2co + f"{i}.npy", h2co)

#     print(N_shocks - i - 1)

# np.save(save_path + "IND_SHOCKS.npy", IND_SHOCK)

# %% average shocks together
# path_corrected = r"D:\DATA_MATT_PATRICK_TRIP_2\06-30-2022\Battalion_28" \
#                  r"\PHASE_CORRECTED_BATT_28_AND_31/"
# path_co_corrected = path_corrected + "co_card1/"
# path_h2co_corrected = path_corrected + "h2co_card2/"
# names_co_corrected = [i.name for i in os.scandir(path_co_corrected)]
# names_h2co_corrected = [i.name for i in os.scandir(path_h2co_corrected)]
#
# key = lambda s: float(s.split(".npy")[0])
# names_co_corrected.sort(key=key)
# names_h2co_corrected.sort(key=key)
# names_co_corrected = [path_co_corrected + i for i in names_co_corrected]
# names_h2co_corrected = [path_h2co_corrected + i for i in names_h2co_corrected]
#
# ind_shocks = np.load(path_corrected + "IND_SHOCKS.npy")
# ppifg_shock_ref = ind_shocks.min() // ppifg
#
# for n, i in enumerate(ind_shocks):
#     ppifg_shock = i // ppifg
#     ppifg_diff = ppifg_shock_ref - ppifg_shock
#
#     if n == 0:
#         co = np.load(names_co_corrected[n])
#         if ppifg_diff < 0:
#             co = co[abs(ppifg_diff):]
#         elif ppifg_diff > 0:
#             assert ppifg_diff < 0, "this shouldn't happen"
#             co = co[:ppifg_diff]
#
#         # ___________________________________________________________________
#         h2co = np.load(names_h2co_corrected[n])
#         if ppifg_diff < 0:
#             h2co = h2co[abs(ppifg_diff):]
#         elif ppifg_diff > 0:
#             assert ppifg_diff < 0, "this shouldn't happen"
#             h2co = h2co[:ppifg_diff]
#
#     else:
#         x = np.load(names_co_corrected[n])
#         if ppifg_diff < 0:
#             x = x[abs(ppifg_diff):]
#         elif ppifg_diff > 0:
#             assert ppifg_diff < 0, "this shouldn't happen"
#             x = x[:ppifg_diff]
#
#         if len(co) > len(x):
#             co = co[:len(x)] + x
#         elif len(co) < len(x):
#             co += x[:len(co)]
#         else:
#             co += x
#
#         # ___________________________________________________________________
#         y = np.load(names_h2co_corrected[n])
#         if ppifg_diff < 0:
#             y = y[abs(ppifg_diff):]
#         elif ppifg_diff > 0:
#             assert ppifg_diff < 0, "this shouldn't happen"
#             y = y[:ppifg_diff]
#
#         if len(h2co) > len(y):
#             h2co = h2co[:len(y)] + y
#         elif len(h2co) < len(y):
#             h2co += y[:len(h2co)]
#         else:
#             h2co += y
#
#     print(len(names_co_corrected) - n - 1)
#
# np.save(path_corrected + "co_averaged_batt_28_and_31.npy", co)
# np.save(path_corrected + "h2co_averaged_batt_28_and_31.npy", h2co)

# %% vacuum background co
# path_vacuum_bckgnd = r"D:\DATA_MATT_PATRICK_TRIP_2\06-30-2022" \
#                      r"\Vacuum_Background_end_of_experiment/"
# path_bckgnd_co = path_vacuum_bckgnd + "4.5um_filter_114204x17506.bin"
#
# co_bckgnd = np.fromfile(path_bckgnd_co, '<h')
# co_bckgnd = co_bckgnd / co_bckgnd.max()
#
# co_bckgnd = co_bckgnd[center:-center]
# co_bckgnd = co_bckgnd[:len(co_bckgnd) // ppifg * ppifg]
# co_bckgnd = np.reshape(co_bckgnd, (114204 - 1, ppifg))
#
# pdiff_co = dpc.get_pdiff(co_bckgnd, ll_freq_co, ul_freq_co, 200)
# h = 0
# step = 250
# while h < len(co_bckgnd):
#     dpc.apply_t0_and_phi0_shift(pdiff_co[h: h + step], co_bckgnd[h: h + step])
#     h += step
#     print(len(co_bckgnd) - h)
#
# avg_co_bckgnd = np.sum(co_bckgnd, axis=0)
# np.save(path_vacuum_bckgnd + "PHASE_CORRECTED/co_vacuum_bckgnd_avg.npy",
#         avg_co_bckgnd)

# %% vacuum background h2co
path_vacuum_bckgnd = (
    r"D:\DATA_MATT_PATRICK_TRIP_2\06-30-2022"
    r"\Vacuum_Background_end_of_experiment/"
)
path_bckgnd_h2co = path_vacuum_bckgnd + "3.5um_filter_114204x17506.bin"

h2co_bckgnd = np.fromfile(path_bckgnd_h2co, "<h")
h2co_bckgnd = h2co_bckgnd / h2co_bckgnd.max()

h2co_bckgnd = h2co_bckgnd[center:-center]
h2co_bckgnd = h2co_bckgnd[: len(h2co_bckgnd) // ppifg * ppifg]
h2co_bckgnd = np.reshape(h2co_bckgnd, (114204 - 1, ppifg))

pdiff_h2co = dpc.get_pdiff(h2co_bckgnd, ll_freq_h2co, ul_freq_h2co, 200)
h = 0
step = 250
while h < len(h2co_bckgnd):
    dpc.apply_t0_and_phi0_shift(
        pdiff_h2co[h: h + step], h2co_bckgnd[h: h + step]
    )
    h += step
    print(len(h2co_bckgnd) - h)

avg_h2co_bckgnd = np.sum(h2co_bckgnd, axis=0)
np.save(
    path_vacuum_bckgnd + "PHASE_CORRECTED/h2co_vacuum_bckgnd_avg.npy",
    avg_h2co_bckgnd,
)
