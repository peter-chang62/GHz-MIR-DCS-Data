"""June 30, 2022: battalions (4, 5) """

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

# _________________________________________________________ load path names ____________________________________________
path_batt_4 = r"D:\DATA_MATT_PATRICK_TRIP_2\06-30-2022\Battalion_4/"
path_batt_4_co = path_batt_4 + "card1/"
path_batt_4_h2co = path_batt_4 + "card2/"

path_batt_5 = r"D:\DATA_MATT_PATRICK_TRIP_2\06-30-2022\Battalion_5/"
path_batt_5_co = path_batt_5 + "card1/"
path_batt_5_h2co = path_batt_5 + "card2/"

# load co (card 1) path names
names_co_batt_4 = [i.name for i in os.scandir(path_batt_4_co)]
names_co_batt_5 = [i.name for i in os.scandir(path_batt_5_co)]
names_co_batt_4.sort(key=key)
names_co_batt_5.sort(key=key)

names_co_batt_4 = [path_batt_4_co + i for i in names_co_batt_4]
names_co_batt_5 = [path_batt_5_co + i for i in names_co_batt_5]

# load co (card 2) path names
names_h2co_batt_4 = [i.name for i in os.scandir(path_batt_4_h2co)]
names_h2co_batt_5 = [i.name for i in os.scandir(path_batt_5_h2co)]
names_h2co_batt_4.sort(key=key)
names_h2co_batt_5.sort(key=key)

names_h2co_batt_4 = [path_batt_4_h2co + i for i in names_h2co_batt_4]
names_h2co_batt_5 = [path_batt_5_h2co + i for i in names_h2co_batt_5]

names_co = names_co_batt_4 + names_co_batt_5
names_h2co = names_h2co_batt_4 + names_h2co_batt_5

# ___________________________________________________ save paths _______________________________________________________
save_path = r"D:\DATA_MATT_PATRICK_TRIP_2\06-30-2022\Battalion_4\PHASE_CORRECTED_BATT_4_AND_5/"
save_path_co = save_path + "co_card1/"
save_path_h2co = save_path + "h2co_card2/"

# _______________________________________ analysis _____________________________________________________________________
# IND_SHOCK = []
#
# N_shocks = len(names_co)
# for i in range(N_shocks):
#     co = np.fromfile(names_co[i], '<h')[:-64]
#     co = co / co.max()
#
#     # load data and reshape
#     ind = np.argmax(co[:ppifg])
#     co = co[ind:]
#     N1 = len(co) // ppifg
#     co = co[:N1 * ppifg]
#     co = co[center:-center]
#     N2 = len(co) // ppifg
#     co = co[:N2 * ppifg]
#     co = np.reshape(co, (N2, ppifg))
#
#     if i == 0:
#         co_global_reference = co[0].copy()
#
#     # locate shock
#     ft = pc.fft(co, axis=1)
#     ft_filtered = ft.copy()
#     for n, m in enumerate(ft_filtered):
#         ft_filtered[n] = np.where(abs(f_MHz) < 5, m, 0)
#     bckgnd = pc.ifft(ft_filtered, axis=1).real.flatten()
#     bckgnd_flipped = bckgnd * -1  # get the reflected shock
#     ind_shock = np.argmax(bckgnd_flipped)
#
#     # phase correction
#     if i > 0:
#         co_pdiff = np.vstack([co_global_reference, co])
#         pdiff = dpc.get_pdiff(co_pdiff, ll_freq_co, ul_freq_co, 200)
#         pdiff = pdiff[1:]
#     else:
#         pdiff = dpc.get_pdiff(co, ll_freq_co, ul_freq_co, 200)
#
#     dpc.apply_t0_shift(pdiff, freq_nounit, ft)
#     td = pc.ifft(ft, axis=1).real
#     hbt = ss.hilbert(td)
#     dpc.apply_phi0_shift(pdiff, hbt)
#     hbt = hbt.real
#
#     # save data
#     IND_SHOCK.append(ind_shock)
#     np.save(save_path_co + f"{i}.npy", hbt)
#
#     # ____________________________ repeat for h2co but use co shock location and data truncation _____________________
#     h2co = np.fromfile(names_h2co[i], '<h')[:-64]
#     h2co = h2co / h2co.max()
#
#     h2co = h2co[ind:]
#     h2co = h2co[:N1 * ppifg]
#     h2co = h2co[center:-center]
#     h2co = h2co[:N2 * ppifg]
#     h2co = np.reshape(h2co, (N2, ppifg))
#
#     if i == 0:
#         h2co_global_reference = h2co[0].copy()
#
#     if i > 0:
#         h2co_pdiff = np.vstack([h2co_global_reference, h2co])
#         pdiff = dpc.get_pdiff(h2co_pdiff, ll_freq_h2co, ul_freq_h2co, 200)
#         pdiff = pdiff[1:]
#     else:
#         pdiff = dpc.get_pdiff(h2co, ll_freq_h2co, ul_freq_h2co, 200)
#     dpc.apply_t0_and_phi0_shift(pdiff, h2co)
#
#     np.save(save_path_h2co + f"{i}.npy", h2co)
#
#     print(N_shocks - i - 1)
#
# np.save(save_path + "IND_SHOCKS.npy", IND_SHOCK)

# _______________________________________ average shocks together ______________________________________________________
path_corrected = r"D:\DATA_MATT_PATRICK_TRIP_2\06-30-2022\Battalion_4\PHASE_CORRECTED_BATT_4_AND_5/"
path_co_corrected = path_corrected + "co_card1/"
path_h2co_corrected = path_corrected + "h2co_card2/"
names_co_corrected = [i.name for i in os.scandir(path_co_corrected)]
names_h2co_corrected = [i.name for i in os.scandir(path_h2co_corrected)]

key = lambda s: float(s.split(".npy")[0])
names_co_corrected.sort(key=key)
names_h2co_corrected.sort(key=key)
names_co_corrected = [path_co_corrected + i for i in names_co_corrected]
names_h2co_corrected = [path_h2co_corrected + i for i in names_h2co_corrected]

ind_shocks = np.load(path_corrected + "IND_SHOCKS.npy")
ppifg_shock_ref = ind_shocks.min() // ppifg

for n, i in enumerate(ind_shocks):
    ppifg_shock = i // ppifg
    ppifg_diff = ppifg_shock_ref - ppifg_shock

    if n == 0:
        co = np.load(names_co_corrected[n])
        if ppifg_diff < 0:
            co = co[abs(ppifg_diff):]
        elif ppifg_diff > 0:
            assert ppifg_diff < 0, "this shouldn't happen"
            co = co[:ppifg_diff]

        # ____________________________________________________________________________________________________________
        h2co = np.load(names_h2co_corrected[n])
        if ppifg_diff < 0:
            h2co = h2co[abs(ppifg_diff):]
        elif ppifg_diff > 0:
            assert ppifg_diff < 0, "this shouldn't happen"
            h2co = h2co[:ppifg_diff]

    else:
        x = np.load(names_co_corrected[n])
        if ppifg_diff < 0:
            x = x[abs(ppifg_diff):]
        elif ppifg_diff > 0:
            assert ppifg_diff < 0, "this shouldn't happen"
            x = x[:ppifg_diff]

        if len(co) > len(x):
            co = co[:len(x)] + x
        elif len(co) < len(x):
            co += x[:len(co)]
        else:
            co += x

        # ____________________________________________________________________________________________________________
        y = np.load(names_h2co_corrected[n])
        if ppifg_diff < 0:
            y = y[abs(ppifg_diff):]
        elif ppifg_diff > 0:
            assert ppifg_diff < 0, "this shouldn't happen"
            y = y[:ppifg_diff]

        if len(h2co) > len(y):
            h2co = h2co[:len(y)] + y
        elif len(h2co) < len(y):
            h2co += y[:len(h2co)]
        else:
            h2co += y

    print(len(names_co_corrected) - n - 1)

np.save(path_corrected + "co_averaged_batt_4_and_5.npy", co)
np.save(path_corrected + "h2co_averaged_batt_4_and_5.npy", h2co)
