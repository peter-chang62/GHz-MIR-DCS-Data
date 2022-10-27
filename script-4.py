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

# _________________________________________________________ load path names ____________________________________________
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

# ___________________________________________________ save paths _______________________________________________________
save_path = r"D:\DATA_MATT_PATRICK_TRIP_2\06-30-2022\Battalion_28\PHASE_CORRECTED_BATT_28_AND_31/"
save_path_co = save_path + "co_card1/"
save_path_h2co = save_path + "h2co_card2/"

# _______________________________________ analysis _____________________________________________________________________
"""different from the previous scripts, to get the incident shock now, I need to find the incident one first and then 
tell it to search for the shock after that. I think this is because the shocks here were a little weaker than the 
shocks analyzed in the previous scripts """

IND_SHOCK = []

N_shocks = len(names_co)
for i in range(N_shocks):
    co = np.fromfile(names_co[i], '<h')[:-64]
    co = co / co.max()

    # load data and reshape
    ind = np.argmax(co[:ppifg])
    co = co[ind:]
    N1 = len(co) // ppifg
    co = co[:N1 * ppifg]
    co = co[center:-center]
    N2 = len(co) // ppifg
    co = co[:N2 * ppifg]
    co = np.reshape(co, (N2, ppifg))

    if i == 0:
        co_global_reference = co[0].copy()

    # locate shock
    ft = pc.fft(co, axis=1)
    ft_filtered = ft.copy()
    for n, m in enumerate(ft_filtered):
        ft_filtered[n] = np.where(abs(f_MHz) < 5, m, 0)
    bckgnd = pc.ifft(ft_filtered, axis=1).real.flatten()
    ind_incident = np.argmax(bckgnd)
    bckgnd_flipped = bckgnd * -1  # get the reflected shock
    ind_shock = np.argmax(bckgnd_flipped[ind_incident:]) + ind_incident

    # phase correction
    if i > 0:
        co_pdiff = np.vstack([co_global_reference, co])
        pdiff = dpc.get_pdiff(co_pdiff, ll_freq_co, ul_freq_co, 200)
        pdiff = pdiff[1:]
    else:
        pdiff = dpc.get_pdiff(co, ll_freq_co, ul_freq_co, 200)

    dpc.apply_t0_shift(pdiff, freq_nounit, ft)
    td = pc.ifft(ft, axis=1).real
    hbt = ss.hilbert(td)
    dpc.apply_phi0_shift(pdiff, hbt)
    hbt = hbt.real

    # save data
    IND_SHOCK.append(ind_shock)
    np.save(save_path_co + f"{i}.npy", hbt)

    # ____________________________ repeat for h2co but use co shock location and data truncation _____________________
    h2co = np.fromfile(names_h2co[i], '<h')[:-64]
    h2co = h2co / h2co.max()

    h2co = h2co[ind:]
    h2co = h2co[:N1 * ppifg]
    h2co = h2co[center:-center]
    h2co = h2co[:N2 * ppifg]
    h2co = np.reshape(h2co, (N2, ppifg))

    if i == 0:
        h2co_global_reference = h2co[0].copy()

    if i > 0:
        h2co_pdiff = np.vstack([h2co_global_reference, h2co])
        pdiff = dpc.get_pdiff(h2co_pdiff, ll_freq_h2co, ul_freq_h2co, 200)
        pdiff = pdiff[1:]
    else:
        pdiff = dpc.get_pdiff(h2co, ll_freq_h2co, ul_freq_h2co, 200)
    dpc.apply_t0_and_phi0_shift(pdiff, h2co)

    np.save(save_path_h2co + f"{i}.npy", h2co)

    print(N_shocks - i - 1)

np.save(save_path + "IND_SHOCKS.npy", IND_SHOCK)
