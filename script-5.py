#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:44:58 2023

@author: peterchang

April 24, surfs 13 and 14
"""

import sys

sys.path.append("include/")
import numpy as np
import clipboard_and_style_sheet as cr
import phase_correction as pc
import digital_phase_correction as dpc
import os
import matplotlib.pyplot as plt
import scipy.signal as ss
from tqdm import tqdm

key = lambda s: float(s.split("LoopCount_")[1].split("_Datetime")[0])
ppifg = 17507
center = ppifg // 2

f_MHz = np.fft.rfftfreq(ppifg, d=1e-9) * 1e-6
freq_nounit = np.fft.rfftfreq(ppifg)

ll_freq_co, ul_freq_co = 0.1549, 0.2211
ll_freq_h2co, ul_freq_h2co = 0.0791, 0.1686

# %% load path names
path = (
    r"/Volumes/Extreme SSD/Research_Projects/Shocktube/"
    "DATA_MATT_PATRICK_TRIP_1/04242022/"
)

path_save = (
    r"/Volumes/Extreme SSD/Research_Projects/Shocktube/"
    "DATA_MATT_PATRICK_TRIP_1/04242022/Surf_13/PHASE_CORRECTED/"
)

# %%
path_batt_13 = path + "Surf_13/"
path_batt_13_co = path_batt_13 + "card1/"
path_batt_13_h2co = path_batt_13 + "card2/"

path_batt_14 = path + "Surf_14/"
path_batt_14_co = path_batt_14 + "card1/"
path_batt_14_h2co = path_batt_14 + "card2/"

# load co (card1) path names
names_co_batt_13 = [i.name for i in os.scandir(path_batt_13_co)]
names_co_batt_14 = [i.name for i in os.scandir(path_batt_14_co)]
names_co_batt_13.sort(key=key)
names_co_batt_14.sort(key=key)
names_co_batt_13 = [path_batt_13_co + i for i in names_co_batt_13]
names_co_batt_14 = [path_batt_14_co + i for i in names_co_batt_14]

# load h2co (card2) path names
names_h2co_batt_13 = [i.name for i in os.scandir(path_batt_13_h2co)]
names_h2co_batt_14 = [i.name for i in os.scandir(path_batt_14_h2co)]
names_h2co_batt_13.sort(key=key)
names_h2co_batt_14.sort(key=key)
names_h2co_batt_13 = [path_batt_13_h2co + i for i in names_h2co_batt_13]
names_h2co_batt_14 = [path_batt_14_h2co + i for i in names_h2co_batt_14]

# combine surf names
names_co = names_co_batt_13 + names_co_batt_14
names_h2co = names_h2co_batt_13 + names_h2co_batt_14

# %% phase correcting surfs 13 and 14

IND_SHOCK = np.zeros(len(names_co))
N_shocks = len(names_co)

for i in tqdm(range(len(names_co))):
    # load data
    co = np.fromfile(names_co[i], "<h")[:-64].astype(float)
    h2co = np.fromfile(names_h2co[i], "<h")[:-64].astype(float)

    # resize data
    ind_throw = np.argmax(co[:ppifg]) + center
    co = co[ind_throw:]
    co = co[: (co.size // ppifg) * ppifg]
    co.resize((co.size // ppifg, ppifg))

    # throw out same number of points as co, this is important to line up the
    # shocks
    h2co = h2co[ind_throw:]
    h2co = h2co[: (h2co.size // ppifg) * ppifg]
    h2co.resize((h2co.size // ppifg, ppifg))

    # locate shocks from low-pass filtered background
    ft = dpc.rfft(co, 1)
    ft_filtered = ft.copy()
    ft_filtered[:] = np.where(f_MHz < 5, ft_filtered, 0)
    bckgnd = dpc.irfft(ft_filtered, 1)
    bckgnd.resize(bckgnd.size)
    ind_incident = np.argmax(bckgnd)
    bckgnd_flipped = bckgnd * -1
    ind_shock = np.argmax(bckgnd_flipped[ind_incident:]) + ind_incident

    # phase_correction
    if i == 0:
        co_global_reference = co[0].copy()
        h2co_global_reference = h2co[0].copy()

        pdiff_co = dpc.get_pdiff(co, ll_freq_co, ul_freq_co, 200)
        pdiff_h2co = dpc.get_pdiff(h2co, ll_freq_h2co, ul_freq_h2co, 200)
    else:
        co_pdiff_co = np.vstack([co_global_reference, co])
        pdiff_co = dpc.get_pdiff(co_pdiff_co, ll_freq_co, ul_freq_co, 200)
        pdiff_co = pdiff_co[1:]

        h2co_pdiff_h2co = np.vstack([h2co_global_reference, h2co])
        pdiff_h2co = dpc.get_pdiff(
            h2co_pdiff_h2co, ll_freq_h2co, ul_freq_h2co, 200
        )
        pdiff_h2co = pdiff_h2co[1:]

    dpc.apply_t0_shift(pdiff_co, freq_nounit, ft)
    td = dpc.irfft(ft, 1)
    hbt = ss.hilbert(td)
    dpc.apply_phi0_shift(pdiff_co, hbt)
    co_corr = hbt.real

    h2co_corr = dpc.apply_t0_and_phi0_shift(pdiff_h2co, h2co, return_new=True)

    # save phase corrected data
    IND_SHOCK[i] = ind_shock
    np.save(path_save + "co_card1/" + f"{i}.npy", co_corr)
    np.save(path_save + "h2co_card2/" + f"{i}.npy", h2co_corr)

# save shock locations
np.save(path_save, "IND_SHOCKS.npy", IND_SHOCK)
