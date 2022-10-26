"""April 24, 2022: surfs 27 and 28 """

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

# ___________________________________________________ get data paths for surf 27 _______________________________________
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

# ___________________________________________________ get data paths for surf 28 _______________________________________
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

# ___________________________________________________ all data paths ___________________________________________________
names_co = names_co_surf27 + names_co_surf28
names_h2co = names_h2co_surf27 + names_h2co_surf28

# ___________________________________________________ save paths _______________________________________________________
save_path = r"D:\ShockTubeData\04242022_Data\Surf_27\PHASE_CORRECTED_DATA/"
save_path_co = save_path + "co_card1/"
save_path_h2co = save_path + "h2co_card2/"

# ______________________________________________ set ppifg _____________________________________________________________
ppifg = 17507
center = ppifg // 2

f_MHz = np.fft.fftshift(np.fft.fftfreq(ppifg, d=1e-9)) * 1e-6
freq_nounit = np.fft.fftshift(np.fft.fftfreq(ppifg))

# _______________________________________ analysis _____________________________________________________________________
CO = []
H2CO = []
IND_SHOCK = []

N_shocks = len(names_co)
# N_shocks = 2
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
    bckgnd_flipped = bckgnd * -1  # get the reflected shock
    ind_shock = np.argmax(bckgnd_flipped)

    # phase correction
    ll_freq_co, ul_freq_co = .1548, 0.2443
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
    # CO.append(hbt)
    np.save(save_path_co + f"{i}.npy", hbt)

    # ____________________________ repeat for h2co but use co shock location and data truncation _______________________
    h2co = np.fromfile(names_h2co[i], '<h')[:-64]
    h2co = h2co / h2co.max()

    h2co = h2co[ind:]
    h2co = h2co[:N1 * ppifg]
    h2co = h2co[center:-center]
    h2co = h2co[:N2 * ppifg]
    h2co = np.reshape(h2co, (N2, ppifg))

    if i == 0:
        h2co_global_reference = h2co[0].copy()

    ll_freq_h2co, ul_freq_h2co = 0.0791, 0.1686
    if i > 0:
        h2co_pdiff = np.vstack([h2co_global_reference, h2co])
        pdiff = dpc.get_pdiff(h2co_pdiff, ll_freq_h2co, ul_freq_h2co, 200)
        pdiff = pdiff[1:]
    else:
        pdiff = dpc.get_pdiff(h2co, ll_freq_h2co, ul_freq_h2co, 200)
    dpc.apply_t0_and_phi0_shift(pdiff, h2co)

    # H2CO.append(h2co)
    np.save(save_path_h2co + f"{i}.npy", h2co)

    print(N_shocks - i - 1)

np.save(save_path + "IND_SHOCKS.npy", IND_SHOCK)
