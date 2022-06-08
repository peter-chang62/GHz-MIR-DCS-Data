import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc
import digital_phase_correction as dpc

"""Data paths """


def save_npy(path, filename, arr):
    with open(path + filename, 'wb') as f:
        np.save(f, arr)


# %%____________________________________________________________________________________________________________________
# path_co = r'D:\ShockTubeData\Data_04232022\Surf_18\card1/'
# path_h2co = r'D:\ShockTubeData\Data_04232022\Surf_18\card2/'
# ppifg = 17511
# center = ppifg // 2

# %%____________________________________________________________________________________________________________________
# path_co = r'D:\ShockTubeData\04242022_Data\Surf_27\card1/'
# path_h2co = r'D:\ShockTubeData\04242022_Data\Surf_27\card2/'
# ppifg = 17507
# center = ppifg // 2

# %%____________________________________________________________________________________________________________________
# path_co = r'D:\ShockTubeData\04242022_Data\Surf_28\card1/'
# path_h2co = r'D:\ShockTubeData\04242022_Data\Surf_28\card2/'
# ppifg = 17507
# center = ppifg // 2

# %%____________________________________________________________________________________________________________________
path_co = r'D:\ShockTubeData\04242022_Data\Surf_22/card1/'
path_h2co = r'D:\ShockTubeData\04242022_Data\Surf_22/card2/'
save_path = r'D:\ShockTubeData\04242022_Data\Surf_22\PHASE_CORRECTED_DATA/'
ppifg = 17507
center = ppifg // 2

"""Initialize arrays and specify what interferograms to look at """

# %%____________________________________________________________________________________________________________________
# The arrays are 3D, with indexing going as N_shock, N_ifg, ppifg
N_ifgs = 70  # look at 50 interferograms post shock
N_shocks = pc.Number_of_files(path_co)
H2CO = np.zeros((N_shocks, N_ifgs, ppifg))
CO = np.zeros((N_shocks, N_ifgs, ppifg))

""" Phase correct each shock"""

# %%____________________________________________________________________________________________________________________
# phase correct data for each shock
ll_freq_h2co = 0.0597
ul_freq_h2co = 0.20
ll_freq_co = 0.15
ul_freq_co = 0.25
IND_MINUS_INDI = np.zeros(N_shocks)
IND_MINUS_INDR = np.zeros(N_shocks)
for n in range(N_shocks):
    # get the data
    co = pc.get_data(path_co, n)

    # throw out data points from the beginning to get down to
    # integer ppifg
    co, _ = pc.adjust_data_and_reshape(co, ppifg)
    co = co.flatten()

    # find the incident and reflected shock location
    ind_i, ind_r = pc.get_ind_total_to_throw(co, ppifg)
    ind = int(np.round(ind_i / ppifg) * ppifg)
    IND_MINUS_INDI[n] = ind - ind_i
    IND_MINUS_INDR[n] = ind - ind_r

    # truncate it down to 20 shocks before incident and 50 shocks
    # after reflected
    co = co[ind - ppifg * 20: ind + ppifg * 50]
    co = co.reshape((70, ppifg))

    # phase correct
    p_co = dpc.get_pdiff(co, ll_freq_co, ul_freq_co, 200)
    dpc.apply_t0_and_phi0_shift(p_co, co)
    CO[n][:] = co

    # the same thing for h2co except that the shock location
    # is taken from the co data (the two streams were simultaneously
    # triggered so this works)
    h2co = pc.get_data(path_h2co, n)
    h2co, _ = pc.adjust_data_and_reshape(h2co, ppifg)
    h2co = h2co.flatten()
    h2co = h2co[ind - ppifg * 20: ind + ppifg * 50]
    h2co = h2co.reshape((70, ppifg))

    p_h2co = dpc.get_pdiff(h2co, ll_freq_h2co, ul_freq_h2co, 200)
    dpc.apply_t0_and_phi0_shift(p_h2co, h2co)
    H2CO[n][:] = h2co

    print(N_shocks - n)

"""Combine shocks"""

# %%____________________________________________________________________________________________________________________
# calculate the average for each shock
avg_shock_H2CO = np.zeros((N_shocks, ppifg))
avg_shock_CO = np.zeros((N_shocks, ppifg))
for n in range(len(H2CO)):
    avg_shock_H2CO[n] = np.mean(H2CO[n], 0)
    avg_shock_CO[n] = np.mean(CO[n], 0)

p_avg_shock_H2CO = dpc.get_pdiff(avg_shock_H2CO, ll_freq_h2co, ul_freq_h2co, 200)
dpc.apply_t0_and_phi0_shift(p_avg_shock_H2CO, avg_shock_H2CO)

p_avg_shock_CO = dpc.get_pdiff(avg_shock_CO, ll_freq_co, ul_freq_co, 200)
dpc.apply_t0_and_phi0_shift(p_avg_shock_CO, avg_shock_CO)

# %%____________________________________________________________________________________________________________________
for n in range(len(H2CO)):
    p = np.repeat(p_avg_shock_H2CO[n][:, np.newaxis], N_ifgs, 1).T
    dpc.apply_t0_and_phi0_shift(p, H2CO[n])

    p = np.repeat(p_avg_shock_CO[n][:, np.newaxis], N_ifgs, 1).T
    dpc.apply_t0_and_phi0_shift(p, CO[n])

    print(len(H2CO) - n)

# %%____________________________________________________________________________________________________________________
# If you want to save the data
save_npy(save_path, f'CO_{CO.shape[0]}x{CO.shape[1]}x{CO.shape[2]}.npy', CO)
save_npy(save_path, f'H2CO_{CO.shape[0]}x{CO.shape[1]}x{CO.shape[2]}.npy', H2CO)
save_npy(save_path, 'ind_minus_indi.npy', IND_MINUS_INDI)
save_npy(save_path, 'ind_minus_indr.npy', IND_MINUS_INDR)

# %%____________________________________________________________________________________________________________________
"""Combine surfs, call this after you've run the above cells and saved the phase corrected data """

surf_27_co = np.load(r'D:\ShockTubeData\04242022_Data\Surf_13\PHASE_CORRECTED_DATA/CO_499x70x17507.npy')
surf_27_h2co = np.load(r'D:\ShockTubeData\04242022_Data\Surf_13\PHASE_CORRECTED_DATA/H2CO_499x70x17507.npy')

surf_28_co = np.load(r'D:\ShockTubeData\04242022_Data\Surf_22\PHASE_CORRECTED_DATA/CO_299x70x17507.npy')
surf_28_h2co = np.load(r'D:\ShockTubeData\04242022_Data\Surf_22\PHASE_CORRECTED_DATA/H2CO_299x70x17507.npy')
save_path = r'D:\ShockTubeData\04242022_Data\Surf_22\PHASE_CORRECTED_DATA/'

# surf_27_co.resize((499, 70, 17507))
# surf_27_h2co.resize((499, 70, 17507))
# surf_28_co.resize((299, 70, 17507))
# surf_28_h2co.resize((299, 70, 17507))

ppifg = 17507
center = ppifg // 2
ll_freq_h2co = 0.0597
ul_freq_h2co = 0.20
ll_freq_co = 0.15
ul_freq_co = 0.25

N_ifgs = surf_28_co.shape[1]

# %%____________________________________________________________________________________________________________________
avg_surf_27_co = np.mean(surf_27_co, axis=(0, 1))
avg_surf_27_h2co = np.mean(surf_27_h2co, axis=(0, 1))
avg_surf_28_co = np.mean(surf_28_co, axis=(0, 1))
avg_surf_28_h2co = np.mean(surf_28_h2co, axis=(0, 1))

avg_co = np.vstack([avg_surf_27_co, avg_surf_28_co])
avg_h2co = np.vstack([avg_surf_27_h2co, avg_surf_28_h2co])

# %%____________________________________________________________________________________________________________________
p_h2co = dpc.get_pdiff(avg_h2co, ll_freq_h2co, ul_freq_h2co, 200)
p_co = dpc.get_pdiff(avg_co, ll_freq_co, ul_freq_co, 200)

dpc.apply_t0_and_phi0_shift(p_h2co, avg_h2co)
dpc.apply_t0_and_phi0_shift(p_co, avg_co)

# %%____________________________________________________________________________________________________________________
p_h2co_rep = np.repeat(p_h2co[1][:, np.newaxis], N_ifgs, 1).T
p_co_rep = np.repeat(p_co[1][:, np.newaxis], N_ifgs, 1).T
for n in range(len(surf_28_h2co)):
    dpc.apply_t0_and_phi0_shift(p_h2co_rep, surf_28_h2co[n])
    dpc.apply_t0_and_phi0_shift(p_co_rep, surf_28_co[n])
    print(len(surf_28_h2co) - n)

# %%____________________________________________________________________________________________________________________
# Save the data
save_npy(save_path, f'CO_{surf_28_co.shape[0]}x{surf_28_co.shape[1]}x{surf_28_co.shape[2]}.npy', surf_28_co)
save_npy(save_path, f'H2CO_{surf_28_co.shape[0]}x{surf_28_co.shape[1]}x{surf_28_co.shape[2]}.npy', surf_28_h2co)

plt.figure()
[plt.plot(i[center - 25:center + 25]) for i in surf_27_co[:, 0]]
[plt.plot(i[center - 25:center + 25]) for i in surf_28_co[:, 0]]
plt.figure()
[plt.plot(i[center - 25:center + 25]) for i in surf_27_h2co[:, 0]]
[plt.plot(i[center - 25:center + 25]) for i in surf_28_h2co[:, 0]]

# %%____________________________________________________________________________________________________________________
"""Now that surfs are combined, average across all the shocks"""
# surf_27_co = np.fromfile(r'D:\ShockTubeData\04242022_Data\Surf_27\PHASE_CORRECTED_DATA/CO_499x70x17507.bin')
# surf_27_h2co = np.fromfile(r'D:\ShockTubeData\04242022_Data\Surf_27\PHASE_CORRECTED_DATA/H2CO_499x70x17507.bin')
#
# surf_28_co = np.fromfile(r'D:\ShockTubeData\04242022_Data\Surf_28\PHASE_CORRECTED_DATA/CO_299x70x17507.bin')
# surf_28_h2co = np.fromfile(r'D:\ShockTubeData\04242022_Data\Surf_28\PHASE_CORRECTED_DATA/H2CO_299x70x17507.bin')
#
# surf_27_co.resize((499, 70, 17507))
# surf_27_h2co.resize((499, 70, 17507))
# surf_28_co.resize((299, 70, 17507))
# surf_28_h2co.resize((299, 70, 17507))
#
# avg_surf_27_co = np.mean(surf_27_co, axis=0)
# avg_surf_27_h2co = np.mean(surf_27_h2co, axis=0)
# avg_surf_28_co = np.mean(surf_28_co, axis=0)
# avg_surf_28_h2co = np.mean(surf_28_h2co, axis=0)
#
# avg_co = (avg_surf_27_co + avg_surf_28_co) / 2
# avg_h2co = (avg_surf_27_h2co + avg_surf_28_h2co) / 2
