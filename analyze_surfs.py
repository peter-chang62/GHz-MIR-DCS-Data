import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc


def rad_to_deg(rad):
    return rad * 180 / np.pi


def deg_to_rad(deg):
    return deg * np.pi / 180


"""Data paths """

# %%
# path_co = r'D:\ShockTubeData\Data_04232022\Surf_18\card1/'
# path_h2co = r'D:\ShockTubeData\Data_04232022\Surf_18\card2/'
# ppifg = 17511
# center = ppifg // 2

# %%
# path_co = r'D:\ShockTubeData\04242022_Data\Surf_27\card1/'
# path_h2co = r'D:\ShockTubeData\04242022_Data\Surf_27\card2/'
# ppifg = 17507
# center = ppifg // 2

# %%
path_co = r'D:\ShockTubeData\04242022_Data\Surf_28\card1/'
path_h2co = r'D:\ShockTubeData\04242022_Data\Surf_28\card2/'
ppifg = 17507
center = ppifg // 2

# %% The arrays are 3D, with indexing going as N_shock, N_ifg, ppifg
N_ifgs = 50  # look at 50 interferograms post shock
N_shocks = pc.Number_of_files(path_co)
H2CO = np.zeros((N_shocks, N_ifgs, ppifg))
# CO = np.zeros((N_shocks, N_ifgs, ppifg))

# %% phase correct data for each shock
for n in range(N_shocks):
    co = pc.get_data(path_co, n)
    ind_i, ind_r = pc.get_ind_total_to_throw(co, ppifg)

    # co = co[ind_r:]
    # co, _ = pc.adjust_data_and_reshape(co, ppifg)
    # co = co[:N_ifgs]
    # co, _ = pc.Phase_Correct(co, ppifg, 50, False)

    h2co = pc.get_data(path_h2co, n)
    h2co = h2co[ind_r:]
    h2co, _ = pc.adjust_data_and_reshape(h2co, ppifg)
    h2co = h2co[:N_ifgs]
    h2co, _ = pc.Phase_Correct(h2co, ppifg, 25, False)

    H2CO[n][:] = h2co
    # CO[n][:] = co

    print(n)

"""Combine shocks for H2CO"""

# %% # calculate the average for each shock
avg_per_shock = np.zeros((N_shocks, ppifg))
for n, i in enumerate(H2CO):
    avg_per_shock[n] = np.mean(i, 0)

# %%
ref = avg_per_shock[0]
SGN = np.zeros(len(avg_per_shock))
SHIFTS = np.zeros(len(avg_per_shock))
for n, i in enumerate(avg_per_shock):
    arr1 = np.vstack([ref, i])
    arr2 = np.vstack([ref, -i])

    ifg1, shift1 = pc.Phase_Correct(arr1, ppifg, 25, False)
    ifg2, shift2 = pc.Phase_Correct(arr2, ppifg, 25, False)
    shift1 = shift1[1]
    shift2 = shift2[1]
    ifg1 = ifg1[1]
    ifg2 = ifg2[1]

    diff1 = np.mean(abs(ref[center - 50:center + 50] - ifg1[center - 50: center + 50]))
    diff2 = np.mean(abs(ref[center - 50:center + 50] - ifg2[center - 50: center + 50]))

    if diff1 < diff2:
        avg_per_shock[n] = ifg1
        SGN[n] = 1
        SHIFTS[n] = shift1
    else:
        avg_per_shock[n] = ifg2
        SGN[n] = -1
        SHIFTS[n] = shift2

    if n % 100 == 0:
        print(len(avg_per_shock) - n)

# %% shift separate shocks so that you can average different shocks together
for n, i in enumerate(H2CO):
    i *= SGN[n]
    i = pc.shift_2d(i, np.repeat(SHIFTS[n], N_ifgs))
    H2CO[n] = i
    print(len(H2CO) - n)

"""Combine shocks for CO"""

# %% calculate the average for each shock
# avg_per_shock = np.zeros((N_shocks, ppifg))
# for n, i in enumerate(CO):
#     avg_per_shock[n] = np.mean(i, 0)

# %% calculate the shift needed to average different shocks together
# avg_per_shock, shifts = pc.Phase_Correct(avg_per_shock, ppifg, 50, True)

# %% shift separate shocks so that you can average different shocks together
# for n, i in enumerate(CO):
#     i = pc.shift_2d(i, np.repeat(shifts[n], N_ifgs))
#     CO[n] = i
#     print(len(CO) - n)
