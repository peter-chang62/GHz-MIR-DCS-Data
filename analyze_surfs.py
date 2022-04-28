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
# path_co = r'D:\ShockTubeData\04242022\Surf_27\card1/'
# path_h2co = r'D:\ShockTubeData\04242022\Surf_27\card2/'
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
    # co = co[:30]
    # co, _ = pc.Phase_Correct(co, ppifg, 50, False)

    h2co = pc.get_data(path_h2co, n)
    h2co = h2co[ind_r:]
    h2co, _ = pc.adjust_data_and_reshape(h2co, ppifg)
    h2co = h2co[:N_ifgs]
    h2co, _ = pc.Phase_Correct(h2co, ppifg, 25, False)

    H2CO[n][:] = h2co
    # CO[n][:] = co

    print(n)

# %% # calculate the average for each shock
avg_per_shock = np.zeros((N_shocks, ppifg))
for n, i in enumerate(H2CO):
    avg_per_shock[n] = np.mean(i, 0)

# %% the ll and ul give the indices for the positive frequency range where H2CO has appreciable signal
# calculate the shift needed to average different shocks together
ll, ul = 1120 + ppifg // 2, 3600 + ppifg // 2
avg_per_shock, shifts, sgns = pc.fix_sign_and_phase_correct(avg_per_shock, ll, ul, ppifg, 25, True, 10)
avg_per_shock, shifts2, sgns2 = pc.fix_sign_and_phase_correct(avg_per_shock, ll, ul, ppifg, 10, True, 10)
shifts += shifts2
sgns *= sgns2

# %% shift separate shocks so that you can average different shocks together
for n, i in enumerate(H2CO):
    i *= sgns[n]
    i = pc.shift_2d(i, np.repeat(shifts[n], N_ifgs))
    H2CO[n] = i
    print(n)
