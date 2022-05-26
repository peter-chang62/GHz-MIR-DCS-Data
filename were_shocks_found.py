import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc

# %%
path_co = r'D:\ShockTubeData\04242022_Data\Surf_28\card1/'
path_h2co = r'D:\ShockTubeData\04242022_Data\Surf_28\card2/'
ppifg = 17507
center = ppifg // 2

# %% The arrays are 3D, with indexing going as N_shock, N_ifg, ppifg
N_ifgs = 70  # look at 50 interferograms post shock
N_shocks = pc.Number_of_files(path_co)
H2CO = np.zeros((N_shocks, N_ifgs, ppifg))
CO = np.zeros((N_shocks, N_ifgs, ppifg))

# %%
"""Looks good! """

fig, ax = plt.subplots(1, 1)
for n in range(N_shocks):
    co = pc.get_data(path_co, n)
    co, _ = pc.adjust_data_and_reshape(co, ppifg)
    shape = co.shape
    co = co.flatten()
    ind_i, ind_r = pc.get_ind_total_to_throw(co, ppifg)

    bckgnd = co.copy()
    bckgnd.resize(shape)
    bckgnd[:, ppifg // 2 - 50:ppifg // 2 + 50] = 0.0
    bckgnd = bckgnd.flatten()

    ind = 1000
    ll, ul = ind_i - ind, ind_r + ind
    ax.clear()
    ax.plot(co[ll:ul])
    ax.axvline(ind, color='r')
    ax.axvline(ul - ll - ind, color='r')
    plt.title(n)
    # plt.pause(.001)
    plt.savefig(f'temp_fig/{n}.png')
