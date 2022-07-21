import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc
import digital_phase_correction as dpc


def save_npy(path, filename, arr):
    with open(path + filename, 'wb') as f:
        np.save(f, arr)


# %%____________________________________________________________________________________________________________________
# data paths
path = r'/media/peterchang/Samsung_T5/MIR stuff/ShockTubeData/DATA_MATT_PATRICK_TRIP_2/06-21-2022/'
batt_folder = "battalion_9/"
path_co = path + batt_folder
ppifg = 17506
center = ppifg // 2

# %%____________________________________________________________________________________________________________________
# The arrays are 3D, with indexing going as N_shock, N_ifg, ppifg
N_ifgs = 70  # look at 50 interferograms post shock
N_shocks = pc.Number_of_files(path_co)
H2CO = np.zeros((N_shocks, N_ifgs, ppifg))
CO = np.zeros((N_shocks, N_ifgs, ppifg))

# %%____________________________________________________________________________________________________________________
# phase correct data for each shock
ll_freq_co = 0.15
ul_freq_co = 0.25
IND_MINUS_INDI = np.zeros(N_shocks)
IND_MINUS_INDR = np.zeros(N_shocks)

ref_co = np.load('06302022_batt4_card1_shock1.npy')
ref_h2co = np.load('06302022_batt4_card2_shock1.npy')

for n in range(N_shocks):
    # get the data
    co = pc.get_data(path_co, n)

    # throw out data points from the beginning to get down to
    # integer ppifg
    co, _ = pc.adjust_data_and_reshape(co, ppifg)
    co = co.flatten()

    # find the incident and reflected shock location
    ind_i, ind_r = pc.get_ind_total_to_throw(co, ppifg)
    ind = int(np.round(ind_i / ppifg) * ppifg)  # based on incident shock!
    IND_MINUS_INDI[n] = ind - ind_i  # important for time binning!
    IND_MINUS_INDR[n] = ind - ind_r  # actually have NOT used this for time binning!

    # truncate it down to 20 shocks before incident and 50 shocks
    # after reflected
    co = co[ind - ppifg * 20: ind + ppifg * 50]
    co = co.reshape((70, ppifg))

    # phase correct
    co = np.vstack([ref_co, co])
    p_co = dpc.get_pdiff(co, ll_freq_co, ul_freq_co, 200)
    dpc.apply_t0_and_phi0_shift(p_co, co)
    co = co[1:]

    CO[n][:] = co

    print(N_shocks - n)

# %%____________________________________________________________________________________________________________________
# # If you want to save the data
save_path = path + "PHASE_CORRECTED_DATA/" + batt_folder
save_npy(save_path, f'CO_{CO.shape[0]}x{CO.shape[1]}x{CO.shape[2]}.npy', CO)
save_npy(save_path, 'ind_minus_indi.npy', IND_MINUS_INDI)
save_npy(save_path, 'ind_minus_indr.npy', IND_MINUS_INDR)
