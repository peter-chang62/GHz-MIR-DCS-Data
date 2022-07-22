import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc
import digital_phase_correction as dpc
from sys import platform

clipboard_and_style_sheet.style_sheet()


def save_npy(path, filename, arr):
    with open(path + filename, 'wb') as f:
        np.save(f, arr)


if platform == "win32":  # on windows
    path_header = "E:\\"

else:  # otherwise on Pop!OS
    path_header = "/media/peterchang/Samsung_T5/"

# %%____________________________________________________________________________________________________________________
# data paths
path = path_header + r'MIR stuff/ShockTubeData/DATA_MATT_PATRICK_TRIP_2/06-23-2022/'
batt_folder = "battalion_18/"
path_co = path + batt_folder
ppifg = 15046
center = ppifg // 2

# %%____________________________________________________________________________________________________________________
# The arrays are 3D, with indexing going as N_shock, N_ifg, ppifg
N_ifgs = 70  # look at 50 interferograms post shock
N_shocks = pc.Number_of_files(path_co)
H2CO = np.zeros((N_shocks, N_ifgs, ppifg))
CO = np.zeros((N_shocks, N_ifgs, ppifg))

# %%____________________________________________________________________________________________________________________
# phase correct data for each shock
ll_freq_co, ul_freq_co = 0.27700940860215084, 0.39620816532258063
IND_MINUS_INDI = np.zeros(N_shocks)

ref_co = np.load('06212022_batt9_shock1.npy')
ind_i_for_check = np.zeros(N_shocks)
bckgnd_for_check = np.zeros((N_shocks, int(1e7)))
ind_search_window = 1992507, 2995176

for n in range(N_shocks):
    # get the data
    co = pc.get_data(path_co, n)

    # throw out data points from the beginning to get down to
    # integer ppifg
    co, _ = pc.adjust_data_and_reshape(co, ppifg)
    co = co.flatten()

    # find the background manually
    # for data taken on 06/21/2022 this was okay, but for data taken on 06/22/2022 the shocks are weak enough
    # that the reflected shock is NOT above the f0 oscillations!
    N = len(co) // ppifg
    bckgnd = co.copy().reshape((N, ppifg))

    # low pass filter the background ___________________________________________________________________________________
    # this one takes longer, so if the shocks are strong I don't bother to filter
    ft_bckgnd = pc.fft(bckgnd, 1)
    ft_bckgnd[:, 200 + center:] = 0.0
    ft_bckgnd[:, :2 * center - 200 - center] = 0.0
    bckgnd = pc.ifft(ft_bckgnd, 1).flatten().real
    bckgnd -= bckgnd.min()
    bckgnd[:ind_search_window[0]] = 0
    bckgnd[ind_search_window[1]:] = 0
    # low pass filter the background ___________________________________________________________________________________

    # don't low pass filter the background (strong shocks) _____________________________________________________________
    # bckgnd[:, center - 50:center + 50] = 0.0
    # bckgnd = bckgnd.flatten()
    # don't low pass filter the background (strong shocks) _____________________________________________________________

    ind_i = np.argmax(bckgnd)

    # if you have enough signal to distinguish reflected from incident _________________________________________________
    bckgnd_ = bckgnd.copy()
    bckgnd_[ind_i - int(1e4): ind_i + int(1e4)] = 0
    ind_r = np.argmax(bckgnd_)
    ind_i, ind_r = sorted([ind_i, ind_r])
    # if you have enough signal to distinguish reflected from incident _________________________________________________

    ind_i_for_check[n] = ind_i
    bckgnd_for_check[n][:len(bckgnd)] = bckgnd

    # find the incident and reflected shock location
    # ind_i, ind_r = pc.get_ind_total_to_throw(co, ppifg)
    ind = int(np.round(ind_i / ppifg) * ppifg)  # based on incident shock!
    IND_MINUS_INDI[n] = ind - ind_i  # important for time binning!

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
# If you want to save the data
save_path = path + "PHASE_CORRECTED_DATA/" + batt_folder
save_npy(save_path, f'CO_{CO.shape[0]}x{CO.shape[1]}x{CO.shape[2]}.npy', CO)
save_npy(save_path, 'ind_minus_indi.npy', IND_MINUS_INDI)

# %%____________________________________________________________________________________________________________________
# verification
# fig, ax = plt.subplots(1, 1)
# for n, i in enumerate(bckgnd_for_check):
#     ax.clear()
#     ax.plot(i[int(ind_i_for_check[n] - 30e4): int(ind_i_for_check[n] + 30e4)])
#     ax.axvline(30e4, color='r')
#     ax.set_title(N_shocks - n)
#     print(N_shocks - n)
#     # plt.savefig(f'fig/{n}.png')
#     plt.pause(.1)
#
# fig, ax = plt.subplots(1, 1)
# ax.plot(ind_i_for_check, 'o')
# avg = np.mean(ind_i_for_check)
# ax.axhline(avg + center, color='k', linestyle='--', label='ppifg / 2')
# ax.axhline(avg - center, color='k', linestyle='--')
# ax.legend(loc='best')
# ax.set_title("shock location")
