import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc
import digital_phase_correction as dpc
import scipy.signal as sig
import nyquist_bandwidths as nq

clipboard_and_style_sheet.style_sheet()

# %% ___________________________________________________________________________________________________________________
# phase correction for shocks
# path = r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\06-23-2022\battalion_5/"
# ppifg = 15046
# center = ppifg // 2
# ll_freq, ul_freq = 0.2878, 0.3917
#
# N_shocks = pc.Number_of_files(path)
# Data = np.zeros((N_shocks, 70, ppifg))
# for n in range(N_shocks):
#     data = pc.get_data(path, n)
#     data, _ = pc.adjust_data_and_reshape(data, ppifg)
#     data = data.flatten()
#     ind_i, ind_r = pc.get_ind_total_to_throw(data, ppifg)
#     ind = int(np.round(ind_i / ppifg) * ppifg)
#     data = data[ind - ppifg * 20: ind + ppifg * 50]
#     data = data.reshape((70, ppifg))
#
#     if n == 0:
#         ref = data[0]
#         p = dpc.get_pdiff(data, ll_freq, ul_freq, 400)
#         dpc.apply_t0_and_phi0_shift(p, data)
#
#     else:
#         data = np.vstack([ref, data])
#         p = dpc.get_pdiff(data, ll_freq, ul_freq, 400)
#         dpc.apply_t0_and_phi0_shift(p, data)
#         data = data[1:]
#
#     Data[n] = data
#     print(N_shocks - n)
#
# avg = np.mean(Data, 0)
# ftavg = pc.fft(avg, 1)

# %% ___________________________________________________________________________________________________________________
# phase correction for vacuum background
# path = r'E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\06-23-2022\Vacuum_Background/'
# bckgnd = np.fromfile(path + 'vacuum_background_132804x15046.bin', '<h')
# bckgnd = bckgnd[ppifg // 2: - ppifg // 2]
# bckgnd.resize((132804 - 1, 15046))
# ppifg = 15046
# center = ppifg // 2
#
# h = 0
# step = 250
# p = dpc.get_pdiff(bckgnd, ll_freq, ul_freq, 400)
# while h < len(bckgnd):
#     dpc.apply_t0_and_phi0_shift(p[h: h + step], bckgnd[h: h + step])
#     h += step
#     print(len(bckgnd) - h)
#
# bckgnd_avg = np.mean(bckgnd, 0)
# bckgnd_ft = pc.fft(bckgnd_avg)

# %% ___________________________________________________________________________________________________________________
# plotting the saved results
avg = np.load('06-23-2022_batt5_phase_corrected_average.npy')
ftavg = pc.fft(avg, 1)
bckgnd_avg = np.load('06-23-2022_vacuum_background.npy')
bckgnd_ft = pc.fft(bckgnd_avg)
absorbance = -np.log(ftavg[19].__abs__() / bckgnd_ft.__abs__())

ppifg = 15046
center = ppifg // 2

fr = 1010e6 - 10007604.8
nq_nu = center * fr
N_window = 9
nu_axis = np.linspace(nq_nu * (N_window - 1), nq_nu * N_window, center)
wl_axis = sc.c * 1e6 / nu_axis

# plotting
fig, ax = plt.subplots(1, 1)
ax.plot(wl_axis[20:], ftavg[19][center:][20:].__abs__(), label='data')
ax.plot(wl_axis[20:], bckgnd_ft[center:][20:].__abs__(), label='background')
ax.get_yaxis().set_visible(True)
ax.set_xlabel("$\lambda \; (\mathrm{\mu m})$")
ax.set_xlim(4.45, 4.82)
ax.set_ylim(0, 5000)
ax.legend(loc='best')

fig2, ax2 = plt.subplots(1, 1)
ax2.plot(wl_axis, absorbance[center:])
ax2.set_xlabel("$\lambda \; (\mathrm{\mu m})$")
ax2.set_ylabel("absorbance")
ax2.set_xlim(4.45, 4.82)
ax2.set_ylim(-.5, 4)
