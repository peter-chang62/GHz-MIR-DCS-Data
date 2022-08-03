import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc
import digital_phase_correction as dpc
from sys import platform


def save_npy(path, filename, arr):
    with open(path + filename, 'wb') as f:
        np.save(f, arr)


if platform == "win32":  # on windows
    path_header = "E:\\"

else:  # otherwise on Pop!OS
    path_header = "/media/peterchang/Samsung_T5/"

# %%____________________________________________________________________________________________________________________
# data paths
path = path_header + r'MIR stuff/ShockTubeData/' \
                     r'DATA_MATT_PATRICK_TRIP_2/06-30-2022/Vacuum_Background_end_of_experiment/'
ppifg = 17506
center = ppifg // 2

# %%____________________________________________________________________________________________________________________
# load and resize the data
name_co = "4.5um_filter_114204x17506.bin"
name_h2co = "3.5um_filter_114204x17506.bin"
co = np.fromfile(path + name_co, '<h')
h2co = np.fromfile(path + name_h2co, '<h')
co = co[center:-center]
h2co = h2co[center:-center]
co.resize((114203, ppifg))
h2co.resize((114203, ppifg))

# %%____________________________________________________________________________________________________________________
# phase correct
ll_freq_h2co = 0.0597
ul_freq_h2co = 0.20
ll_freq_co = 0.15
ul_freq_co = 0.25

pdiff_co = dpc.get_pdiff(co, ll_freq_co, ul_freq_co, 200)
pdiff_h2co = dpc.get_pdiff(h2co, ll_freq_h2co, ul_freq_h2co, 200)

step = 250
h = 0
while h < len(co):
    dpc.apply_t0_and_phi0_shift(pdiff_co[h: h + step], co[h: h + step])
    dpc.apply_t0_and_phi0_shift(pdiff_h2co[h: h + step], h2co[h: h + step])
    print(len(co) - h)
    h += step

# %%____________________________________________________________________________________________________________________
# save phase corrected data
save_path = path + "PHASE_CORRECTED_DATA/"
avg_co = np.mean(co, 0)
avg_h2co = np.mean(h2co, 0)
save_npy(save_path, "4.5um_filter_phase_corrected.npy", avg_co)
save_npy(save_path, "3.5um_filter_phase_corrected.npy", avg_h2co)
