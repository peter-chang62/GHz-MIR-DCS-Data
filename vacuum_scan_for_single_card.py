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
path = path_header + r'MIR stuff\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\06-21-2022\Vacuum_Background/'
name = 'vacuum_background_132804x15046.bin'
ppifg = 15046
center = ppifg // 2

data = np.fromfile(path + name, '<h')
data = data[center: - center]
data.resize((132803, 15046))

# %%____________________________________________________________________________________________________________________
ll, ul = 0.275, 0.39
pdiff = dpc.get_pdiff(data, ll, ul, 200)

step = 250
h = 0
while h < len(data):
    dpc.apply_t0_and_phi0_shift(pdiff[h: h + step], data[h: h + step])
    print(len(data) - h)
    h += step

# %%____________________________________________________________________________________________________________________
save_path = path + "PHASE_CORRECTED_DATA/"
avg = np.mean(data, 0)
save_npy(save_path, "vacuum_background.npy", avg)
