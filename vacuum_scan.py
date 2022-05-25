import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc
import digital_phase_correction as dpc

clipboard_and_style_sheet.style_sheet()

# %%
ppifg = 17507
center = ppifg // 2
# data = np.fromfile(r'D:\ShockTubeData\04242022_Data\Vacuum_Background/card2_114204x17507.bin', '<h')
data = np.fromfile(r'D:\ShockTubeData\04242022_Data\Vacuum_Background/card1_114204x17507.bin', '<h')
data, _ = pc.adjust_data_and_reshape(data, ppifg)

# %%
ll_freq_h2co = 0.0597
ul_freq_h2co = 0.20
ll_freq_co = 0.15
ul_freq_co = 0.25
# pdiff_h2co = dpc.get_pdiff(data, ppifg, ll_freq_h2co, ul_freq_h2co, 200)
pdiff_co = dpc.get_pdiff(data, ppifg, ll_freq_co, ul_freq_co, 200)

# %%
h = 0
step = 250
while h < len(data):
    # dpc.apply_t0_and_phi0_shift(pdiff_h2co[h: h + step], data[h: h + step])
    dpc.apply_t0_and_phi0_shift(pdiff_co[h: h + step], data[h: h + step])

    h += step
    print(len(data) - h)

avg = np.mean(data, 0)
fft = pc.fft(avg)
