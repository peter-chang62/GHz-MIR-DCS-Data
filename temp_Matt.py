import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc
import digital_phase_correction as dpc

path_co = r'E:\MIR stuff\ShockTubeData\04242022\Surf_27\card1/'
ppifg = 17507
center = ppifg // 2

N_shocks = pc.Number_of_files(path_co)
ind_Delta_T = np.zeros(N_shocks)
for n in range(N_shocks):
    co = pc.get_data(path_co, n)
    co, ind_throw = pc.adjust_data_and_reshape(co, ppifg)
    assert ppifg > ind_throw
    bckgnd = co.copy()
    bckgnd[:, center - 50: center + 50] = 0
    bckgnd = bckgnd.flatten()
    ind_i = np.argmax(bckgnd)
    ind_Delta_T[n] = ind_i + ind_throw
    print(N_shocks - n)

frep = 1010e6 - 9998061.1
dt = 1 / frep
Delta_T = ind_Delta_T * dt
Delta_T -= .0008
avg_us = np.mean(Delta_T) * 1e6
