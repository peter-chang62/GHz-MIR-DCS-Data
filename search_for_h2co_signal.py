import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import phase_correction as pc

h2co = np.fromfile('h2co_surf_27_and_28_avged_across_shocks_50x17507.bin')
h2co_bckgnd = np.fromfile('h2co_vacuum_background.bin')

co = np.fromfile('co_surf_27_and_28_avged_across_shocks_50x17507.bin')
co_bckgnd = np.fromfile('co_vacuum_background.bin')

h2co.resize((50, 17507))
co.resize((50, 17507))

# %%
fft_h2co_bckgnd = pc.fft(h2co_bckgnd).__abs__()

fig, ax = plt.subplots(1, 1)
# ll, ul = 12800, 13150
ll, ul = 11400, 12400
for i in h2co[:20]:
    ax.clear()
    fft = pc.fft(i)
    ax.plot(-np.log(fft[ll:ul].__abs__() / fft_h2co_bckgnd[ll:ul].__abs__()))
    # ax.set_ylim(0, .55)
    ax.set_ylim(0, .4)
    plt.pause(1)
