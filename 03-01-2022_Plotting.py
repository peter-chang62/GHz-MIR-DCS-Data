import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import mkl_fft

# %%
data = np.fromfile("Data/03-01-2022/H2CO_filter_phasecorr_9995x199728.bin", 'h')
data.resize(9995, 199728)
data = data.astype(np.float64)

# %%

# %%
for n, dat in enumerate(data):
    data[n] = (data[n] / (n + 1)) + (data[n - 1] * (n / (n + 1)))

    if n % 100 == 0:
        print(n)
