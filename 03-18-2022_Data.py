import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet

# %%
data = np.fromfile("Data/03-18-2022/CO_56khz_dfrep_223860x35736.bin", '<h')
ppifg = 35736 // 2
data = data[ppifg // 2: - ppifg // 2]
data = data.reshape((223860 - 1, ppifg))

# %%
N_zoom = 200

ref = data[0, ppifg // 2 - N_zoom: ppifg // 2 + N_zoom + 1]
ftref = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(ref)))
ftref = np.pad(ftref, ([2 ** 11, 2 ** 11]), constant_values=0)

omega = np.fft.fftfreq(ppifg) * 2 * np.pi

for n, i in enumerate(data[::]):
    # zoom in and fft 400 points around the centerburst
    zoom = i[ppifg // 2 - N_zoom: ppifg // 2 + N_zoom + 1]
    zoom = zoom - np.mean(zoom)
    fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(zoom)))

    # filter out f0
    fft[np.argmax(fft[:len(fft) // 2].__abs__())] = 0.0
    fft[np.argmax(fft[len(fft) // 2:].__abs__()) + len(fft) // 2] = 0.0

    # calculate the cross-correlation and its maximum
    fft = np.pad(fft, ([2 ** 11, 2 ** 11]), constant_values=0)
    fft *= ftref.conj()
    fft = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(fft)))
    shift = (np.argmax(fft.real) - len(fft) // 2) * len(zoom) / len(fft)

    # shift the data using fft
    fft = np.fft.fft(np.fft.ifftshift(data[n]))
    fft *= np.exp(1j * omega * shift)
    data[n] = np.fft.fftshift(np.fft.ifft(fft)).real

    # progress update
    if n % 100 == 0:
        print(f'{n} / {len(data)}')

# %% plotting
avg = np.mean(data, 0)
fftavg = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(avg)))

t = np.arange(-ppifg // 2, ppifg // 2)
freq = np.fft.fftshift(np.fft.fftfreq(ppifg, 1e-9) * 1e-6)
plt.figure()
plt.plot(t, avg)
plt.figure()
plt.plot(freq, fftavg.__abs__())
plt.ylim(0, 35000)
plt.xlim(0)

# %% saved phase corrected data
data.tofile(f"Data/03-18-2022/CO_56khz_dfrep_PHASE_CORRECTED_{223860 - 1}x{ppifg}.bin")
