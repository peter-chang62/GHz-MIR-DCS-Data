import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import figure_out_phasecorr as pc


def rad_to_deg(rad):
    return rad * 180 / np.pi


def deg_to_rad(deg):
    return deg * np.pi / 180


# %%
path_co = r'D:\ShockTubeData\Data_04232022\Surf_18\card1/'
path_h2co = r'D:\ShockTubeData\Data_04232022\Surf_18\card2/'
ppifg = 17511

# %%
co = pc.get_data(path_co, 0)
ind_i, ind_r = pc.get_ind_total_to_throw(co, ppifg)
h2co = pc.get_data(path_h2co, 0)
h2co = h2co[ind_r:]
h2co, ind = pc.adjust_data_and_reshape(h2co, ppifg)

# %%
# array = np.hstack([h2co[0], h2co[1]])
# array = array[ppifg - 500: ppifg + 500]

array = np.copy(h2co[0])
# array[ppifg // 2 - 25:ppifg // 2 + 25] = 0
array = array[ppifg // 2 - 500: ppifg // 2 + 500]

fft_array = pc.fft(array)

center = len(fft_array) // 2
max1 = np.argmax(fft_array[center:].__abs__()) + center
max2 = np.argmax(fft_array[max1 + 10:].__abs__()) + max1 + 10

freq = np.fft.fftshift(np.fft.fftfreq(len(fft_array)))
freq1 = freq[max1]
freq2 = freq[max2]
val1_pos = fft_array[max1]
val2_pos = fft_array[max2]
val1_neg = fft_array[2 * center - max1]
val2_neg = fft_array[2 * center - max2]

# plt.figure()
# plt.plot(freq, fft_array.__abs__(), '.')
# plt.axvline(freq1, color='r')
# plt.axvline(freq2, color='r')
# plt.axvline(-freq1, color='r')
# plt.axvline(-freq2, color='r')

N = len(fft_array)
t = np.arange(-N // 2, N // 2)


def signal(t, freq1, freq2, val1_neg, val1_pos, val2_neg, val2_pos):
    return np.exp(1j * 2 * np.pi * freq1 * t) * val1_pos + np.exp(- 1j * 2 * np.pi * freq1 * t) * val1_neg + \
           np.exp(1j * 2 * np.pi * freq2 * t) * val2_pos + np.exp(- 1j * 2 * np.pi * freq2 * t) * val2_neg


s = signal(t, freq1, freq2, val1_neg, val1_pos, val2_neg, val2_pos) / N
plt.figure()
plt.plot(array)
plt.plot(s.real)

# %%
plt.figure()
plt.plot(array)
plt.plot(array - s.real)

filtered_fft = pc.fft(array - s.real)
plt.figure()
plt.plot(filtered_fft.__abs__())
