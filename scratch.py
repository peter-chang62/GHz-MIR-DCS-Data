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
H2CO = np.zeros((248, ppifg))
for n in range(248):
    co = pc.get_data(path_co, n)
    ind_i, ind_r = pc.get_ind_total_to_throw(co, ppifg)

    h2co = pc.get_data(path_h2co, n)
    h2co = h2co[ind_r:]
    h2co, _ = pc.adjust_data_and_reshape(h2co, ppifg)
    h2co = h2co[:100]

    h2co = pc.Phase_Correct(h2co, ppifg, 25, False)
    H2CO[n] = np.mean(h2co, 0)
    print(n)

# %%
FFT_H2CO = pc.fft(H2CO, 1)
DIFF = np.zeros(len(FFT_H2CO))
ll, ul = 1120 + ppifg // 2, 3600 + ppifg // 2

ref = FFT_H2CO[0]
freq = np.fft.fftfreq(len(ref))
for n, i in enumerate(FFT_H2CO):
    phase1 = np.unwrap(np.arctan2(ref[ll:ul].imag, ref[ll:ul].real))
    phase2 = np.unwrap(np.arctan2(i[ll:ul].imag, i[ll:ul].real))

    pfit1 = np.polyfit(np.arange(len(phase1)), phase1, 1)
    pfit2 = np.polyfit(np.arange(len(phase2)), phase2, 1)
    p1 = np.poly1d(pfit1)
    p2 = np.poly1d(pfit2)

    diff = pfit1 - pfit2

    remainder = diff % (2 * np.pi)
    diff = np.where(remainder < np.pi, 0, np.pi)

    phase_diff = diff[1]  # higher order first
    i *= np.exp(1j * phase_diff)
    FFT_H2CO[n] = i
    DIFF[n] = diff[1]  # higher order first

    print(n)

# %%
back = pc.ifft(FFT_H2CO, 1)

# %%
back = back.real
back = (back.T - np.mean(back, 1)).T
# back = (back.T / np.max(abs(back), 1)).T

# %%
back = pc.Phase_Correct(back, ppifg, 25, True)

# %%
plt.figure()
avg = np.mean(back, 0)
fft = pc.fft(avg)
plt.plot(fft.__abs__())

# %%
# fft1 = pc.fft(H2CO[0])
# fft2 = pc.fft(H2CO[-1])
#
# # %%
# ll, ul = 1120 + ppifg // 2, 3600 + ppifg // 2
# fft1 = fft1[ll:ul]
# fft2 = fft2[ll:ul]
#
# # %%
# phase1 = np.unwrap(np.arctan2(fft1.imag, fft1.real))
# phase2 = np.unwrap(np.arctan2(fft2.imag, fft2.real))
#
# # %%
# plt.plot(rad_to_deg(phase1))
# plt.plot(rad_to_deg(phase2))
#
# # %%
# pfit1 = np.polyfit(np.arange(len(phase1)), phase1, 1)
# pfit2 = np.polyfit(np.arange(len(phase1)), phase2, 1)
# p1 = np.poly1d(pfit1)
# p2 = np.poly1d(pfit2)
#
# # %%
# plt.plot(rad_to_deg(p1(np.arange(len(phase1)))))
# plt.plot(rad_to_deg(p2(np.arange(len(phase1)))))
#
# # %%
# fft1 = pc.fft(H2CO[0])
# fft2 = pc.fft(H2CO[-1])
# diff = pfit1 - pfit2
# phase_diff = diff[1]
# fft2 *= np.exp(1j * phase_diff)
#
# # %%
# plt.figure()
# plt.plot(fft1.real)
# plt.plot(fft2.real)
# plt.plot(fft1.imag)
# plt.plot(fft2.imag)
# plt.xlim(ppifg // 2, 14000)
# plt.ylim(-4000, 8000)
#
# # %%
# back = pc.ifft(fft2)
# plt.plot(H2CO[0])
# plt.plot(H2CO[-1])
# plt.plot(back.real)
