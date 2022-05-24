import matplotlib.pyplot as plt
import numpy as np
import clipboard_and_style_sheet
import mkl_fft
import phase_correction as pc
import scipy.integrate as si
import scipy.signal as ss


def apply_t0_shift(pdiff, freq, fft):
    fft[:] *= np.exp(1j * freq * pdiff[:, 0][:, np.newaxis])


def apply_phi0_shift(pdiff, hbt):
    hbt[:] *= np.exp(1j * pdiff[:, 1][:, np.newaxis])


# def apply_f0_shift(fcdiff, hbt, t):
#     hbt[:] *= np.exp(1j * 2 * np.pi * fcdiff[:, np.newaxis] * t)


def get_pdiff(data, ppifg, ll_freq, ul_freq, Nzoom=200):
    """
    :param data:
    :param ppifg:
    :param ll_freq:
    :param ul_freq:
    :return: pdiff, fft
    """

    center = ppifg // 2
    zoom = data[:, center - Nzoom // 2:center + Nzoom // 2]
    zoom = (zoom.T - np.mean(zoom, 1)).T

    # not fftshifted
    fft = pc.fft(zoom, 1)
    freq = np.fft.fftshift(np.fft.fftfreq(len(fft[0])))
    ll, ul = np.argmin(abs(freq - ll_freq)), np.argmin(abs(freq - ul_freq))

    phase = np.unwrap(np.arctan2(fft.imag, fft.real))
    phase = phase.T  # column order for polynomial fitting
    p = np.polyfit(freq[ll:ul], phase[ll:ul], 1).T
    pdiff = p[0] - p

    apply_t0_shift(pdiff, freq, fft)
    return pdiff, fft


# def get_fcdiff(fft, ll_freq, ul_freq):
#     """
#     :param data:
#     :param ppifg:
#     :param ll_freq:
#     :param ul_freq:
#     :return: fcdiff, hilbert.real
#     """
#
#     freq = np.fft.fftshift(np.fft.fftfreq(len(fft[0])))
#     ll, ul = np.argmin(abs(freq - ll_freq)), np.argmin(abs(freq - ul_freq))
#     freq = np.repeat(freq[ll:ul][:, np.newaxis], len(fft), axis=1).T
#
#     num = si.simps(freq * fft[:, ll:ul].__abs__(), freq, axis=1)
#     denom = si.simps(fft[:, ll:ul].__abs__(), freq, axis=1)
#     fc = num / denom
#     fcdiff = fc[0] - fc
#
#     return fcdiff


# def get_all_corrections(data, ppifg, ll_freq, ul_freq, Nzoom=200):
#     pdiff1, fft1 = get_pdiff(data, ppifg, ll_freq, ul_freq, Nzoom)
#     fcdiff = get_fcdiff(fft1, ll_freq, ul_freq)
#
#     td = pc.ifft(fft1, 1).real
#     hilbert = ss.hilbert(td, axis=1)
#     t = np.arange(-len(td[0]) // 2, len(td[0]) // 2, 1)
#     apply_f0_shift(fcdiff, hilbert, t)
#     hilbert = hilbert.real
#
#     pdiff2, fft2 = get_pdiff(hilbert, hilbert.shape[1], ll_freq, ul_freq, Nzoom)
#
#     return pdiff1, fcdiff, pdiff2


# %%
path_h2co = r'D:\ShockTubeData\04242022_Data\Surf_27\card2/'
ppifg = 17507
center = ppifg // 2
data = np.fromfile(r'D:\ShockTubeData\04242022_Data\Vacuum_Background/card2_114204x17507.bin', '<h')
data, _ = pc.adjust_data_and_reshape(data, ppifg)

ll_freq = 0.0597
ul_freq = 0.20

# %%
# pdiff1, fcdiff, pdiff2 = get_all_corrections(data, ppifg, ll_freq, ul_freq, 200)
pdiff, _ = get_pdiff(data, ppifg, ll_freq, ul_freq, 200)

# %%
h = 0
step = 250
freq = np.fft.fftshift(np.fft.fftfreq(len(data[0])))
t = np.arange(-len(freq) // 2, len(freq) // 2)
while h < len(data[::]):
    sec = data[h: h + step]

    fft = pc.fft(sec, 1)
    apply_t0_shift(pdiff[h: h + step], freq, fft)
    td = pc.ifft(fft, 1).real

    # hilbert = ss.hilbert(td)
    # apply_f0_shift(fcdiff[h: h + step], hilbert, t)
    # td = hilbert.real

    hilbert = ss.hilbert(td)
    apply_phi0_shift(pdiff[h: h + step], hilbert)
    hilbert = hilbert.real

    sec = hilbert.real
    data[h:h + step] = sec

    h += step
    print(len(data) - h)
