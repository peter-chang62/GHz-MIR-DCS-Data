import matplotlib.pyplot as plt
import numpy as np
import clipboard_and_style_sheet
import mkl_fft
import phase_correction as pc
import scipy.integrate as si
import scipy.signal as ss


def apply_t0_shift(pdiff, freq, fft):
    # the polynomial fits the spectral phase in radians,
    # so the factor of 2 pi is already there
    fft[:] *= np.exp(1j * freq * pdiff[:, 0][:, np.newaxis])


def apply_phi0_shift(pdiff, hbt):
    # the polynomial fits the spectral phase in radians,
    # so the factor of 2 pi is already there
    hbt[:] *= np.exp(1j * pdiff[:, 1][:, np.newaxis])


def get_pdiff(data, ppifg, ll_freq, ul_freq, Nzoom=200):
    """
    :param data: 2D array of IFG's, row column order
    :param ppifg: int, length of each interferogram
    :param ll_freq: lower frequency limit for spectral phase fit, given on -.5 to .5 scale
    :param ul_freq: upper frequency limit for spectral phase fit, given on -.5 to .5 scale
    :param Nzoom: the apodization window for the IFG, don't worry about f0 since you are fitting the spectral phase,
    not doing a cross-correlation, you need to apodize or else your SNR isn't good enough to have a good fit, so
    plot it first before specifying this parameter, generally 200 is pretty good
    :return: pdiff, polynomial coefficients, higher order first
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

    return pdiff


# %%
path_h2co = r'D:\ShockTubeData\04242022_Data\Surf_27\card2/'
ppifg = 17507
center = ppifg // 2
data = np.fromfile(r'D:\ShockTubeData\04242022_Data\Vacuum_Background/card2_114204x17507.bin', '<h')
data, _ = pc.adjust_data_and_reshape(data, ppifg)

ll_freq = 0.0597
ul_freq = 0.20

# %%
pdiff = get_pdiff(data, ppifg, ll_freq, ul_freq, 200)

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

    hbt = ss.hilbert(td)
    apply_phi0_shift(pdiff[h: h + step], hbt)
    hbt = hbt.real

    data[h:h + step] = hbt.real

    h += step
    print(len(data) - h)

# %%
avg = np.mean(data, 0)
fft = pc.fft(avg)
