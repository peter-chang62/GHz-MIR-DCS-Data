import numpy as np
import phase_correction as pc
import scipy.signal as ss
import matplotlib.pyplot as plt


# useful for plotting and determining good apodization window
# to fit spectral phase
def get_phase(dat, N_apod, plot=True, linestyle='-'):
    ppifg = len(dat)
    center = ppifg // 2
    fft = pc.fft(dat[center - N_apod // 2: center + N_apod // 2])
    phase = np.unwrap(np.arctan2(fft.imag, fft.real))
    freq = np.fft.fftshift(np.fft.fftfreq(len(phase)))

    if plot:
        plt.figure()
        plt.plot(freq, pc.normalize(phase), linestyle=linestyle)
        plt.plot(freq, pc.normalize(fft.__abs__()), linestyle=linestyle)
    return freq, phase, fft.__abs__()


def apply_t0_shift(pdiff, freq, fft):
    # the polynomial fits the spectral phase in radians,
    # so the factor of 2 pi is already there
    fft[:] *= np.exp(1j * freq * pdiff[:, 0][:, np.newaxis])


def apply_quadr_phase(p, freq, fft, f0):
    # the polynomial fits the spectral phase in radians,
    # so the factor of 2 pi is already there
    fft[:] *= np.exp(1j * (freq - f0) ** 2 * p[:, 0][:, np.newaxis])


def apply_phi0_shift(pdiff, hbt):
    # the polynomial fits the spectral phase in radians,
    # so the factor of 2 pi is already there
    hbt[:] *= np.exp(1j * pdiff[:, 1][:, np.newaxis])


def fit_phase_quadr(data, ll_freq, ul_freq, Nzoom=200):
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

    center = len(data[0]) // 2
    zoom = data[:, center - Nzoom // 2:center + Nzoom // 2]
    zoom = (zoom.T - np.mean(zoom, 1)).T

    # not fftshifted
    fft = pc.fft(zoom, 1)
    freq = np.fft.fftshift(np.fft.fftfreq(len(fft[0])))
    ll, ul = np.argmin(abs(freq - ll_freq)), np.argmin(abs(freq - ul_freq))

    f0 = (ul_freq - ll_freq) / 2 + ll_freq

    phase = np.unwrap(np.arctan2(fft.imag, fft.real))
    phase = phase.T  # column order for polynomial fitting
    p = np.polyfit(freq[ll:ul] - f0, phase[ll:ul], 2).T  # fit up to a quadratic

    return p


def apply_t0_and_phi0_shift(pdiff, data):
    freq = np.fft.fftshift(np.fft.fftfreq(len(data[0])))
    fft = pc.fft(data, 1)
    apply_t0_shift(pdiff, freq, fft)
    td = pc.ifft(fft, 1).real

    hbt = ss.hilbert(td)
    apply_phi0_shift(pdiff, hbt)
    hbt = hbt.real

    data[:] = hbt.real


# %% ___________________________________________________________________________________________________________________
ppifg = 198850
center = ppifg // 2
data = np.fromfile(r'D:\ShockTubeData\static cell/cell_with_mixture_and_4_5_filter_at200_4998x198850.bin', '<h')
data, _ = pc.adjust_data_and_reshape(data, ppifg)

ll_freq, ul_freq = 0.325, 0.345  # 4.5 um

# %% ___________________________________________________________________________________________________________________
data = data[:400]
p_quad = fit_phase_quadr(data, ll_freq, ul_freq, 400)
freq = np.fft.fftshift(np.fft.fftfreq(len(data[0])))
f0 = (ul_freq - ll_freq) / 2 + ll_freq
fft = pc.fft(data, 1)
apply_quadr_phase(- p_quad, freq, fft, f0)
data_corr = pc.ifft(fft, 1).real
