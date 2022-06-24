import numpy as np
import phase_correction as pc
import scipy.signal as ss
import matplotlib.pyplot as plt


# useful for plotting and determining good apodization window
# to fit spectral phase
def get_phase(dat, N_apod, plot=True):
    ppifg = len(dat)
    center = ppifg // 2
    fft = pc.fft(dat[center - N_apod // 2: center + N_apod // 2])
    phase = np.unwrap(np.arctan2(fft.imag, fft.real))
    freq = np.fft.fftshift(np.fft.fftfreq(len(phase)))

    if plot:
        plt.figure()
        plt.plot(freq, pc.normalize(phase), '.-')
        plt.plot(freq, pc.normalize(fft.__abs__()), '.-')
    return freq, phase, fft.__abs__()


def apply_t0_shift(pdiff, freq, fft):
    # the polynomial fits the spectral phase in radians,
    # so the factor of 2 pi is already there
    fft[:] *= np.exp(1j * freq * pdiff[:, 0][:, np.newaxis])


def apply_phi0_shift(pdiff, hbt):
    # the polynomial fits the spectral phase in radians,
    # so the factor of 2 pi is already there
    hbt[:] *= np.exp(1j * pdiff[:, 1][:, np.newaxis])


def get_pdiff(data, ll_freq, ul_freq, Nzoom=200):
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

    phase = np.unwrap(np.arctan2(fft.imag, fft.real))
    phase = phase.T  # column order for polynomial fitting
    p = np.polyfit(freq[ll:ul], phase[ll:ul], 1).T
    pdiff = p[0] - p

    return pdiff


def apply_t0_and_phi0_shift(pdiff, data):
    freq = np.fft.fftshift(np.fft.fftfreq(len(data[0])))
    fft = pc.fft(data, 1)
    apply_t0_shift(pdiff, freq, fft)
    td = pc.ifft(fft, 1).real

    hbt = ss.hilbert(td)
    apply_phi0_shift(pdiff, hbt)
    hbt = hbt.real

    data[:] = hbt.real
