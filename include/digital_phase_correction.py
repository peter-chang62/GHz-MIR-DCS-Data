import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt


normalize = lambda x: x / np.max(np.abs(x))


def rfft(x, axis=None):
    """
    Parameters
    ----------
    x : numpy array
        array on which to perform an fft.
    axis : int, optional
        axis over which to perform the fft. The default is None which is
        equivalent to axis=0
    Returns
    -------
    fft: numpy array
        fourier transform of x.

    """
    if axis is None:
        return np.fft.rfft(np.fft.fftshift(x))
    else:
        return np.fft.rfft(np.fft.fftshift(x, axes=axis), axis=axis)


def irfft(x, axis=None):
    """
    Parameters
    ----------
    x : numpy array
        array on which to perform an ifft.
    axis : int, optional
        axis over which to perform the ifft. The default is None which is
        equivalent to axis=0
    Returns
    -------
    fft: numpy array
        fourier transform of x.

    """
    if axis is None:
        return np.fft.ifftshift(np.fft.irfft(x))
    else:
        return np.fft.ifftshift(np.fft.irfft(x, axis=axis), axes=axis)


# useful for plotting and determining good apodization window
# to fit spectral phase
def get_phase(dat, N_apod, plot=True):
    """
    Parameters
    ----------
    dat : 1D numpy array
        one interferogram
    N_apod : int
        apodization window.
    plot : boolean, optional
        plot the phase and amplitude. The default is True.

    Returns
    -------
    freq : 1D numpy array
        frequency axis (-.5 to .5).
    phase : 1D numpy array
        phase in radians.
    spectrum: 1D numpy array
        spectrum
    """
    ppifg = len(dat)
    center = ppifg // 2
    zoom = dat[center - N_apod // 2 : center + N_apod // 2]
    fft = rfft(zoom)
    phase = np.unwrap(np.arctan2(fft.imag, fft.real))
    freq = np.fft.rfftfreq(len(zoom))

    if plot:
        plt.figure()
        plt.plot(freq, normalize(phase), ".-")
        plt.plot(freq, normalize(fft.__abs__()), ".-")
    return freq, phase, fft.__abs__()


def apply_t0_shift(pdiff, freq, fft):
    """
    Parameters
    ----------
    pdiff : Nx2 numpy array
        array that contains the linear fit coefficients of the fft phase.
    freq : 1D numpy array
        frequency axis that corresonds to pdiff. If pdiff is calculated from
        get_pdiff then the frequency axis runs from -.5 to .5
    fft : 2D numpy array
        2D numpy array with fft's of interferograms one each row.

    Returns
    -------
    None.

    """
    # the polynomial fits the spectral phase in radians,
    # so the factor of 2 pi is already there
    fft[:] *= np.exp(1j * freq * pdiff[:, 0][:, np.newaxis])


def apply_phi0_shift(pdiff, hbt):
    """

    Parameters
    ----------
    pdiff : Nx2 numpy array
        array that contains the linear fit coefficients of the fft phase.
    hbt : 2D numpy array
        hilbert transform of interferograms on each row.

    Returns
    -------
    None.

    """
    # the polynomial fits the spectral phase in radians,
    # so the factor of 2 pi is already there
    hbt[:] *= np.exp(1j * pdiff[:, 1][:, np.newaxis])


def get_pdiff(data, ll_freq, ul_freq, Nzoom=200):
    """
    Parameters
    ----------
    data : 2D array
        array of IFG's, row column order.
    ll_freq : float
        lower frequency limit for spectral phase fit, given on -.5 to .5 scale.
    ul_freq : float
        upper frequency limit for spectral phase fit, given on -.5 to .5 scale.
    Nzoom : integer, optional
        the apodization window. You need to apodize or else your SNR isn't good
        enough to have a good fit, so plot it first before specifying this
        parameter, generally 200 is pretty goodThe default is 200.
    Returns
    -------
    pdiff : 2D array
        polynomial coefficients for linear phase fit, given higher order first.
    """

    # apodize the data and subtract constant offset
    center = len(data[0]) // 2
    zoom = data[:, center - Nzoom // 2 : center + Nzoom // 2]
    zoom = (zoom.T - np.mean(zoom, 1)).T

    # if looking at things in the frequency domain, note that rfft takes an
    # fftshifted input, and returns an already ifftshifted output. The latter
    # occurs from the fact that it only returns the positive frequency side of
    # the fft
    fft = rfft(zoom, 1)
    freq = np.fft.rfftfreq(len(zoom[0]))
    ll, ul = np.argmin(abs(freq - ll_freq)), np.argmin(abs(freq - ul_freq))

    phase = np.unwrap(np.arctan2(fft.imag, fft.real))
    phase = phase.T  # column order for polynomial fitting
    p = np.polyfit(freq[ll:ul], phase[ll:ul], 1).T
    pdiff = p[0] - p

    return pdiff


def apply_t0_and_phi0_shift(pdiff, data, return_new=False):
    """
    Parameters
    ----------
    pdiff : Nx2 numpy array
        array that contains the linear fit coefficients of the fft phase.
    data : 2D numpy array
        1 interferogram per row.
    return_new: boolean, optional
        if True, then a new phase corrected array is returned. otherwise, data
        is altered in place

    Returns
    -------
    None.
    """
    freq = np.fft.rfftfreq(len(data[0]))
    fft = rfft(data, 1)
    apply_t0_shift(pdiff, freq, fft)
    td = irfft(fft, 1)

    hbt = ss.hilbert(td)
    apply_phi0_shift(pdiff, hbt)
    hbt = hbt.real

    if return_new:
        return hbt
    else:
        data[:] = hbt
