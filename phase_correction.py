import numpy as np
from scipy.signal import windows as wd
import os
import matplotlib.pyplot as plt

try:
    import mkl_fft
except:
    mkl_fft = np.fft


def fft(x, axis=None):
    """
    calculates the 1D fft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if axis is None:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def ifft(x, axis=None):
    """
    calculates the 1D ifft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if axis is None:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def normalize(vec):
    return vec / np.max(abs(vec))


def rad_to_deg(rad):
    return rad * 180 / np.pi


def deg_to_rad(deg):
    return deg * np.pi / 180


def Number_of_files(path):
    names = [i.name for i in os.scandir(path)]
    return len(names)


def get_data(path, N_file):
    """
    :param path: path to data folder containing all the segment files
    :param N_file: which segment file to analyze (starting from 1)
    :return: data as a 1D array
    """
    names = [i.name for i in os.scandir(path)]
    key = lambda f: int(f.split('LoopCount_')[1].split('_Datetime')[0])
    names = sorted(names, key=key)

    # load a single data file and throw out the time stamp
    data = np.fromfile(path + names[N_file], '<h')
    data = data[:-64]

    return data


def get_ind_total_to_throw(data, ppifg):
    """
    :param data: data as a 1D array
    :param ppifg: points per interferogram (int)

    :return:
    the index that marks the incident shock,
    the index that marks the reflected shock
    """
    center = ppifg // 2

    # skip to the max of the first interferogram, and then ppifg // 2 after that
    start = data[:ppifg]
    ind_THREW_OUT = np.argmax(start)
    data = data[ind_THREW_OUT:]
    N = len(data) // ppifg
    data = data[:N * ppifg]
    data = data[ppifg // 2: - ppifg // 2]
    N = len(data) // ppifg
    data = data.reshape(N, ppifg)

    # how do you find the start of the transient?
    bckgnd = np.copy(data)
    # remove all the interferograms
    bckgnd[:, center - 50:center + 50] = 0.0
    # the first maximum of the baseline gives the incident wave
    ind_incident = np.argmax(bckgnd.flatten()[:int(3e6)])
    # clear the incident wave, and look for the reflected one
    # don't let it look too far, in case there are subsequent shocks from reflection off the far end wall
    skip = ind_incident + int(1e4)
    ind_reflected = np.argmax(bckgnd.flatten()[skip:skip + int(2e5)]) + skip

    # skip to the max of the first interferogram, then ppifg // 2 after that, and then add on ind_reflected
    ind_incident += ind_THREW_OUT + ppifg // 2
    ind_reflected += ind_THREW_OUT + ppifg // 2

    return ind_incident, ind_reflected


def adjust_data_and_reshape(data, ppifg):
    """
    :param data:
    :param ppifg:

    :return: data truncated (both start and end) to an integer number of interferograms and reshaped.
    because it might be important to know, I also return the number of points I truncated
    from the start of the data
    """

    # skip to the max of the first interferogram, and then NEGATIVE or POSITIVE ppifg // 2 after that
    start = data[:ppifg]
    ind = np.argmax(abs(start))
    if ind > ppifg // 2:
        ind -= ppifg // 2
    elif ind < ppifg // 2:
        ind += ppifg // 2

    data = data[ind:]
    N = len(data) // ppifg
    data = data[:N * ppifg]
    N = len(data) // ppifg
    data = data.reshape(N, ppifg)
    return data, ind


def shift_2d(data, shifts):
    ft = mkl_fft.fft(np.fft.ifftshift(data, axes=1), axis=1)  # fftshifted
    freq = np.fft.fftfreq(len(ft[0]))
    phase = np.zeros(data.shape).astype(np.complex128)
    phase[:] = 1j * 2 * np.pi * freq
    phase = (phase.T * shifts).T
    phase = np.exp(phase)
    ft *= phase
    ft = np.fft.fftshift(mkl_fft.ifft(ft, axis=1), axes=1)
    phase_corr = ft.real
    return phase_corr


def t0_correct_via_cross_corr(data, N_zoom=50, plot=True):
    """
    :param data: data as a 2D array
    :param ppifg: points per interferogram
    :param N_zoom: number of data points in the zoomed in window from calculating the phase correction
    :param plot: plot the phase correction (diagnostics)

    :return: phase corrected data as a 2D array
    """
    center = len(data[0]) // 2

    # zoomed in data
    zoom = data[:, center - (N_zoom + 0): center + (N_zoom + 1)].astype(float)
    zoom = (zoom.T - np.mean(zoom, 1)).T

    # appodize to remove f0, use a window of size 50
    window = wd.blackman(N_zoom)
    left = (len(zoom[0]) - N_zoom) // 2
    right = len(zoom[0]) - N_zoom - left
    window = np.pad(window, (left, right), constant_values=0)

    WINDOWS = np.zeros(zoom.shape)
    for n, i in enumerate(zoom):
        ind = np.argmax(abs(i) ** 2)
        roll = ind - len(zoom[0]) // 2
        WINDOWS[n] = np.roll(window, roll)

    zoom_appod = zoom * WINDOWS

    # calculate the shifts
    fft_zoom = fft(zoom_appod, 1)
    ref = fft_zoom[0]
    fft_zoom *= np.conj(ref)
    fft_zoom = np.pad(fft_zoom, ([0, 0], [2 ** 10, 2 ** 10]), constant_values=0.0)
    fft_zoom = ifft(fft_zoom, 1)
    ind = np.argmax(fft_zoom, axis=1) - len(fft_zoom[0]) // 2
    shift = ind * len(zoom[0]) / len(fft_zoom[0])

    # shift correct data
    phase_corr = shift_2d(data, shift)

    if plot:
        # a view of the appodization method for removal of f0
        fig, ax = plt.subplots(1, 2, figsize=np.array([11.9, 4.8]))
        ax[0].plot(normalize(zoom[0]))
        ax[0].plot(WINDOWS[0])
        ax[1].plot(normalize(zoom_appod[0]))

        # check the phase correction
        fig, ax = plt.subplots(1, 2, figsize=np.array([11.9, 4.8]))
        [ax[1].plot(i[center - 100:center + 100]) for i in phase_corr[:50]]
        [ax[1].plot(i[center - 100:center + 100]) for i in phase_corr[-50:]]
        [ax[0].plot(i[center - 100:center + 100]) for i in data[:50]]
        [ax[0].plot(i[center - 100:center + 100]) for i in data[-50:]]
        ax[0].set_title("un corrected")
        ax[1].set_title("corrected")

    return phase_corr, shift
