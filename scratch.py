import numpy as np
import mkl_fft
import PyQt5.QtWidgets as qt
from scipy.signal import windows as wd
import os
import matplotlib.pyplot as plt


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


def get_ind_total_to_throw(data, ppifg):
    center = ppifg // 2

    # %% skip to the max of the first interferogram, and then ppifg // 2 after that
    start = data[:ppifg]
    ind_THREW_OUT = np.argmax(start)
    data = data[ind_THREW_OUT:]
    N = len(data) // ppifg
    data = data[:N * ppifg]
    data = data[ppifg // 2: - ppifg // 2]
    N = len(data) // ppifg
    data = data.reshape(N, ppifg)

    # %% how do you find the start of the transient?
    hey = np.copy(data)
    # remove all the interferograms
    hey[:, center - 50:center + 50] = 0.0
    # the first maximum of the baseline gives the incident wave
    ind_incident = np.argmax(hey.flatten()[:int(3e6)])
    # clear the incident wave, and look for the reflected one
    # don't let it look too far, in case there are subsequent shocks from reflection off the far end wall
    skip = ind_incident + int(1e4)
    ind_reflected = np.argmax(hey.flatten()[skip:skip + int(2e5)]) + skip

    # %% to reference the time location of the shock wave of H2CO based on CO:
    # skip to the max of the first interferogram, then ppifg // 2 after that, and then add on ind_reflected
    IND_TOTAL_TO_THROW = ind_THREW_OUT + ppifg // 2 + ind_reflected

    return IND_TOTAL_TO_THROW, hey, ind_incident, ind_reflected


def analyze(path, ppifg, N_toanalyze, plot=True, IND_TOTAL_TO_THROW=None, N_zoom=50):
    center = ppifg // 2

    # %%
    names = [i.name for i in os.scandir(path)]
    key = lambda f: int(f.split('LoopCount_')[1].split('_Datetime')[0])
    names = sorted(names, key=key)

    # %% load a single data file and throw out the time stamp
    data = np.fromfile(path + names[N_toanalyze], '<h')
    data = data[:-64]

    if IND_TOTAL_TO_THROW is None:
        plot_hey = True
        IND_TOTAL_TO_THROW, hey, ind_incident, ind_reflected = get_ind_total_to_throw(data, ppifg)
    else:
        plot_hey = False

    data = data[IND_TOTAL_TO_THROW:]

    # %% skip to the max of the first interferogram, and then NEGATIVE or POSITIVE ppifg // 2 after that
    start = data[:ppifg]
    ind_THREW_OUT = np.argmax(abs(start))
    if ind_THREW_OUT > ppifg // 2:
        ind_THREW_OUT -= ppifg // 2
    elif ind_THREW_OUT < ppifg // 2:
        ind_THREW_OUT += ppifg // 2

    data = data[ind_THREW_OUT:]
    N = len(data) // ppifg
    data = data[:N * ppifg]
    N = len(data) // ppifg
    data = data.reshape(N, ppifg)

    # %% zoomed in data
    zoom = data[:, center - 200: center + 201].astype(float)
    zoom = (zoom.T - np.mean(zoom, 1)).T

    # %% appodize to remove f0, use a window of size 50
    window = wd.blackman(N_zoom)
    left = (len(zoom[0]) - N_zoom) // 2
    right = len(zoom[0]) - N_zoom - left
    window = np.pad(window, (left, right), constant_values=0)
    zoom_appod = zoom * window

    # %% calculate the shifts
    fft_zoom = fft(zoom_appod, 1)
    ref = fft_zoom[0]
    fft_zoom *= np.conj(ref)
    fft_zoom = np.pad(fft_zoom, ([0, 0], [2 ** 10, 2 ** 10]), constant_values=0.0)
    fft_zoom = ifft(fft_zoom, 1)
    ind = np.argmax(fft_zoom, axis=1) - len(fft_zoom[0]) // 2
    shift = ind * len(zoom[0]) / len(fft_zoom[0])

    # %% shift correct data
    ft = np.fft.fft(np.fft.ifftshift(data, axes=1), axis=1)
    freq = np.fft.fftfreq(len(ft[0]))
    phase = np.zeros(data.shape).astype(np.complex128)
    phase[:] = 1j * 2 * np.pi * freq
    phase = (phase.T * shift).T
    phase = np.exp(phase)
    ft *= phase
    ft = np.fft.fftshift(np.fft.ifft(ft, axis=1), axes=1)
    phase_corr = ft.real

    # %% a lot of diagnostic plots
    if plot:
        # check the phase correction
        fig, ax = plt.subplots(1, 2, figsize=np.array([11.9, 4.8]))
        [ax[1].plot(i[center - 100:center + 100]) for i in phase_corr[:50]]
        [ax[1].plot(i[center - 100:center + 100]) for i in phase_corr[-50:]]
        [ax[0].plot(i[center - 100:center + 100]) for i in data[:50]]
        [ax[0].plot(i[center - 100:center + 100]) for i in data[-50:]]
        ax[0].set_title("un corrected")
        ax[1].set_title("corrected")

        # a view of the appodization method for removal of f0
        fig, ax = plt.subplots(1, 2, figsize=np.array([11.9, 4.8]))
        ax[0].plot(normalize(zoom[0]))
        ax[0].plot(window)
        ax[1].plot(normalize(zoom[0] * window))

        # average time domain
        plt.figure()
        avg = np.mean(phase_corr, 0)
        plt.plot(avg)

        # average frequency domain
        plt.figure()
        plt.plot(fft(avg).__abs__())

        if plot_hey:
            # a view of where the program thought the reflected shock was
            plt.figure()
            plt.plot(hey.flatten())
            plt.axvline(ind_incident, color='r')
            plt.axvline(ind_reflected, color='r')
            start = ind_reflected + ind_THREW_OUT
            plt.plot(np.arange(start, start + len(data.flatten())), data.flatten())

    return phase_corr, IND_TOTAL_TO_THROW


# %%
path_co = r'C:\Users\fastdaq\Desktop\ShockTubeData\Surf_18\card1/'
path_h2co = r'C:\Users\fastdaq\Desktop\ShockTubeData\Surf_18\card2/'
ppifg = 17511

# %%
co, ind = analyze(path_co, ppifg, 15, True)
h2co, _ = analyze(path_h2co, ppifg, 15, True, ind, N_zoom=25)
