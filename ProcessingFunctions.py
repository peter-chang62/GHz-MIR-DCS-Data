import numpy as np
import matplotlib.pyplot as plt

"""This find the points per interferogram method works so long as you don't have only a few points per interferogram. 
the returned points per interferogram is the average of the distance between interferogram maxes """


def _get_initial_npts_guess(dat, level_percent=40):
    level = np.max(dat) * level_percent * .01
    ind = (dat > level).nonzero()[0]
    dind = np.diff(ind)
    return int(np.round(np.mean(dind[dind > 1000])))


def _find_npts_from_guess(dat, guess):
    N = len(dat[guess // 2:]) // guess
    arr = dat[guess // 2:][:N * guess]
    arr = arr.reshape(N, guess)
    ind = np.argmax(arr, axis=1)
    xind = np.arange(len(ind)) * guess + ind
    N = np.mean(np.diff(xind))
    return int(np.round(N)), N


def find_npts(dat, level_percent=40):
    guess = _get_initial_npts_guess(dat, level_percent)
    return _find_npts_from_guess(dat, guess)


def plot_section(arr, npts, npts_plot):
    for i in range(1, len(arr) // npts):
        plt.plot(arr[npts // 2:][npts * (i - 1):npts * i][npts // 2 - npts_plot:npts // 2 + npts_plot])
