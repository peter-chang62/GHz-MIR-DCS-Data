import numpy as np
import matplotlib.pyplot as plt

"""This find the points per interferogram method works so long as you don't have only a few points per interferogram. 
the returned points per interferogram is the average of the distance between interferogram maxes """


def _get_initial_npts_guess(dat, level_percent=40):
    level = np.max(abs(dat)) * level_percent * .01
    ind = (dat > level).nonzero()[0]
    dind = np.diff(ind)
    return int(np.round(np.mean(dind[dind > 1000]))), level


def _find_npts_from_guess(dat, guess):
    N = len(dat[guess // 2:]) // guess
    arr = dat[guess // 2:][:N * guess]
    arr = arr.reshape(N, guess)
    ind = np.argmax(arr, axis=1)
    xind = np.arange(len(ind)) * guess + ind
    N = np.mean(np.diff(xind))
    return int(np.round(N)), N


def _find_npts(dat, level_percent=40):
    guess, level = _get_initial_npts_guess(dat, level_percent)
    return [*_find_npts_from_guess(dat, guess), level]


# def find_npts(dat, level_percent=40):
#     N, N_float, level = _find_npts(dat, level_percent)
#     return _find_npts(dat[N // 2:-N // 2], level_percent)


def plot_section(arr, npts, npts_plot):
    for i in range(1, len(arr) // npts):
        plt.plot(arr[npts // 2:][npts * (i - 1):npts * i][npts // 2 - npts_plot:npts // 2 + npts_plot])


def find_npts(dat, level_percent=40):
    level = np.max(abs(dat)) * level_percent * .01
    ind = (dat > level).nonzero()[0]
    diff = np.diff(ind)
    ind_diff = (diff > 1000).nonzero()[0]

    h = 0
    trial = []
    for i in (ind_diff + 1):
        trial.append(ind[h:i])
        h = i

    # plt.figure()
    # plt.plot(dat)
    # [plt.plot(i, dat[i], 'o') for i in trial]

    ind_maxes = []
    for i in trial:
        ind_maxes.append(i[np.argmax(dat[i])])

    # plt.plot(ind_maxes, dat[ind_maxes], 'ko')

    mean = np.mean(np.diff(ind_maxes))

    return int(np.round(mean)), mean, level
