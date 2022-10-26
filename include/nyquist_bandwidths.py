import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import scipy.constants as sc


# clipboard_and_style_sheet.style_sheet()


def bandwidth(fr, dfr):
    return fr ** 2 / (2 * dfr)


def calc_num_for_mod_cond(num, denom):
    N_smaller = num // denom
    N_bigger = N_smaller + 1
    eps_bigger = N_bigger * denom - num
    eps_smaller = N_smaller * denom - num
    return num + eps_bigger, num + eps_smaller


def calc_denom_for_mod_cond(num, denom):
    N_smaller = num // denom
    N_bigger = N_smaller + 1
    eps_bigger = (num - N_bigger * denom) / N_bigger
    eps_smaller = (num - N_smaller * denom) / N_smaller
    return denom + eps_bigger, denom + eps_smaller


def calc_fr_for_ppifg(fr, dfr, ppifg):
    eps = ppifg * dfr - fr
    return fr + eps


def calc_dfr_for_ppifg(fr, dfr_guess, ppifg):
    eps = (fr - ppifg * dfr_guess) / ppifg
    return dfr_guess + eps


# if True, then aliasing is True (bad)
# if False, then aliasing is False (good)
def is_aliasing(vi, vf, dnu):
    n1 = vi // dnu
    n2 = vf // dnu + 1
    cond = n2 - n1 > 1
    return cond, n2 - n1, n2


def find_allowed_nyquist_bandwidths(vi, vf):
    """
    :param vi: start of power spectrum
    :param vf: end of power spectrum

    For Nyquist bandwidths that lie within the region: (vf - vi < dnu < vf), your Nyquist bandwidth is larger than
    the bandwidth of your power spectrum, but your signal does not lie within the first Nyquist zone. Here,
    you run the danger that your power spectrum might span the boundary between two Nyquist zones. As a result,
    in this region, you end up getting "windows" of allowed Nyquist bandwidths.

    For that region, this function returns a list of allowed Nyquist bandwidth windows
    [(start, end), (start, end), ...], i.e. any Nyquist bandwidth that is (start < dnu < end) will work

    * In the end, the allowed Nyquist bandwidths you can use are the ones returned by this function,
    and any bandwidth that is larger than vf *
    """

    dnu_min = vf - vi
    N_vf = int(vf // dnu_min)
    N_vi = int(vi // dnu_min)
    bounds_vi = vi / np.arange(1, N_vi + 1)
    bounds_vf = vf / np.arange(2, N_vf + 1)

    bounds_vf = bounds_vf[::-1]
    bounds_vi = bounds_vi[::-1]
    key = lambda i: i[0]
    bounds_list = [bounds_vf, bounds_vi]
    bounds_list.sort(key=key)
    dnu_windows = np.array(list(zip(*bounds_list)))
    return dnu_windows


def find_allowed_dfr(vi, vf, fr):
    dnu_windows = find_allowed_nyquist_bandwidths(vi, vf)
    dfr_windows = fr ** 2 / (2 * dnu_windows)
    dfr_windows = dfr_windows[::-1, ::-1]
    return dfr_windows


def return_allowed_indices_dnu(dnu, vi, vf):
    dnu_windows = find_allowed_nyquist_bandwidths(vi, vf)
    ind = np.hstack([np.logical_and(dnu > i[0], dnu < i[1]).nonzero()[0] for i in dnu_windows] + \
                    [(dnu > vf).nonzero()[0]])
    return ind
