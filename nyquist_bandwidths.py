import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import scipy.constants as sc

clipboard_and_style_sheet.style_sheet()

start = sc.c / 4.55e-6
end = sc.c / 3.45e-6


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


def calc_dfr_for_ppifg(fr, dfr, ppifg):
    eps = (fr - ppifg * dfr) / ppifg
    return dfr + eps


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
    bounds_vf = end / np.arange(2, N_vf + 1)

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

# %%
# fr = 1e9
# allowed_dnu_windows = find_allowed_nyquist_bandwidths(start, end)
# allowed_dfr_windows = find_allowed_dfr(start, end, fr)
#
# dfr = np.linspace(50, fr ** 2 / (2 * (end - start)), 5000)
# dnu = bandwidth(fr, dfr)
# ind = return_allowed_indices_dnu(dnu, start, end)
#
# x = np.zeros(len(dfr))
# plt.plot(dfr[ind], x[ind], '.')
# [plt.axvline(i) for i in allowed_dfr_windows.flatten()]
# plt.axvline(fr ** 2 / (2 * end))
#
# """Given an fr an dfr, we can also calculate the nearest fr, or nearest dfr that would give us an integer ppifg.
#
# In the following, fr and dfr do not give an integer ppifg. However, if we change fr -> corr_fr1 or fr -> corr_fr2
# and keep the same dfr, then we will get an integer ppifg.
#
# Likewise, if we change dfr -> corr_dfr1 or dfr -> corr_dfr2 and keep the same fr, then we will get an integer ppifg
#
# In the above, the two possible corrected rep rates, or two possible corrected delta freps give ppifg that differ
# by one.
#
# Generally it is easier to apply corrections to dfr, since those corrections are much much smaller than
# corrections to fr. This is expected since ppifg = fr / dfr and fr >> dfr.
#
# Secondly, if applying corrections to dfr it is important that when applying corrections you keep fr1 = fr fixed,
# and only change fr2 = fr + dfr > fr1 """
#
# # fr = 1e9
# # dfr = 23e3
# # corr_dfr1, corr_dfr2 = calc_denom_for_mod_cond(fr, dfr)
# # corr_fr1, corr_fr2 = calc_num_for_mod_cond(fr, dfr)
# # print("original ppifg: ", fr / dfr)
# # print("corrected ppifg fr/corr_dfr1 :", fr / corr_dfr1)
# # print("corrected ppifg fr/corr_dfr2 :", fr / corr_dfr2)
# # print("corrected ppifg corr_fr1/dfr :", corr_fr1 / dfr)
# # print("corrected ppifg corr_fr2/dfr :", corr_fr2 / dfr)
