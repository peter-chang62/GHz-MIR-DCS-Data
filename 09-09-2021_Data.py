import scipy.ndimage.interpolation as sni
import numpy as np
import matplotlib.pyplot as plt
import GaGeIFGProcessing as gp
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()

normalize = lambda vec: vec / np.max(abs(vec))


def plot_section(arr, npts, npts_sec):
    if len(arr.shape) > 1:
        [plt.plot(i[npts // 2 - npts_sec: npts // 2 + npts_sec]) for i in arr]
    else:
        plt.plot(arr[npts // 2 - npts_sec:npts // 2 + npts_sec])


# get the points per interferogram
google_drive_path = "G:/.shortcut-targets-by-id" \
                    "/1x47gJys_dhP6gubqgzefkAm5X9gEfoKP/GHz MIR DCS" \
                    "Data/210921/"
file = "interferogram_8.asc"
path = google_drive_path + file

ifg = gp.IFG(path, read_chunk=True, chunksize=2e6)
npts = ifg.ppifg()

# using ppifg to get the data
it = ifg.get_iter(chunksize=npts, offset=npts // 2)

# pull 100 interferograms
IFG = []
for n in range(100):
    IFG.append(next(it).values)

# pull all the interferograms
# IFG = [i.values for i in ifg.get_iter(chunksize=npts, offset=npts // 2)]
# length = np.array([len(i) for i in IFG])
# if not np.all(length == length[0]):
#    IFG = IFG[:-1]

# only average 10 interferograms
# IFG = IFG[0:10]

IFG = np.array(IFG)
Y = IFG[:, :, 1]
Y = (Y.T - np.mean(Y, 1)).T

# calculate the shifts needed
Sec = Y[:, npts // 2 - 200: npts // 2 + 200]
FTSec = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(Sec, axes=1), axis=1),
                         axes=1)
FTSec = np.pad(FTSec, ((0, 0), (2000, 2000)), constant_values=0.)
ref = FTSec[0]
ref = np.repeat(ref[:, np.newaxis], len(FTSec), 1).T
corr = ref * np.conj(FTSec)
corr = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(corr, axes=1), axis=1),
                        axes=1)
corr = corr.real
best = np.argmax(corr, 1) - len(ref[0]) // 2
shift = best * len(Sec[0]) / len(ref[0])

# shift using fft
FT = np.fft.fft(np.fft.fftshift(Y, axes=1), axis=1)
f = np.fft.fftfreq(len(FT[0]))
phase = (np.repeat(f[:, np.newaxis], len(FT), 1) * shift).T
FT *= np.exp(-1j * 2 * np.pi * phase)
final = np.fft.ifftshift(np.fft.ifft(FT, axis=1), axes=1).real
avg = np.mean(final.real, 0)

# or shifting by interpolating directly in time domain
# final = np.zeros(Y.shape)
# for n, i in enumerate(Y):
#     final[n] = sni.shift(i, shift[n], order=1)
# averaged = np.mean(final, 0)

# calculate the FFT of the averaged ifg
avg_fft = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(avg)))
T = IFG[:, :, 0]
dT = np.mean(np.diff(T, axis=1))
freq = np.fft.fftshift(np.fft.fftfreq(len(avg_fft), dT))

# get rid of f0 or else plotting is annoying
ind = np.logical_and(freq >= -300e6, freq <= 300e6).nonzero()
# avg_fft[ind]=0

# save the averaged data
# np.hstack([T[0][:, np.newaxis], avg[:, np.newaxis]]).tofile(
#     "background_avg" + str(len(avg)) + ".txt")
