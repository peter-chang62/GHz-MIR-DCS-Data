import numpy as np
import pandas as pd


class IFG:
    def __init__(self, path, read_chunk=False, chunksize=None, level=.025):
        dataframe = get_dataframe(path, read_chunk, chunksize)
        self.iter = None
        if read_chunk:
            self.iter = dataframe
            self.dataframe = next(self.iter)
        else:
            self.dataframe = dataframe

        self.path = path
        self.read_chunk = read_chunk
        self.x = self.dataframe.values[:, 0]
        self.y = self.dataframe.values[:, 1]
        self.level = level
        self.N = None
        self.Nm = None
        self.h = None
        self.step = int(1e5)

    def next_frame(self):
        if not self.read_chunk:
            print("dataframe is not an iterable")
            return

        self.dataframe = next(self.iter)
        self.x = self.dataframe.values[:, 0]
        self.y = self.dataframe.values[:, 1]

    def get_iter(self, chunksize, offset=0):
        if not self.read_chunk:
            print("dataframe is not an iterable")
            return

        return get_dataframe(self.path, True, chunksize, offset)

    def ppifg(self):
        self._N_spacing()
        self.Nm = int(np.round(np.mean(self.N)))
        return self.Nm

    def _N_spacing(self):
        self.h = 1
        _ = np.ceil(len(self.y) / self.step) * self.step
        y_ = np.pad(self.y, (0, int(_ - len(self.y))), constant_values=0)
        y__ = y_.reshape(len(y_) // self.step, self.step)
        ind = np.argmax(y__, axis=1)
        maxes = np.max(y__, axis=1)
        xind = np.arange(len(ind)) * self.step + ind
        xind_ = xind[maxes > self.level]

        self.N = np.diff(xind_)
        if np.std(self.N) > 100:
            self.N = self.N[self.N > 100]
            self.step = int(np.round(np.mean(self.N)))
            self._N_spacing()
            self.h += 1


def get_dataframe(path, read_chunk=False, chunksize=None, offset=0):
    if read_chunk:
        frame = pd.read_csv(path, sep='\t', skiprows=13 + offset,
                            chunksize=chunksize, header=None)
    else:
        frame = pd.read_csv(path, sep='\t', skiprows=13 + offset,
                            header=None)
    return frame
