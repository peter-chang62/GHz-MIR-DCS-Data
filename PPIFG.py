import gc

import numpy as np
import pandas as pd


class IFG:
    def __init__(self, path, read_chunk=False, chunksize=None,
                 level_percent=20.):
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
        self.level = max(self.y) * level_percent * .01
        self.N = None
        self.Nm = None
        self.h = None
        self.step = int(1e3)

        self.thresh = 100
        self.h = 1

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
        _ = np.ceil(len(self.y) / self.step) * self.step
        y_ = np.pad(self.y, (0, int(_ - len(self.y))), constant_values=0)
        y__ = abs(y_.reshape(len(y_) // self.step, self.step))
        ind = np.argmax(y__, axis=1)
        maxes = np.max(y__, axis=1)
        xind = np.arange(len(ind)) * self.step + ind
        xind_ = xind[maxes > self.level]
        self.N = np.diff(xind_)
        self.xind = xind_
        if np.std(self.N) > self.thresh:
            self.N = self.N[self.N > 100]
            self.step = int(np.round(np.mean(self.N)))
            print("trying again, ", self.h)
            if self.h > 3:
                self.thresh *= 2
            gc.collect()
            self.h += 1
            self._N_spacing()
        else:
            self.h = 1


class IFG_from_memory:
    def __init__(self, data, level_percent=20.):
        self.x = data[:, 0]
        self.y = data[:, 1]
        self.level = max(self.y) * level_percent * .01

        self.N = None
        self.Nm = None
        self.h = None
        self.step = int(1e5)

        self.thresh = 100
        self.h = 1

    def ppifg(self):
        return IFG.ppifg(self)

    def _N_spacing(self):
        return IFG._N_spacing(self)


def get_dataframe(path, read_chunk=False, chunksize=None, offset=0):
    if read_chunk:
        frame = pd.read_csv(path, sep='\t', skiprows=13 + offset,
                            chunksize=chunksize, header=None)
    else:
        frame = pd.read_csv(path, sep='\t', skiprows=13 + offset,
                            header=None)
    return frame
