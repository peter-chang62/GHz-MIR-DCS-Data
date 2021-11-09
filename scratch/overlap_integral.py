import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class MFD:
    def __init__(self):
        mfd_data = np.genfromtxt("SM_INF3_MFD.txt")
        self.wl_um = mfd_data[:, 0]
        self.mfd = mfd_data[:, 1]

        self.grid = interp1d(self.wl_um, self.mfd, 'cubic', bounds_error=True)

def func(r):
    pass
