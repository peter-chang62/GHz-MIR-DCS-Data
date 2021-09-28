import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

import numpy as np

class Plotting:
    """
    PLDPlot.py

    Plots things in a nice way
    """
    def __init__(self):
        # Set plot default to seaborn style
        plt.style.use('seaborn-muted')
        # Adjust font
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 14
        plt.rcParams['mathtext.fontset'] = 'cm'
        # Adjust line widths
        plt.rcParams["lines.markersize"] = 2.5*plt.rcParams["lines.linewidth"]
        self.defLW = plt.rcParams["lines.linewidth"]

    def plot(self, dataX, dataY, strX="Wavelength [nm]", strY="Absorbance [A.U.]", strLabels=None, fResid=False):
        """
        Plots dataY against dataX, with optional residuals subplot
        Assumes that dataX is common to all columns of dataY

        INPUTS:
            dataX  = data of x axis, numpy array of size [n]
            dataY  = data of y axis, numpy array of size [n,m]
            strX   = optional, x-label
            strY   = optional, y-label
            strLabels = data labels
            fResid = optional, whether to plot residuals as well but will only
                     work when more than one column of data is given

        OUTPUTS:
            None
        """

        self.fig = plt.figure()
        try:
            sizeData = dataY.shape
            for i in range(sizeData[1]):
                plt.plot(dataX, dataY[:,i])
        except AttributeError: # List of arrays rather than numpy array
            for spectrum in dataY:
                plt.plot(dataX, spectrum)

        plt.xlabel(strX)
        plt.ylabel(strY)
        plt.grid(True)

        ax = plt.gca()
        ax.grid(linestyle=':', alpha=0.75)
        if strLabels is not None:
            plt.legend(strLabels)
        plt.show()

