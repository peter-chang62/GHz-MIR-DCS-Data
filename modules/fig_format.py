# -*- coding: utf-8 -*-
"""
Matplotlib paper figure formatting

Created on Wed Sep 16 10:35:07 2020

@author: Nate the Average
"""

import matplotlib.pyplot as plt
from cycler import cycler

plt.rcParams.update({'figure.autolayout':False,'lines.linewidth':0.8})
# additional formatting for paper
plt.rcParams['font.sans-serif'][0] = 'Arial'
plt.rcParams['font.size'] = 8.5
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.labelright'] = False

# color sequence change defaults
default_cycle = plt.rcParams['axes.prop_cycle'] # list of colors
new_cycle = cycler(color=['r','g','b','y','k'])
plt.rc('axes', prop_cycle=new_cycle)