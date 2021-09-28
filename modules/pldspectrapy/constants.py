"""
constants.py

Similar to a header file in C++, this script will contain a multitude of
useful constants to be referenced.

ALL UNITS MUST BE BASE SI.
"""
try: # python 2
    range = xrange
except NameError: # python 3
    pass
try: # python 2
    import itertools.izip as zip
except ImportError: # python 3
    pass
from collections import OrderedDict

SPEED_OF_LIGHT = 299792458 # (m/s)

BAR_2_ATM = 0.986923 # (atm/bar)
M_2_CM    = 100 # (cm/m)

HITRAN_160 = {
        # Formatting guide to pull from PAR files to be consistent with hapi
        'molec_id': {'index':(0,2), 'par_format': '%2d',
                     'labfit_id': 0, 'labfit_format': '%3d',
                     'description': 'Hitran indicator, 1 for H2O'},
        'local_iso_id': {'index':(2,3), 'par_format':'%1d',
                         'labfit_id': 1, 'labfit_format': '%2d',
                         'description': 'isotopologue, almost always 1'},
        'nu': {'index':(3,15), 'par_format': '%12.6f',
               'labfit_id': 2, 'labfit_format': '%14.7f',
               'description': 'linecenter (cm-1)'},
        'sw': {'index':(15,25), 'par_format': '%10.3E',
               'labfit_id': 3, 'labfit_format': '%13.5E',
               'description': ' S296 (cm-1/atm)'},
        'gamma_air': {'index':(35,40), 'par_format': '%5.4f',
                      'labfit_id': 4, 'labfit_format': '%10.5f',
                      'description': 'foreign-broadening (cm-1/atm)'},
        'gamma_self': {'index':(40,45), 'par_format': '%5.3f',
                       'labfit_id': 10, 'labfit_format': '%10.5f',
                       'description': 'self-broadening half-width-half-max (cm-1/atm)'},
        'elower': {'index': (45,55), 'par_format': '%10.4f',
                   'labfit_id': 5, 'labfit_format': '%14.7f',
                   'description': 'lower-state energy E" (cm-1)'},
        'n_air': {'index': (55,59), 'par_format': '%4.2f',
                  'labfit_id': 6, 'labfit_format': '%8.4f',
                  'description': 'foreign-broadening temperature exponent'},
        'delta_air': {'index': (59,67), 'par_format': '%8.6f',
                      'labfit_id': 7, 'labfit_format': '%11.7f',
                      'description': 'foreign-shift (cm-1/atm)'},
        'quanta': {'index': (67, 127), 'par_format': '%60s',
                   'labfit_id': 17, 'labfit_format': '%60s', # for 4-line Labfit
                   'description': 'quantum numbers for upper-vibration, lower-vibration, upper-rotation, lower-rotation'}
        }

MOLECULE_NAMES = ('H2O','CO2','O3','N2O','CO','CH4','O2','NO','SO2','NO2','NH3',
                  'HNO3','OH','HF','HCl','HBr','HI','ClO','OCS','H2CO','HOCl',
                  'N2','HCN','CH3Cl','H2O2','C2H2','C2H6','PH3','COF2','SF6',
                  'H2S','HCOOH','HO2','O','ClONO2','NO+','HOBr','C2H4','CH3OH',
                  'CH3Br','CH3CN','CF4','C4H2','HC3N','H2','CS','SO3')
# convert from name to HITRAN id using MOLECULE_NAMES.index(name) + 1
# eg to get H2O = 1, use MOLECULE_NAMES.index('H2O') + 1
MOLECULE_IDS = OrderedDict(zip(MOLECULE_NAMES, range(1, len(MOLECULE_NAMES)+1)))