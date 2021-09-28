from __future__ import print_function
from __future__ import division

import numpy as np

try:
    from . import pldhapi as hapi
    from .constants import SPEED_OF_LIGHT, MOLECULE_IDS
except (ValueError, SystemError, ImportError):
    from . import pldhapi as hapi
    from constants import SPEED_OF_LIGHT, MOLECULE_IDS


class SpectraSim:
    """
    PLDSim.py

    Handles the simulation of various spectra

    """
    def __init__(self, strDataFolder = "linelists"):
        """
        Constructs and downloads molecule information

        TODO:
         - Check for existing folders
        """
        # Create data folder
        hapi.db_begin(strDataFolder)
        self.wvn_start = None
        self.y_absorbance = []
        self.num_molecs = 0

    def def_environment(self, temperature, pressure, pathlength, nm_start, nm_stop, step=0.01):
        """
        Defines the environmental variables for the simulation

        INPUTS:
            temperature = system temperature, [K]
            pressure    = system pressure, [atm]
            pathlength  = system pathlength, [m]
            nm_start    = start of the spectral range, [nm]
            nm_stop     = stop of the spectral range, [nm]
            step        = stepsize of simulation; defaults to 0.01, [cm-1]

        OUTPUTS:
            None

        """
        assert nm_stop > nm_start, "Stop wavelength must be larger than the start"

        self.temperature = temperature
        self.pressure    = pressure
        self.pathlength  = pathlength

        # Pull in spectral fitting region, but allow this code to guess which units
        if(nm_start > 1e12):
            print("Assuming def_environment() frequencies given in Hz")
            self.nm_start = 1e7 * SPEED_OF_LIGHT / nm_start
            self.nm_stop  = 1e7 * SPEED_OF_LIGHT / nm_stop
            self.wvn_start = nm_start / SPEED_OF_LIGHT
            self.wvn_stop  = nm_stop / SPEED_OF_LIGHT
        elif(nm_start > 5000):
            # Assume wavenumbers in near-IR
            print("Assuming def_environment() wavenumbers given in cm-1")
            self.wvn_start = nm_start
            self.wvn_stop = nm_stop
            self.nm_start = 1e7 / nm_start
            self.nm_stop  = 1e7 / nm_stop
        elif(nm_start > 500):
            # Assume wavelengths in nm
            print("Assuming def_environment() wavelengths given in nm")
            self.nm_start = nm_start
            self.nm_stop = nm_stop
            self.wvn_start = 1e7 / nm_start
            self.wvn_stop  = 1e7 / nm_stop
        else:
            print("Assuming def_environment() wavelengths given in microns")
            self.nm_start = 1000 * nm_start
            self.nm_stop = 1000 * nm_stop
            self.wvn_start = 1e7 / self.nm_start
            self.wvn_stop = 1e7 / self.nm_stop

        if self.wvn_start > self.wvn_stop:
            wvn_stop = self.wvn_start
            self.wvn_start = self.wvn_stop
            self.wvn_stop = wvn_stop

        self.wvn_step     = step             # [cm-1]
        self.nm_step     = step/(1e7)*1e9   # [nm]


        self.x_wvn      = np.arange(self.wvn_start, self.wvn_stop, step)
        self.x_nm      = 1e7/self.x_wvn

        # If running inside Fitting object, update the fit parameters dictionary as well
        try:
            self.fit_params.add('Temp',value = temperature, min = 290, max = 1600, vary = False)
            # can modify these floats and ranges later by saying,
            #  self.fit_params['Temp].vary = True
            self.fit_params.add('press', value=pressure, min=pressure*0.1, max=pressure*3, vary=False)
            self.fit_params.add('Pathlength', value = pathlength, vary=False)
        except:
            pass

    def def_molecules(self, molec_names, mole_fractions, dbSource = None):
        """
        Downloads molecule information and stores them in self.listMolecs

        INPUTS:
            molec_names = array of molecule strings
            mole_fractions = array of mole fractions
            dbSource    = list of linelist sources, eg if all from HitranOnline, use 'HitranOnline',
                          otherwise make array of .data file names to search for.
                          eg: if getting H2O from custom linelist called 'H2O_Labfit.data' in /data directory,
                              but CO2 coming from HitranOnline, then call
                              def_molecules(['H2O','CO2'],[0.1,0.1], ['H2O_Labfit',None])

        OUTPUTS:
            None

        """
        # Verify correct inputs
        assert isinstance(molec_names, list), "A list of variables, even just one, needs to be used as input"
        assert len(mole_fractions) == len(molec_names), "The input array lengths do not match"
        assert self.wvn_start is not None, "Run def_environment() first to establish the spectral range"

        # Verify that all of the specified molecules exist in the HITRAN database
        fMolecExist = all(elem in MOLECULE_IDS for elem in molec_names)
        if not fMolecExist:
            raise ValueError("One or more of the input molecules do not exist in HITRAN, please see print_Hitran() for a possible list")

        # Store the values
        self.listStrMolecs = molec_names
        self.listMoleFrac  = mole_fractions
        self.num_molecs     = len(molec_names)
        self.num_molecIso   = np.ones((self.num_molecs, )) # Default the isotopes to 1
        # Create list of HITRAN molecule reference numbers
        # and fetch the HITRAN data
        self.listMolecsDbNum = np.zeros((self.num_molecs,))
        for i in range(self.num_molecs):
            self.listMolecsDbNum[i] = MOLECULE_IDS[molec_names[i]]
            hapi.fetch(molec_names[i], self.listMolecsDbNum[i], self.num_molecIso[i], self.wvn_start-1, self.wvn_stop+1)

        # Define 2D matrix of absorbance values
#        self.y_absorbance = np.zeros((int((self.wvn_stop - self.wvn_start)/self.wvn_step)+1, self.num_molecs))

        # If running inside Fitting object, update the fit parameters dictionary as well
        if dbSource is None:
            dbSource = []
            for i in range(self.num_molecs):
                dbSource.append(None)
        try:
            for index, Xi in enumerate(mole_fractions):
                self.fit_params.add('X_'+molec_names[index], value = Xi, min = 0.1*Xi, max = min(1,10*Xi), vary = True)
                # Now custom add attributes to this dictionary key for fitting
                self.fit_params['X_'+molec_names[index]].molID = self.listMolecsDbNum[index]
                self.fit_params['X_'+molec_names[index]].iso = self.num_molecIso[index]
                if dbSource is not None:
                    self.fit_params['X_'+molec_names[index]].dbSource = dbSource[i]
        except:
            pass

        # Make an object with all the multispecies molecule attributes
        i = 0
        self.molec = Molecules({molec_names[i]:[self.listMolecsDbNum[i], self.num_molecIso[i],
                                         'Voigt',dbSource[i]]})
        for i in range(1,self.num_molecs):
            self.molec.md.update({molec_names[i]:[self.listMolecsDbNum[i], self.num_molecIso[i],
                                         'Voigt',dbSource[i]]})

    def def_diluent(self, moleFracCO2=0):
        """
        Creates a dict of diluent

        INPUTS:
            moleFracCO2 = mole fraction of CO2, defaults to 0

        OUTPUTS:
            None

        """
        #self.diluent = {'self':molefractions[jj],'air':1-molefractions[jj], 'co2':0}
        print("Currently not used")

    def simulate(self):
        """
        Generates the absorbance spectra with x-axis in wavelength [nm]

        INPUTS:
            None
            But you should run 2 other functions first:
                def_environment() and def_molecules()

        OUTPUTS:
            self.y_absorbance = Spectral data as numpy array,
                             different column for each absorbing molecule
                             produced in def_molecules()
            (self.x_nm wavelength array already produced in def_environment())

        """
        assert len(self.listStrMolecs) > 0, "Must have molecules defined. Run def_molecules() and try again"
        assert self.temperature is not None, "You should run def_environment() to define your T,P,L"
        assert self.pressure is not None
        assert self.pathlength is not None

        if self.pressure > 2:
            print('Assuming pressure is ' + repr(self.pressure) +
                  " atm. Convert from Torr or mbar to atm in def_environment() if you don't want this.")

#        self.y_absorbance = np.zeros((self.stop_pnt - self.start_pnt, self.num_molecs))
        self.y_absorbance = []
        # Loop through defined molecules and calculate the absorption coefficients
        for i in range(self.num_molecs):
            [nu,coefs] = hapi.absorptionCoefficient_Voigt(((self.listMolecsDbNum[i], self.num_molecIso[i], self.listMoleFrac[i]),), (self.listStrMolecs[i]),
                OmegaStep=self.wvn_step, OmegaRange=[self.wvn_start, self.wvn_stop], HITRAN_units=False,
                Environment={'p':self.pressure,'T':self.temperature},Diluent={'self':self.listMoleFrac[i],'air':1-self.listMoleFrac[i], 'co2':0},
                IntensityThreshold = 0)

            self.y_absorbance.append(coefs * self.pathlength)

#            [nu1, absorp] = hapi.absorptionSpectrum(nu, coefs,Environment ={'l':self.pathlength*M_2_CM, 'p':self.pressure*BAR_2_ATM,'T':self.temperature})
#            self.y_absorbance[:,i] = -np.log(1-absorp)  # Converts from absorption to absorbance

    def print_Hitran(self):
        """
        Prints the list of available molecules in HITRAN
        """
        print("Available molecules:\n")
        print(*MOLECULE_IDS, sep=', ')



class Molecules:
    '''
    For spectral fitting, want to associate extra attributes to each molecular species.
    This class produces a nice way to pass these attributes to the fitting function.
    '''
    def __init__(self, molecule_dict):
        '''
        Add a set of fit molecules with the appropriate attributes

        EG, for H2O would have molID = 1, iso = 1, dbSource = None
        Or for custom H2O from ArH2O_Labfit.data file, would have
            molID = 1, iso = 1, dbSource = 'ArH2O_Labfit'

        '''

        molID = None
        iso = None
        model = 'Voigt'
        dbSource = None

        self.md = molecule_dict
        self.items = molecule_dict.items

        for mol, vals in molecule_dict.items():
            self.__setattr__(mol, vals)


    def __getitem__(self, i):
        return self.md[i]

    def list_code(self):
        return("Molecule ID, isotope ID, Voigt / other lineshape model, linelistFileName (for non-HitranOnline)")
