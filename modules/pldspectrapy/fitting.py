from lmfit import Model, Parameters
import numpy as np

try:
    from . import pldhapi as hapi
    from .sim import SpectraSim # Inheritance: nest this object for coupling to hapi
    from .combtools import CombTools
    from .constants import SPEED_OF_LIGHT, M_2_CM
except (ValueError, SystemError, ImportError):
    import pldhapi as hapi
    from sim import SpectraSim # Inheritance: nest this object for coupling to hapi
    from combtools import CombTools
    from constants import SPEED_OF_LIGHT, M_2_CM


class Fitting(CombTools, SpectraSim):
    """
    Fitting.py

    Fits DCS spectra using HAPI and lmfit, and also holds the spectra data but not the IG data.

    By default lmfit uses the Levenberg-Marquadt fitting routine. This could be explored further
    for non-quadratic cost functions, which would likely involve using lmfit's Minimize function.

    -TODO:
       Run on transmission or absorption
       RUn baseline_remove() with initial guess spectrum rather than no-spectrum fit

    """

    def __init__(self):

        CombTools.__init__(self)
        SpectraSim.__init__(self)

        # Data containers
        self.data_spectra = None # Numpy matrix of spectral data in columns
        self.THz_step  = None # Scaling of data wave in [THz]
        self.num_spectra  = None # Number of spectral data columns

        # Environmental parameters
        self.temperature  = None # [K]
        self.pressure     = None # [bar]
        self.pathlength   = None # [m]
        # Add species as necessary

#        self.params = Parameters() # initialize fitting object

        # Laser parameters for frequency axis
        self.x_nm  = None # [nm]
        self.x_wvn = None # [cm^{-1}]
        self.x_Hz = None # [THz]
        self.wvn_start     = None # [cm^{-1}]
        self.x_data = None
        self.absorption_meas = [] # list of absorption measurement arrays

        # Fitting parameters for baseline_remove()
        self.bl_order    = 5    # number of Chebyshev baseline coeffs across fit spectrum
        self.chunks     = 1    # number of polynomial chunks across baseline
        self.bl_type     = 'Chebyshev' # Set to 'Chebyshev' or 'Taylor' for Chebyshev or Taylor polynomial
        self.fit_transmission = False # Either fit to absorbance or normalized transmission
                                     # Absorbance = -ln(I/I0), transmission = I/I0

        # Fitting parameters for fit()
        self.bl_order_fit = self.bl_order

        # Or pass along everything in nested object
#        self.hapiSims = SpectraSim() # this is confusing buggy version of inheritance
#        self.hapiSims.fit_params = Parameters() # also want everything in
        self.fit_params = Parameters()


    def fit(self):
        """
        Fits the spectroscopic data with HAPI and lmfit. The user can optionally provide windowing
        vectors, which may improve the results. Additionally, a different core fitting routine can
        be input.

        This function is really the leftover fitting setup structure not covered in def_environment() or def_molecules()

        INPUTS:
            absorptionData   = baseline-corrected absorption measurement as single-column numpy array.

        OUTPUTS:
            self.result output object, containing best things

        ASSUMPTIONS:
            self.XXXX is already coarsely baseline-corrected in absorbance, not transmission

        ERRORS:

        TODO:
            - Fill out
            Might want to set up much of the Parameters() as nested object when calling def_environment()
              with default float ranges + options for custom in def_environment() call.
            Then just pass the self.params object to a svelte fitting function here.
        """

        assert self.temperature is not None, "Run def_environment() to set up thermodynamic model"
        assert self.num_molecs is not None, "Run def_molecules() to add absorbing species"
        # Already run def_environment() before calling this function, so self.pressure, etc. must also exist

        ## Set up fit variables
        # Already set up thermodynamic parameters in def_environment and def_molecules.
        # edit whether floating molecules here...

        # And add baseline parameters
        for ii in range(self.bl_order_fit):
            self.fit_params.add('Cheb'+str(ii), value = 0,vary=True)
            if ii > 0: # assign fit range
                self.fit_params['Cheb' + str(ii)].min = -0.01
                self.fit_params['Cheb' + str(ii)].max =  0.01
        self.fit_params.add('shift', value = 0, min = -1, max = 1, vary = False)

        ## Set up measurement
        # and add y-scaling
#        x_data = self.x_Hz[self.start_pnt:self.stop_pnt]/SPEED_OF_LIGHT # SPEED_OF_LIGHT defined in SpectralConstants
#        self.x_data = x_data
        if len(self.absorption_meas) is 0:
            try:
                self.y_data= self.data_spectra[self.start_pnt:self.stop_pnt,0] # first first time-step of FFT
            except IndexError: # data_spectra is 1D
                self.y_data = self.data_spectra[self.start_pnt:self.stop_pnt]
            self.y_data = -np.log(self.y_data) # convert to absorbance
        else:
            self.y_data = self.absorption_meas[0]
        y_data = self.y_data

#        ## Run fit algorithm
        gmodel = Model(makeSpectraOutsideLoop, independent_vars = ['xx','molDict'])
        self.gmodel = gmodel # for debugging
        molecs = self.molec
        result = gmodel.fit(y_data, params = self.fit_params, xx = self.x_data, molDict = molecs)
           # didn't set molDict as independent_vars, so Model.fit assumes it is a parameter which it can't find in the fit_params class.
           # But independent_vars isn't right either, because fit() wants to turn those into numpy float arrays, and you can't do that with a dictionary
        self.result = result

    def fit_repeat(self):
        self.result2 = self.gmodel.fit(self.y_data, params = self.fit_params,
                                       xx = self.x_data, molDict = self.molec)


    def load_environment_sim(self, PreviousSim):
        """
        Loads the fitting environment from a previously defined simulation class

        INPUTS:
            PreviousSim = PLDSim class

        OUTPUTS:
            Loads the following
                self.temperature [K]
                self.pressure [bar]
                self.pathlength [m]
            And the frequency axis vectors
                self.x_Hz [THz]
                self.x_nm  [nm]
                self.x_wvn [cm^{-1}]

        ASSUMPTIONS:

        ERRORS:

        TODO:
        """

        self.temperature  = PreviousSim.temperature
        self.pressure     = PreviousSim.pressure
        self.pathlength   = PreviousSim.pathlength
        self.x_nm  = PreviousSim.x_nm
        self.x_wvn = PreviousSim.x_wvn
        self.x_Hz = SPEED_OF_LIGHT / self.x_nm


    def load_spectra(self, strFilename, colWaveScale=0, charDelim=',', getFreq = False):
        """
        Loads spectroscopic data from a text file, eg: exported from Igor.

        INPUTS:
            strFilename  = string of relative path to text file including extension.
            colWaveScale = column index of wave scaling values. Default of 0 will start load process
                           at 0th column
            charDelim    = delimeter character, default as ',' for CSV

        OUTPUTS:
            Loads self.data_spectra
            Loads self.num_spectra

        ASSUMPTIONS:
            - File exists and contains column-data
            - Scaling column is not [THz]
            - Apart from colWaveScale column of text file, all other columns are FFT

        ERRORS:
            usually frequency axis is first column, but that will give colWaveScale = 0 so

        TODO:
            - Perhaps add exception handling, although the calling code could as well
            - Synchronize with def_environment() for x-axis scaling parameters
        """

        # Load the data
#        self.data_spectra = np.genfromtxt(open(strFilename, 'r'), delimiter=charDelim) # Anthony's function
        self.data_spectra = np.loadtxt(strFilename) # Nate prefers this function, can interpret any delimeter as long as consistent
        self.num_spectra  = self.data_spectra.shape[1]

        if getFreq:
            # Determine the number of spectra
            self.num_spectra  -= 1 # 1 column is frequency, others are spectra
            # Extract scaling
            self.x_Hz = self.data_spectra[:,colWaveScale]
            # test whether data scale is Hz, cm-1, or nm by context
            if(self.x_Hz[0] < 1e12):
                if(self.x_Hz[0] > 5000):
                    # Assume wavenumbers in near-IR
                    print("Assuming measurement file wavenumbers given in cm-1")
                    self.x_Hz *= SPEED_OF_LIGHT * M_2_CM
                elif(self.x_Hz[0] > 500):
                    # Assume wavelengths in nm
                    print("Assuming measurement file wavelengths given in nm")
                    self.x_Hz = SPEED_OF_LIGHT * 1e7 / self.x_Hz
                else:
                    print("Assuming measurement file wavelengths given in microns")
                    self.x_Hz = SPEED_OF_LIGHT * 1e4 / self.x_Hz
            self.THz_step = (self.x_Hz[-1] - self.x_Hz[0]) / (len(self.x_Hz)-1)
            self.wvn_step = self.THz_step / SPEED_OF_LIGHT / M_2_CM
            self.x_wvn = self.x_Hz / SPEED_OF_LIGHT / M_2_CM
            self.x_nm = 1e7 / self.x_wvn
            # Remove frequency vector from the FFT data
            self.data_spectra = np.delete(self.data_spectra, colWaveScale, axis=1)

            # And rescale to smaller fit window
            if self.wvn_start is None:
                self.wvn_start = float(input("Enter start wavenumber (cm-1)"))
                self.wvn_stop  = float(input("Enter stop wavenumber (cm-1)"))
            self.start_pnt = np.argmin(abs(self.x_Hz/SPEED_OF_LIGHT/M_2_CM - self.wvn_start))
            self.stop_pnt = np.argmin(abs(self.x_Hz/SPEED_OF_LIGHT/M_2_CM - self.wvn_stop))
        else:
            # Set default scaling
            self.x_Hz = np.arange(start=0, stop=self.num_spectra)

        self.fit_range()


    def baseline_remove(self,spectrumToFloat = 0, fVerbose=0):
        """
        Baseline corrects the spectra using Chebyshev polynomials.
        Note: Chebyshev polynomials are used since they reduce Runge's Phenomenon (polynomial ringing)
        about the approximation

        Absorbance = - log ( I_meas / I0)  = - log(I_meas) + log(I0)
        fit a Chebyshev to -log(I_meas) so want to add that baseline to get Absorbance_blRemoved

        Or you can specify blType = 'Taylor' for piecewise-Taylor polynomial baseline as in Igor's blRemove() function

        If you want to run multiple time-steps with similar baseline shape,
        it's probably better to only call this function baseline_remove() for
        the first time-step, and then send next time-steps into the T,X fitting function
        after dividing out the first baseline model
        eg. timeStep2 = -np.log(self.data_spectra[:,1]) / baseline_array.
        (Fit with chebfit(), which doesn't take initial guess coefficients.)

        INPUTS:
            spectrumToFloat is the index of the transmission 2d file to fit.
            fVerbose = flag to turn on/off verbose print statements. 0 -> Off (Default), 1 -> On
        OUTPUTS:
            Baseline array for the input y_data. Available for dividing any subsequent time-series transmission spectra.
              using equation NextTimeStep_blRemove = -ln(transmission_nextTimeStep) / baseline_array
            2nd optional output: baseline-removed absorption spectrum of the input y_data

        ASSUMPTIONS:
            No absorption model.
            Fit absorption spectra, not transmission? (or doesn't matter if no absorption model?)
            Fit entire spectrum with 1 Chebyshev polynomial (of the 1st kind, scaled -1<x<1)
            y_data is the FFT

        ERRORS:
            ValueError: raised if blType is not properly defined

        TODO:
            Validate piecewise-Taylor function (done)
            Set up return statement to return as many elements as called, not a tuple.
            Spectral fitting window being not entire Nyquist window.

            # Example of Validation code
            # Start in ~/Documents/pldspectrapy
            from pldspectrapy.pldfitting import PLDFitting
            sim = PLDFitting()
            sim.data_spectra = np.exp(1) * np.ones(2000) # make 1 flat spectrum
            sim.bl_type = 'Taylor'; sim.chunks = 5 # Testing Taylor method
            sim.baseline_remove()
        """

        ## First load in data

        try:
        # Want to insert everywhere some if not exist, then don't run for these variables not called in _init_
            bl_order = self.bl_order    # number of Chebyshev baseline coeffs across fit spectrum
            chunks  = self.chunks
            blType = self.bl_type
        except:
            bl_order = float(input('Enter number of baseline fit coefficients'))
            blType = input("Select 'Chebyshev' or 'Taylor' baseline (don't need quotes")
            chunks = float(input('Enter number of baseline polynomial fit regions across entire spectrum (suggest 1)'))
#            self.bl_order = bl_order # Want to be able to run this function without producing an object

        # Get single transmission spectrum, either the first
        assert self.data_spectra is not None, "Add at least one spectrum to self.data_spectra numpy array first."
        # Want to take 1st row of object from LoadIGs and phase-correct, or something like that.
        try:
            y_data = self.data_spectra[self.start_pnt:self.stop_pnt,spectrumToFloat]
        except IndexError:
            y_data = self.data_spectra[self.start_pnt:self.stop_pnt] # only 1-column spectrum

        ## Other data prep
        y_data = -np.log(y_data) # convert transmission to absorbance
        blcoefs = [] # 2D list of baseline coefficients

#        # And subtract initial guess absorbance model
        if self.num_molecs is not 0:
            self.simulate()
            # bug with off-by-one lengths, just collect data again
            if len(y_data) is not len(self.y_absorbance[0]):
                self.fit_range()
                self.simulate()
                try:
                    y_data = -np.log(self.data_spectra[self.start_pnt:self.stop_pnt,spectrumToFloat])
                except IndexError:
                    y_data = -np.log(self.data_spectra[self.start_pnt:self.stop_pnt]) # only 1-column spectrum
            for spectra in self.y_absorbance:
                y_data -= spectra
        else:
            if fVerbose:
                print("Fitting baseline without nominal absorbance model.")


        ## Then calculate baseline
        if blType.lower() == 'chebyshev':
            # Fit entire spectrum with a single Chebyshev baseline model
            if fVerbose:
                print('Fitting whole spectrum with a ' + str(bl_order) + '-order Chebyshev polynomial')
            x_scale = np.linspace(-1,1,len(y_data))
            blcoef =np.polynomial.chebyshev.chebfit(x_scale,y_data, bl_order)
            blcoefs.append(blcoef) # Group the blcoefs together in the same list entry for readability
                                   # But size (1,bl_order) will screw up the sizing of baseline_array to (bl_order, nTeeth)
            baseline_array = np.polynomial.chebyshev.chebval(x_scale,blcoef) # make baseline model
            absorbance_without_baseline = y_data - baseline_array # make baseline-corrected

        elif blType.lower() == 'taylor':
            # Fit spectrum in chunks with several Taylor polynomials
            # This algorithm taken from Igor's WorkerFunc_blRemove()

            # Debug the code on y_data = np.ones(), to see if weight function returns all 1.

            borderp = 200 # for multiple baseline polynomial chunks, want linear spline region spanning this many points

            if fVerbose:
                print('Fitting ' + str(chunks) + ' sequential ' + str(self.bl_order) + '-order Taylor-polynomial baseline model.')

            pnts_per_chunk = int(len(y_data)/chunks)
            borderp = int(min(borderp, np.floor(pnts_per_chunk/2)-1)) # Avoid bug for 2 overlapping baseline-interpolation regions
            if fVerbose:
                print('borderp = ',borderp)

            baseline_array = np.zeros((len(y_data)))

            for ii in range(chunks):
                # Add triangular-weighted overlap region to Taylor-polynomial fit chunk
                # Equations simplify to full y_data region for chunks = 1
                start_pnt = int(max(ii*pnts_per_chunk-borderp, 0))
                stop_pnt  = int(min((ii+1)*pnts_per_chunk + borderp, len(y_data)))
                x_scale = np.linspace(0,1,stop_pnt-start_pnt) # which x-scaling to use for Taylor series?
                if fVerbose:
                    print(start_pnt,stop_pnt)

                #  How does Igor use spline-fitting for piecewise-continuous Taylor series?
                blcoef = np.polyfit(x_scale,y_data[start_pnt:stop_pnt],bl_order)
                blcoefs.append(blcoef)
                if chunks > 1:
                    baselineChunk = np.polyval(blcoef,x_scale)
                    upRamp = np.linspace(0,1,2*borderp)
                    downRamp = np.linspace(1,0,2*borderp)
                    # now apply the spline
                    # TODO:
                    # - Test that bounds are still within array limits, else adjust bounds to limits
                    if ii < (chunks-1):
                        baselineChunk[-(2*borderp):] *= downRamp
                    if ii > 0:
                        baselineChunk[0:2*borderp] *= upRamp
                    baseline_array[start_pnt:stop_pnt] += baselineChunk

                else:
                    baseline_array = np.polyval(blcoef, x_scale)

            absorbance_without_baseline = y_data - baseline_array

        else:
            raise ValueError("self.bl_type is invalid. Please set to 'Chebyshev' or 'Taylor', case insensitive")

        # Add the a priori absorbance model back into the data
        for spectra in self.y_absorbance:
            absorbance_without_baseline += spectra

        self.blcoefs = blcoefs
        self.baseline_array = baseline_array
        self.absorption_meas.append(absorbance_without_baseline)

        return blcoefs, baseline_array, absorbance_without_baseline
        # TODO:
        #  Don't want to return a tuple, but instead just return (in MATLAB form) as many arguments as asked
        # How does Python do that?

    def baseline_from_absorbance(self, spectrum, baseline_order = 0):
        '''
        Iterate on baseline_remove()
        '''

        if baseline_order is 0:
            baseline_order = self.bl_order
        for spectra in self.y_absorbance:
            spectrum -= spectra
        else:
            print("Fitting baseline without nominal absorbance model.")
        x_scale = np.linspace(-1,1,len(spectrum))
        blcoef =np.polynomial.chebyshev.chebfit(x_scale, spectrum, baseline_order)
        baseline_array = np.polynomial.chebyshev.chebval(x_scale,blcoef)
        absorbance_without_baseline = spectrum - baseline_array
        # and add back in the absorbance model
        for spectra in self.y_absorbance:
            absorbance_without_baseline += spectra

        return absorbance_without_baseline



def makeSpectraOutsideLoop(xx,molDict,**params):
    # NAM: test arbitrary sized dictionary of parameters
    # using KWARGS with double asterisk as recommended by StackOverflow
    # https://stackoverflow.com/questions/32386179/creating-a-python-lmfit-model-with-arbitrary-number-of-parameters#32427928
    """
    Transmission fitting function used for master fitting code.

    INPUTS:
        xx      = comb teeth frequencies [cm^{-1}]
        molDict = class containing extra attributes for each molecule
                  in order [molID, iso, Voigt?, linelistFile]
        params  = Parameters() class
    To-do:
        Finish adapting from gasifier_fit and baselineFit_furnace Nate's scripts
        -Handle multispecies in naming convention or in self.num_molec setup
        -Naming convention in Parameters() to handle non-HitranOnline linelists
        -Change transmission fit to absorbance fit with extra low-order Chebyshev baseline model

        -Want to run from purely **params, rather than inside class Fitting
        -  because get error 'module' object is not callable

    How to call?
    makeSpectraOutsideLoop(x_Cheb,m,**Example.fit_params)
    """
#    try:
##            xx = self.x_wvn
#        xx = self.x_Hz[self.start_pnt:self.stop_pnt]/SPEED_OF_LIGHT
#    except ValueError:
#        print("Run def_environment() to get comb tooth frequency array")
#        return 0

    # Initialize the absorption spectrum
    Absorption_All = np.zeros(len(xx))
    xScale = np.linspace(-1,1,len(xx))
    step = (max(xx)-min(xx))/(len(xx)-1)

    Temp  = params['Temp']       # Kelvin
    press = params['press']      # atm
    shift = params['shift']      # cm^{-1}
    Lpath = params['Pathlength'] # cm


    # Add in each absorbing molecule
    for key in list(params.keys()):
        # look through each fit parameter for the X_i ones
        if 'X_' in key[:2]:
            molName = key[2:]
            # now extract attributes
            Xi = params[key] # molefraction
            molID, iso, lineshape, linelistFile = molDict[molName]
            if linelistFile is None:
                linelistFile = molName # Default is HitranOnline through hapi.fetch
            if lineshape is 'Voigt':
                nu, coef = hapi.absorptionCoefficient_Voigt(((molID,iso, Xi),),(linelistFile),
                OmegaStep=step,OmegaRange=[min(xx)+shift, max(xx)+shift],HITRAN_units=False,
                Environment={'p':press,'T':Temp},Diluent={'air':1-Xi,'self':Xi})
            else:
                if lineshape is 'SDVoigt':
                    nu, coef = hapi.absorptionCoefficient_SDVoigt(((molID,iso, Xi),),(linelistFile),
                    OmegaStep=step,OmegaRange=[min(xx)+shift, max(xx)+shift],HITRAN_units=False,
                    Environment={'p':press,'T':Temp},Diluent={'air':1-Xi,'self':Xi})
                pass
               # want to run non-Voigt models here.
               # Igor has ability to set up a function call as a string, and then execute.
               # So funcCall = 'hapi.absorptionCoefficient_' + nonVoigtModel + 'input variable structure...'

            Absorption_All += coef

    Absorption_All *= Lpath # convert from coefficient to absorbance

    # Add baseline coefficients manually, removing the need for self.bl_order_fit
    for key in list(params.keys()):
        if 'Cheb' in key[:4]:
            ChebOrder = float(key[4:]) # Chebyshev order of current value
            ChebCoef = params[key]
            Absorption_All += ChebCoef * np.cos(ChebOrder * np.arccos(xScale))

    return Absorption_All #absorption model
