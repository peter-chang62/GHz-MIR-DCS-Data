# py2/3 compatibility
from __future__ import division
try: # python 2
    range = xrange
except NameError: # python 3
    pass

from os import path as _path
from os.path import pardir as _pardir
try:
    from .editmodule import _load_module, _finalize_modifications
except (ValueError, SystemError, ImportError):
    from editmodule import _load_module, _finalize_modifications

_hapi_path, _old_locals = _load_module('hapi', _path.realpath(_path.join(_path.split(__file__)[0], _pardir)), locals())

# Need to specifically redefine protected variables:
__ComplexType__ = complex128
__IntegerType__ = int64
__FloatType__ = float64


from hapi import arange_ # added by Scott after getting weird errors (is the same as arrange with minor modification for small step sizes)
# in the end this fix didn't work, so I changed arange_ back to arange. I"m still new. This is not a change that should be perpetuated


###############################################################################
#                        Insert Modified Code Below                           #
"""
Lab group calculation updates labelled by 'pld_update' with contributor and date.
Update lineshape parameter-handling to be consistent with our in-house high-temperature databases.
-Added lineshift temperature-dependence exponent n_delta_air=1 to absorptionCoefficient_
 (Voigt and SDVoigt).
 Hapi assumes linear pressure-shift model parameter deltap_air, except HitranOnline doesn't have many of those.
 In our version, hapi looks for deltap_, and if it doesn't find it it uses a power-law model, looks for n_delta_,
 and if it can't find that assumes n_delta_ = 1
-Lorentz-broadening note: hapi looks for n_self, and if it doesn't find that in the database, assumes n_self = n_air for the line.
 If you want to hard-code a particular n_self = 0.694 or similar, must add comma-separated value to Hitran database file.
- SDVoigt has same temperature dependence in Gamma0 as Gamma0
"""

# pld_update Nate Malarich 6June2019: allow shift temperature-scaling exponent
# Only look for this parameter if deltap doesn't exist (what if all values are zero?)
PARAMETER_META["n_delta_self"] = {"default_fmt":"%7.4f"}
PARAMETER_META["n_delta_air"] = {"default_fmt":"%7.4f"}


def absorptionCoefficient_SDVoigt(Components=None,SourceTables=None,partitionFunction=PYTIPS,
                                Environment=None,OmegaRange=None,OmegaStep=None,OmegaWing=None,
                                IntensityThreshold=DefaultIntensityThreshold,
                                OmegaWingHW=DefaultOmegaWingHW,
                                GammaL='gamma_air', HITRAN_units=True, LineShift=True,
                                File=None, Format=None, OmegaGrid=None,
                                WavenumberRange=None,WavenumberStep=None,WavenumberWing=None,
                                WavenumberWingHW=None,WavenumberGrid=None,
                                Diluent={},EnvDependences=None):
    """
    INPUT PARAMETERS:
        Components:  list of tuples [(M,I,D)], where
                        M - HITRAN molecule number,
                        I - HITRAN isotopologue number,
                        D - relative abundance (optional)
        SourceTables:  list of tables from which to calculate cross-section   (optional)
        partitionFunction:  pointer to partition function (default is PYTIPS) (optional)
        Environment:  dictionary containing thermodynamic parameters.
                        'p' - pressure in atmospheres,
                        'T' - temperature in Kelvin
                        Default={'p':1.,'T':296.}
        WavenumberRange:  wavenumber range to consider.
        WavenumberStep:   wavenumber step to consider.
        WavenumberWing:   absolute wing for calculating a lineshape (in cm-1)
        WavenumberWingHW:  relative wing for calculating a lineshape (in halfwidths)
        IntensityThreshold:  threshold for intensities
        GammaL:  specifies broadening parameter ('gamma_air' or 'gamma_self')
        HITRAN_units:  use cm2/molecule (True) or cm-1 (False) for absorption coefficient
        File:   write output to file (if specified)
        Format:  c-format of file output (accounts for significant digits in WavenumberStep)
    OUTPUT PARAMETERS:
        Wavenum: wavenumber grid with respect to parameters WavenumberRange and WavenumberStep
        Xsect: absorption coefficient calculated on the grid
    ---
    DESCRIPTION:
        Calculate absorption coefficient using SDVoigt profile.
        Absorption coefficient is calculated at arbitrary temperature and pressure.
        User can vary a wide range of parameters to control a process of calculation.
        The choise of these parameters depends on properties of a particular linelist.
        Default values are a sort of guess which gives a decent precision (on average)
        for a reasonable amount of cpu time. To increase calculation accuracy,
        user should use a trial and error method.
    ---
    EXAMPLE OF USAGE:
        nu,coef = absorptionCoefficient_SDVoigt(((2,1),),'co2',WavenumberStep=0.01,
                                              HITRAN_units=False,GammaL='gamma_self')
    ---
    """

    # warn('To get the most up-to-date version of SDVoigt please check http://hitran.org/hapi. This version is customized by Yang et al 2018')

    # Paremeters OmegaRange,OmegaStep,OmegaWing,OmegaWingHW, and OmegaGrid
    # are deprecated and given for backward compatibility with the older versions.
    if WavenumberRange:  OmegaRange=WavenumberRange
    if WavenumberStep:   OmegaStep=WavenumberStep
    if WavenumberWing:   OmegaWing=WavenumberWing
    if WavenumberWingHW: OmegaWingHW=WavenumberWingHW
    if WavenumberGrid:   OmegaGrid=WavenumberGrid

    # "bug" with 1-element list
    Components = listOfTuples(Components)
    SourceTables = listOfTuples(SourceTables)

    # determine final input values
    Components,SourceTables,Environment,OmegaRange,OmegaStep,OmegaWing,\
    IntensityThreshold,Format = \
       getDefaultValuesForXsect(Components,SourceTables,Environment,OmegaRange,
                                OmegaStep,OmegaWing,IntensityThreshold,Format)

    # warn user about too large omega step
    if OmegaStep>0.1: warn('Big wavenumber step: possible accuracy decline')

    # get uniform linespace for cross-section
    #number_of_points = (OmegaRange[1]-OmegaRange[0])/OmegaStep + 1
    #Omegas = linspace(OmegaRange[0],OmegaRange[1],number_of_points)
    if OmegaGrid is not None:
        Omegas = npsort(OmegaGrid)
    else:
        #Omegas = arange(OmegaRange[0],OmegaRange[1],OmegaStep)
        Omegas = arange_(OmegaRange[0],OmegaRange[1],OmegaStep) # fix
    number_of_points = len(Omegas)
    Xsect = zeros(number_of_points)

    # reference temperature and pressure
    Tref = __FloatType__(296.) # K
    pref = __FloatType__(1.) # atm

    # actual temperature and pressure
    T = Environment['T'] # K
    p = Environment['p'] # atm

    # create dictionary from Components
    ABUNDANCES = {}
    NATURAL_ABUNDANCES = {}
    for Component in Components:
        M = Component[0]
        I = Component[1]
        if len(Component) >= 3:
            ni = Component[2]
        else:
            try:
                ni = ISO[(M,I)][ISO_INDEX['abundance']]
            except KeyError:
                raise Exception('cannot find component M,I = %d,%d.' % (M,I))
        ABUNDANCES[(M,I)] = ni
        NATURAL_ABUNDANCES[(M,I)] = ISO[(M,I)][ISO_INDEX['abundance']]

    # precalculation of volume concentration
    if HITRAN_units:
        factor = __FloatType__(1.0)
    else:
        factor = volumeConcentration(p,T)

    # setup the default empty environment dependence function
    if not EnvDependences:
        EnvDependences = lambda ENV,LINE:{}
    Env = Environment.copy()
    Env['Tref'] = Tref
    Env['pref'] = pref

    # setup the Diluent variable
    GammaL = GammaL.lower()
    if not Diluent:
        if GammaL == 'gamma_air':
            Diluent = {'air':1.}
        elif GammaL == 'gamma_self':
            Diluent = {'self':1.}
        else:
            raise Exception('Unknown GammaL value: %s' % GammaL)

    # Simple check
    for key in Diluent:
        val = Diluent[key]
        if val < 0 and val > 1:
            raise Exception('Diluent fraction must be in [0,1]')

    # SourceTables contain multiple tables
    for TableName in SourceTables:

        # get the number of rows
        nline = LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows']

        # get parameter names for each table
        parnames = LOCAL_TABLE_CACHE[TableName]['data'].keys()

        # loop through line centers (single stream)
        for RowID in range(nline):

            # Get the custom environment dependences
            Line = {}
            for parname in parnames:
                Line[parname] = LOCAL_TABLE_CACHE[TableName]['data'][parname][RowID]
            CustomEnvDependences = EnvDependences(Env,Line)

            # get basic line parameters (lower level)
            LineCenterDB = LOCAL_TABLE_CACHE[TableName]['data']['nu'][RowID]
            LineIntensityDB = LOCAL_TABLE_CACHE[TableName]['data']['sw'][RowID]
            LowerStateEnergyDB = LOCAL_TABLE_CACHE[TableName]['data']['elower'][RowID]
            MoleculeNumberDB = LOCAL_TABLE_CACHE[TableName]['data']['molec_id'][RowID]
            IsoNumberDB = LOCAL_TABLE_CACHE[TableName]['data']['local_iso_id'][RowID]

            # filter by molecule and isotopologue
            if (MoleculeNumberDB,IsoNumberDB) not in ABUNDANCES: continue

            # partition functions for T and Tref
            SigmaT = partitionFunction(MoleculeNumberDB,IsoNumberDB,T)
            SigmaTref = partitionFunction(MoleculeNumberDB,IsoNumberDB,Tref)

            # get all environment dependences from voigt parameters

            #   intensity
            if 'sw' in CustomEnvDependences:
                LineIntensity = CustomEnvDependences['sw']
            else:
                LineIntensity = EnvironmentDependency_Intensity(LineIntensityDB,T,Tref,SigmaT,SigmaTref,
                                                                LowerStateEnergyDB,LineCenterDB)

            #   FILTER by LineIntensity: compare it with IntencityThreshold
            if LineIntensity < IntensityThreshold: continue

            #   doppler broadening coefficient (GammaD)
            cMassMol = 1.66053873e-27 # hapi
            m = molecularMass(MoleculeNumberDB,IsoNumberDB) * cMassMol * 1000
            GammaD = sqrt(2*cBolts*T*log(2)/m/cc**2)*LineCenterDB

            #   pressure broadening coefficient
            Gamma0 = 0.; Shift0 = 0.; Gamma2 = 0.; Shift2 = 0.
            for species in Diluent:
                species_lower = species.lower()

                abun = Diluent[species]

                gamma_name = 'gamma_' + species_lower
                try:
                    Gamma0DB = LOCAL_TABLE_CACHE[TableName]['data'][gamma_name][RowID]
                except:
                    Gamma0DB = 0.0

                n_name = 'n_' + species_lower
                try:
                    TempRatioPowerDB = LOCAL_TABLE_CACHE[TableName]['data'][n_name][RowID]
                    if species_lower == 'self' and TempRatioPowerDB == 0.:
                        TempRatioPowerDB = LOCAL_TABLE_CACHE[TableName]['data']['n_air'][RowID] # same for self as for air
                except:
                    #TempRatioPowerDB = 0
                    TempRatioPowerDB = LOCAL_TABLE_CACHE[TableName]['data']['n_air'][RowID]

                # Add to the final Gamma0
                Gamma0 += abun*CustomEnvDependences.get(gamma_name, # default ->
                          EnvironmentDependency_Gamma0(Gamma0DB,T,Tref,p,pref,TempRatioPowerDB))

                delta_name = 'delta_' + species_lower
                try:
                    Shift0DB = LOCAL_TABLE_CACHE[TableName]['data'][delta_name][RowID]
                except:
                    Shift0DB = 0.0

                # Nate Malarich pld_update 06/06/2019, select linear or power-law shift model
                deltap_name = 'deltap_' + species_lower
                # try:
                #     deltap = LOCAL_TABLE_CACHE[TableName]['data'][deltap_name][RowID]
                # except:
                #     deltap = 0.0

                deltap = 0.0
                try:
                    deltap = LOCAL_TABLE_CACHE[TableName]['data'][deltap_name][RowID]
                    power_law_shift = False
                except KeyError:
                    power_law_shift = True

                # For lineshift consistent with Labfit
                # If deltap information not available, get custom n_delta_air (default 1 or 0?)
                if power_law_shift is False:
                    Shift0 += abun*CustomEnvDependences.get(delta_name, # default ->
                            ((Shift0DB + deltap*(T-Tref))*p/pref))
                else:
                    # PARAMETER_META doesn't have any keys for shift temperature-exponents, stuck with hardcode
                    try:
                        n_delta = LOCAL_TABLE_CACHE[TableName]['data']['n_delta_air'][RowID]
                    except:
                        n_delta = 1
                    Shift0 += abun*CustomEnvDependences.get(delta_name,
                                (Shift0DB * (Tref/T)**n_delta * p))
                # End pld_update

                SD_name = 'sd_' + species_lower
                try:
                    SDDB = LOCAL_TABLE_CACHE[TableName]['data'][SD_name][RowID]
                except:
                    SDDB = 0.0

                # pld_update 06/06/2019 Nate Malarich
                # add temperature-scaling to speed-dependence
                Gamma2 += abun*CustomEnvDependences.get(SD_name,
                           SDDB * EnvironmentDependency_Gamma0(Gamma0DB,T,Tref,p,pref,TempRatioPowerDB))
                # end pld_update

            #   get final wing of the line according to Gamma0, OmegaWingHW and OmegaWing
            OmegaWingF = max(OmegaWing,OmegaWingHW*Gamma0,OmegaWingHW*GammaD)

            BoundIndexLower = bisect(Omegas,LineCenterDB-OmegaWingF)
            BoundIndexUpper = bisect(Omegas,LineCenterDB+OmegaWingF)
            lineshape_vals = PROFILE_SDVOIGT(LineCenterDB,GammaD,Gamma0,Gamma2,Shift0,Shift2,Omegas[BoundIndexLower:BoundIndexUpper])[0]
            Xsect[BoundIndexLower:BoundIndexUpper] += factor / NATURAL_ABUNDANCES[(MoleculeNumberDB,IsoNumberDB)] * \
                                                      ABUNDANCES[(MoleculeNumberDB,IsoNumberDB)] * \
                                                      LineIntensity * lineshape_vals

    if File: save_to_file(File,Format,Omegas,Xsect)
    return Omegas,Xsect


def absorptionCoefficient_Voigt(Components=None,SourceTables=None,partitionFunction=PYTIPS,
                                Environment=None,OmegaRange=None,OmegaStep=None,OmegaWing=None,
                                IntensityThreshold=DefaultIntensityThreshold,
                                OmegaWingHW=DefaultOmegaWingHW,
                                GammaL='gamma_air', HITRAN_units=True, LineShift=True,
                                File=None, Format=None, OmegaGrid=None,
                                WavenumberRange=None,WavenumberStep=None,WavenumberWing=None,
                                WavenumberWingHW=None,WavenumberGrid=None,
                                Diluent={},EnvDependences=None):
    """
    INPUT PARAMETERS:
        Components:  list of tuples [(M,I,D)], where
                        M - HITRAN molecule number,
                        I - HITRAN isotopologue number,
                        D - relative abundance (optional)
        SourceTables:  list of tables from which to calculate cross-section   (optional)
        partitionFunction:  pointer to partition function (default is PYTIPS) (optional)
        Environment:  dictionary containing thermodynamic parameters.
                        'p' - pressure in atmospheres,
                        'T' - temperature in Kelvin
                        Default={'p':1.,'T':296.}
        WavenumberRange:  wavenumber range to consider.
        WavenumberStep:   wavenumber step to consider.
        WavenumberWing:   absolute wing for calculating a lineshape (in cm-1)
        WavenumberWingHW:  relative wing for calculating a lineshape (in halfwidths)
        GammaL:  specifies broadening parameter ('gamma_air' or 'gamma_self')
        HITRAN_units:  use cm2/molecule (True) or cm-1 (False) for absorption coefficient
        File:   write output to file (if specified)
        Format:  c-format of file output (accounts for significant digits in WavenumberStep)
    OUTPUT PARAMETERS:
        Wavenum: wavenumber grid with respect to parameters WavenumberRange and WavenumberStep
        Xsect: absorption coefficient calculated on the grid
    ---
    DESCRIPTION:
        Calculate absorption coefficient using Voigt profile.
        Absorption coefficient is calculated at arbitrary temperature and pressure.
        User can vary a wide range of parameters to control a process of calculation.
        The choise of these parameters depends on properties of a particular linelist.
        Default values are a sort of guess which gives a decent precision (on average)
        for a reasonable amount of cpu time. To increase calculation accuracy,
        user should use a trial and error method.
    ---
    EXAMPLE OF USAGE:
        nu,coef = absorptionCoefficient_Voigt(((2,1),),'co2',WavenumberStep=0.01,
                                              HITRAN_units=False,GammaL='gamma_self')
    ---
    """

    # Paremeters OmegaRange,OmegaStep,OmegaWing,OmegaWingHW, and OmegaGrid
    # are deprecated and given for backward compatibility with the older versions.
    if WavenumberRange:  OmegaRange=WavenumberRange
    if WavenumberStep:   OmegaStep=WavenumberStep
    if WavenumberWing:   OmegaWing=WavenumberWing
    if WavenumberWingHW: OmegaWingHW=WavenumberWingHW
    if WavenumberGrid:   OmegaGrid=WavenumberGrid

    # "bug" with 1-element list
    Components = listOfTuples(Components)
    SourceTables = listOfTuples(SourceTables)

    # determine final input values
    Components,SourceTables,Environment,OmegaRange,OmegaStep,OmegaWing,\
    IntensityThreshold,Format = \
       getDefaultValuesForXsect(Components,SourceTables,Environment,OmegaRange,
                                OmegaStep,OmegaWing,IntensityThreshold,Format)

    # warn user about too large omega step
    if OmegaStep>0.1: warn('Big wavenumber step: possible accuracy decline')

    # get uniform linespace for cross-section
    #number_of_points = (OmegaRange[1]-OmegaRange[0])/OmegaStep + 1
    #Omegas = linspace(OmegaRange[0],OmegaRange[1],number_of_points)
    if OmegaGrid is not None:
        Omegas = npsort(OmegaGrid)
    else:
        Omegas = arange(OmegaRange[0],OmegaRange[1],OmegaStep)
        #Omegas = arange_(OmegaRange[0],OmegaRange[1],OmegaStep) # fix
    number_of_points = len(Omegas)
    Xsect = zeros(number_of_points)

    # reference temperature and pressure
    Tref = __FloatType__(296.) # K
    pref = __FloatType__(1.) # atm

    # actual temperature and pressure
    T = Environment['T'] # K
    p = Environment['p'] # atm

    # create dictionary from Components
    ABUNDANCES = {}
    NATURAL_ABUNDANCES = {}
    for Component in Components:
        M = Component[0]
        I = Component[1]
        if len(Component) >= 3:
            ni = Component[2]
        else:
            try:
                ni = ISO[(M,I)][ISO_INDEX['abundance']]
            except KeyError:
                raise Exception('cannot find component M,I = %d,%d.' % (M,I))
        ABUNDANCES[(M,I)] = ni
        NATURAL_ABUNDANCES[(M,I)] = ISO[(M,I)][ISO_INDEX['abundance']]

    # precalculation of volume concentration
    if HITRAN_units:
        factor = __FloatType__(1.0)
    else:
        factor = volumeConcentration(p,T)

    # setup the default empty environment dependence function
    if not EnvDependences:
        EnvDependences = lambda ENV,LINE:{}
    Env = Environment.copy()
    Env['Tref'] = Tref
    Env['pref'] = pref

    # setup the Diluent variable
    GammaL = GammaL.lower()
    if not Diluent:
        if GammaL == 'gamma_air':
            Diluent = {'air':1.}
        elif GammaL == 'gamma_self':
            Diluent = {'self':1.}
        else:
            raise Exception('Unknown GammaL value: %s' % GammaL)

    # Simple check
    for key in Diluent:
        val = Diluent[key]
        if val < 0 and val > 1:
            raise Exception('Diluent fraction must be in [0,1]')

    # SourceTables contain multiple tables
    for TableName in SourceTables:

        # get the number of rows
        nline = LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows']

        # get parameter names for each table
        parnames = LOCAL_TABLE_CACHE[TableName]['data'].keys()

        # loop through line centers (single stream)
        for RowID in range(nline):

            # Get the custom environment dependences
            Line = {}
            for parname in parnames:
                Line[parname] = LOCAL_TABLE_CACHE[TableName]['data'][parname][RowID]
            CustomEnvDependences = EnvDependences(Env,Line)

            # get basic line parameters (lower level)
            LineCenterDB = LOCAL_TABLE_CACHE[TableName]['data']['nu'][RowID]
            LineIntensityDB = LOCAL_TABLE_CACHE[TableName]['data']['sw'][RowID]
            LowerStateEnergyDB = LOCAL_TABLE_CACHE[TableName]['data']['elower'][RowID]
            MoleculeNumberDB = LOCAL_TABLE_CACHE[TableName]['data']['molec_id'][RowID]
            IsoNumberDB = LOCAL_TABLE_CACHE[TableName]['data']['local_iso_id'][RowID]

            # filter by molecule and isotopologue
            if (MoleculeNumberDB,IsoNumberDB) not in ABUNDANCES: continue

            # partition functions for T and Tref
            SigmaT = partitionFunction(MoleculeNumberDB,IsoNumberDB,T)
            SigmaTref = partitionFunction(MoleculeNumberDB,IsoNumberDB,Tref)

            # get all environment dependences from voigt parameters

            #   intensity
            if 'sw' in CustomEnvDependences:
                LineIntensity = CustomEnvDependences['sw']
            else:
                LineIntensity = EnvironmentDependency_Intensity(LineIntensityDB,T,Tref,SigmaT,SigmaTref,
                                                                LowerStateEnergyDB,LineCenterDB)

            #   FILTER by LineIntensity: compare it with IntencityThreshold
            if LineIntensity < IntensityThreshold: continue

            #   doppler broadening coefficient (GammaD)
            cMassMol = 1.66053873e-27 # hapi
            m = molecularMass(MoleculeNumberDB,IsoNumberDB) * cMassMol * 1000
            GammaD = sqrt(2*cBolts*T*log(2)/m/cc**2)*LineCenterDB

            #   pressure broadening coefficient
            Gamma0 = 0.; Shift0 = 0.;
            for species in Diluent:
                species_lower = species.lower()

                abun = Diluent[species]

                gamma_name = 'gamma_' + species_lower
                try:
                    Gamma0DB = LOCAL_TABLE_CACHE[TableName]['data'][gamma_name][RowID]
                except:
                    Gamma0DB = 0.0

                n_name = 'n_' + species_lower
                try:
                    TempRatioPowerDB = LOCAL_TABLE_CACHE[TableName]['data'][n_name][RowID]
                    if species_lower == 'self' and TempRatioPowerDB == 0.:
                        TempRatioPowerDB = LOCAL_TABLE_CACHE[TableName]['data']['n_air'][RowID] # same for self as for air
                except:
                    #TempRatioPowerDB = 0
                    TempRatioPowerDB = LOCAL_TABLE_CACHE[TableName]['data']['n_air'][RowID]

                # Add to the final Gamma0
                Gamma0 += abun*CustomEnvDependences.get(gamma_name, # default ->
                          EnvironmentDependency_Gamma0(Gamma0DB,T,Tref,p,pref,TempRatioPowerDB))

                delta_name = 'delta_' + species_lower
                try:
                    Shift0DB = LOCAL_TABLE_CACHE[TableName]['data'][delta_name][RowID]
                except:
                    Shift0DB = 0.0


                deltap_name = 'deltap_' + species_lower
                try:
                    deltap = LOCAL_TABLE_CACHE[TableName]['data'][deltap_name][RowID]
                except:
                    deltap = 0.0
                # Added this code to account for temperature dependent line shifts when no deltap information is available
                # Amanda Makowiecki pld_update 5/17/2018
                if deltap!=0:
                    Shift0 += abun*CustomEnvDependences.get(delta_name, # default ->
                              ((Shift0DB + deltap*(T-Tref))*p/pref))
                else:
                    Shift0 += abun*Shift0DB*Tref/T*p
                # End pld_update


            #   get final wing of the line according to Gamma0, OmegaWingHW and OmegaWing
            OmegaWingF = max(OmegaWing,OmegaWingHW*Gamma0,OmegaWingHW*GammaD)

            BoundIndexLower = bisect(Omegas,LineCenterDB-OmegaWingF)
            BoundIndexUpper = bisect(Omegas,LineCenterDB+OmegaWingF)
            lineshape_vals = PROFILE_VOIGT(LineCenterDB+Shift0,GammaD,Gamma0,Omegas[BoundIndexLower:BoundIndexUpper])[0]
            Xsect[BoundIndexLower:BoundIndexUpper] += factor / NATURAL_ABUNDANCES[(MoleculeNumberDB,IsoNumberDB)] * \
                                                      ABUNDANCES[(MoleculeNumberDB,IsoNumberDB)] * \
                                                      LineIntensity * lineshape_vals

    if File: save_to_file(File,Format,Omegas,Xsect)
    return Omegas,Xsect


###############################################################################
#                  Finalize the modifications (DO NOT EDIT)                   #
m = _finalize_modifications(_hapi_path, _old_locals, locals())
del _path, _pardir, _load_module, _finalize_modifications, _hapi_path, _old_locals