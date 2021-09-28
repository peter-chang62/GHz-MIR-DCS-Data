 # -*- coding: utf-8 -*-
"""
Universal time-domain codes.

TODO:
    working example of how to use these functions
    plotting function for multispecies fit

For implementation into pldspectrapy.
Can handle multispecies fitting, each with their own full path characteristics.
You can apply a constraint to match, say, pathlength, pressure, temperature of each.

Created on Tue Nov  5 13:30:40 2019

@author: Nate Malarich
"""
# built-in modules
import numpy as np
import matplotlib.pyplot as plt

# Modules from the internet
from lmfit import Model

# In-house modules (in this case a lab-custom version of hapi for accurate high-T)
from packfind import find_package
find_package('pldspectrapy')
import pldspectrapy.pldhapi as hapi
from pldspectrapy.constants import SPEED_OF_LIGHT

plt.rcParams.update({'font.size':11,'figure.autolayout': True})
'''
First codes for setting up x-axis

'''
def largest_prime_factor(n):
    '''
    Want 2 * (x_stop - x_start - 1) to have small largest_prime_factor
    '''
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n

def bandwidth_select_td(x_array, band_fit, max_prime_factor = 500):
    '''
    Tweak bandwidth selection for swift time-domain fitting.
    
    Time-domain fit does inverse FFT for each nonlinear least-squares iteration,
    and speed of FFT goes with maximum prime factor.
    
    INPUTS:
        x_array = x-axis for measurement transmission spectrum
        band_fit = [start_frequency, stop_frequency]
    '''
    x_start = np.argmin(np.abs(x_array - band_fit[0]))
    x_stop = np.argmin(np.abs(x_array - band_fit[1]))
       
    len_td = 2 * (np.abs(x_stop - x_start) - 1) # np.fft.irfft operation
    prime_factor = largest_prime_factor(len_td)
    while prime_factor > max_prime_factor:
        x_stop -= 1
        len_td = 2 * (np.abs(x_stop - x_start) - 1)
        prime_factor = largest_prime_factor(len_td)
    return x_start, x_stop

def weight_func(len_fd, bl, etalons = []):
    '''
    Time-domain weighting function, set to 0 over selected baseline, etalon range
    INPUTS:
        len_fd = length of frequency-domain spectrum
        bl = number of points at beginning to attribute to baseline
        etalons = list of [start_point, stop_point] time-domain points for etalon spikes
    '''
    weight = np.ones(2*(len_fd-1))
    weight[:bl] = 0; weight[-bl:] = 0
    for et in etalons:
        weight[et[0]:et[1]] = 0
        weight[-et[1]:-et[0]] = 0
    return weight

'''
Wrapper codes for producing absorption models in time-domain.
To be called using lmfit nonlinear least-squares
'''


def spectra_single(xx, mol_id, iso, molefraction, pressure, 
                   temperature, pathlength, shift, name = 'H2O', flip_spectrum=False):
    '''
    Spectrum calculation for adding multiple models with composite model.
    
    See lmfit model page on prefix, parameter hints, composite models.
    
    INPUTS:
        xx -> wavenumber array (cm-1)
        name -> name of file (no extension) to pull linelist
        mol_id -> Hitran integer for molecule
        iso -> Hitran integer for isotope
        molefraction
        pressure -> (atmospheres)
        temperature -> kelvin
        pathlength (centimeters)
        shift -> (cm-1) calculation relative to Hitran
        flip_spectrum -> set to True if Nyquist window is 0.5-1.0
    
    '''

    nu, coef = hapi.absorptionCoefficient_Voigt(((int(mol_id), int(iso), molefraction),),
            name, HITRAN_units=False,
            OmegaGrid = xx + shift,
            Environment={'p':pressure,'T':temperature},
            Diluent={'self':molefraction,'air':(1-molefraction)})
    coef *= hapi.abundance(int(mol_id), int(iso)) # assume natural abundance
    if flip_spectrum:
        absorp = np.fft.irfft(coef[::-1] * pathlength)
    else:
        absorp = np.fft.irfft(coef * pathlength)
    return absorp

def spectra_single_lmfit(prefix='', sd = False):
    '''
    Set up lmfit model with function hints for single absorption species
    '''
    if sd:
        mod = Model(spectra_sd, prefix = prefix)
    else:
        mod = Model(spectra_single, prefix = prefix)
    mod.set_param_hint('mol_id',vary = False)
    mod.set_param_hint('iso', vary=False)
    mod.set_param_hint('pressure',min=0)
    mod.set_param_hint('temperature',min=0)
    mod.set_param_hint('pathlength',min=0)
    mod.set_param_hint('molefraction',min=0,max=1)
    mod.set_param_hint('shift',value=0,min=-.2,max=.2)
    pars = mod.make_params()
    # let's set up some default thermodynamics
    pars[prefix + 'mol_id'].value = 1
    pars[prefix + 'iso'].value = 1
    pars[prefix + 'pressure'].value = 640/760
    pars[prefix + 'temperature'].value = 296
    pars[prefix + 'pathlength'].value = 100
    pars[prefix + 'molefraction'].value = 0.01
    
    return mod, pars

def spectra_sd(xx, mol_id, iso, molefraction, pressure, 
                   temperature, pathlength, shift, name = 'H2O', flip_spectrum=False):
    '''
    Spectrum calculation for adding multiple models with composite model.
    
    See lmfit model page on prefix, parameter hints, composite models.
    
    INPUTS:
        xx -> wavenumber array (cm-1)
        name -> name of file (no extension) to pull linelist
        mol_id -> Hitran integer for molecule
        iso -> Hitran integer for isotope
        molefraction
        pressure -> (atmospheres)
        temperature -> kelvin
        pathlength (centimeters)
        shift -> (cm-1) calculation relative to Hitran
        flip_spectrum -> set to True if Nyquist window is 0.5-1.0
    
    '''

    nu, coef = hapi.absorptionCoefficient_SDVoigt(((int(mol_id), int(iso), molefraction),),
            name, HITRAN_units=False,
            OmegaGrid = xx + shift,
            Environment={'p':pressure,'T':temperature},
            Diluent={'self':molefraction,'air':(1-molefraction)})
    coef *= hapi.abundance(int(mol_id), int(iso)) # assume natural abundance
    if flip_spectrum:
        absorp = np.fft.irfft(coef[::-1] * pathlength)
    else:
        absorp = np.fft.irfft(coef * pathlength)
    return absorp


'''
Tools for plotting results and baseline removal.
'''
def lmfit_uc(Fit, str_param):
    '''
    Get statistical fitting uncertainty of some fit parameter named str_param
    INPUTS:
        Fit = lmfit Model result object (Fit = mod.fit(...))
        str_param = name of parameter to extract
    warning: some fits are unstable and cannot calculate statistical uncertainties
    '''
    fit_report = Fit.fit_report()
    for line in fit_report.split('\n'):
        if (str_param + ':') in line:
            foo = line.split()
            fit_value = (float(foo[1]))
            fit_uc = (float(foo[3]))
    
    return fit_uc

def plot_fit(x_data, Fit, tx_title = True):
    '''
    Plot lmfit time-domain result.
    INPUTS:
        x_data: x-axis (wavenumber) for fit
        Fit: lmfit object result from model.fit()
    
    '''
    y_datai = Fit.data
    fit_datai = Fit.best_fit
    weight = Fit.weights
    # plot frequency-domain fit
    data_lessbl = np.real(np.fft.rfft(y_datai - (1-weight) * (y_datai - fit_datai)))
    model = np.real(np.fft.rfft(fit_datai))
    # plot with residual
    fig, axs = plt.subplots(2,1, sharex = 'col')
    axs[0].plot(x_data, data_lessbl, x_data, model)
    axs[1].plot(x_data, data_lessbl - model)
    axs[0].set_ylabel('Absorbance'); #axs[0].legend(['data','fit'])
    axs[1].set_ylabel('Residual'); axs[1].set_xlabel('Wavenumber ($cm^{-1}$)')
    if tx_title:
        t_fit = Fit.best_values['temperature']
        x_fit = Fit.best_values['molefraction']
        axs[0].set_title('Combustor fit T = ' + f'{t_fit:.0f}' + 'K, ' + 
                  f'{100*x_fit:.1f}' + '% H2O')
    # and time-domain fit
    plt.figure()
    plt.plot(fit_datai)
    plt.plot(y_datai - fit_datai)
    plt.plot(weight)
#    plt.legend(['model','residual','weighting function'])
    
    return data_lessbl

def plot_fit_multispecies(x_data, Fit):
    '''
    Plot baseline-subtracted fit and time-domain fit for multispecies fit.
    
    INPUTS:
        pars_multispecies = lmfit parameters object set up by 
                    "full_pars = ...
                    
    bl = baseline
    res = residual
                    
    TODO:
        Is multipath_mod in the Fit result?
        What is .data linelist name of each multispecies molecule?
          lmfit seems clunky for automated fit-plotting of multiple species
          from multiple linelists
    '''
    y_datai = Fit.data
    fit_datai = Fit.best_fit
    weight = Fit.weights
    # plot frequency-domain fit
    bl = (1 - Fit.weights) * (Fit.data - Fit.best_fit) # time-domain fit mismatch in non-baseline region
    res = (Fit.weights) * (Fit.data - Fit.best_fit)
    data_lessbl = np.real(np.fft.rfft(Fit.data - bl))
    # plot with residual
    fig, axs = plt.subplots(2,1, sharex = 'col')
    axs[0].plot(x_data, data_lessbl, label='data')
    axs[1].plot(x_data, np.real(np.fft.rfft(res)))
    axs[0].set_ylabel('Absorbance'); #axs[0].legend(['data','fit'])
    axs[1].set_ylabel('Residual'); axs[1].set_xlabel('Wavenumber ($cm^{-1}$)')    
    
#    # calculate each multispecies frequency-domain fit
    spectra_each = Fit.eval_components()
    for key, val in spectra_each.items():
        axs[0].plot(x_data, np.fft.rfft(val), label=key)
    axs[0].legend()
#    pars_out = pars_multispecies.copy()
#    for key, val in Fit.best_values.items():
#        pars_out[key].value = val
#    # Actually want to initialize the H2O of the next segment at the 1st segment value
#    full_pars = pars_out.copy()
#    comps = multipath_mod.eval_components(xx=x_data,params=pars_out,CH4name = 'CH4')
    
    return data_lessbl
    
    