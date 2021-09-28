# -*- coding: utf-8 -*-
"""
Change linelist format between Python dataframe, Labfit, and HITRAN formats.

Useful if you want to pull custom Labfit linelist into hapi simulation for fitting,
or if you want to perform various Python analyses on any linelist.
Python pandas dataframes are nice instead of Excel for linelist analysis,
 because you can save all your data manipulations in a script for repeatability,
 whereas in Excel you may have had some cat-on-the-keyboard copy-paste actions
 which mix up the parameters between different lines.
 
Example of usage:
    from packfind import find_package
    find_package('pldspectrapy')
    import linelist_conversions as db
    df_paul = db.par_to_df(r'data\H2O_PaulLF.data')
    df_paul2 = db.labfit_to_df('PaulData_SD_Avgn_AJ', htp=False)
    db.df_to_par(df_paul2, 'H2O_PaulLF_dummy',extra_params=db.SDVoigt_FORMAT)
    

Linelist formats:
df = pandas dataframe. Can use load_linelist() in linelist_analysis.py
      to convert Labfit DTL file into dataframe.
par = Hitran linelist data file. Uses 160-length strings w/ comma-separated extras.
        Used in conjunction with .header files for hapi calculations.
inp/rei = Labfit input or rei file, with 4 or 5-line space-separated string format
             for each transition.
dtl = Labfit file which contains complete linelist if successful fit convergence.
shelf = Python hard-drive-written dictionary format used in LabfitSetup.py

Created on Mon Jul 22 09:27:16 2019

@author: ForAmericanEyesOnly (Nate Malarich)
"""
import os
import numpy as np
import pandas as pd
import json
from pldspectrapy.constants import HITRAN_160, MOLECULE_NAMES
from hapi import HITRAN_DEFAULT_HEADER

HITRAN_160_EXTRA = {
        'ierr': {'index':[127,133], "default": "   EEE"},
        'iref': {'index':[133,145], "default": "         EEE"},
        'other': {'index':[145,160], "default": 'E    0.0    0.0'}
        }
SDVoigt_FORMAT =   {
    "delta_self": "%9.6f",
    "n_self": "%7.4f",
    "sd_self": "%9.6f",
	"n_delta_air": "%7.4f"
    }

def par_to_df(hit_file, nu_min = 0, nu_max = 10000):
    '''
    Turn standard par file into dataframe with hapi names.
    '''                
    # Look for extra parameters in header file
    extra = []
    header_file = hit_file.split('.')[0] + '.header'
    if os.path.exists(header_file):
        with open(header_file,'r') as header:
            head = json.loads(header.read())
        try:
            extra = head['extra']
            extra_separator = head['extra_separator']
        except KeyError:
            pass
    with open(hit_file,'r') as air_linelist:
        linelist = []
        nu_i = HITRAN_160['nu']['index']
        for hit_line in air_linelist.readlines():
            # 100x faster to make list of dictionaries than append rows to dataframe
            row_dict = {}
            if len(hit_line) > 1:
                if float(hit_line[nu_i[0]:nu_i[1]]) < nu_min:
                    continue
                elif float(hit_line[nu_i[0]:nu_i[1]]) > nu_max:
                    break
                for name, props in HITRAN_160.items():
                    value = hit_line[props['index'][0]:props['index'][1]]
                    param_type = props['par_format'][-1]
                    if param_type == 's':
                        row_dict[name] = value
                    elif param_type == 'd':
                        row_dict[name] = int(value)
                    else:
                        try:
                            row_dict[name] = float(value)
                        except ValueError:
                            row_dict[name] = 0.0
                    if 'quanta' in name:
                        row_dict['quanta_index'] = ' '.join(value.split())
                for name, props in HITRAN_160_EXTRA.items():
                    row_dict[name] = hit_line[props['index'][0]:props['index'][1]]
                if len(extra) > 0:
                    extras = hit_line.split(extra_separator)[1:]
                    for name, value in zip(extra, extras):
                        row_dict[name] = float(value)
                linelist.append(row_dict)
        titles = list(HITRAN_160.keys())
        titles.append('quanta_index')
        titles.extend(list(HITRAN_160_EXTRA.keys()))
        titles.extend(extra)
        df_air = pd.DataFrame(linelist, columns = titles)
#        df_air['Smax'] = df_air['sw']*(.518*df_air['elower']/296)**(-2.77) * np.exp(-1.435 * df_air['elower']*(1/(.518 * df_air['elower']) - 1/296))
    return df_air
    
def df_to_par(df, par_name, suffix = '', extra_params = {}, save_dir = None):
    '''
    Turns dataframe linelist into .data file for hapi.
    
    INPUTS:
        df -> can be straight from load_linelist(),
             or output from match_to_hitran() if you set suffix='_hit'
        suffix -> if concatenate linelist, select from one part of concatenated
        extra_params -> dictionary such as SDVoigt_FORMAT for params to add
                    want at minimum to add n_self
                    If you have extra parameters artificially set to 0,
                    then you don't want to include these in the list.
         
    '''                                    
#    line_suffix = '0000000000 0 0 0 0    0.00   0.00\n'
    # First change some HITRAN defaults so formatting writes correctly
    HITRAN_DEFAULT_HEADER['default']['line_mixing_flag'] = 'E'
    HITRAN_DEFAULT_HEADER['default']['gp'] = 0
    HITRAN_DEFAULT_HEADER['default']['gpp'] = 0
    if save_dir is None:
        save_dir = os.getcwd()
    with open(os.path.join(save_dir,par_name + '.data'), 'w') as out:
        for i in range(len(df)):
            added_quanta = False
            for name in HITRAN_DEFAULT_HEADER['order']:
                # Many of these string-writing operations are more difficult than they need to be.
                # Writing code for each exception.
                try:
                    value = df[name+suffix].iloc[i]
                except KeyError:
                    value = HITRAN_DEFAULT_HEADER['default'][name]
                # Now write the value to file, with several special cases
                if 'molec_id' in name:
                    value = MOLECULE_NAMES.index(value) + 1
                    out.write(HITRAN_DEFAULT_HEADER['format'][name] % value)
                elif 'local_iso_id' in name:
                    value = float(value)
                    out.write(HITRAN_DEFAULT_HEADER['format'][name] % value)
                elif name == 'gamma_air' or name == 'delta_air' or name == 'n_air':
                    # remove leading zero (formatting actually different for these two)
                    value_str = HITRAN_DEFAULT_HEADER['format'][name] % value
                    str_len = int(HITRAN_DEFAULT_HEADER['format'][name].split('.')[0][1:])
                    if len(value_str) > str_len:
                        # need to remove digits from the beginning of this
                        if value >= 0:
                            out.write(value_str[1:])
                        else:
                            out.write('-.' + value_str.split('.')[1])
                    else:
                        out.write(value_str)
                else:
                    if 'quanta' in name:
                        # only count the quanta once
                        if added_quanta is False:
                            out.write('%60s' % df['quanta'+suffix].iloc[i][:60])
                            added_quanta = True
                    else:
#                        print(name, HITRAN_DEFAULT_HEADER['format'][name] % value)
                        # all other parameters are normal
                        try:
                            out.write(HITRAN_DEFAULT_HEADER['format'][name] % value)
                        except:
                            print(name)
                            print(value)
            # And add non-standard Hitran lines at the end w/o suffix
            for name, str_format in extra_params.items():
                out.write(',')
                out.write(str_format % df[name].iloc[i])                
#                out.write(',')
#            out.write(line_suffix)
            out.write('\n')
    # And write accompanying header file
    with open(os.path.join(save_dir,par_name + '.header'),'w') as f:
        HITRAN_DEFAULT_HEADER['table_name'] = par_name
        if len(extra_params) > 0:
            # Add extra header entries hapi-formatted for non_voigt extra params
            HITRAN_DEFAULT_HEADER["extra"] = list(extra_params.keys())
            HITRAN_DEFAULT_HEADER["extra_format"] = extra_params
            HITRAN_DEFAULT_HEADER["extra_separator"] = ","
        f.write(json.dumps(HITRAN_DEFAULT_HEADER, indent=2))
        
def calc_ierr(df_row, err_ref):
    '''
    Udpate Hitran uncertainty indices based on labfit float uncertainties
    INPUTS:
        df_row: pandas row of dataframe from match_to_hitran()
        err_ref: 18-element list of Ierr and Iref
    OUTPUT:
        err_ref: 18-element updated list of Ierr and Iref
    '''             
    # now modify the error and reference
    float_names = {'nu':0,'sw':1,'gamma_self':3}
    for name, position in float_names.items():
        uc = df_row['uc_' + name] / df_row[name + '_lf']
        if uc > 0:
            if name == 'nu':
                code = int(-np.log10(df_row['uc_nu'])) + 1
            else:
                if uc < .01:
                    code = 8
                elif uc < .02:
                    code = 7
                elif uc < .05:
                    code = 6
                elif uc < .1:
                    code = 5
                elif uc < .2:
                    code = 4
                else:
                    code = 3
            err_ref[position] = repr(code)
            err_ref[6+2*position:6+2*(1+position)] = 'ps' # Paul Schroeder fit
    return err_ref

def leading_zero(str_format, value_float):
    '''
    Hitran's float format will often add zero to make string length longer than you want.
    This script corrects the bug.
    '''
    value_str = str_format % value_float
    if len(value_str) > int(str_format.split('.')[0][1:]):
        # need to remove digits from the beginning of this
        if value_float >= 0:
            value_str = (value_str[1:])
        else:
            value_str = ('-.' + value_str.split('.')[1])
    return value_str

"""
Pulls Labfit linelist from .dtl file for analysis

Created on Sat Mar 30 11:36:42 2019

@author: ForAmericanEyesOnly
"""


def labfit_to_df(file_name, label_extension = '', htp = False):
    '''
    Get sorted Labfit output linelist from .dtl file into Python pandas dataframe..
    
    INPUTS:
        file_name = full path to Labfit .dtl file (ignore .dtl extension)
        label_extension = suffix to add to all dataframe titles 
                         (useful if you concatenate two dataframes afterwards)
        htp = True if using Hartman-Tran version of Labfit with extra row of fit parameters,
               set to False if using standard 4-line Labfit.
    '''

    linelist = [] # contains all linelist numeric parameters
    linelist_str = [] # other line parameters: index, molecule, iso, quanta
    
    titles_str = ['index','molec_id','local_iso_id','quanta']
    titles_htp = ['nu','uc_nu','sw','uc_sw', # matches HITRAN_160 and LABFIT_EXTRA string names
                  'elower','uc_elower','mass','uc_mass','gamma_air','uc_gamma_air',
                  'n_air','uc_n_air','gamma_self','uc_gamma_self',
                  'n_self','uc_n_self','delta_air','uc_delta_air',
                  'n_delta_air','uc_n_delta_air','delta_self','uc_delta_self',
                  'n_delta_self','uc_n_d_self',
                  'beta_g_self','uc_beta_g_self','y_self','uc_y_self',
                  'SD_air','uc_SD_air', 'sd_self','uc_sd_s',
                  'nu_vc_f','uc_nu_vc_f','nu_vc_s','uc_nu_vc_s', # I'm not 100% sure of the 10 params after n_delta_self
                  'eta_for','uc_eta_f','eta_self','uc_eta_s','dumb_for','uc_dumb_for',
                  'dumb_self','uc_dumb_self']
    if htp is False:
        titles_htp = titles_htp[:30] # the 4-line Labfit formatting
        titles_htp[-2:] = ['sd_self','uc_sd_self']
    if label_extension != '':
        for index, string in enumerate(titles_htp):
            titles_htp[index] = string + label_extension
    
    with open(file_name + '.dtl','r') as dtl:
        dtl_lines = dtl.readlines()
    found_list = False
    has_finished = False
    cur_line = len(dtl_lines)
    while has_finished is False:
        cur_line -= 1
        line = dtl_lines[cur_line]
        if len(line) > 200: # this is final linelist in detail file
            linelist.append(np.asfarray(line[95:].split()))
            found_list = True
            other_params = line[:18].split()
            other_params[0] = float(other_params[0])
            other_params.append(line[30:90])
            linelist_str.append(other_params)
        else:
            if found_list: # already went through all lines in linelist
                has_finished = True
    
    df_linelist = pd.DataFrame.from_records(linelist, columns = titles_htp) # aggregate information into pandas dataframe
    
    linelist_str_df = pd.DataFrame.from_records(linelist_str, columns = titles_str) # and add 4 string elements to front
    df_linelist = pd.concat([linelist_str_df, df_linelist], axis = 1)
    df_linelist = pd.DataFrame.sort_values(df_linelist, 'index')

    df_linelist = df_linelist.set_index('index') # and make the dataframe indexing equivalent to Labfit indexing
    
    return df_linelist
