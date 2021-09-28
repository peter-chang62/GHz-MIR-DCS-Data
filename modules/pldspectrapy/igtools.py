# Author: rjw
'''
Python code base for manipulating DAQ interferogram files
into phase-corrected data.

Quick user guide of useful commands:
IG = open_daq_files(filename_string_without_extension)
len(IG) # number of interferograms
SNR_peak_to_peak = (np.max(IG[0]) - np.min(IG[0])) / np.std(IG[0][:500])
spectrum_raw = np.abs(np.fft.fft(IG[0])[:len(IG[0])//2])
IG.set_mode(PC) # work from .cor phase corrected file instead of raw file
IG.set_mode(RAW) # work from raw file, this is default
s = Spectra(IG, num_pcs = len(IG)) # phase-correct entire interferogram into one spectrum over Nyquist = 0.1:0.4
transmission = np.abs(s[0])
# Or just phase-correct first 100 IGs over Nyquist window 0.17-0.44
# (search for 'DEFAULTS[' in this script to see default values)
s_quick_look = Spectra(IG, num_pcs = 100, stop = 100, pc_lim_low = .17, pc_lim_high = .44)
transmission = np.abs(s[0])

Word to the wise:
Be careful about real vs complex values when converting from interferogram <-> spectra
.cor files are complex interferograms, and .raw files are real interferograms
'''
# TODO:
#   Add in DCSLoaded data for IGs, not just spectra
#       How to handle DCS data that is either spectra or IGs?
#   Move phase correction to functions instead of in class initializations?
#       Might make custom phase correction routines easier?
#   Redo file organization methods to work with new daqfiles class
#
#   How to calc p2p based on input data?
#       NI raw data range +/-; VC707 depends on num_hwavgs

from __future__ import division
from six import iteritems
try: # python 2
    range = xrange
except NameError: # python 3
    pass
try: # python 2
    import itertools.izip as zip
except ImportError: # python 3
    pass
try: # python 2
    basestring
except NameError:
    basestring = (str, bytes)

import os
import time
import sys
import traceback
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from shutil import copy
from zipfile import ZipFile, ZIP_DEFLATED
from itertools import repeat
from scipy.signal import hilbert
from collections import OrderedDict
from enum import IntEnum
from re import search as re_search
from numbers import Integral

try:
    from .constants import SPEED_OF_LIGHT as c
except (ValueError, SystemError, ImportError):
    from constants import SPEED_OF_LIGHT as c

# Define modes for DAQ file organization
class FileModes(IntEnum):
    COPY = 0
    MOVE = 1
    ZIPCOPY = 2
    ZIPMOVE = 3

# Data mode for DAQFiles
class DataMode(IntEnum):
    RAW = 0
    PC = 1

class IterableInt(int):
    def __iter__(self):
        return repeat(self)
    def __getitem__(self, i):
        return self

def ceildiv(a, b):
    return -(-a // b)

def unwrap(a, r=2*np.pi, thresh=0.5):
    """
    An unwrap function with an arbitrary range (numpy locked at 2pi)
    Currently assumes a is 1d array
    """
    try: # If range r is not integer but a is make sure b is not integer
        if isinstance(a[0], Integral) and not isinstance(r, Integral):
            b = np.array(a, np.float64)
        else:
            b = np.copy(a)
    except IndexError:
        return np.empty(0)
    r = abs(r); thresh_value = r*abs(thresh)
    d = np.diff(a)
    b[1:] -= np.cumsum(np.subtract(d>thresh_value, d<-thresh_value, dtype=np.int8))*r
    return b
    # Iterative solution
    correction = 0
    for ii in range(1, len(a)):
        diff = a[ii] - a[ii-1]
        if abs(diff) > thresh_value:
            correction += np.sign(diff)*r
        b[ii] -= correction
    return b

def force_fit_line(x, y, thresh=0.05):
    """
    2 step line fit: fits a linear model to a set of data, then removes
    outliers and refits a line.

    INPUTS:
        x = array_like, shape (M,): x-coordinates of the M sample
            points (x[i], y[i]).
        y = array_like, shape (M,): y-coordinates of the sample points.
        thresh = fractional threshold, in terms of the max residual from the
            initial line fit, above which those points will be ignored in the
            second line fit. Can be set to None to weight fits based on
            1/residual.

    OUTPUTS:
        p = Fit line coefficients (slope, y-intercept)

    """
    linefit = np.polyfit(x, y, 1)
    line = np.poly1d(linefit)
    r = np.abs(line(x) - y)

    if thresh is not None:
        thresh_value = thresh*np.max(r)
        weights = (r < thresh_value).astype(int)
        if np.sum(weights) < 5:
            thresh = None
    if thresh is None:
        weights = 1/r

    return np.polyfit(x, y, 1, w=weights)

def str_lst_search(list_of_strings, substring):
    """
    Looks for a string within a list of strings

    INPUTS:
        list_of_strings = list of strings to search within
        substring       = target string to find

    OUTPUTS:
        (row, location) = tuple containing the row and location, both int,
                          of the target string. Otherwise returns (-1,-1) as
                          failure condition
    """
    for row, string in enumerate(list_of_strings):
        location = string.find(substring)
        if location > -1:
            return (row, location)
    return (-1, -1)

# Define constants for concatenation, parsing, and testing
EXT_LOG_FILE = 'log'
EXT_RAW_FILE = 'raw'
EXT_PCD_FILE = 'cor'
EXT_PCP_FILE = 'corpar'
EXTS = (EXT_LOG_FILE, EXT_RAW_FILE, EXT_PCD_FILE, EXT_PCP_FILE)

class NIConstants:
    start_time = 'Start time'
    start_time_fmt_1 = 'Start time: %m/%d/%Y %H:%M:%S\n'
    start_time_fmt_2 = 'Start time: %m/%d/%Y %I:%M:%S %p\n'
    frame_length = 'Frame length'
    num_hwavgs = 'Num of Avg'
    num_pcs_realtime = '# phasecorrections'
    pc_lim_low = 'LowFreq'
    pc_lim_high = 'HighFreq'

    # The LabVIEW software uses big-endian style
    # More information: https://en.wikipedia.org/wiki/Endianness
    dtype_raw = np.dtype(np.float64).newbyteorder('>')
    dtype_pc = np.dtype(np.complex128).newbyteorder('>')


class VC707Constants:
    num_hwavgs = 'num_hwavgs'
    num_hwavgs_prop = '@num_hwavgs'
    frame_length = 'frame_length'
    p2p_range = 'p2p_range'
    p2p_min = 'p2p_min'
    p2p_max = 'p2p_max'
    p2p_total = 'p2p_total'
    pc_win = 'pc_win'
    pc_lim_low = 'pc_lim_low'
    pc_lim_high = 'pc_lim_high'
    start_time = 'start_time_local'
    start_time_utc = 'start_time_utc'
    byte_order = 'byte_order'
    trigger_name = 'trigger_name'
    timeout = 'timeout'
    max_time = 'max_time'
    runtime = 'runtime'
    num_igs_accepted = 'igms_accepted'
    num_igs_received = 'igms_received'
    time_fmt = '%Y%m%d%H%M%S'
    notes = 'notes'
    num_pcs_realtime = 'num_pcs'
    pc_time = 'pc_time'
    igs_per_pc = 'igs_per_pc'

    Fr1 = 'fr1'
    Fr2 = 'fr2'
    DFr = 'dfr'
    Fr1_setpoint = 'fr1_setpoint'

    dtype_raw = np.dtype(np.int32)
    dtype_pc = np.dtype(np.complex128)


def get_lines(item):
    try:
        seeking_old = item.tell()
        item.seek(0)
        line_list = item.readlines()
        item.seek(seeking_old)
        return line_list
    except AttributeError:
        try:
            return item.readlines()
        except AttributeError:
            try:
                with open(item, 'r') as f:
                    return f.readlines()
            except (IOError, TypeError):
                try:
                    return item.split('\n')
                except AttributeError:
                    return item # assume item is list of string settings


def parse_ni_log(log):
    line_list = get_lines(log)

    # Grab the start time row
    time_row, _ = str_lst_search(line_list, NIConstants.start_time)
    # Parse to datetime
    try:
        settings = {'start_time':datetime.strptime(line_list[time_row], NIConstants.start_time_fmt_1)}
    except ValueError:
        settings = {'start_time':datetime.strptime(line_list[time_row], NIConstants.start_time_fmt_2)}

    for setting, cast in (('frame_length', int), ('num_hwavgs', int),
                          ('num_pcs_realtime', int), ('pc_lim_low', float),
                          ('pc_lim_high', float)):
        name_row, _ = str_lst_search(line_list, getattr(NIConstants, setting))
        search_result = re_search('<Val>(.*)</Val>', line_list[name_row + 1])
        settings[setting] = cast(search_result.group(1))
    return settings


def parse_vc707_log(log):
    line_list = get_lines(log)
    a = {}
    for l in line_list:
        # TODO: if notes is multiple lines will only read in 1st line
        try:
            setting_pair = l.strip().split(' = ')
            if len(setting_pair) != 2:
                name = setting_pair[0].strip().lower()
                if name == VC707Constants.notes:
                    v = ' = '.join(setting_pair[1:])
                    a[name] = v
            a[setting_pair[0].strip().lower()] = setting_pair[1].strip()
        except:
            pass

    settings = {}
    settings['frame_length'] = int(round(float(a.pop(VC707Constants.frame_length))))
    settings['p2p_total'] = float(a.pop(VC707Constants.p2p_total))
    settings['start_time'] = datetime.strptime(a.pop(VC707Constants.start_time), VC707Constants.time_fmt)
    settings['start_time_utc'] = datetime.strptime(a.pop(VC707Constants.start_time_utc), VC707Constants.time_fmt)
    settings['byte_order'] = a.pop(VC707Constants.byte_order)
    settings['runtime'] = float(a.pop(VC707Constants.runtime))
    settings['num_igs_accepted'] = int(round(float(a.pop(VC707Constants.num_igs_accepted))))
    settings['num_igs_received'] = int(round(float(a.pop(VC707Constants.num_igs_received))))
    try:
        settings['timeout'] = float(a.pop(VC707Constants.timeout))
    except (KeyError, ValueError):
        settings['timeout'] = float(a.pop(VC707Constants.max_time))
    try:
        settings['num_hwavgs'] = int(round(float(a.pop(VC707Constants.num_hwavgs))))
    except (KeyError, ValueError, OverflowError):
        settings['num_hwavgs'] = int(round(float(a.pop(VC707Constants.num_hwavgs_prop))))
    try:
        p2p_range = a.pop(VC707Constants.p2p_range).strip('()[] ')
        split = p2p_range.split(',')
        if len(split) != 2:
            split = p2p_range.split(' ')
            if len(split) != 2:
                raise ValueError()
        p2p_range = tuple(float(x) for x in split)
        settings['p2p_min'], settings['p2p_max'] = p2p_range
    except (KeyError, ValueError):
        settings['p2p_min'] = float(a.pop(VC707Constants.p2p_min))
        settings['p2p_max'] = float(a.pop(VC707Constants.p2p_max))
    try:
        pc_win = a.pop(VC707Constants.pc_win).strip('()[] ')
        split = pc_win.split(',')
        if len(split) != 2:
            split = pc_win.split(' ')
            if len(split) != 2:
                raise ValueError()
        pc_win = tuple(float(x) for x in split)
        settings['pc_lim_low'], settings['pc_lim_high'] = p2p_range
    except (KeyError, ValueError):
        settings['pc_lim_low'] = float(a.pop(VC707Constants.pc_lim_low))
        settings['pc_lim_high'] = float(a.pop(VC707Constants.pc_lim_high))

    try:
        settings['num_pcs_realtime'] = int(round(float(a.pop(VC707Constants.num_pcs_realtime))))
    except (KeyError, ValueError, OverflowError):
        settings['pc_time'] = float(a.pop(VC707Constants.pc_time))
        igs_per_pc = a.pop(VC707Constants.igs_per_pc).strip('()[] ')
        if len(igs_per_pc) > 0:
            split = igs_per_pc.split(',')
            if len(split) < 2:
                split = igs_per_pc.split(' ')
            settings['igs_per_pc'] = tuple(int(round(float(x))) for x in split)
        else:
            settings['igs_per_pc'] = 0
    # Parameters termed to be optional
    try:
        settings['trigger_name'] = a.pop(VC707Constants.trigger_name)
        if settings['trigger_name'].lower() == 'none':
            settings['trigger_name'] = None
    except KeyError:
        pass
    try:
        settings['notes'] = a.pop(VC707Constants.notes)
    except KeyError:
        pass
    try:
        settings['fc'] = float(a.pop(VC707Constants.Fr1))
    except (KeyError, ValueError):
        pass
    try:
        settings['fr2'] = float(a.pop(VC707Constants.Fr2))
    except (KeyError, ValueError):
        pass
    try:
        settings['dfr'] = float(a.pop(VC707Constants.DFr))
    except (KeyError, ValueError):
        pass
    try:
        settings['fc_set'] = float(a.pop(VC707Constants.Fr1_setpoint))
    except (KeyError, ValueError):
        pass
    # parameters not currently loaded:
    # pc_mos, p2p_thresh, pc_width, timezone
    return settings


class Settings(object):

    def __init__(self, obj):
        self.DEFAULTS = obj.DEFAULTS
        self.obj = obj
        self.__objdict = obj.__dict__

    def __getitem__(self, i):
        return self.__objdict.__getitem__(i)

    def __setitem__(self, i, v):
        self.__objdict.__setitem__(i, v)

    def __iter__(self):
        for setting in self.DEFAULTS:
            yield setting, self.__objdict[setting]
        raise StopIteration

    def __str__(self):
        return '{' + ', '.join('%s: %s' % item for item in self) + '}'

    def reset(self, settings=None):
        if settings is None:
            for setting, default in iteritems(self.DEFAULTS):
                self.__objdict[setting] = default
        else:
            for setting in self.DEFAULTS:
                try:
                    self.__objdict[setting] = settings[setting]
                except KeyError:
                    pass


class ConfigurableObject(object):

    DEFAULTS = {} # can be OrderedDict if order is important

    def __init__(self, **kwargs):
        for setting, default in iteritems(self.DEFAULTS):
            try:
                self.__setattr__(setting, kwargs.pop(setting))
            except KeyError:
                self.__setattr__(setting, default)
        self.settings = Settings(self)
        if len(kwargs):
            print('Warning initializing %s instance: unknown kwargs %s' % (self.__class__.__name__, kwargs))

    def reset(self, settings=None):
        self.settings.reset(settings)


class DCSData(ConfigurableObject):

    DEFAULTS = OrderedDict()
    DEFAULTS['start_time'] = None
    DEFAULTS['start_time_utc'] = None
    DEFAULTS['timeout'] = None
    DEFAULTS['runtime'] = None
    DEFAULTS['num_igs_accepted'] = None
    DEFAULTS['num_igs_received'] = None
    DEFAULTS['frame_length'] = None
    DEFAULTS['num_hwavgs'] = None
    DEFAULTS['num_pcs'] = None
    DEFAULTS['p2p_min'] = 0.0
    DEFAULTS['p2p_max'] = float('inf')
    DEFAULTS['p2p_scale'] = None
    DEFAULTS['pc_lim_low'] = 0.1
    DEFAULTS['pc_lim_high'] = 0.4
    DEFAULTS['p2p_total'] = None
    DEFAULTS['byte_order'] = None
    DEFAULTS['trigger_name'] = None
    DEFAULTS['data_source'] = None
    # NAM added
    DEFAULTS['fc'] = 200e6
    DEFAULTS['fr2'] = None
    DEFAULTS['dfr'] = None
    DEFAULTS['fc_set'] = None

    def reset(self):
        print('Cannot reset')

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data.__getitem__(i)


class DAQFiles(DCSData):
    """
    Class to interface with DAQ files on the disk without loading to RAM
    """
    DEFAULTS = OrderedDict(DCSData.DEFAULTS)
    DEFAULTS['num_pcs_realtime'] = None

    DTYPE_RAW = None
    DTYPE_PC = None

    def __init__(self, filename):
        """
        INPUTS:
            filename = string of the filename of DAQ file

        OUTPUTS:
            None

        ERRORS:
            TODO
        """
        # self.base_path = directory + base file name of log file (no extension)
        # self.base_dir = directory (or zip file) conatining DAQ files
        # self.base_fname = log file name with all extensions stripped
        self.base_path = os.path.realpath(filename.strip())

        self.file_log, self.data_raw, self.data_raw_source, self.data_pc, self.data_pc_source, self.file_corpar = [None]*6
        self.is_zip_source = False

        try:
            # If it's a .zip, then use the zfile library to peak inside
            if self.base_path[-4:].lower() == '.zip':
                self.zfile = ZipFile(self.base_path, 'r')
                self.is_zip_source = True
                # Iterate through file heirarchy
                for fname in self.zfile.namelist():
                    f_stripped = fname.strip()
                    for ext in EXTS:
                        f_ext = f_stripped[-(len(ext)+1):].lower()
                        if f_ext == '.' + ext or f_ext == '_' + ext:
                            if ext == EXT_LOG_FILE and self.file_log is None:
                                self.file_log = self.zfile.open(fname, 'r')
                                self.base_fname = f_stripped[:-len(ext)]
                                self.base_fname.rstrip('_.')
                            elif ext == EXT_RAW_FILE and self.data_raw_source is None:
                                self.data_raw_source = fname
                            elif ext == EXT_PCD_FILE and self.data_pc_source is None:
                                self.data_pc_source = fname
                            elif ext == EXT_PCP_FILE and self.file_corpar is None:
                                self.file_corpar = fname
                            break
                if self.file_log:
                    self.base_dir = self.base_path
                    self.base_path = os.path.join(self.base_dir, self.base_fname)

            else:
                for ext in EXTS: # if base_path had an extension then remove it
                    bp_ext = self.base_path[-(len(ext)+1):].lower()
                    if bp_ext == '.' + ext or bp_ext == '_' + ext:
                        self.base_path = self.base_path[:-(len(ext)+1)]
                        break
                self.base_dir, self.base_fname = os.path.split(self.base_path)
                # Open files as read-only, or read-only-binary defined by extension type
                for ext in EXTS:
                    for fname in (self.base_path + '.' + ext, self.base_path + '_' + ext):
                        if os.path.isfile(fname):
                            if ext == EXT_LOG_FILE:
                                self.file_log = open(fname, 'r')
                            elif ext == EXT_RAW_FILE:
                                self.data_raw = np.memmap(fname, dtype=self.DTYPE_RAW, mode='r')
                                self.data_raw_source = fname
                            elif ext == EXT_PCD_FILE:
                                self.data_pc = np.memmap(fname, dtype=self.DTYPE_PC, mode='r')
                                self.data_pc_source = fname
                            elif ext == EXT_PCP_FILE:
                                self.file_corpar = fname
                            break

            if not self.file_log:
                raise ValueError('DAQ Files %s did not contain a log file' % self.base_path)
            self._finalize_init()
        except:
            self.close()
            raise

        # TODO: For now assume a rep rate clock
        # This assumes a stable, correctly locked DCS
        if self.dfr is None:
            self.is_fc_known = False
            self.fc = 200e6 # Hz of clock signal
            self.dfr = self.fc/self.frame_length
            self.nyquist_window = 2.0

        self.has_data = False
        if self.is_zip_source:
            if self.data_raw_source is not None:
                self.has_data = True
            elif self.data_pc_source is not None:
                self.has_data = True
            else:
                self.mode = None
                self.data = None
        else:
            if self.data_raw is not None:
                self.data_raw.shape = (-1, self.frame_length)
                if len(self.data_raw):
                    self.has_data = True
                    self.set_mode(DataMode.RAW)
                else:
                    self.data_raw = None
                    self.data_raw_source = None
            if self.data_pc is not None:
                self.data_pc.shape = (-1, self.frame_length)
                if len(self.data_pc):
                    if not self.has_data:
                        self.has_data = True
                        self.set_mode(DataMode.PC)
                else:
                    self.data_pc = None
                    self.data_pc_source = None
            if not self.has_data:
                # TODO: how to handle this, what to set things to
                self.mode = None
                self.data = None

    def _finalize_init(self):
        # must call super(DAQFiles, self).__init__()
        raise NotImplementedError

    def set_mode(self, mode):
        # TODO: handle zip_source mode set
        if mode == DataMode.RAW:
            if self.data_raw is None:
                raise ValueError('Has no raw IGs')
            self.data = self.data_raw
            self.num_pcs = 1
            self.num_igs_per = IterableInt(1)
        elif mode == DataMode.PC:
            if self.data_pc is None:
                raise ValueError('Has no phase corrected IGs')
            self.data = self.data_pc
            self.num_pcs = self.num_pcs_realtime
            self.num_igs_per = np.empty(len(self.data), dtype=np.uint32)
            self.num_igs_per[:] = self.num_pcs
            if len(self.data)*self.num_pcs > self.num_igs_accepted:
                self.num_igs_per[-1] = self.num_igs_accepted - (len(self.data) - 1)*self.num_pcs
        else:
            raise ValueError('Data reading mode (%r) unknown' % mode)
        self.mode = mode
        self.data_source = self.data.filename

    def calc_p2p_total(self):
        if self.p2p_total is not None:
            return
        if self.data_raw is not None:
            self.p2p_total = 0
            for ig in self.data_raw:
                p2p = (max(ig) - min(ig))*self.p2p_scale
                self.p2p_total += p2p
        elif self.data_pc is not None:
            self.p2p_total = 0
            for ig in self.data_pc:
                self.p2p_total += max(abs(ig))*2*self.p2p_scale
        else:
            raise ValueError('Has no data')

    def is_open_elsewhere(self):
        """
        Determine if the file is open elsewhere by trying to rename it

        TODO:
            - Modify so this works on *Unix
            - Does this work if we have the files open? probably not
            - And for the zip files?
        """
        try:
            os.rename(self.file_log.name, self.file_log.name)
            return False
        except WindowsError:
            return True

    def close(self):
        """
        Close everything
        This helps to make sure that other programs can gain access, and Windows won't freak out
        """
        self.data = None
        self.data_raw = None
        self.data_pc = None
        if self.file_log:
            self.file_log.close()
        if self.is_zip_source:
            self.zfile.close()

    def __del__(self):
        """
        If this object is deleted, then make sure to close everything
        """
        self.close()


class DAQFilesNI(DAQFiles):
    DTYPE_RAW = NIConstants.dtype_raw
    DTYPE_PC = NIConstants.dtype_pc

    def _finalize_init(self):
        # Generate the DAQ log from the input filename
        log_settings = parse_ni_log(self.file_log.readlines())
        log_settings['byte_order'] = 'big'
        # Set objects DAQ attributes from the log
        super(DAQFiles, self).__init__(**log_settings)

        if self.data_raw is None:
            # NI DAQ puts no extension on raw data, try to find it
            if self.is_zip_source:
                if self.base_fname in self.zfile.namelist():
                    self.data_raw_source = self.base_fname
            else:
                try:
                    self.data_raw = np.memmap(self.base_path, dtype=self.DTYPE_RAW, mode='r')
                except (IOError, ValueError):
                    pass
        if self.data_raw is not None:
            self.num_igs_accepted = len(self.data_raw)
        elif self.data_pc is not None:
            self.num_igs_accepted = len(self.data_pc)*self.num_pcs_realtime
        self.num_igs_received = self.num_igs_accepted
        self.p2p_scale = 1


class DAQFilesVC707(DAQFiles):
    DTYPE_RAW = VC707Constants.dtype_raw
    DTYPE_PC = VC707Constants.dtype_pc

    def _finalize_init(self):
        # Generate the DAQ log from the input filename
        log_settings = parse_vc707_log(self.file_log.readlines())
        # Set objects DAQ attributes from the log
        super(DAQFiles, self).__init__(**log_settings)
        try:
            self.igs_per_pc = log_settings['igs_per_pc']
        except:
            pass

        if self.byte_order == 'little':
            bo = '<'
        elif self.byte_order == 'big':
            bo = '>'
        else:
            raise ValueError('Unrecognized byte order %r' % self.byte_order)
        if self.data_raw is not None:
            self.data_raw.dtype = self.data_raw.dtype.newbyteorder(bo)
        if self.data_pc is not None:
            self.data_pc.dtype = self.data_pc.dtype.newbyteorder(bo)
        self.p2p_scale = 1/((pow(2, 14 - 1) - 1)*self.num_hwavgs)


def open_daq_files(filename):
    err_str = '\nAttempt to open %s as DAQFiles gave following exceptions\n' % filename
    types = (DAQFilesVC707, DAQFilesNI)
    for t in types:
        try:
            return t(filename)
        except:
            err_str += '\nException initializing as %s:' % t.__name__
            err_str += traceback.format_exc().lstrip('Traceback (most recent call last):')
    raise ValueError(err_str.replace('\n', '\n\t'))


class DCSDataLoaded(DCSData):
    # TODO: set start, stop , step to reflect which igs from raw data are being used?
    DEFAULTS = OrderedDict(DCSData.DEFAULTS)
    DEFAULTS['num_pcs'] = 1
    DEFAULTS['start'] = None
    DEFAULTS['stop'] = None
    DEFAULTS['step'] = None

    def __init__(self, data_source, **kwargs):
        for setting in DCSData.DEFAULTS:
            if setting in ('p2p_min', 'p2p_max', 'num_pcs', 'byte_order', 'start', 'stop', 'step'):
                continue
            if setting in ('pc_lim_low', 'pc_lim_high'):
                if setting in kwargs:
                    continue
            kwargs[setting] = data_source.settings[setting]
        kwargs['byte_order'] = sys.byteorder
        super(DCSDataLoaded, self).__init__(**kwargs)

        if self.pc_lim_low < 0:
            self.pc_lim_low = 0.0
        elif self.pc_lim_low > 0.5:
            self.pc_lim_low = 0.5
        if self.pc_lim_high > 0.5:
            self.pc_lim_high = 0.5
        elif self.pc_lim_high < 0.0:
            self.pc_lim_high = 0.0
        if self.pc_lim_high < self.pc_lim_low:
            t = self.pc_lim_high
            self.pc_lim_high = self.pc_lim_low
            self.pc_lim_low = t

        s = slice(self.start, self.stop, self.step)
        self.start, self.stop, self.step = s.indices(len(data_source))
        data = data_source[s]
        num_pcs = self.num_pcs
        if num_pcs > len(data):
            num_pcs = len(data)
        elif num_pcs < 1:
            num_pcs = 1
        self.num_pcs = data_source.num_pcs * num_pcs
        self._load_data(data, data_source.num_igs_per[s], num_pcs)

    def _load_data(self, data, input_igs_per, num_pcs):
        raise NotImplementedError


class Spectra(DCSDataLoaded):

    def _load_data(self, data, input_igs_per, num_pcs):#, match_peak = False):
        """
        Phase correction happens here
        TODO: assumes data is IG, but should be able to accept spectrum
        """

        is_input_complex = np.iscomplexobj(data)
        num_spectra = ceildiv(len(data), num_pcs)
        len_spectrum = self.frame_length//2 + 1 # based on rfft result
        self.spectra = np.empty((num_spectra, len_spectrum), dtype=np.complex128)
        # Have to make this 2d in order to iterate over rows
        self.num_igs_per = np.zeros((num_spectra, 1), dtype=np.uint32)
        self.p2p_total = 0

        pc_start_pt = int(round(self.pc_lim_low*len_spectrum*2))
        pc_end_pt = int(round(self.pc_lim_high*len_spectrum*2))

        iter_data = iter(data)
        iter_num = iter(input_igs_per)

        try:
            for spectrum, num_igs in zip(self.spectra, self.num_igs_per):
                for i in range(num_pcs):
                    ig = next(iter_data); num_igs_in = next(iter_num)
                    p2p = np.max(ig.real)
                    if p2p < 0:
                        p2p = 0
                    else:
                        p2p *= 2*self.p2p_scale
                    if p2p < self.p2p_min or p2p > self.p2p_max:
                        continue
                    if is_input_complex:
                        real = ig.real; imag = ig.imag
                        p2p = np.sqrt(np.max(real*real + imag*imag))*2*self.p2p_scale
                        if p2p < self.p2p_min or p2p > self.p2p_max:
                            continue
                        fft = np.fft.fft(ig)[:len_spectrum]
                    else:
                        p2p = np.max(ig.real)
                        if p2p < 0:
                            p2p = 0
                        else:
                            p2p *= 2*self.p2p_scale
                        if p2p < self.p2p_min or p2p > self.p2p_max:
                            continue
#                        if match_peak:
#                            peak_pt = np.argmax(np.abs(ig))
#                            ig = np.roll(ig, -peak_pt)
                        fft = np.fft.rfft(ig)
                    if num_igs == 0:
                        if num_pcs > 1:
                            if pc_start_pt == pc_end_pt:
                                phase_ref_index = pc_start_pt
                            else:
                                real = fft.real[pc_start_pt:pc_end_pt]
                                imag = fft.imag[pc_start_pt:pc_end_pt]
                                mag2 = (real*real) + (imag*imag)
                                phase_ref_index = np.argmax(mag2) + pc_start_pt
                            phase_ref = np.angle(fft[phase_ref_index])
                        spectrum[:] = fft
                    else:
                        dphase = np.angle(fft[phase_ref_index]) - phase_ref
                        spectrum += fft*np.exp(-1j*dphase)
                    num_igs += num_igs_in
                    self.p2p_total += p2p
                if num_igs == 0:
                    spectrum[:] = 0
        except StopIteration:
            pass
        # Turn this back to 1d array
        self.num_igs_per.shape = (num_spectra,)
        self.data = self.spectra

    def fliplr(self):
        self.data = np.fliplr(self.data)

    def phase_average_further(self, match_peaks = False):
        '''
        After first calling Spectra object and producing complex phase-averaged FFTs,
        further phase-correct FFTs
        Same phase-correction algorithm as above, except start w/ ffts
        OUTPUT:
            object is now single real-valued transmission spectrum.

        TODO: Debug Fabrizzio algorithm with np.roll
        '''
        len_spectrum = self.frame_length//2 + 1 # based on rfft result
        pc_start_pt = int(round(self.pc_lim_low*len_spectrum*2))
        pc_end_pt = int(round(self.pc_lim_high*len_spectrum*2))
        spectrum_out = self.data[0]
        try:
            for num_igs, spectrum in enumerate(self.data):
                if match_peaks:
                    ig = np.fft.irfft(spectrum)
                    peak_pt = np.argmax(np.abs(ig))
                    spectrum = np.fft.fft(np.roll(ig, -peak_pt))[:len_spectrum]
                if num_igs == 0:
                    if self.num_pcs > 1:
                        if pc_start_pt == pc_end_pt:
                            phase_ref_index = pc_start_pt
                        else:
                            mag2 = np.abs(spectrum[pc_start_pt:pc_end_pt])
                            phase_ref_index = np.argmax(mag2) + pc_start_pt
                        phase_ref = np.angle(spectrum[phase_ref_index])
                    spectrum_out[:] = spectrum
                else:
                    dphase = np.angle(spectrum[phase_ref_index]) - phase_ref
                    spectrum_out += spectrum * np.exp(-1j * dphase)

        except StopIteration:
            pass

        self.data = np.abs(spectrum_out)


class IGs(DCSDataLoaded):

    def _load_data(self, data, input_igs_per, num_pcs):
        """
        Phase correction happens here (TODO: currently only coadd)
        TODO: assumes data is IG, but should be able to accept spectrum
        """
        num_igs_coor = ceildiv(len(data), num_pcs)
        self.igs = np.zeros((num_igs_coor, self.frame_length), dtype=np.float64)
        # Have to make this 2d in order to iterate over rows
        self.num_igs_per = np.zeros((num_igs_coor, 1), dtype=np.uint32)

        iter_data = iter(data)
        iter_num = iter(input_igs_per)

        try:
            for ig, num_igs in zip(self.igs, self.num_igs_per):
                for i in range(num_pcs):
                    ig_in = next(iter_data); num_igs_in = next(iter_num)
                    p2p = np.max(ig_in.real)
                    if p2p < 0:
                        p2p = 0
                    else:
                        p2p *= 2*self.p2p_scale
                    if p2p < self.p2p_min or p2p > self.p2p_max:
                        continue
                    if num_igs == 0:
                        if num_pcs > 1:
                            pass
                        ig[:] = ig_in
                    else:
                        ig += ig_in
                    num_igs += num_igs_in

                if num_igs == 0:
                        ig[:] = 0
        except StopIteration:
            pass
        # Turn this back to 1d array
        self.num_igs_per.shape = (num_igs_coor,)
        self.data = self.igs


def pc_crosscorrelate(igs, sample_width=2**11):
    sample_width = pow(2, int(round(np.log2(sample_width))))
    i_center = None
    igc = np.zeros(len(igs[0]), dtype=np.complex128)
    x_range = np.arange(2)
    for ig in igs:
        # ig = ig - np.mean(ig)
        if i_center is None:
            i_center = np.argmax(ig)
            start = i_center - sample_width//2; stop = start + sample_width
            centerburst = ig[start:stop]
            # make ig_center 1 point longer so result of convolve will be 2^n in length
            stop += 1
        ig_center = ig[start:stop]
        xcreal = np.correlate(centerburst, ig_center, 'full')
        xc = hilbert(xcreal)
        xcmag = abs(xc)
        i_max = np.argmax(xcmag)

        fit = np.polyfit(range(i_max-1, i_max+2), xcmag[i_max-1:i_max+2], 2)
        delay = fit[1]/(-2*fit[0])
        delay0 = int(delay); delay2 = delay0 + 2
        phase = np.interp(delay % 1, x_range, np.angle(xc[delay0:delay2]))
        igc += ig*np.exp(1j*phase)
    return igc


def pc_hilbert(igs, sample_width=2**8):
    sample_width = pow(2, int(round(np.log2(sample_width))))
    i_center = None
    igc = np.zeros(len(igs[0]), dtype=np.complex128)
    x_range = np.arange(2)
    for ig in igs:
        # ig = ig - np.mean(ig)
        if i_center is None:
            i_center = np.argmax(ig)
            start = i_center - sample_width//2; stop = start + sample_width
        h = hilbert(ig[start:stop])
        hmag = abs(h)
        i_max = np.argmax(hmag)

        fit = np.polyfit(range(i_max-1, i_max+2), xcmag[i_max-1:i_max+2], 2)
        delay = fit[1]/(-2*fit[0])
        delay0 = int(delay); delay2 = delay0 + 2
        phase = np.interp(delay % 1, x_range, np.angle(xc[delay0:delay2]))
        igc += ig*np.exp(1j*phase)
    return igc


def pc_truncated(igs, pc_lim_low=0.1, pc_lim_high=0.4, sample_width=2**11):
    sample_width = pow(2, int(round(np.log2(sample_width))))
    ig0 = igs[0]
    is_input_complex = np.iscomplexobj(ig0)
    center = np.argmax(ig0)
    start = center - sample_width//2; stop = start + sample_width
    igc = np.array(ig0, dtype=np.complex128)

    if is_input_complex:
        s = np.fft.fft(ig0[start:stop])
        pc_start = int(round(pc_lim_low*len(s)))
        pc_stop = int(round(pc_lim_high*len(s)))
    else:
        s = np.fft.rfft(ig0[start:stop])
        len_2 = 2*len(s)
        pc_start = int(round(pc_lim_low*len_2))
        pc_stop = int(round(pc_lim_high*len_2))
    r = s.real[pc_start:pc_stop]
    i = s.imag[pc_start:pc_stop]
    mag = r*r + i*i
    ref_index = pc_start + np.argmax(mag)
    phase_ref = np.angle(s[ref_index])

    for ig in igs[1:]:
        if is_input_complex:
            s = np.fft.fft(ig[start:stop])
        else:
            s = np.fft.rfft(ig[start:stop])
        dphase = np.angle(s[ref_index]) - phase_ref
        igc += ig*np.exp(-1j*dphase)

    return igc


def pc_nearest_2(igs, pc_lim_low=0.1, pc_lim_high=0.4):
    ig0 = igs[0]
    ig_width = pow(2, int(np.log2(len(ig0))))
    spectrum_width = ig_width//2 + 1 # based on output of rfft
    is_input_complex = np.iscomplexobj(ig0)
    center = np.argmax(ig0)
    start = center - ig_width//2; stop = start + ig_width

    if is_input_complex:
        sc = np.fft.fft(ig0[start:stop])[:spectrum_width]
    else:
        sc = np.fft.rfft(ig0[start:stop])
    len_2 = 2*len(sc)
    pc_start = int(round(pc_lim_low*len_2))
    pc_stop = int(round(pc_lim_high*len_2))
    r = sc.real[pc_start:pc_stop]
    i = sc.imag[pc_start:pc_stop]
    mag = r*r + i*i
    ref_index = pc_start + np.argmax(mag)
    phase_ref = np.angle(sc[ref_index])

    for ig in igs[1:]:
        if is_input_complex:
            s = np.fft.fft(ig[start:stop])[:spectrum_width]
        else:
            s = np.fft.rfft(ig[start:stop])
        dphase = np.angle(s[ref_index]) - phase_ref
        sc += s*np.exp(-1j*dphase)

    return sc


def get_walking_rate2(data_source, igs_to_use=1000, plot=False):
    """
    INPUTS:
        data_source: DCSData object to calc values on
        igs_to_use = the max number of igs to use in the ref laser calc
        plot = boolean, whether to plot the ig walking results

    OUTPUTS:
        walk_rate = the number of points the peak of IG moves each
            delta fc time step

    ERRORS:
        None
    """
    if igs_to_use > len(data_source):
        igs_to_use = len(data_source)

    # Have to make this 2d in order to iterate over rows
    maxes = np.empty((igs_to_use, 1), dtype=np.uint32)
    for max, ig in zip(maxes, data_source.data[:igs_to_use]):
        max[0] = np.argmax(ig.real)
        # Turn this back to 1d array
    maxes.shape = (igs_to_use,)
    maxes = unwrap(maxes, data_source.frame_length ,0.95)

    x = tuple(range(igs_to_use))
    linefit = force_fit_line(x, maxes)

    num_igs_avgd = data_source.num_pcs*data_source.num_hwavgs
    walk_rate = linefit[0]/num_igs_avgd

    if plot:
        line = np.poly1d(linefit)
        plt.figure()
        plt.plot(x, maxes, x, line(x))

    return walk_rate


def get_walking_rate(data_source, igs_to_use=1000, plot=False):
    """
    INPUTS:
        data_source: DCSData object to calc values on
        igs_to_use = the max number of igs to use in the ref laser calc
        plot = boolean, whether to plot the ig walking results

    OUTPUTS:
        walk_rate = the number of points the peak of IG moves each
            delta fc time step

    ERRORS:
        None
    """
    if igs_to_use > len(data_source):
        igs_to_use = len(data_source)

    data = data_source.data[:igs_to_use].real
    fl = data_source.frame_length
    dtype = data.dtype.newbyteorder('N')

    means = np.mean(data, axis=1)
    maxlocs = np.argmax(data, axis=1)
    maxes = np.empty(igs_to_use, dtype=dtype)
    for i, (d, maxloc) in enumerate(zip(data, maxlocs)):
        maxes[i] = d[maxloc]
    # Move halfway around the ig from the max and assume it is all noise
    # at this point
    noiselocs = (maxlocs + fl//2) % fl
    # Unwrap the maxlocs to have walk be a continuous line
    maxlocs = unwrap(maxlocs, fl, 0.95)

    # Take noise_pts num of points around each noiseloc and find the max;
    # call this the noise level
    noise_pts = fl//100
    start_largest = fl - noise_pts
    starts = noiselocs - noise_pts//2
    stops = starts + noise_pts
    noise = np.empty(igs_to_use, dtype=dtype)
    for i, (d, start, stop) in enumerate(zip(data, starts, stops)):
        if start < 0:
            start = 0
            stop = noise_pts
        elif stop > fl:
            start = start_largest
            stop = fl
        noise[i] = np.amax(d[start:stop])

    # sf = signal factor, is a measure of how noisy the spectrum is
    sf = (maxes - means)/(noise - means) - 1

    # Begin fitting lines; first one weight by the noise factor
    x = tuple(range(igs_to_use))
    linefit = np.polyfit(x, maxlocs, 1, w=pow(sf, 2.5))
    line1 = np.poly1d(linefit)

    # Second line weighting is binary based on residual thresholding
    residuals = np.abs(line1(x) - maxlocs)
    thresh = 0.05
    thresh_value = thresh*np.amax(residuals)
    weights = (residuals < thresh_value).astype(int)
    linefit = np.polyfit(x, maxlocs, 1, w=weights)
    line2 = np.poly1d(linefit)

    if plot:
        plt.figure()
        plt.plot(x, maxlocs, x, line1(x), x, line2(x))
        plt.legend(('raw', 'signal fit', 'final fit'))

    num_igs_avgd = data_source.num_pcs*data_source.num_hwavgs
    walk_rate = linefit[0]/num_igs_avgd
    return walk_rate

def estimate_ref_laser(data_source, igs_to_use=1000, plot=False):
    """
    INPUTS:
        data_source: DCSData object to calc values on
        igs_to_use = the max number of igs to use in the ref laser calc
        plot = boolean, whether to plot the ig walking results

    OUTPUTS:
        laser_wl = the estimated wavelength of the reference laser used
            in generating these IGs

    ERRORS:
        None
    """
    walk_rate = get_walking_rate(data_source, igs_to_use, plot)
    frame_length_true = data_source.frame_length + walk_rate
    if data_source.dfr < 0: # clocking on faster comb
        dfNYQ = (frame_length_true - 1)*data_source.fc
    else: # clocking on slower comb
        dfNYQ = (frame_length_true + 1)*data_source.fc

    # Estimate which nyqueist window the laser is locked to
    laser_wl_est = 1565.0 # nm
    laser_f_est = 1e9*c/laser_wl_est # hz
    NYQref_estimate = int(round(laser_f_est/dfNYQ))
    laser_f = dfNYQ * NYQref_estimate

    laser_wl = 1e9*c/(laser_f)
    return laser_wl


def organize_daq_files(input_dir, output_dir, dirfmt='%Y%m%d', filefmt='%Y%m%d%H%M%S', time_thresh=1000, use_utc=False, mode=FileModes.MOVE):
    """
    Organize the DAQ files into a logical structure of subdirectories based on date

    INPUTS:
        input_dir   = string of DAQ log directory, does not need ending '\'
        output_dir  = string of directory to move everything to
        dirfmt      = string of subdirectory format to reorganize things
        filefmt     = string of datetime format stored in DAQ file to parse
        time_thresh = timeout time to find files, [sec]
        mode        = int of what to do with the files, defined in top of file

    OUTPUTS:
        None
    """
    if use_utc:
        raise NotImplementedError
    # Create directory strings
    input_dir = os.path.realpath(input_dir)
    output_dir = os.path.realpath(output_dir)
    # Look for the input files
    try:
        input_files = os.listdir(input_dir)
    except WindowsError: # TODO: make os independant
        print("Do not have access to input directory '%s'" % input_dir)
        return
    # Initialize log list, and start timing for information and debug
    now = time.time()
    daq_logs = []

    # Look for log files in input_dir
    # Doing 2 interations is not most efficient, but allows you to determine
    # number of files that must be moved before heavy lifting
    for f in input_files:
        ext = f[-(len(EXT_LOG_FILE) + 1):].lower()
        if ext != ('.' + EXT_LOG_FILE) and ext != ('_' + EXT_LOG_FILE):
            continue
        if now - os.path.getmtime(os.path.join(input_dir, f)) < time_thresh:
            continue
        daq_logs.append(f)

    num_daq_files = len(daq_logs)
    if not num_daq_files:
        print("No DAQ file sets found to organize in '%s'" % input_dir)
        return
    print('Found %i DAQ file sets; begin organization' % num_daq_files)
    percent_factor = 100.0/num_daq_files

    # Rearrange DAQ files into logical directories
    for index, daq_log in enumerate(daq_logs):
        if index:
            time_elapsed = (time.time() - now)/60 # minutes
            time_total = time_elapsed/index*num_daq_files
            time_remaining = '%0.2f' % (time_total - time_elapsed)
        else:
            time_remaining = 'N/A'
        print('%05.2f%% complete, %s minutes remaining' % (index*percent_factor, time_remaining))
        print('   Processing ' + os.path.join(input_dir, daq_log))
        # Get DAQFilesNI object for this particular daq_log, which contains the open fileid, and close it
        daq_files = open_daq_files(os.path.join(input_dir, daq_log))
        daq_files.close()
        # Skip the file if it's open somewhere else
        if daq_files.is_open_elsewhere():
            continue
        # Don't worry about empty files
        if not daq_files.has_data:
            if mode == FileModes.MOVE or mode == FileModes.ZIPMOVE:
                print('   has no data; remove files')
                for f in (daq_files.file_log, daq_files.data_raw_source, daq_files.data_pc_source, daq_files.file_pcp):
                    if f is not None:
                        if isinstance(f, basestring):
                            os.remove(f)
                        else:
                            os.remove(f.name)
            else:
                print('   has no data; skip')
            continue

        # Grab log info for generating subdirectories and create them
        start_time = daq_files.start_time
        save_path = os.path.join(output_dir, start_time.strftime(dirfmt))
        save_name = start_time.strftime(filefmt)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # COPY or MOVE the file
        if mode == FileModes.COPY or mode == FileModes.MOVE:
            save_name = os.path.join(save_path, save_name)
            for f, ext in zip((daq_files.file_log, daq_files.data_raw_source, daq_files.data_pc_source, daq_files.file_pcp), EXTS):
                if not f: # File not included in this set of DAQ Files
                    continue
                if isinstance(f, basestring):
                    fname = os.path.realpath(f)
                else:
                    fname = os.path.realpath(f.name)
                fname_new = save_name + '.' + ext
                if mode == FileModes.COPY:
                    copy(fname, fname_new)
                else:
                    os.rename(fname, fname_new)
        # Do the same, only perform operation on the .zip
        elif mode == FileModes.ZIPCOPY or mode == FileModes.ZIPMOVE:
            with ZipFile(os.path.join(save_path, save_name + '.zip'), 'a', ZIP_DEFLATED, True) as zfile:
                for f, ext in zip((daq_files.file_log, daq_files.file_raw, daq_files.file_pc, daq_files.file_pcp), EXTS):
                    if not f: # File not included in this set of DAQ Files
                        continue
                    if isinstance(f, basestring):
                        fname = os.path.realpath(f)
                    else:
                        fname = os.path.realpath(f.name)
                    zfile.write(fname, save_name + '.' + ext)
                    if mode == FileModes.ZIPMOVE:
                        os.remove(fname)
    print('100.0% complete')

def organize_subdirectories(base_dir, output_dir, dirfmt='%Y%m%d', filefmt='%Y%m%d%H%M%S', time_thresh=1000, use_utc=False, mode=FileModes.MOVE):
    """
    Organize the subdirectories of DAQ files into the structure defined in "organize_daq_files"

    INPUTS:
        base_dir    = string of the root directory of the directory tree to reorganize
        output_dir  = string of the root directory for the output
        dirfmt      = string of subdirectory format to reorganize things
        filefmt     = string of datetime format stored in DAQ file to parse
        time_thresh = timeout time to find files, [sec]
        mode        = int of what to do with the files, defined in top of file

    OUTPUTS:
        None

    ERRORS:
        None
    """
    for folder in os.listdir(base_dir):
        folder = os.path.join(base_dir, folder)
        if not os.path.isdir(folder):
            continue
        organize_daq_files(folder, output_dir, dirfmt, filefmt, time_thresh, use_utc, mode)