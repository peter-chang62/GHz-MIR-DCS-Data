import numpy as np

try:
    from .constants import SPEED_OF_LIGHT, M_2_CM
except (ValueError, SystemError, ImportError):
    from constants import SPEED_OF_LIGHT, M_2_CM


class CombTools:
    """
    CombTools.py

    Contains helpful functions when using the dual-comb spectrometer

    TODO:
     - None
    """

    def __init__(self):
        self.x_Hz = None
        self.x_nm = None
        self.x_wvn = None

        # And initialize the fitting window
        self.wvn_start = None
        self.wvn_stop = None
        self.wvn_step = None

    def freq_axis_2CWlasers(self, v_low, frep_sig, points_per_ig):
        '''
        Calculates frequency axis from laboratory frequency comb method.
        The complex locking scheme using 2 CW lasers

        Inputs:
            v_low    = (Hz)     CW-laser-referenced Nyquist window edge
            frep_sig = (Hz)      pulse repetition rate of Signal comb, the one not clocking the DAQ
            points_per_ig   = (integer) points per interferogram

        OUTPUT:
            x_freq   = frequency of each comb tooth (cm-1)
        '''

        n = points_per_ig//2 + 1 # FFT gives one extra point
        x_freq = np.asarray(range(n))
        x_freq = v_low + x_freq * frep_sig
#        for tooth in range(nPnts):
#            x_freq[tooth] = (v_low*1e12 + tooth * frep_sig)
        self.x_Hz = x_freq
        self.x_wvn = x_freq / (SPEED_OF_LIGHT*M_2_CM)
        self.x_nm = 1e7 / self.x_wvn

        self.wvn_step = frep_sig / (SPEED_OF_LIGHT*M_2_CM)

        self.fit_range() # define which comb teeth over fitting window
        return x_freq

    def freq_axis_mobile_comb(self, v_CW, frep_Clocked, frep_Other, Nyq = 0):
        '''
        Calculates frequency axis from f-2f self-referenced mobile frequency combs.

        INPUTS:
            v_CW          = (THz) frequency of MyRio CW reference laser
            frep_Clocked  = (Hz)  pulse repetition rate of comb into DAQ
            frep_Other    = (Hz)  pulse repetition rate of unclocked comb
            Nyq           =       distance in 1/2-integers from CW laser to edge of Nyquist window
                                  eg if v_CW = 200THz and window from 205-206THz, then Nyq = 2.5
        OUTPUTS:
           x_freq         = (cm-1) frequency of each comb tooth

           !!!!!!!!!!!
           Nate is not sure that this calculation is correct
           !!!!!!!!!!!
        '''
        assert self.data_spectra is not [], "Add FFT to self.data_spectra first"
        dfrep = abs(frep_Other - frep_Clocked) # comb difference frequency (Hz)
        try:
            n_pnts = np.shape(self.data_spectra)[0]
        except AttributeError:
            n_pnts =  round(1/2 * frep_Clocked / dfrep) # or frep_Other?
        except IndexError:
            print("Add FFT to self.data_spectra first")
            n_pnts = round(1/2 * frep_Clocked / dfrep)
        df_Nyq = frep_Clocked * 2 * n_pnts
        Hz_step = 1/2 * (frep_Clocked + frep_Other)
        self.wvn_step = Hz_step / (SPEED_OF_LIGHT * M_2_CM)

        x_freq = np.zeros((nPnts))
        if ((2*Nyq) % 2 is 0):
            # Nyquist window from 0-0.5
            v_low = v_CW + Nyq * df_Nyq
            x_freq[:] = range(n_pnts)
#            x_freq = x_freq * frep_Other + v_low
            x_freq = x_freq * Hz_step + v_low
#            for tooth in range(nPnts):
#                x_freq[tooth] = (v_low + tooth * frep_Other)
        else:
            # Nyquist window from 0.5-1, other calculation method.
            x_freq += 0 #placeholder
            # To do.

        self.x_Hz = x_freq
        self.x_wvn = x_freq / (SPEED_OF_LIGHT*M_2_CM)
        self.x_nm = 1e7 / self.x_wvn

        self.fit_range()
        return x_freq

    def fit_range(self):
        '''
        Given frequency axis and fit window, determine which comb teeth to fit.
        INPUT:
            x_wvn = comb tooth axis
            wvn_start = fit region
        OUTPUT:
            start_pnt = which index along frequency axis to start fitting

        '''
        assert self.wvn_start is not None, "Run def_environment() to determine fitting region"
        assert self.x_wvn is not None, "Run a scale***() function to determine the frequency axis from locking spreadsheet"

        self.start_pnt = np.argmin(np.abs(self.x_wvn - self.wvn_start))
        self.stop_pnt = np.argmin(np.abs(self.x_wvn - self.wvn_stop))
        self.wvn_start = self.x_wvn[self.start_pnt]
        self.wvn_stop = self.x_wvn[self.stop_pnt-1] # -1 accounts for Python indexing

        self.x_data = self.x_wvn[self.start_pnt:self.stop_pnt]
        
    
def mobile_axis2(IG, f_opt = 25e6, wvn_spectroscopy = 7000):
    '''
    Calculate frequency axis for self-referenced dual-comb.
    
    f_opt = optical lock frequency and sign (MHz). Where f_CW = f_tooth + f_opt
    wvn_spectroscopy = one frequency inside filtered spectra (cm-1)

    IG is an object from pldspectrapy/igtools.py with all the log file information from comb locking
    IG.fc = instantaneous repetition rate of clock comb
    IG.fr2 = instantaneous repetition rate of unclocked comb
    IG.frame_length = points per interferogram. If interferogram centerburst is not moving, 
                                                this tells us 1/df_rep
    '''
    # Calculate CW laser frequency from individual comb f_reps, IG length, optical beat offset
    f_cw_approx = 191.5e12 # approximate CW reference laser frequency (Hz)
    df_tooth = .5*(IG.fc + IG.fr2)
    df_nyq = IG.frame_length * IG.fr2
    nyq_cw = np.argmin(np.abs(f_cw_approx - df_nyq * np.asarray(range(20))))
    f_cw = df_nyq * nyq_cw + f_opt
    print('CW laser =', f_cw, 'Hz')
    
    # determine Nyquist window of spectroscopy
    f_spectroscopy = wvn_spectroscopy * SPEED_OF_LIGHT * M_2_CM
    nyq, sign = divmod(f_spectroscopy, df_nyq)
    if sign / df_nyq < 0.5:
        flip_spectrum = False
    else:
        nyq += 1
        flip_spectrum = True
    
    # Last make frequency axis array
    x_hz = np.arange(IG.frame_length/2+1)
    if flip_spectrum:
        x_hz *= -1
    x_hz = x_hz * df_tooth
    x_hz += nyq * df_nyq - f_opt
    IG.x_wvn = x_hz / SPEED_OF_LIGHT / M_2_CM
    
    return IG.x_wvn

def mobile_axis(IG, f_opt = 25e6, wvn_spectroscopy = 7000):
    '''
    Calculate frequency axis for self-referenced dual-comb.
    Based on setpoint and dfr rather than instantaneous fr1, fr2 which shift by 5 Hz
    
    f_opt = optical lock frequency and sign (MHz). Where f_CW = f_tooth + f_opt
    wvn_spectroscopy = one frequency inside filtered spectra (cm-1)

    IG is an object from pldspectrapy/igtools.py with all the log file information from comb locking
    IG.fc = repetition rate setpoint of clock comb (sets DAQ clock)
    IG.fr2 = repetition rate of unclocked comb
    IG.frame_length = points per interferogram. If interferogram centerburst is not moving, 
                                                this tells us 1/df_rep
    '''
    # Calculate CW laser frequency from individual comb f_reps, IG length, optical beat offset
    f_cw_approx = 191.5e12 # approximate CW reference laser frequency (Hz)
    fr2_set = IG.fc_set - IG.dfr
    df_tooth = .5*(IG.fc_set + fr2_set)
    df_nyq = IG.frame_length * fr2_set
    nyq_cw = np.argmin(np.abs(f_cw_approx - df_nyq * np.asarray(range(20))))
    f_cw = df_nyq * nyq_cw + f_opt
    print('CW laser =', f_cw, 'Hz')
    
    # determine Nyquist window of spectroscopy
    f_spectroscopy = wvn_spectroscopy * SPEED_OF_LIGHT * M_2_CM
    nyq, sign = divmod(f_spectroscopy, df_nyq)
    if sign / df_nyq < 0.5:
        flip_spectrum = False
    else:
        nyq += 1
        flip_spectrum = True
    
    # Last make frequency axis array
    x_hz = np.arange(IG.frame_length/2+1)
    if flip_spectrum:
        x_hz *= -1
    x_hz = x_hz * df_tooth
    x_hz += nyq * df_nyq - f_opt
    IG.x_wvn = x_hz / SPEED_OF_LIGHT / M_2_CM
    
    return IG.x_wvn