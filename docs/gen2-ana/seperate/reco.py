import os
import numpy as np
import astropy.units as u
from scipy.fft import fft, fftfreq
from scipy.signal import stft
import copy

from helper import *

def roll_2D(array, shifts):

    # Get the number of rows and columns
    num_rows, num_cols = array.shape

    # Create an array of indices for the rolled array
    rolled_indices = np.arange(num_cols)  # Original column indices
    shifted_indices = (rolled_indices - shifts[:, np.newaxis]) % num_cols  # Apply shifts, negative shift has same definition as np.roll

    # Use advanced indexing to create the shifted array
    shifted_arr = array[np.arange(num_rows)[:, np.newaxis], shifted_indices]

    return shifted_arr

def mask_2D(a2, bins):

    array = copy.deepcopy(a2)

    #Create a mask for the elements to be set to zero
    row_indices = np.arange(array.shape[0])[:, np.newaxis]  # Create a column vector for row indices
    mask = np.arange(array.shape[1]) < bins[row_indices]  # Create a mask based on bin positions

    # Set elements larger than the bin positions to 0
    array[~mask] = 0

    return array


class Reconstruction():

    def __init__(self, 
                 sim, 
                 res_dt,
                 distance, 
                 temp_para):

        # define a few attributes
        self.sim = sim
        self.sim._res_dt = res_dt
        self.distance = distance
        self.tlength = len(self.sim.time)
        self.temp_para = dict(temp_para) # deep copy of temp_para dict is saved to avoid implicit changes in attributes when looping over values

        self._file = os.path.dirname(os.path.abspath(__file__))

        # rescale result
        self.sim.rebin_result(dt = self.sim._res_dt)
        self.sim.scale_result(distance=distance)
        
    def set_sim(self, sim):
        """Set simulation

        Args:
            sim (asteria.simulation.Simulation): ASTERIA Simulation Class
        """
        self.sim = sim

    def set_distance(self, distance):
        """Set distance and rescale simulation

        Args:
            distance (astropy.units.quantity.Quantity): distance for simulation
        """
        self.distance = distance
        self.sim.scale_result(distance=distance)

    def set_trials(self, trials):
        """Set number of trials

        Args:
            trials (int): number of trials
        """
        self.trials = trials

    def set_temp_para(self, temp_para):
        """Set template parameter dictionary

        Args:
            temp_para (dict): template parameter dictionary
        """
        self.temp_para = temp_para

    def _background(self):
        """Calculates the background hits in res_dt time steps for all sensors and combines the hits into three detector scopes
        IceCube 86 (IC86) : i3 + dc
        IceCube-Gen2      : i3 + dc + md
        IceCube-Gen2 + WLS: i3 + dc + md + ws
        """ 
        self._bkg = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}} # empty dictionary

        size = self.trials * self.tlength * 2 # we need distinct noise for signal and null hypothesis -> 2 times the data

        # pull random background for subdetectors
        bkg_i3 = self.sim.detector.i3_bg(dt=self.sim._res_dt, size=size).reshape(2, self.trials, self.tlength)
        bkg_dc = self.sim.detector.dc_bg(dt=self.sim._res_dt, size=size).reshape(2, self.trials, self.tlength)
        bkg_md = self.sim.detector.md_bg(dt=self.sim._res_dt, size=size).reshape(2, self.trials, self.tlength)
        bkg_ws = self.sim.detector.ws_bg(dt=self.sim._res_dt, size=size).reshape(2, self.trials, self.tlength)
        
        # combine subdetector background into IC86 and Gen2 background rate
        bkg_ic86 = bkg_i3 + bkg_dc
        bkg_gen2 = bkg_i3 + bkg_dc + bkg_md
        bkg_wls  = bkg_i3 + bkg_dc + bkg_md + bkg_ws

        for h, hypo in enumerate(["null", "signal"]): # loop over hypothesis
            self._bkg[hypo]["ic86"] = bkg_ic86[h] * (1 + self.bkg_var)
            self._bkg[hypo]["gen2"] = bkg_gen2[h] * (1 + self.bkg_var)
            self._bkg[hypo]["wls"] = bkg_wls[h] * (1 + self.bkg_var)

        return
    
    def _average_background(self):
        """Calculates the average background hits in res_dt time steps by multiplying the mean detector noise rate with the number 
        of sensors for each sensor type.
        """
        self._avg_bkg = {"ic86": None, "gen2": None, "wls": None} # empty dictionary

        avg_bkg_ic86 = (self.sim.detector.n_i3_doms * self.sim.detector.i3_dom_bg_mu + 
                        self.sim.detector.n_dc_doms * self.sim.detector.dc_dom_bg_mu) * 1/u.s * self.sim._res_dt.to(u.s)
        avg_bkg_gen2 = (self.sim.detector.n_i3_doms * self.sim.detector.i3_dom_bg_mu + 
                        self.sim.detector.n_dc_doms * self.sim.detector.dc_dom_bg_mu + 
                        self.sim.detector.n_md * self.sim.detector.md_bg_mu) * 1/u.s * self.sim._res_dt.to(u.s)
        avg_bkg_wls  = (self.sim.detector.n_i3_doms * self.sim.detector.i3_dom_bg_mu + 
                        self.sim.detector.n_dc_doms * self.sim.detector.dc_dom_bg_mu + 
                        self.sim.detector.n_md * self.sim.detector.md_bg_mu + 
                        self.sim.detector.n_ws * self.sim.detector.ws_bg_mu) * 1/u.s * self.sim._res_dt.to(u.s)

        self._avg_bkg["ic86"] = avg_bkg_ic86 * (1 + self.bkg_var)
        self._avg_bkg["gen2"] = avg_bkg_gen2 * (1 + self.bkg_var)
        self._avg_bkg["wls"] = avg_bkg_wls * (1 + self.bkg_var)

        return
    
    def _template(self, temp_para):
        """Returns a generic modulation (template) of the same length and binning like the simulated light curve.

        Args:
            temp_para (dict): template parameter dictionary

        Raises:
            ValueError: Valid values for temp_para["position"] are "left", "center" and "right"

        Returns:
            template (numpy.ndarray): template to add to signal hits
        """

        # template parameters
        frequency = temp_para["frequency"] # frequency
        amplitude = temp_para["amplitude"] # amplitude
        time_start = temp_para["time_start"] # start time
        time_end = temp_para["time_end"] # end time
        duration = temp_para["duration"]
        position = temp_para["position"] # positioning relative to start time

        # transform all parameters in units of [s] and [1/s]
        frequency, dt, time_start, time_end = frequency.to(1/u.s).value, self.sim._res_dt.to(u.s).value, time_start.to(u.s).value, time_end.to(u.s).value
        
        # find how many full periods fit into the template_window
        n_periods = (duration*frequency).astype(np.int64) #only full periods
        duration_period = n_periods/frequency

        # x is time
        x = self.sim.time.value
        # y is the periodic signal not yet cropped to the duration
        y = np.sin(2 * np.pi * frequency[:, np.newaxis] * x) * amplitude # scale sinus by amplitude factor
        
        # mask values greater than duration of period to True
        mask = x > duration_period[:, np.newaxis]
        # set the values where mask is True to 0
        y[mask] = 0

        # Vectorized application of the Hanning window
        non_masked_counts = np.sum(~mask, axis=1)  # Number of non-masked (to be hanned) elements in each row
        hanning_windows = np.array([np.hanning(count) for count in non_masked_counts], dtype = object)  # Hanning windows for each row
        # Apply the Hanning window to each row of y
        y[~mask] *= np.concatenate([hanning_windows[i] for i in range(len(hanning_windows))])

        # find bins corresponding to start, end and "full period" end time
        bin_start, bin_end, bin_end_new = (time_start/dt).astype(np.int64), (time_end/dt).astype(np.int64), ((time_start+duration_period)/dt).astype(np.int64) ##
        # prepare SASI template
        bins_template = (duration_period/dt).astype(np.int64) # number of bins in template
        #template = np.zeros_like(self.sim.time.value) # empty array of tlength signal window
        #template[:bins_template] = y # place template in the beginning
        template = y
        
        # template can be placed next to time_start (left), time_end (right) and in the middle between time_start and time_end_new
        if position == "center":
            bin_roll = bin_start + ((bin_end-bin_end_new)/2).astype(np.int64)
        elif position == "left":
            bin_roll = bin_start
        elif position == "right":
            bin_roll = bin_end-bins_template
        else:
            raise ValueError('{} locator does not exist. Choose from "center", "left", "right".'.format(position))
        
        #template = np.tile(template, (self.trials, 1))
        # shift template in position
        template = roll_2D(template, bin_roll)
        
        return template
    

    def _signal_generic(self, smoothing = False):
        """Calculates the signal hits for a flat (null hypothesis) and non-flat (signal hypothesis) SN light curve. For the latter
        counts from a generic oscillation template are added to the flat light curve.
                
        Args:
            smoothing (bool, optional): Smooth lightcurve. Defaults to False.
        """
        self._sig = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}} # empty dictionary
        
        # 1) Null hypothesis, i.e. flat, no modulation SN light curve
        # get signal hits in res_dt binning for all sensor types
        t, sig_i3 = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='i3')
        t, sig_dc = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='dc')
        t, sig_md = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='md')
        t, sig_ws = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='ws')

        # combine signal hits into IC86, Gen2 and Gen2+WLS
        sig_ic86 = sig_i3 + sig_dc
        sig_gen2 = sig_i3 + sig_dc + sig_md
        sig_wls  = sig_i3 + sig_dc + sig_md + sig_ws

        self._sig["null"]["ic86"] = sig_ic86 * (1 + self.sig_var)
        self._sig["null"]["gen2"] = sig_gen2 * (1 + self.sig_var)
        self._sig["null"]["wls"] = sig_wls * (1 + self.sig_var)

        if smoothing:
            # binning needed to smoothen a frequency f, low frequency cut of 100 Hz
            frequency = 100*u.Hz
            duration = self.sim.time[-1]-self.sim.time[0]
            samples = (duration/self.sim._res_dt.to(u.s)).value
            binning  = int((1/frequency*samples/duration).value) #binning needed to filter out sinals with f>f_lb_sas

            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self._sig["null"][det] = moving_average(self._sig["null"][det], n = binning, const_padding = True)

        # 2) Signal hypothesis, i.e. oscillations in the SN light curve
        # The idea is we add the modulations from a template to the null hypothesis counts
        for det in ["ic86", "gen2", "wls"]: # loop over detector
            temp_para_det = dict(self.temp_para) # copy of template dictionary
            
            if self.temp_para["time_start"].min() < self.sim.time[0].to(u.ms):
                raise ValueError("time_start = {} smaller than simulation time start of {}".format(self.temp_para["time_start"], self.sim.time[0]))
            elif self.temp_para["time_end"].min() > self.sim.time[-1].to(u.ms):
                raise ValueError("time_end = {} larger than simulation time end of {}".format(self.temp_para["time_end"], self.sim.time[-1]))

            # scale amplitude relative to maximum of light curve
            temp_para_det["amplitude"] = temp_para_det["amplitude"] * np.max(self._sig["null"][det])
      
            # get template counts
            template = self._template(temp_para_det)
            
            # combine flat light curve with template
            sig = self._sig["null"][det] + template
        
            # make sure that high amplitude fluctuations do not cause negative counts
            sig = np.maximum(sig, 0)

            # scale signal by 1 + signal variation.
            self._sig["signal"][det] = sig * (1 + self.sig_var)

        return
    
    def _signal_sampled(self):
        """Add Poissonian fluctuations of signal
        """
        self._sig_sample = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                            "signal" : {"ic86": None, "gen2": None, "wls": None}} # empty dictionary
        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self._sig_sample[hypo][det] = np.random.normal(self._sig[hypo][det], np.sqrt(np.abs(self._sig[hypo][det])), size=(self.trials, self.tlength))
        
        return
              
    def _hypothesis(self, residual = False, hanning = False):
        """Combines signal and background hits for the null and signal hypothesis.
        Uses residual counts via apply_residuals() if residual = True.
        Applies hanning via apply_hanning() if hanning = True.

        Args:
            residual (bool, optional): Use residuals. Defaults to False.
            hanning (bool, optional): Apply hanning. Defaults to False.
        """
        self._comb = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                      "signal" : {"ic86": None, "gen2": None, "wls": None}} # empty dictionary

        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self._comb[hypo][det] = self._sig_sample[hypo][det] + self._bkg[hypo][det]
        
        if residual:
            self.apply_residual()

        if hanning:
            self.apply_hanning()

        return
    
    def apply_residual(self):
        """ Applies residual. The residual is defined as residual = ( sampled signal + background - averaged background ) / flat signal unsampled.
        Make sure you apply this only once to avoid rescaling repeatedly. This method overwrites the content of _comb.
        """

        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self._comb[hypo][det] = ((self._comb[hypo][det] - self._avg_bkg[det])/self._sig[hypo][det])-1

        return
    
    def apply_hanning(self):
        """ Applies hanning. Scales signal by a hanning window of size tlength. Make sure you apply this only once to avoid rescaling repeatedly. 
        This method overwrites the content of _comb.
        """

        hann = np.hanning(self.tlength)

        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self._comb[hypo][det] *= hann

        return
    
    def apply_tmask(self, time_win, hypo = None, det = None):
        """ Applies time mask. Cuts signal window to values given in time_win for both FFT and STF method. 
        This method overwrites the content of _comb. If keywords hypo and det are set to None, the cut is applied to all
        hypothesis and subdetectors. The new, cut time is _time_new with length tlength_new.
        Args:
            time_win (list of astropy.units.quantity.Quantity): lower and higher time cut
            hypo (str): hypothesis ("null" or "signal"), default None
            det (str): subdetector ("ic86", "gen2" or "wls"), default None
        """
        time_low, time_high = time_win
        tmask = np.logical_and(self._time>=time_low, self._time<=time_high) # time mask

        if hypo is None and det is None:
            if self.mode == "FFT":
                for hypo in ["null", "signal"]: # loop over hypothesis
                    for det in ["ic86", "gen2", "wls"]: # loop over detector
                        self._comb[hypo][det] = self._comb[hypo][det][:,tmask]
                
            elif self.mode == "STF":
                for hypo in ["null", "signal"]: # loop over hypothesis
                    for det in ["ic86", "gen2", "wls"]: # loop over detector
                        self._stf[hypo][det] = self._stf[hypo][det][:,:,tmask]
            
        else:
            if self.mode == "FFT":
                self._comb[hypo][det] = self._comb[hypo][det][:,tmask]
                
            elif self.mode == "STF":
                self._stf[hypo][det] = self._stf[hypo][det][:,:,tmask]
            
        self._time_new = self._time[tmask]
        self.tlength_new = len(self._time_new) # new length of time array

        return

    def apply_fmask(self, freq_win, hypo = None, det = None):
        """ Applies frequency mask. Cuts frequency spectrum to values given in freq_win for both FFT and STF method. 
        This method overwrites the content of _fft or _stf. If keywords hypo and det are set to None, the cut is applied to all
        hypothesis and subdetectors. The new, cut frequency is _freq_new with length flength_new.
        Args:
            freq_win (list of astropy.units.quantity.Quantity): lower and higher frequency cut
            hypo (str): hypothesis ("null" or "signal"), default None
            det (str): subdetector ("ic86", "gen2" or "wls"), default None
        """        
        freq_low, freq_high = freq_win
        fmask = np.logical_and(self._freq >=freq_low, self._freq<=freq_high) # frequency mask

        if hypo is None and det is None:
            if self.mode == "FFT":
                for hypo in ["null", "signal"]: # loop over hypothesis
                    for det in ["ic86", "gen2", "wls"]: # loop over detector
                        self._fft[hypo][det] = self._fft[hypo][det][:,fmask]
                
            elif self.mode == "STF":
                for hypo in ["null", "signal"]: # loop over hypothesis
                    for det in ["ic86", "gen2", "wls"]: # loop over detector
                        self._stf[hypo][det] = self._stf[hypo][det][:,fmask,:]

        else:
            if self.mode == "FFT":
                self._fft[hypo][det] = self._fft[hypo][det][:,fmask]
                
            elif self.mode == "STF":
                self._stf[hypo][det] = self._stf[hypo][det][:,fmask,:]
            
        self._freq_new = self._freq[fmask] # new frequency array
        self.flength_new = len(self._freq_new) # new length of frequency array

        return

    def fft(self, fft_para):

        time_res = fft_para["time_res"]
        time_win = fft_para["time_win"]
        freq_res = fft_para["freq_res"]
        freq_win = fft_para["freq_win"]

        if time_res != self.sim._res_dt:
            raise ValueError('fft_para["time_res"] = {} but ana.sim._res_dt = {}. Make sure to execute ana.run with the same resolution as you set in fft_para'.format(time_res, self.sim._res_dt))
        
        self._time = self.sim.time
        self.apply_tmask(time_win) # apply time mask   

        self._fft = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}} # empty dictionary
        self._fft0 = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}}
        self.freq_reco = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}}

        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                # calculate FFT, power = (fourier modes) ** 2
                self._fft[hypo][det] = (2.0/self.tlength_new * np.abs(fft(self._comb[hypo][det], axis = -1)[:,1:self.tlength_new//2]))**2
                self._fft0[hypo][det] = self._fft[hypo][det][0] #save one non-manipulated copy for summary plot
                # return frequencies
                self._freq = fftfreq(self.tlength_new, self.sim._res_dt)[1:self.tlength_new//2].to(u.Hz)

                self.apply_fmask(freq_win, det = det) # apply frequency mask
        
                # max of FFT is used to build TS distribution
                self.freq_reco[hypo][det] = self._freq_new[np.argmax(self._fft[hypo][det], axis=-1)].value

        return

    def stf(self, stf_para):

        hann_len = stf_para["hann_len"] # length of hann window
        hann_res = stf_para["hann_res"] # desired frequency resolution in Hann window
        hann_hop = stf_para["hann_hop"] # hann hop = number of time bins the window is moved
        freq_sam = stf_para["freq_sam"] # sampling frequency of entire signal (1 kHz for 1 ms binning)
        time_win = stf_para["time_win"]
        freq_win = stf_para["freq_win"]


        hann_len = int(hann_len.to_value(u.ms)) # define hann window
        hann_res = hann_res.to_value(u.Hz)
        hann_hop = int(hann_hop.to_value(u.ms)) #ShortTimeFFT does not accept numpt.int
        freq_sam = freq_sam.to_value(u.Hz)
        freq_mfft = int(freq_sam/hann_res)  #oversampling of hann window, relates to frequency resolution
        hann_ovl = int(hann_len - hann_hop) # define hann overlap

        # STFT is computationally expensive, batching will ensure that RAM is not completly used up
        bat_step = 10000 # size of batches
        trial_batch = np.arange(0, self.trials, step=bat_step) #chunk data in batches of bat_step

        self._stf = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}} # empty dictionary
        self._stf0 = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}}
        self.freq_reco = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}}
        self.time_reco = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}}
        self.ts = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}}

        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector

                # empty lists filled in batch loop
                ts, fit_freq, fit_time = [], [], []

                for bat in trial_batch: # loop over batches
                    
                    # avoid padding as this will introduce artefacts in the FT
                    self._freq, self._time, self._stf[hypo][det] = stft(self._comb[hypo][det][bat:bat+bat_step], 
                                                                                        fs = freq_sam, window = "hann", 
                                                                                        nperseg = hann_len, noverlap = hann_ovl, 
                                                                                        boundary = None, padded = False, 
                                                                                        return_onesided = True)
                    self._freq *= u.Hz
                    self._time = (self._time * u.s).to(u.ms) # time in units of ms
                    
                    self._stf0[hypo][det] = np.abs(self._stf[hypo][det][0]) ** 2 #save one non-manipulated copy for summary plot

                    self.apply_tmask(time_win, hypo = hypo, det = det)
                    self.apply_fmask(freq_win, hypo = hypo, det = det)

                    # take square of absolute for power
                    self._stf[hypo][det] = np.abs(self._stf[hypo][det]) ** 2

                    # get maximum of power
                    ts.append(np.nanmax(self._stf[hypo][det], axis = (1,2)))
                    # get time and freq index position of maximum 
                    ind_freq, ind_time = argmax_lastNaxes(self._stf[hypo][det], 2)
                    # get corresponding time and freq of bin
                    fit_freq.append(self._freq_new[ind_freq])
                    fit_time.append(self._time_new[ind_time])

                self.ts[hypo][det] = np.array(ts).flatten()
                self.freq_reco[hypo][det], self.time_reco[hypo][det] = np.array(fit_freq).flatten(), np.array(fit_time).flatten() # reconstructed parameters

        return
  
    def run(self, mode, ft_para, sig_var, bkg_var, trials, model = "generic", smoothing = False):
        """Runs complete analysis chain including for time-integrated fast fourier transform (FFT)
        and short-time fourier transform (STF). It computes background and signal hits, 
        combines them, performs either FFT or STFT and returns the frequency and time of the hottest point   

        Args:
            mode (str): analysis mode (FFT or STF)
            ft_para (dict): parameters of the fourier transform (FFT or STF)
            sig_var (float): percentage variation of signal rate
            bkg_var (float): percentage variation of background rate
            trials (int): Number of trials.
            model (str): composition of signal trial ("generic", "model", "mix")
            smoothing (bool): Applies high-pass (moving average) filter.

        Raises:
            ValueError: model takes three valid values: "generic", "model" and "mix.
            ValueError: mode takes two valid values: "FFT" and "STF".
        """

        self.mode = mode
        self.sig_var = sig_var
        self.bkg_var = bkg_var
        self.trials = trials

        if self.mode != "FFT" and self.mode != "STF":
            raise ValueError('{} mode does not exist. Choose from "FFT" and "STF"'.format(self.mode))

        # load and combine data
        self._background()
        self._average_background()
        if model == "generic":
            self._signal_generic(smoothing = smoothing)
        elif model == "model":
            self._signal_model()
        elif model == "mix":
            self._signal_mix()
        else:
            raise ValueError('{} model type does not exist. Choose from "generic", "model", "mix".'.format(model))
        self._signal_sampled()

        if self.mode == "FFT":
            self._hypothesis(hanning = ft_para["hanning"])
            self.fft(ft_para)

        elif self.mode == "STF":
            self._hypothesis()
            self.stf(ft_para)