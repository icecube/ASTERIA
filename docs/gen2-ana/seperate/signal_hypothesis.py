import os
import numpy as np
import astropy.units as u
from scipy.fft import fft, fftfreq
from scipy.signal import stft
from scipy.stats import norm

from helper import *
from plthelper import plot_summary_fft, plot_summary_stf

class Signal_Hypothesis():

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

    def get_dir_name(self):
        
        self.mixing_scheme = self.sim.mixing_scheme
        if self.sim.hierarchy.name == "NORMAL":
            self.hierarchy = "normal"
        elif self.sim.hierarchy.name == "INVERTED":
            self.hierarchy = "inverted"

        # select correct directory for systemics
        if self.mixing_scheme == "NoTransformation":
            self.bkg_dir_name = "default"

            if self.temp_para["time_start"] == 150 * u.ms and self.temp_para["time_end"] == 300 * u.ms:
                self.scan_dir_name = "default"
            else:
                self.scan_dir_name = "syst_time_{:.0f}_{:.0f}ms".format(self.temp_para["time_start"].value, self.temp_para["time_end"].value)

            if self.sig_var != 0:
                self.bkg_dir_name = "syst_det_sig_{:+.0f}%".format(self.sig_var*100)
                self.scan_dir_name = self.bkg_dir_name
            elif self.bkg_var != 0:
                self.bkg_dir_name = "syst_det_bkg_{:+.0f}%".format(self.bkg_var*100)
                self.scan_dir_name = self.bkg_dir_name

        elif self.mixing_scheme == "CompleteExchange":
            self.bkg_dir_name = "syst_mix_comp_exch"
            self.scan_dir_name = self.bkg_dir_name

        elif self.mixing_scheme == "AdiabaticMSW":
            if self.hierarchy == "normal":
                self.bkg_dir_name = "syst_mix_MSW_NH"
                self.scan_dir_name = self.bkg_dir_name

            elif self.hierarchy == "inverted":
                self.bkg_dir_name = "syst_mix_MSW_IH"
                self.scan_dir_name = self.bkg_dir_name
        
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

    def set_sig_trials(self, sig_trials):
        """Set number of signal trials

        Args:
            sig_trials (int): number of signal trials
        """
        self.sig_trials = sig_trials

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
        self._bkg = {"ic86": None, "gen2": None, "wls": None} # empty dictionary

        size = self.sig_trials * self.tlength

        # pull random background for subdetectors
        bkg_i3 = self.sim.detector.i3_bg(dt=self.sim._res_dt, size=size).reshape(self.sig_trials, self.tlength)
        bkg_dc = self.sim.detector.dc_bg(dt=self.sim._res_dt, size=size).reshape(self.sig_trials, self.tlength)
        bkg_md = self.sim.detector.md_bg(dt=self.sim._res_dt, size=size).reshape(self.sig_trials, self.tlength)
        bkg_ws = self.sim.detector.ws_bg(dt=self.sim._res_dt, size=size).reshape(self.sig_trials, self.tlength)
        
        # combine subdetector background into IC86 and Gen2 background rate
        bkg_ic86 = bkg_i3 + bkg_dc
        bkg_gen2 = bkg_i3 + bkg_dc + bkg_md
        bkg_wls  = bkg_i3 + bkg_dc + bkg_md + bkg_ws

        self._bkg["ic86"] = bkg_ic86 * (1 + self.bkg_var)
        self._bkg["gen2"] = bkg_gen2 * (1 + self.bkg_var)
        self._bkg["wls"] = bkg_wls * (1 + self.bkg_var)

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
        position = temp_para["position"] # positioning relative to start time

        # transform all parameters in units of [s] and [1/s]
        frequency, dt, time_start, time_end = frequency.to(1/u.s).value, self.sim._res_dt.to(u.s).value, time_start.to(u.s).value, time_end.to(u.s).value

        # find how many full periods fit into the template_window
        template_window = time_end-time_start
        n_periods = int(template_window*frequency) #only full periods
        template_duration = n_periods/frequency

        # produce sinusodial signal for the template duration
        x = np.arange(0,template_duration,dt)
        y = np.sin(2 * np.pi * frequency * x) * amplitude # scale sinus by amplitude factor

        # smooth SASI on-set by applying hanning window over the template duration
        y *= np.hanning(len(y))

        # find bins corresponding to start, end and "full period" end time
        bin_start, bin_end, bin_end_new = int(time_start/dt), int(time_end/dt), int((time_start+template_duration)/dt) ##
        
        # prepare SASI template
        bins_template = len(x) # number of bins in template
        template = np.zeros_like(self.sim.time.value) # empty array of tlength signal window
        template[:bins_template] = y # place template in the beginning
        
        # template can be placed next to time_start (left), time_end (right) and in the middle between time_start and time_end_new
        if position == "center":
            bin_roll = bin_start + int((bin_end-bin_end_new)/2)
        elif position == "left":
            bin_roll = bin_start
        elif position == "right":
            bin_roll = bin_end-bins_template
        else:
            raise ValueError('{} locator does not exist. Choose from "center", "left", "right".'.format(position))
        
        # shift template in position
        template = np.roll(template, bin_roll)
        
        return template
    
    def _signal_model(self):
        """Calculates the signal hits for a flat (null hypothesis) and non-flat (signal hypothesis) SN light curve. For the latter
        counts from a generic oscillation template are added to the flat light curve.
        """
        self._sig = {"ic86": None, "gen2": None, "wls": None} # empty dictionary
        
        # 1) Signal hypothesis, i.e. oscillations in the SN light curve
        # get signal hits in res_dt binning for all sensor types
        t, sig_i3 = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='i3')
        t, sig_dc = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='dc')
        t, sig_md = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='md')
        t, sig_ws = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='ws')

        # combine signal hits into IC86, Gen2 and Gen2+WLS
        sig_ic86 = sig_i3 + sig_dc
        sig_gen2 = sig_i3 + sig_dc + sig_md
        sig_wls  = sig_i3 + sig_dc + sig_md + sig_ws

        self._sig["ic86"] = sig_ic86 * (1 + self.sig_var)
        self._sig["gen2"] = sig_gen2 * (1 + self.sig_var)
        self._sig["wls"] = sig_wls * (1 + self.sig_var)

        return
    
    def _signal_generic(self, smoothing = False):
        """Calculates the signal hits for a flat (null hypothesis) and non-flat (signal hypothesis) SN light curve. For the latter
        counts from a generic oscillation template are added to the flat light curve.
                
        Args:
            smoothing (bool, optional): Smooth lightcurve. Defaults to False.
        """
        self._sig = {"ic86": None, "gen2": None, "wls": None} # empty dictionary
        
        # 1) get signal hits in res_dt binning for all sensor types
        t, sig_i3 = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='i3')
        t, sig_dc = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='dc')
        t, sig_md = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='md')
        t, sig_ws = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='ws')

        # combine signal hits into IC86, Gen2 and Gen2+WLS
        sig_ic86 = sig_i3 + sig_dc
        sig_gen2 = sig_i3 + sig_dc + sig_md
        sig_wls  = sig_i3 + sig_dc + sig_md + sig_ws

        self._sig["ic86"] = sig_ic86
        self._sig["gen2"] = sig_gen2
        self._sig["wls"] = sig_wls

        if smoothing:
            # binning needed to smoothen a frequency f, low frequency cut of 100 Hz
            frequency = 100*u.Hz
            duration = self.sim.time[-1]-self.sim.time[0]
            samples = (duration/self.sim._res_dt.to(u.s)).value
            binning  = int((1/frequency*samples/duration).value) #binning needed to filter out sinals with f>f_lb_sas

            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self._sig[det] = moving_average(self._sig[det], n = binning, const_padding = True)

        # 2) The idea is we add the modulations from a template to the null hypothesis counts
        for det in ["ic86", "gen2", "wls"]: # loop over detector
            temp_para_det = dict(self.temp_para) # copy of template dictionary

            if self.temp_para["time_start"] < self.sim.time[0]:
                raise ValueError("time_start = {} smaller than simulation time start of {}".format(self.temp_para["time_start"], self.sim.time[0]))
            elif self.temp_para["time_end"] > self.sim.time[-1]:
                raise ValueError("time_end = {} larger than simulation time end of {}".format(self.temp_para["time_end"], self.sim.time[-1]))

            # scale amplitude relative to maximum of light curve
            temp_para_det["amplitude"] = temp_para_det["amplitude"] * np.max(self._sig[det])
      
            # get template counts
            template = self._template(temp_para_det)
            
            # combine flat light curve with template
            sig = self._sig[det] + template
        
            # make sure that high amplitude fluctuations do not cause negative counts
            sig = np.maximum(sig, 0)

            # scale signal by 1 + signal variation.
            self._sig[det] = sig * (1 + self.sig_var)

        return
    
    def _signal_mix(self):
        """Calculates the signal hits for a flat (null hypothesis) and non-flat (signal hypothesis) SN light curve. For the latter
        counts from a generic oscillation template are added to the flat light curve.
        """
        self._sig = {"ic86": None, "gen2": None, "wls": None} # empty dictionary
        
        # 1) The idea is that we first get the SASI wiggles and then smoothen them out with a moving average filter
        t, sig_i3 = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='i3')
        t, sig_dc = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='dc')
        t, sig_md = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='md')
        t, sig_ws = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='ws')

        # combine signal hits into IC86, Gen2 and Gen2+WLS
        sig_ic86 = sig_i3 + sig_dc
        sig_gen2 = sig_i3 + sig_dc + sig_md
        sig_wls  = sig_i3 + sig_dc + sig_md + sig_ws

        self._sig["ic86"] = sig_ic86
        self._sig["gen2"] = sig_gen2
        self._sig["wls"] = sig_wls

        # binning needed to smoothen a frequency f, for Tamborra 2014, 20 M: f_sasi = 80 Hz
        frequency = 80*u.Hz
        duration = self.sim.time[-1]-self.sim.time[0]
        samples = (duration/self.sim._res_dt.to(u.s)).value

        binning  = int((1/frequency*samples/duration).value) #binning needed to filter out sinals with f>f_lb_sasi

        for det in ["ic86", "gen2", "wls"]: # loop over detector

            self._sig[det] = moving_average(self._sig[det], n = binning, const_padding = True)

        # 2) The idea is we add the modulations from a template to the null hypothesis counts
        for det in ["ic86", "gen2", "wls"]: # loop over detector
            temp_para_det = dict(self.temp_para) # copy of template dictionary

            if self.temp_para["time_start"] < self.sim.time[0]:
                raise ValueError("time_start = {} smaller than simulation time start of {}".format(self.temp_para["time_start"], self.sim.time[0]))
            elif self.temp_para["time_end"] > self.sim.time[-1]:
                raise ValueError("time_end = {} larger than simulation time end of {}".format(self.temp_para["time_end"], self.sim.time[-1]))

            # scale amplitude relative to maximum of light curve
            temp_para_det["amplitude"] = temp_para_det["amplitude"] * np.max(self._sig[det])
      
            # get template counts
            template = self._template(temp_para_det)
            
            # combine flat light curve with template
            sig = self._sig[det] + template
        
            # make sure that high amplitude fluctuations do not cause negative counts
            sig = np.maximum(sig, 0)

            # scale signal by 1 + signal variation.
            self._sig[det] = sig * (1 + self.sig_var)
        return

    def _signal_sampled(self):
        """Add Poissonian fluctuations of signal
        """
        self._sig_sample = {"ic86": None, "gen2": None, "wls": None} # empty dictionary
        for det in ["ic86", "gen2", "wls"]: # loop over detector
            self._sig_sample[det] = np.random.normal(self._sig[det], np.sqrt(np.abs(self._sig[det])), size=(self.sig_trials, self.tlength))
    
        return
              
    def _hypothesis(self, residual = False, hanning = False):
        """Combines signal and background hits for the null and signal hypothesis.
        Uses residual counts via apply_residuals() if residual = True.
        Applies hanning via apply_hanning() if hanning = True.

        Args:
            residual (bool, optional): Use residuals. Defaults to False.
            hanning (bool, optional): Apply hanning. Defaults to False.
        """
        self._comb = {"ic86": None, "gen2": None, "wls": None} # empty dictionary

        for det in ["ic86", "gen2", "wls"]: # loop over detector
            self._comb[det] = self._sig_sample[det] + self._bkg[det]
        
        if residual:
            self.apply_residual()

        if hanning:
            self.apply_hanning()

        return
    
    def apply_residual(self):
        """ Applies residual. The residual is defined as residual = ( sampled signal + background - averaged background ) / flat signal unsampled.
        Make sure you apply this only once to avoid rescaling repeatedly. This method overwrites the content of _comb.
        """

        for det in ["ic86", "gen2", "wls"]: # loop over detector
            self._comb[det] = ((self._comb[det] - self._avg_bkg[det])/self._sig[det])-1

        return
    
    def apply_hanning(self):
        """ Applies hanning. Scales signal by a hanning window of size tlength. Make sure you apply this only once to avoid rescaling repeatedly. 
        This method overwrites the content of _comb.
        """

        hann = np.hanning(self.tlength)

        for det in ["ic86", "gen2", "wls"]: # loop over detector
            self._comb[det] *= hann

        return
    
    def apply_tmask(self, time_win, det = None):
        """ Applies time mask. Cuts signal window to values given in time_win for both FFT and STF method. 
        This method overwrites the content of _comb. If keywords hypo and det are set to None, the cut is applied to all
        hypothesis and subdetectors. The new, cut time is _time_new with length tlength_new.
        Args:
            time_win (list of astropy.units.quantity.Quantity): lower and higher time cut
            det (str): subdetector ("ic86", "gen2" or "wls"), default None
        """
        time_low, time_high = time_win
        tmask = np.logical_and(self._time>=time_low, self._time<=time_high) # time mask

        if det is None:
            if self.mode == "FFT":
                for det in ["ic86", "gen2", "wls"]: # loop over detector
                    self._comb[det] = self._comb[det][:,tmask]
                
            elif self.mode == "STF":
                for det in ["ic86", "gen2", "wls"]: # loop over detector
                    self._stf[det] = self._stf[det][:,:,tmask]

        else:
            if self.mode == "FFT":
                self._comb[det] = self._comb[det][:,tmask]
                
            elif self.mode == "STF":
                self._stf[det] = self._stf[det][:,:,tmask]
            
        self._time_new = self._time[tmask]
        self.tlength_new = len(self._time_new) # new length of time array

        return

    def apply_fmask(self, freq_win, det = None):
        """ Applies frequency mask. Cuts frequency spectrum to values given in freq_win for both FFT and STF method. 
        This method overwrites the content of _fft or _stf. If keywords hypo and det are set to None, the cut is applied to all
        hypothesis and subdetectors. The new, cut frequency is _freq_new with length flength_new.
        Args:
            freq_win (list of astropy.units.quantity.Quantity): lower and higher frequency cut
            det (str): subdetector ("ic86", "gen2" or "wls"), default None
        """        
        freq_low, freq_high = freq_win
        fmask = np.logical_and(self._freq >=freq_low, self._freq<=freq_high) # frequency mask

        if det is None:
            if self.mode == "FFT":
                for det in ["ic86", "gen2", "wls"]: # loop over detector
                    self._fft[det] = self._fft[det][:,fmask]
                
            elif self.mode == "STF":
                for det in ["ic86", "gen2", "wls"]: # loop over detector
                    self._stf[det] = self._stf[det][:,fmask,:]
            
        else:
            if self.mode == "FFT":
                self._fft[det] = self._fft[det][:,fmask]
                
            elif self.mode == "STF":
                self._stf[det] = self._stf[det][:,fmask,:]
            
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

        self._fft = {"ic86": None, "gen2": None, "wls": None} # empty dictionary
        self._fft0 = {"ic86": None, "gen2": None, "wls": None} # just for monitoring, no cuts applied        
        self.ts = {"ic86": None, "gen2": None, "wls": None}

        for det in ["ic86", "gen2", "wls"]: # loop over detector
            # calculate FFT, power = (fourier modes) ** 2
            self._fft[det] = (2.0/self.tlength_new * np.abs(fft(self._comb[det], axis = -1)[:,1:self.tlength_new//2]))**2
            self._fft0[det] = self._fft[det][0] #save one non-manipulated copy for summary plot
            # return frequencies
            self._freq = fftfreq(self.tlength_new, self.sim._res_dt)[1:self.tlength_new//2].to(u.Hz)

            self.apply_fmask(freq_win, det = det) # apply frequency mask
    
            # max of FFT is used to build TS distribution
            self.ts[det] = np.nanmax(self._fft[det], axis = -1)

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
        bat_step = 5000 # size of batches
        trial_batch = np.arange(0, self.sig_trials, step=bat_step) #chunk data in batches of bat_step

        self._stf = {"ic86": None, "gen2": None, "wls": None} # empty dictionary
        self._stf0 = {"ic86": None, "gen2": None, "wls": None}
        self.ts = {"ic86": None, "gen2": None, "wls": None}

        for det in ["ic86", "gen2", "wls"]: # loop over detector

            # empty lists filled in batch loop
            ts = []
            
            for bat in trial_batch: # loop over batches
                
                # avoid padding as this will introduce artefacts in the FT
                self._freq, self._time, self._stf[det] = stft(self._comb[det][bat:bat+bat_step], 
                                                                                    fs = freq_sam, window = "hann", 
                                                                                    nperseg = hann_len, noverlap = hann_ovl, 
                                                                                    boundary = None, padded = False, 
                                                                                    return_onesided = True)
                self._freq *= u.Hz
                self._time = (self._time * u.s).to(u.ms) # time in units of ms
                
                self._stf0[det] = np.abs(self._stf[det][0]) ** 2 #save one non-manipulated copy for summary plot

                self.apply_tmask(time_win, det = det)
                self.apply_fmask(freq_win, det = det)

                # take square of absolute for power
                self._stf[det] = np.abs(self._stf[det]) ** 2

                # maximum (hottest pixel) in array of 2D STF, returns array of length sig_trials
                # value used for ts distribution
                ts.append(np.nanmax(self._stf[det], axis = (1,2)))
        
            self.ts[det] = np.array(ts).flatten()

        return

    def get_ts_stat(self):

        # Load the 50%, 16% and 84% quantiles for the background hypothesis
        filename = self._file + "/files/background/{}/{}/QUAN_model_{}_{:.0f}_mode_{}_mix_{}_hier_{}_sig_var_{:+.0f}%_bkg_var_{:+.0f}%_bkg_trials_{:1.0e}_bins_{:1.0e}.npz".format(
            self.mode, self.bkg_dir_name, self.temp_para["model"]["name"], self.temp_para["model"]["param"]["progenitor_mass"].value, 
            self.mode, self.mixing_scheme, self.hierarchy,
            self.sig_var * 100, self.bkg_var * 100,
            self.bkg_trials, self.bkg_bins)
           
        bkg_quan = np.load(filename)

        # from here on we carry along the null hypothesis
        self.ts_stat = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                        "signal" : {"ic86": None, "gen2": None, "wls": None}}
        
        for det in ["ic86", "gen2", "wls"]: # loop over detector
            self.ts_stat["null"][det] = bkg_quan[det][bkg_quan["dist"] == self.distance.value][0]

        # Compute the 50%, 16% and 84% quantiles for the signal hypothesis
        for det in ["ic86", "gen2", "wls"]: # loop over detector
            # median, 16% and 84% quantiles of TS distribution
            self.ts_stat["signal"][det] = np.array([np.median(self.ts[det]), np.quantile(self.ts[det], 0.16), np.quantile(self.ts[det], 0.84)])

        return

    def get_zscore(self):
        filename = self._file + "/files/background/{}/{}/HIST_model_{}_{:.0f}_mode_{}_mix_{}_hier_{}_sig_var_{:+.0f}%_bkg_var_{:+.0f}%_bkg_trials_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.npz".format(
            self.mode, self.bkg_dir_name, self.temp_para["model"]["name"], self.temp_para["model"]["param"]["progenitor_mass"].value, 
            self.mode, self.mixing_scheme, self.hierarchy,
            self.sig_var * 100, self.bkg_var * 100,
            self.bkg_trials, self.bkg_bins, self.distance.value)
        
        bkg_hist = np.load(filename)        
        self.pvalue = {"ic86": None, "gen2": None, "wls": None} # empty dictionary
        self.zscore = {"ic86": None, "gen2": None, "wls": None} # empty dictionary

        for det in ["ic86", "gen2", "wls"]: # loop over detector
            p, z = [], []
            for i in range(3): # loop over median, 16% and 84% quantiles of TS distribution

                # p-value of signal given a background distribution                
                # bkg_hist data is normalized, np.sum != 1 because bin size needs to be take care of
                pp = np.sum(bkg_hist[det][1][bkg_hist[det][0] > self.ts_stat["signal"][det][i]]) * (bkg_hist[det][0][1]-bkg_hist[det][0][0])

                # two-sided Z score corresponding to the respective p-value, survival probability = 1 - cdf
                #zz = norm.isf(p/2)
                # one-side Z score corresponding to the respective p-value, ppf = inverse cdf
                zz = norm.ppf(1-pp)
                p.append(pp)
                z.append(zz)
            self.pvalue[det] = np.array(p)
            self.zscore[det] = np.array(z)

        return
  
    def run(self, mode, ft_para, sig_var, bkg_var, sig_trials, bkg_trials, bkg_bins = None, 
            model = "generic", smoothing = False):
        """Runs complete analysis chain including for time-integrated fast fourier transform (FFT)
        and short-time fourier transform (STF). It computes background and signal hits, 
        combines them, performs either FFT or STFT and calculates the TS distribution and significance.    

        Args:
            mode (str): analysis mode (FFT or STF)
            ft_para (dict): parameters of the fourier transform (FFT or STF)
            sig_var (float): percentage variation of signal rate
            bkg_var (float): percentage variation of background rate
            sig_trials (int): Number of signal trials.
            bkg_trials (int): Number of background trials.
            bkg_bins (int): Number of histogram bins of background distribution
            model (str): composition of signal trial ("generic", "model", "mix")
            smoothing (bool): Applies high-pass (moving average) filter.

        Raises:
            ValueError: model takes three valid values: "generic", "model" and "mix.
            ValueError: mode takes two valid values: "FFT" and "STF".
        """

        self.mode = mode
        self.sig_var = sig_var
        self.bkg_var = bkg_var
        self.sig_trials = sig_trials
        self.bkg_trials = bkg_trials
        self.bkg_bins = bkg_bins

        self.get_dir_name() # build bkg file and scan directory name from input

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
            self.get_ts_stat()

        elif self.mode == "STF":
            self._hypothesis()
            self.stf(ft_para)
            self.get_ts_stat()
      
        self.get_zscore()
        
    def dist_scan(self, distance_range, mode, ft_para, sig_var, bkg_var, sig_trials, bkg_trials, bkg_bins, 
                  model = "generic", smoothing = False, verbose = None):
        """Calls run method for a range of distances and saves z-score and TS value for all detectors in an array.   

        Args:
            distance_range (np.ndarray): Distance range array
            mode (str): analysis mode (FFT or STF)
            ft_para (dict): parameters of the fourier transform (FFT or STF)
            sig_var (float): percentage variation of signal rate
            bkg_var (float): percentage variation of background rate
            sig_trials: Number of signal trials.
            bkg_trials (int): Number of background trials.
            bkg_bins (int): Number of histogram bins of background distribution
            model (str): composition of signal trial ("generic", "model", "mix")
            smoothing (bool): Applies high-pass (moving average) filter.
        """
        # prepare empty lists for distance loop
        pvalue = {"ic86": [], "gen2": [], "wls": []}
        zscore = {"ic86": [], "gen2": [], "wls": []}
        ts_stat = {"null" : {"ic86": [], "gen2": [], "wls": []}, "signal" : {"ic86": [], "gen2": [], "wls": []}}

        for dist in distance_range:

            if verbose == "debug":
                print("Distance: {:.1f}".format(dist))

            self.set_distance(distance=dist) # set simulation to distance
            self.run(mode, ft_para, sig_var, bkg_var, sig_trials, bkg_trials, bkg_bins, model, smoothing)

            if verbose == "debug":

                if self.mode == "FFT": plot_summary_fft(self, relative = True)
                if self.mode == "STF": plot_summary_stf(self, relative = True)

            for det in ["ic86", "gen2", "wls"]: # loop over detector
                pvalue[det].append(self.pvalue[det])
                zscore[det].append(self.zscore[det])
                ts_stat["null"][det].append(self.ts_stat["null"][det])
                ts_stat["signal"][det].append(self.ts_stat["signal"][det])

        # for each key return array of length (3, len(dist_range))
        Pvalue = {"ic86": [], "gen2": [], "wls": []}
        Zscore = {"ic86": [], "gen2": [], "wls": []}

        for det in ["ic86", "gen2", "wls"]: 
            Pvalue[det] = np.transpose(np.array(pvalue[det]))
            Zscore[det] = np.transpose(np.array(zscore[det]))

        Ts_stat = {}
        for key, nested_dict in ts_stat.items():
            Ts_stat[key] = {}
            for nested_key, value in nested_dict.items():
                Ts_stat[key][nested_key] = np.transpose(np.array(value))

        return Pvalue, Zscore, Ts_stat
