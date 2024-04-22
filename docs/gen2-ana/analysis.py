import numpy as np
import astropy.units as u
from scipy.fft import fft, fftfreq
from scipy.signal import stft
from scipy.stats import skewnorm, norm

from helper import argmax_lastNaxes, moving_average
from asteria.simulation import Simulation as sim

class Analysis():

    def __init__(self, 
                 sim, 
                 res_dt,
                 distance, 
                 trials, 
                 temp_para):

        # define a few attributes
        self.sim = sim
        self.sim._res_dt = res_dt
        self.distance = distance
        self.trials = trials
        self.tlength = len(self.sim.time)
        self.temp_para = dict(temp_para) # deep copy of temp_para dict is saved to avoid implicit changes in attributes when looping over values

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
        self._bkg = {"ic86": None, "gen2": None, "wls": None} # empty dictionary

        size = self.trials * self.tlength

        # pull random background for subdetectors
        bkg_i3 = self.sim.detector.i3_bg(dt=self.sim._res_dt, size=size).reshape(self.trials, self.tlength)
        bkg_dc = self.sim.detector.dc_bg(dt=self.sim._res_dt, size=size).reshape(self.trials, self.tlength)
        bkg_md = self.sim.detector.md_bg(dt=self.sim._res_dt, size=size).reshape(self.trials, self.tlength)
        bkg_ws = self.sim.detector.ws_bg(dt=self.sim._res_dt, size=size).reshape(self.trials, self.tlength)
        
        # combine subdetector background into IC86 and Gen2 background rate
        bkg_ic86 = bkg_i3 + bkg_dc
        bkg_gen2 = bkg_i3 + bkg_dc + bkg_md
        bkg_wls  = bkg_i3 + bkg_dc + bkg_md + bkg_ws

        self._bkg["ic86"] = bkg_ic86
        self._bkg["gen2"] = bkg_gen2
        self._bkg["wls"] = bkg_wls

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

        self._avg_bkg["ic86"] = avg_bkg_ic86
        self._avg_bkg["gen2"] = avg_bkg_gen2
        self._avg_bkg["wls"] = avg_bkg_wls

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
        self._sig = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}} # empty dictionary
        
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

        self._sig["signal"]["ic86"] = sig_ic86
        self._sig["signal"]["gen2"] = sig_gen2
        self._sig["signal"]["wls"] = sig_wls

        # 2) Null hypothesis, i.e. flat, no modulation SN light curve
        # The idea is use a moving average filter to smoothen the SASI wiggles out

        # binning needed to smoothen a frequency f, for Tamborra 2014, 20 M: f_sasi = 80 Hz
        frequency = 80*u.Hz
        duration = self.sim.time[-1]-self.sim.time[0]
        samples = (duration/self.sim._res_dt.to(u.s)).value

        binning  = int((1/frequency*samples/duration).value) #binning needed to filter out sinals with f>f_lb_sasi

        for det in ["ic86", "gen2", "wls"]: # loop over detector

            self._sig["null"][det] = moving_average(self._sig["signal"][det], n = binning, const_padding = True)

        return
    
    def _signal(self):
        """Calculates the signal hits for a flat (null hypothesis) and non-flat (signal hypothesis) SN light curve. For the latter
        counts from a generic oscillation template are added to the flat light curve.
        
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

        self._sig["null"]["ic86"] = sig_ic86
        self._sig["null"]["gen2"] = sig_gen2
        self._sig["null"]["wls"] = sig_wls

        # 2) Signal hypothesis, i.e. oscillations in the SN light curve
        # The idea is we add the modulations from a template to the null hypothesis counts
        for det in ["ic86", "gen2", "wls"]: # loop over detector
            temp_para_det = dict(self.temp_para) # copy of template dictionary

            if self.temp_para["time_start"] < self.sim.time[0]:
                raise ValueError("time_start = {} smaller than simulation time start of {}".format(self.temp_para["time_start"], self.sim.time[0]))
            elif self.temp_para["time_end"] > self.sim.time[-1]:
                raise ValueError("time_end = {} larger than simulation time end of {}".format(self.temp_para["time_end"], self.sim.time[-1]))

            # scale amplitude relative to maximum of light curve
            temp_para_det["amplitude"] = temp_para_det["amplitude"] * np.max(self._sig["null"][det])
      
            # get template counts
            template = self._template(temp_para_det)
            
            # combine flat light curve with template
            sig = self._sig["null"][det] + template
        
            # make sure that high amplitude fluctuations do not cause negative counts
            sig = np.maximum(sig, 0)

            self._sig["signal"][det] = sig

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
                self._comb[hypo][det] = self._sig_sample[hypo][det] + self._bkg[det]
        
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
    
    def apply_tmask(self, time_win):
        """ Applies time mask. Cuts signal window to values given in time_win. This method overwrites the content of _comb.
        The new, cut time is _time_new with length tlength_new.
        Args:
            time_win (list of astropy.units.quantity.Quantity): lower and higher time cut
        """
        time_low, time_high = time_win
        tmask = np.logical_and(self.sim.time>=time_low, self.sim.time<=time_high) # time mask
        
        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self._comb[hypo][det] = self._comb[hypo][det][:,tmask]

        self._time_new = self.sim.time[tmask] # new times
        self.tlength_new = len(self._time_new) # new length of time array

        return

    def apply_fmask(self, freq_win):
        """ Applies frequency mask. Cuts frequency spectrum to values given in freq_win. This method overwrites the content of _fft.
        The new, cut frequency is _freq_new with length flength_new.
        Args:
            freq_win (list of astropy.units.quantity.Quantity): lower and higher frequency cut
        """        
        freq_low, freq_high = freq_win
        fmask = np.logical_and(self._freq >=freq_low, self._freq<=freq_high) # frequency mask

        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self._fft[hypo][det] = self._fft[hypo][det][:,fmask]

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

        self.apply_tmask(time_win) # apply time mask   

        self._fft = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}} # empty dictionary

        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                # calculate FFT, power = (fourier modes) ** 2
                self._fft[hypo][det] = (2.0/self.tlength_new * np.abs(fft(self._comb[hypo][det], axis = -1)[:,1:self.tlength_new//2]))**2
                # return frequencies
                self._freq = fftfreq(self.tlength_new,self.sim._res_dt)[1:self.tlength_new//2].to(u.Hz)

        # apply frequency cuts
        self.apply_fmask(freq_win)
        
        self.ts = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                       "signal" : {"ic86": None, "gen2": None, "wls": None}} # empty dictionary
        self.ffit = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                         "signal" : {"ic86": None, "gen2": None, "wls": None}} # empty dictionary

        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                # max of FFT is used to build TS distribution
                self.ts[hypo][det] = np.nanmax(self._fft[hypo][det], axis = -1)
                self.ffit[hypo][det] = self._freq_new[np.argmax(self._fft[hypo][det], axis=-1)].value

        return

    def stf(self, stf_para):

        hann_len = stf_para["hann_len"] # length of hann window
        hann_res = stf_para["hann_res"] # desired frequency resolution in Hann window
        hann_hop = stf_para["hann_hop"] # hann hop = number of time bins the window is moved
        freq_sam = stf_para["freq_sam"] # sampling frequency of entire signal (1 kHz for 1 ms binning)
        time_int = stf_para["time_int"] # should power be summed over time?, True or False

        hann_len = int(hann_len.to_value(u.ms)) # define hann window
        hann_res = hann_res.to_value(u.Hz)
        hann_hop = int(hann_hop.to_value(u.ms)) #ShortTimeFFT does not accept numpt.int
        freq_sam = freq_sam.to_value(u.Hz)
        freq_mfft = int(freq_sam/hann_res)  #oversampling of hann window, relates to frequency resolution
        hann_ovl = int(hann_len - hann_hop) # define hann overlap

        # STFT is computationally expensive, batching will ensure that RAM is not completly used up
        bat_step = 5000 # size of batches
        trial_batch = np.arange(0, self.trials, step=bat_step) #chunk data in batches of bat_step

        self._stf = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}} # empty dictionary
        self._log = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}} # empty dictionary
        self.ts = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                   "signal" : {"ic86": None, "gen2": None, "wls": None}} # empty dictionary
        self.ffit = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}} # empty dictionary
        self.tfit = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                     "signal" : {"ic86": None, "gen2": None, "wls": None}} # empty dictionary

        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector

                # empty lists filled in batch loop
                ts = []
                fit_freq, fit_time = [], []

                for bat in trial_batch: # loop over batches
                    
                    # avoid padding as this will introduce artefacts in the FT
                    self.freq_mid, self.time_mid, self._stf[hypo][det] = stft(self._comb[hypo][det][bat:bat+bat_step], 
                                                                                      fs = freq_sam, window = "hann", 
                                                                                      nperseg = hann_len, noverlap = hann_ovl, 
                                                                                      boundary = None, padded = False, 
                                                                                      return_onesided = True)
                    self.time_mid *= 1000 # time in units of ms

                    # take square of absolute for power
                    self._stf[hypo][det] = np.abs(self._stf[hypo][det]) ** 2

                    # take logarithm of max normalized power
                    self._log[hypo][det] = np.log10(self._stf[hypo][det]/np.nanmax(self._stf[hypo][det]))
                    
                    # take difference between median of time averaged spectrum and spectrum
                    self._log[hypo][det] = self._log[hypo][det] - np.repeat(np.nanmedian(self._log[hypo][det], axis = -1), self._log[hypo][det].shape[-1]).reshape(self._log[hypo][det].shape) 

                    # take only values where difference is larger than zero, i.e. only overfluctuations
                    self._log[hypo][det] = np.maximum(self._log[hypo][det], 0)

                    # maximum (hottest pixel) in array of 2D STF, returns array of length trials
                    # value used for ts distribution
                    ts.append(np.nanmax(self._log[hypo][det], axis = (1,2)))

                    # get time and freq index position of maximum 
                    ind_freq, ind_time = argmax_lastNaxes(self._log[hypo][det], 2)
                    # get corresponding time and freq of bin
                    fit_freq.append(self.freq_mid[ind_freq])
                    fit_time.append(self.time_mid[ind_time])
                
                self.ts[hypo][det] = np.array(ts).flatten()
                self.ffit[hypo][det], self.tfit[hypo][det] = np.array(fit_freq).flatten(), np.array(fit_time).flatten()

        return

    def get_ts_stat(self):

        self._ts_bkg_fit = {"ic86": None, "gen2": None, "wls": None} # empty dictionary
        self.ts_stat = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                        "signal" : {"ic86": None, "gen2": None, "wls": None}}
        
        for det in ["ic86", "gen2", "wls"]: # loop over detector
            # fitted background TS distribution
            self._ts_bkg_fit[det] = skewnorm(*skewnorm.fit(self.ts["null"][det]))
        
        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector

                # median, 16% and 84% quantiles of TS distribution
                self.ts_stat[hypo][det] = np.array([np.median(self.ts[hypo][det]), np.quantile(self.ts[hypo][det], 0.16), np.quantile(self.ts[hypo][det], 0.84)])

        return

    def get_zscore(self):

        self.zscore = {"ic86": None, "gen2": None, "wls": None} # empty dictionary

        for det in ["ic86", "gen2", "wls"]: # loop over detector
            z = []
            for i in range(3): # loop over median, 16% and 84% quantiles of TS distribution

                # p-value of signal given a background distribution
                p = self._ts_bkg_fit[det].sf(self.ts_stat["signal"][det][i])

                # two-sided Z score corresponding to the respective p-value, survival probability = 1 - cdf
                zz = norm.isf(p/2)
                z.append(zz)

            self.zscore[det] = np.array(z)

        return
  
    def run(self, mode, ft_para, trials = None):
        """Runs complete analysis chain including for time-integrated fast fourier transform (FFT)
        and short-time fourier transform (STF). It computes background and signal hits, 
        combines them, performs either FFT or STFT and calculates the TS distribution and significance.    

        Args:
            mode (str): analysis mode (FFT or STF)
            ft_para (dict): parameters of the fourier transform (FFT or STF)
            trials (int, optional): Number of trials. Defaults to None.

        Raises:
            ValueError: mode takes two valid values: "FFT" and "STF".
        """

        if trials is not None:
            self.trials = trials

        # load and combine data
        self._background()
        self._average_background()
        #self._signal()
        self._signal_model()
        self._signal_sampled()

        if mode == "FFT":
            self._hypothesis(hanning = ft_para["hanning"])
            self.fft(ft_para)

        elif mode == "STF":
            self._hypothesis()
            self.stf(ft_para)
            
        else:
            raise ValueError('{} mode does not exist. Choose from "FFT" and "STF"'.format(mode))
      
        self.get_ts_stat()
        self.get_zscore()
        
    def dist_scan(self, distance_range, mode, ft_para, trials = None):

        if trials is not None:
            self.trials = trials
        
        # prepare empty lists for distance loop
        zscore = {"ic86": [], "gen2": [], "wls": []}
        ts_stat = {"null" : {"ic86": [], "gen2": [], "wls": []}, "signal" : {"ic86": [], "gen2": [], "wls": []}}

        for dist in distance_range:

            print("Distance: {:.1f}".format(dist))

            self.set_distance(distance=dist) # set simulation to distance
            self.run(mode, ft_para)

            for det in ["ic86", "gen2", "wls"]: # loop over detector
                zscore[det].append(self.zscore[det])
                ts_stat["null"][det].append(self.ts_stat["null"][det])
                ts_stat["signal"][det].append(self.ts_stat["signal"][det])

        # for each key return array of length (3, len(dist_range))
        Zscore = {}
        for key, value in zscore.items():
            Zscore[key] = np.transpose(np.array(value))

        Ts_stat = {}
        for key, nested_dict in ts_stat.items():
            Ts_stat[key] = {}
            for nested_key, value in nested_dict.items():
                Ts_stat[key][nested_key] = np.transpose(np.array(value))

        return Zscore, Ts_stat



    """
    ToDo
    - implement SASI on lumi model basis, correct flat model
    - write template loop, optimize fit
    - test functions
    - units: be consistent with time units e.g. always s, ms, distance kpc  
    """