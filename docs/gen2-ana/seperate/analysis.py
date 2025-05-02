import numpy as np
import astropy.units as u
from scipy.fft import fft, fftfreq
from scipy.signal import stft
from scipy.stats import norm, skewnorm, lognorm

from helper import *
from asteria.simulation import Simulation as sim

class Analysis():

    def __init__(self, 
                 sim, 
                 res_dt,
                 distance):

        # define a few attributes
        self.sim = sim
        self.sim._res_dt = res_dt
        self.distance = distance
        self.tlength = len(self.sim.time)

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
        duration = self.sim.time[-1]-self.sim.time[0]
        samples = (duration/self.sim._res_dt.to(u.s)).value

        binning  = int((1/self.smoothing_frequency*samples/duration).value) #binning needed to filter out sinals with f>f_lb_sasi

        for det in ["ic86", "gen2", "wls"]: # loop over detector

            self._sig["null"][det] = moving_average(self._sig["signal"][det], n = binning, const_padding = True)

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
    
    def _maxampl(self):
        """ Takes the maximum amplitude of the signal and null hypothesis for each detector as a proxy of the TS distribution. 
        """
        self._time = self.sim.time # get simulation time
        if (self._time[0] < self.time_win[0]) or (self._time[-1] > self.time_win[1]):
            self.apply_tmask(time_win=self.time_win)
        
        self.ts = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                   "signal" : {"ic86": None, "gen2": None, "wls": None}}
        
        for hypo in ["null", "signal"]: # loop over hypothesis        
            for det in ["ic86", "gen2", "wls"]: # loop over detector    
                self.ts[hypo][det] = np.nanmax(self._comb[hypo][det], axis = -1)

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
            for hypo in ["null", "signal"]: # loop over hypothesis
                for det in ["ic86", "gen2", "wls"]: # loop over detector
                    self._comb[hypo][det] = self._comb[hypo][det][:,tmask]
            
        else:
            self._comb[hypo][det] = self._comb[hypo][det][:,tmask]
            
        self._time_new = self._time[tmask]
        self.tlength_new = len(self._time_new) # new length of time array

        return

    def get_ts_stat(self, distr = lognorm):

        self._ts_bkg_fit = {"ic86": None, "gen2": None, "wls": None} # empty dictionary
        self.ts_stat = {"null" : {"ic86": None, "gen2": None, "wls": None}, 
                        "signal" : {"ic86": None, "gen2": None, "wls": None}}
        
        for det in ["ic86", "gen2", "wls"]: # loop over detector
            # fitted background TS distribution
            self._ts_bkg_fit[det] = distr(*distr.fit(self.ts["null"][det]))

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
  
    def run(self, mode, trials, time_win, smoothing_frequency):
        """Runs complete analysis chain for the background distribution for the maximum amplitude method. 
        It computes background and flat signal hits, combines them and calculates the TS distribution 
        from the maximum in the time domain.     

        Args:
            mode (str): analysis mode
            bkg_trials (int): Number of background trials.
            time_win (float): Time window.
            smoothing_frequency (float): Low pass frequency for smooting of signal hypothesis.
        """

        self.mode = mode
        self.trials = trials
        self.time_win = time_win
        self.smoothing_frequency = smoothing_frequency

        # analysis chain
        self._background()
        self._average_background()
        self._signal_model()
        self._signal_sampled()
        self._hypothesis()
        self._maxampl()
        self.get_ts_stat()
        self.get_zscore()
        
    def dist_scan(self, distance_range, mode, trials, 
                  time_win, smoothing_frequency, verbose = None):
        """Calls run method for a range of distances and saves z-score and TS value for all detectors in an array.   

        Args:
            distance_range (np.ndarray): Distance range array
            mode (str): analysis mode
            bkg_trials (int): Number of background trials.
            time_win (float): Time window.
            smoothing_frequency (float): Low pass frequency for smooting of signal hypothesis.
            verbose (bool, optional): Verbose level. Defaults to None.
        """

        self.dist_range = distance_range

        # prepare empty lists for distance loop
        zscore = {"ic86": [], "gen2": [], "wls": []}
        ts_stat = {"null" : {"ic86": [], "gen2": [], "wls": []}, "signal" : {"ic86": [], "gen2": [], "wls": []}}

        for dist in distance_range:

            if verbose == "debug":
                print("Distance: {:.1f}".format(dist))

            self.set_distance(distance=dist) # set simulation to distance
            self.run(mode, trials, time_win, smoothing_frequency)

            for det in ["ic86", "gen2", "wls"]: # loop over detector
                zscore[det].append(self.zscore[det])
                for hypo in ["null", "signal"]:
                    ts_stat[hypo][det].append(self.ts_stat[hypo][det])


        # for each key return array of length (3, len(dist_range))
        Zscore = {"ic86": [], "gen2": [], "wls": []}
        Ts_stat = {"null" : {"ic86": [], "gen2": [], "wls": []}, "signal" : {"ic86": [], "gen2": [], "wls": []}}

        for det in ["ic86", "gen2", "wls"]: # loop over detector  
            Zscore[det] = np.transpose(np.array(zscore[det]))
            for hypo in ["null", "signal"]: # loop over hypothesis
                Ts_stat[hypo][det] = np.transpose(np.array(ts_stat[hypo][det]))

        return Zscore, Ts_stat