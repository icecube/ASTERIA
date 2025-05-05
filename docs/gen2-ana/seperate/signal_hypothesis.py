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
                 distance):

        # define a few attributes
        self.sim = sim
        self.sim._res_dt = res_dt
        self.distance = distance
        self.tlength = len(self.sim.time)

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

        self._sig["ic86"] = sig_ic86
        self._sig["gen2"] = sig_gen2
        self._sig["wls"] = sig_wls

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
    
    def _maxampl(self):
        """ Takes the maximum amplitude of the signal and null hypothesis for each detector as a proxy of the TS distribution. 
        """
        self._time = self.sim.time # get simulation time
        if (self._time[0] < self.time_win[0]) or (self._time[-1] > self.time_win[1]):
            self.apply_tmask(time_win=self.time_win)
        
        self.ts = {"ic86": None, "gen2": None, "wls": None}
        
        for det in ["ic86", "gen2", "wls"]: # loop over detector    
                self.ts[det] = np.nanmax(self._comb[det], axis = -1)
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
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self._comb[det] = self._comb[det][:,tmask]

        else:
            self._comb[det] = self._comb[det][:,tmask]
            
        self._time_new = self._time[tmask]
        self.tlength_new = len(self._time_new) # new length of time array

        return

    def get_ts_stat(self):
        # Load the 50%, 16% and 84% quantiles for the background hypothesis
        filename = self._file + "/files/background/{}/{}/{}/QUAN_model_{}_mode_{}_mix_{}_hier_{}_bkg_trials_{:1.0e}_bins_{:1.0e}.npz".format(
            self.mode, self.bkg_dir_name, self.model, self.model,
            self.mode, self.mixing_scheme, self.hierarchy,
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
        filename = self._file + "/files/background/{}/{}/{}/HIST_model_{}_mode_{}_mix_{}_hier_{}_bkg_trials_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.npz".format(
                self.mode, self.bkg_dir_name, self.model, self.model, 
                self.mode, self.mixing_scheme, self.hierarchy,
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
  
    def run(self, mode, sig_trials, bkg_trials, time_win, bkg_bins = None):
        """Runs complete analysis chain including for time-integrated fast fourier transform (FFT)
        and short-time fourier transform (STF). It computes background and signal hits, 
        combines them, performs either FFT or STFT and calculates the TS distribution and significance.    

        Args:
            mode (str): analysis mode (FFT or STF)
            sig_trials (int): Number of signal trials.
            bkg_trials (int): Number of background trials.
            bkg_bins (int): Number of histogram bins of background distribution
            time_win (tuple): lower and higher time cut

        Raises:
            ValueError: model takes three valid values: "generic", "model" and "mix.
            ValueError: mode takes two valid values: "FFT" and "STF".
        """

        self.mode = mode
        self.sig_trials = sig_trials
        self.bkg_trials = bkg_trials
        self.time_win = time_win
        self.bkg_bins = bkg_bins

        # model name (RDF_1_2 or RDF_1_7) is saved in path
        path = self.sim.metadata["model"]["param"].split(",")[1]
        self.model = path.split("/")[-2]

        self.get_dir_name() # build bkg file and scan directory name from input

        # load and combine data
        self._background()
        self._average_background()
        self._signal_model()
        self._signal_sampled()
        self._hypothesis()
        self._maxampl()
        self.get_ts_stat()
        self.get_zscore()
        
    def dist_scan(self, distance_range, mode, sig_trials, bkg_trials, time_win, bkg_bins, verbose = None):
        """Calls run method for a range of distances and saves z-score and TS value for all detectors in an array.   

        Args:
            distance_range (np.ndarray): Distance range array
            mode (str): analysis mode (FFT or STF)
            sig_trials: Number of signal trials.
            bkg_trials (int): Number of background trials.
            time_win (tuple): lower and higher time cut
            bkg_bins (int): Number of histogram bins of background distribution
        """
        # prepare empty lists for distance loop
        zscore = {"ic86": [], "gen2": [], "wls": []}
        ts_stat = {"null" : {"ic86": [], "gen2": [], "wls": []}, "signal" : {"ic86": [], "gen2": [], "wls": []}}

        for dist in distance_range:

            if verbose == "debug":
                print("Distance: {:.1f}".format(dist))

            self.set_distance(distance=dist) # set simulation to distance
            self.run(mode, sig_trials, bkg_trials, time_win, bkg_bins)

            for det in ["ic86", "gen2", "wls"]: # loop over detector
                zscore[det].append(self.zscore[det])
                ts_stat["null"][det].append(self.ts_stat["null"][det])
                ts_stat["signal"][det].append(self.ts_stat["signal"][det])

        # for each key return array of length (3, len(dist_range))
        Zscore = {"ic86": [], "gen2": [], "wls": []}

        for det in ["ic86", "gen2", "wls"]: 
            Zscore[det] = np.transpose(np.array(zscore[det]))

        Ts_stat = {}
        for key, nested_dict in ts_stat.items():
            Ts_stat[key] = {}
            for nested_key, value in nested_dict.items():
                Ts_stat[key][nested_key] = np.transpose(np.array(value))

        return Zscore, Ts_stat
