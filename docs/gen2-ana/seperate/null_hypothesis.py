import numpy as np
import astropy.units as u
from scipy.fft import fft, fftfreq
from scipy.signal import stft

from helper import argmax_lastNaxes, moving_average

class Null_Hypothesis():

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

    def set_bkg_trials(self, bkg_trials):
        """Set number of background trials

        Args:
            bkg_trials (int): number of background trials
        """
        self.bkg_trials = bkg_trials

    def _background(self):
        """Calculates the background hits in res_dt time steps for all sensors and combines the hits into three detector scopes
        IceCube 86 (IC86) : i3 + dc
        IceCube-Gen2      : i3 + dc + md
        IceCube-Gen2 + WLS: i3 + dc + md + ws
        """ 
        self._bkg = {"ic86": None, "gen2": None, "wls": None} # empty dictionary

        size = self.bkg_trials * self.tlength

        # pull random background for subdetectors
        bkg_i3 = self.sim.detector.i3_bg(dt=self.sim._res_dt, size=size).reshape(self.bkg_trials, self.tlength)
        bkg_dc = self.sim.detector.dc_bg(dt=self.sim._res_dt, size=size).reshape(self.bkg_trials, self.tlength)
        bkg_md = self.sim.detector.md_bg(dt=self.sim._res_dt, size=size).reshape(self.bkg_trials, self.tlength)
        bkg_ws = self.sim.detector.ws_bg(dt=self.sim._res_dt, size=size).reshape(self.bkg_trials, self.tlength)
        
        # combine subdetector background into IC86 and Gen2 background rate
        bkg_ic86 = bkg_i3 + bkg_dc
        bkg_gen2 = bkg_i3 + bkg_dc + bkg_md
        bkg_wls  = bkg_i3 + bkg_dc + bkg_md + bkg_ws

        self._bkg["ic86"] = bkg_ic86 * (1 + self.bkg_var)
        self._bkg["gen2"] = bkg_gen2* (1 + self.bkg_var)
        self._bkg["wls"] = bkg_wls* (1 + self.bkg_var)

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
    
    def _signal_generic(self, smoothing = False,):
        """Calculates the signal hits for a flat (null hypothesis).
                
        Args:
            smoothing (bool, optional): Smooth lightcurve. Defaults to False.
        """
        self._sig = {"ic86": None, "gen2": None, "wls": None} # empty dictionary
        
        # Null hypothesis, i.e. flat, no modulation SN light curve
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

        if smoothing:
            # binning needed to smoothen a frequency f, low frequency cut of 100 Hz
            frequency = 100*u.Hz
            duration = self.sim.time[-1]-self.sim.time[0]
            samples = (duration/self.sim._res_dt.to(u.s)).value
            binning  = int((1/frequency*samples/duration).value) #binning needed to filter out sinals with f>f_lb_sas

            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self._sig[det] = moving_average(self._sig[det], n = binning, const_padding = True)

        return
    
    def _signal_sampled(self):
        """Add Poissonian fluctuations of signal
        """
        self._sig_sample = {"ic86": None, "gen2": None, "wls": None} # empty dictionary
        for det in ["ic86", "gen2", "wls"]: # loop over detector
            self._sig_sample[det] = np.random.normal(self._sig[det], np.sqrt(np.abs(self._sig[det])), size=(self.bkg_trials, self.tlength))
        
        return
              
    def _hypothesis(self, residual = False, hanning = False):
        """Combines signal and background hits for the null hypothesis.
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
        This method overwrites the content of _comb. If keyword det is set to None, the cut is applied to all
        subdetectors. The new, cut time is _time_new with length tlength_new.
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
        This method overwrites the content of _fft or _stf. If keyword det is set to None, the cut is applied to all
        subdetectors. The new, cut frequency is _freq_new with length flength_new.
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

        self._fft = {"ic86": None, "gen2": None, "wls": None}
        self.ts = {"ic86": None, "gen2": None, "wls": None}
        self.ffit = {"ic86": None, "gen2": None, "wls": None}

        for det in ["ic86", "gen2", "wls"]: # loop over detector
            # calculate FFT, power = (fourier modes) ** 2
            self._fft[det] = (2.0/self.tlength_new * np.abs(fft(self._comb[det], axis = -1)[:,1:self.tlength_new//2]))**2
            # return frequencies
            self._freq = fftfreq(self.tlength_new,self.sim._res_dt)[1:self.tlength_new//2].to(u.Hz)

            self.apply_fmask(freq_win, det = det) # apply frequency mask
    
            # max of FFT is used to build TS distribution
            self.ts[det] = np.nanmax(self._fft[det], axis = -1)
            self.ffit[det] = self._freq_new[np.argmax(self._fft[det], axis=-1)].value

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
        trial_batch = np.arange(0, self.bkg_trials, step=bat_step) #chunk data in batches of bat_step

        self._stf = {"ic86": None, "gen2": None, "wls": None}
        self.ts = {"ic86": None, "gen2": None, "wls": None}
        self.ffit = {"ic86": None, "gen2": None, "wls": None}
        self.tfit = {"ic86": None, "gen2": None, "wls": None}

        for det in ["ic86", "gen2", "wls"]: # loop over detector

            # empty lists filled in batch loop
            ts = []
            fit_freq, fit_time = [], []

            for bat in trial_batch: # loop over batches
                
                # avoid padding as this will introduce artefacts in the FT
                self._freq, self._time, self._stf[det] = stft(self._comb[det][bat:bat+bat_step], 
                                                                                    fs = freq_sam, window = "hann", 
                                                                                    nperseg = hann_len, noverlap = hann_ovl, 
                                                                                    boundary = None, padded = False, 
                                                                                    return_onesided = True)
                self._freq *= u.Hz
                self._time = (self._time * u.s).to(u.ms) # time in units of ms
                
                self.apply_tmask(time_win, det = det)
                self.apply_fmask(freq_win, det = det)

                # take square of absolute for power
                self._stf[det] = np.abs(self._stf[det]) ** 2

                # maximum (hottest pixel) in array of 2D STF, returns array of length bkg_trials
                # value used for ts distribution
                ts.append(np.nanmax(self._stf[det], axis = (1,2)))

                # get time and freq index position of maximum 
                ind_freq, ind_time = argmax_lastNaxes(self._stf[det], 2)
                # get corresponding time and freq of bin
                fit_freq.append(self._freq_new[ind_freq])
                fit_time.append(self._time_new[ind_time])
            
            self.ts[det] = np.array(ts).flatten()
            self.ffit[det], self.tfit[det] = np.array(fit_freq).flatten(), np.array(fit_time).flatten()

        return
  
    def run(self, mode, ft_para, sig_var, bkg_var, bkg_trials, model = "generic", smoothing = False):
        """Runs complete analysis chain for the background distribution for both
        time-integrated fast fourier transform (FFT) and short-time fourier transform (STF). 
        It computes background and flat signal hits, combines them, performs either FFT or STFT 
        and calculates the TS distribution.    

        Args:
            mode (str): analysis mode (FFT or STF)
            ft_para (dict): parameters of the fourier transform (FFT or STF)
            sig_var (float): percentage variation of signal rate
            bkg_var (float): percentage variation of background rate            
            bkg_trials (int): Number of background trials.
            model (str): composition of signal trial ("generic", "model", "mix")
            smoothing (bool): Applies high-pass (moving average) filter.

        Raises:
            ValueError: mode takes two valid values: "FFT" and "STF".
        """

        self.mode = mode
        self.bkg_trials = bkg_trials
        self.sig_var = sig_var
        self.bkg_var = bkg_var

        if self.mode != "FFT" and self.mode != "STF":
            raise ValueError('{} mode does not exist. Choose from "FFT" and "STF"'.format(self.mode))

        # load and combine data
        self._background()
        self._average_background()
        self._signal_generic(smoothing=smoothing)
        self._signal_sampled()

        if self.mode == "FFT":
            self._hypothesis(hanning = ft_para["hanning"])
            self.fft(ft_para)

        elif self.mode == "STF":
            self._hypothesis()
            self.stf(ft_para)