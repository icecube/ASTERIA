import numpy as np
import astropy.units as u
from scipy.fft import fft, fftfreq
from scipy.signal import ShortTimeFFT, stft
from scipy.signal.windows import hann
from scipy.stats import skewnorm, norm

from asteria.simulation import Simulation as sim

def argmax_lastNaxes(A, N):
    # extension of argmax over several axis
    s = A.shape
    new_shp = s[:-N] + (np.prod(s[-N:]),)
    max_idx = A.reshape(new_shp).argmax(-1)
    return np.unravel_index(max_idx, s[-N:])




# test functions
# units: closer to front end (_E_V) remove units at backend, add in frontend
# units: be consistent with time units e.g. always s, ms, distance kpc  
# wls: detector, read mdom table twice and rescale,

# analysis class inherits methods of simulation class
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

        #define empty dictionaries that will be filled with data later
        self._bkg = {"ic86": None, "gen2": None, "wls": None}
        self._avg_bkg = {"ic86": None, "gen2": None, "wls": None}
        self._sig = {"null" : {"ic86": None, "gen2": None, "wls": None}, "signal" : {"ic86": None, "gen2": None, "wls": None}}
        self._sig_sample = {"null" : {"ic86": None, "gen2": None, "wls": None}, "signal" : {"ic86": None, "gen2": None, "wls": None}}
        self._comb = {"null" : {"ic86": None, "gen2": None, "wls": None}, "signal" : {"ic86": None, "gen2": None, "wls": None}}
        self._fft = {"null" : {"ic86": None, "gen2": None, "wls": None}, "signal" : {"ic86": None, "gen2": None, "wls": None}}
        self._stf = {"null" : {"ic86": None, "gen2": None, "wls": None}, "signal" : {"ic86": None, "gen2": None, "wls": None}}
        self._log = {"null" : {"ic86": None, "gen2": None, "wls": None}, "signal" : {"ic86": None, "gen2": None, "wls": None}}
        self.stf_ts = {"null" : {"ic86": None, "gen2": None, "wls": None}, "signal" : {"ic86": None, "gen2": None, "wls": None}}
        self.stf_fit_freq = {"null" : {"ic86": None, "gen2": None, "wls": None}, "signal" : {"ic86": None, "gen2": None, "wls": None}}
        self.stf_fit_time = {"null" : {"ic86": None, "gen2": None, "wls": None}, "signal" : {"ic86": None, "gen2": None, "wls": None}}
        self._ts_bkg_fit = {"ic86": None, "gen2": None, "wls": None} 
        self.ts_stat = {"null" : {"ic86": None, "gen2": None, "wls": None}, "signal" : {"ic86": None, "gen2": None, "wls": None}}
        self.zscore = {"ic86": None, "gen2": None, "wls": None}


        # rescale result
        self.sim.rebin_result(dt = self.sim._res_dt)
        self.sim.scale_result(distance=distance)

    def set_sim(self, sim):
        self.sim = sim

    def set_distance(self, distance):
        self.distance = distance
        self.sim.scale_result(distance=distance)

    def set_trials(self, trials):
        self.trials = trials

    def set_temp_para(self, temp_para):
        self.temp_para = temp_para

    def _background(self):

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
        
        #average background given by the mean of the sensor distribution and scaled to the full detector
        #rate is indirectly in Hz but noise rate is in units of binning e.g. 1 ms
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

    def _flat_signal(self):

        #detector_signal s0 is not drawn from distribution
        t, sig_i3 = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='i3')
        t, sig_dc = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='dc')
        t, sig_md = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='md')
        t, sig_ws = self.sim.detector_signal(dt=self.sim._res_dt, subdetector='ws')

        # combine subdetector signal into IC86 and Gen2 background rate
        sig_ic86 = sig_i3 + sig_dc
        sig_gen2 = sig_i3 + sig_dc + sig_md
        sig_wls  = sig_i3 + sig_dc + sig_md + sig_ws

        self._sig["null"]["ic86"] = sig_ic86
        self._sig["null"]["gen2"] = sig_gen2
        self._sig["null"]["wls"] = sig_wls

        return
    
    def _flat_signal_sampled(self):

        #add Poissonian fluctuations of signal
        for det in ["ic86", "gen2", "wls"]: # loop over detector
            self._sig_sample["null"][det] = np.random.normal(self._sig["null"][det], np.sqrt(np.abs(self._sig["null"][det])), size=(self.trials, self.tlength))
               
        return

    def get_template(self, temp_para):

        # template parameters
        frequency = temp_para["frequency"] # SASI frequency
        amplitude = temp_para["amplitude"] # SASI amplitude
        time_start = temp_para["time_start"] #SASI start time
        time_end = temp_para["time_end"] #SASI end time
        position = temp_para["position"] #SASI positioning relative to start time

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

    def _sasi_signal(self):

        for det in ["ic86", "gen2", "wls"]: # loop over detector
       
            temp_para_det = dict(self.temp_para)

            if self.temp_para["time_start"] < self.sim.time[0]:
                raise ValueError("time_start = {} smaller than simulation time start of {}".format(self.temp_para["time_start"], self.sim.time[0]))
            elif self.temp_para["time_end"] > self.sim.time[-1]:
                raise ValueError("time_end = {} larger than simulation time end of {}".format(self.temp_para["time_end"], self.sim.time[-1]))

            # scale amplitude relative to max light curve of IC86, Gen2, WLS
            temp_para_det["amplitude"] = temp_para_det["amplitude"] * np.max(self._sig["null"][det])
      
            template = self.get_template(temp_para_det)
            
            # combine flat light curve with template
            sig = self._sig["null"][det] + template
        
            # make sure that high amplitude fluctuations do not cause negative counts
            sig = np.maximum(sig, 0)

            self._sig["signal"][det] = sig
        
        return

    def _sasi_signal_sampled(self):
                
        #add Poissonian fluctuations of signal
        for det in ["ic86", "gen2", "wls"]: # loop over detector
            self._sig_sample["signal"][det] = np.random.normal(self._sig["signal"][det], np.sqrt(np.abs(self._sig["signal"][det])), size=(self.trials, self.tlength))
        
        return
              
    def _hypothesis(self, residual = False, hanning = False):

        # combine signal and background for null and signal hypothesis
        # apply residual or hanning if wanted

        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self._comb[hypo][det] = self._sig_sample[hypo][det] + self._bkg[det]
        
        if residual:
            self.apply_residual()

        if hanning:
            self.apply_hanning()

        return
    
    def apply_residual(self):

        # residual = ( combined (sampled) signal + background - averaged background ) / flat signal unsampled
        # applying residual multiple times will repeatedly rescale combined hits which is undesirable
        # make sure you only apply this once

        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self._comb[hypo][det] = ((self._comb[hypo][det] - self._avg_bkg[det])/self._sig[hypo][det])-1

        return
    
    def apply_hanning(self):
        # get hanning window
        # applying hanning multiple times will repeatedly hann combined hits which is undesirable
        # make sure you only apply this once

        hann = np.hanning(self.tlength)

        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self._comb[hypo][det] *= hann

        return
    
    def apply_tmask(self, time_win):
        #apply time cut before FFT
        time_low, time_high = time_win
        tmask = np.logical_and(self.sim.time>=time_low, self.sim.time<=time_high)
        
        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self._comb[hypo][det] = self._comb[hypo][det][:,tmask]

        self._time_new = self.sim.time[tmask]
        self.tlength_new = len(self._time_new)

        return

    def apply_fmask(self, freq_win):
        #apply frequency cut after FFT
        freq_low, freq_high = freq_win
        fmask = np.logical_and(self._freq >=freq_low, self._freq<=freq_high)

        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self._fft[hypo][det] = self._fft[hypo][det][:,fmask]

        self._freq_new = self._freq[fmask]
        self.flength_new = len(self._freq_new)

        return

    def fft(self, fft_para):

        time_res = fft_para["time_res"]
        time_win = fft_para["time_win"]
        freq_res = fft_para["freq_res"]
        freq_win = fft_para["freq_win"]

        if time_res != self.sim._res_dt:
            raise ValueError('fft_para["time_res"] = {} but ana.sim._res_dt = {}. Make sure to execute ana.run with the same resolution as you set in fft_para'.format(time_res, self.sim._res_dt))

        # time cut    
        self.apply_tmask(time_win)

        # power = (fourier modes) ** 2
        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                # calculate FFT
                self._fft[hypo][det] = (2.0/self.tlength_new * np.abs(fft(self._comb[hypo][det], axis = -1)[:,1:self.tlength_new//2]))**2
                # return frequencies
                self._freq = fftfreq(self.tlength_new,self.sim._res_dt)[1:self.tlength_new//2]

        # apply frequency cuts
        self.apply_fmask(freq_win)
        
        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                # max of FFT is used to build TS distribution
                self.stf_ts[hypo][det] = np.nanmax(self._fft[hypo][det], axis = -1)
                self.stf_fit_freq[hypo][det] = self._freq_new[np.argmax(self._fft[hypo][det], axis=-1)]

        return

    def stf(self, stf_para):

        hann_len = stf_para["hann_len"] # length of hann window
        hann_res = stf_para["hann_res"] # desired frequency resolution in Hann window
        hann_hop = stf_para["hann_hop"] # hann hop = number of time bins the window is moved
        freq_sam = stf_para["freq_sam"] # sampling frequency of entire signal (1 kHz for 1 ms binning)

        hann_len = int(hann_len.to_value(u.ms)) # define hann window
        hann_res = hann_res.to_value(u.Hz)
        hann_hop = int(hann_hop.to_value(u.ms)) #ShortTimeFFT does not accept numpt.int
        freq_sam = freq_sam.to_value(u.Hz)
        freq_mfft = int(freq_sam/hann_res)  #oversampling of hann window, relates to frequency resolution
        hann_ovl = int(hann_len - hann_hop) # define hann overlap

        """
        stf = ShortTimeFFT(hann_win, hop = hann_hop, fs = freq_sam)#, mfft= freq_mfft)

        # low and high value of time and frequency range
        time_low, time_high = np.array(stf.extent(self.tlength)[:2]) * 1000 # in ms units
        freq_low, freq_high = np.array(stf.extent(self.tlength)[2:]) # in Hz

        # edge values
        self.stf_time_edge = np.arange(time_low, (time_high + hann_hop), step = hann_hop)-hann_hop/2
        self.stf_freq_edge = np.arange(freq_low, (freq_high + hann_res), step = hann_res)-hann_res/2

        # mid values
        self.stf_time_mid = (self.stf_time_edge[1:] + self.stf_time_edge[:-1])/2
        self.stf_freq_mid = (self.stf_freq_edge[1:] + self.stf_freq_edge[:-1])/2
        """
        bat_step = 5000 # size of batches
        trial_batch = np.arange(0, self.trials, step=bat_step) #chunk data in batches of bat_step

        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector

                # empty lists filled in batch loop
                stf_ts = []
                fit_freq, fit_time = [], []

                for bat in trial_batch: # loop over batches

                    #self._stf[hypo][det] = stf.spectrogram(self._comb[hypo][det][bat:bat+1000], axis = -1) 
                    self.stf_freq_mid, self.stf_time_mid, self._stf[hypo][det] = stft(self._comb[hypo][det][bat:bat+bat_step], 
                                                                                      fs = freq_sam, window = "hann", 
                                                                                      nperseg = hann_len, noverlap = hann_ovl, 
                                                                                      boundary = None, padded = False, 
                                                                                      return_onesided = True)
                    self.stf_time_mid *= 1000 # time in units of ms

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
                    stf_ts.append(np.nanmax(self._log[hypo][det], axis = (1,2)))

                    # get time and freq index position of maximum 
                    ind_freq, ind_time = argmax_lastNaxes(self._log[hypo][det], 2)
                    # get corresponding time and freq of bin
                    fit_freq.append(self.stf_freq_mid[ind_freq])
                    fit_time.append(self.stf_time_mid[ind_time])
                
                self.stf_ts[hypo][det] = np.array(stf_ts).flatten()
                self.stf_fit_freq[hypo][det], self.stf_fit_time[hypo][det] = np.array(fit_freq).flatten(), np.array(fit_time).flatten()

        return

    def get_ts_stat(self):
        
        for det in ["ic86", "gen2", "wls"]: # loop over detector
            # fitted background TS distribution
            self._ts_bkg_fit[det] = skewnorm(*skewnorm.fit(self.stf_ts["null"][det]))
        
        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector

                # median, 16% and 84% quantiles of TS distribution
                self.ts_stat[hypo][det] = np.array([np.median(self.stf_ts[hypo][det]), np.quantile(self.stf_ts[hypo][det], 0.16), np.quantile(self.stf_ts[hypo][det], 0.84)])

        return

    def get_zscore(self):

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
  
    def run(self, fft_para, stf_para, mode = "STF"):

        # load and combine data
        self._background()
        self._average_background()
        self._flat_signal()
        self._flat_signal_sampled()
        self._sasi_signal()
        self._sasi_signal_sampled()

        if mode == "FFT":
            self._hypothesis(hanning=True)
            self.fft(fft_para)

        elif mode == "STF":
            self._hypothesis
            self.stf(stf_para)
            
        else:
            raise ValueError('{} mode does not exist. Choose from "FFT" and "STF"'.format(mode))
      
        self.get_ts_stat()
        self.get_zscore()
        
    def dist_scan(self, distance_range, fft_para, stf_para, mode = "STF"):
        
        # prepare empty lists for distance loop
        zscore = {"ic86": [], "gen2": [], "wls": []}
        ts_stat = {"null" : {"ic86": [], "gen2": [], "wls": []}, "signal" : {"ic86": [], "gen2": [], "wls": []}}

        for dist in distance_range:

            print("Distance: {:.1f}".format(dist))

            self.set_distance(distance=dist) # set simulation to distance
            self.run(fft_para, stf_para, mode = "STF")

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
    -implement SASI on lumi model basis, correct flat model
    -write template loop, optimize fit
    """