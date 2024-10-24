import os
import sys
from tqdm import tqdm
from asteria.simulation import Simulation
from reco import *
import scipy.stats as stats
import matplotlib.pyplot as plt 
from helper import *
from plthelper import *

def strip_units(data):
    if isinstance(data, dict):
        return {key: strip_units(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [strip_units(item) for item in data]
    elif isinstance(data, u.Quantity):
        return data.value  # Strip the units
    else:
        return data  # Return the data as is if it's not a Quantity

# Class generating and resorting data for background trials
class Reconstruction_Trials():

    def __init__(self,
                 sim,
                 para,
                 verbose = None):
        """Class responsible for simulating the background TS distribution.

        Args:
            sim (asteria.simulation.Simulation): ASTERIA Simulation Class
            para (dict): Dictionary containing all analysis parameters.
            verbose (bool, optional): Verbose level. Defaults to None.
        """

        self.sim = sim
        self.para = para
        self.verbose = verbose

        # read in keywords of para
        self.model = self.para["model"]
        self.hierarchy = self.para["hierarchy"]
        self.mixing_scheme = self.para["mixing_scheme"]
        self.distance = self.para["distance"]
        self.reco_para = self.para["reco_para"]
        self.ft_mode = self.para["ft_mode"]
        self.ft_para = self.para["ft_para"]
        self.trials = self.para["trials"]
        self.sig_var = self.para["sig_var"]
        self.bkg_var = self.para["bkg_var"]

        self.get_dir_name()
        #self.reco_dir_name = "test" #overwrite directory name

        self._file = os.path.dirname(os.path.abspath(__file__))

        np.random.seed(0)

    def set_distance(self, distance):
        self.distance = distance

    def set_amplitude(self, amplitude):
        self.reco_para["ampl"] = np.array([amplitude])

    def get_dir_name(self):
        # select correct directory for systemics
        if self.mixing_scheme == "NoTransformation":
            self.reco_dir_name = "default"
            if self.sig_var != 0:
                self.reco_dir_name = "syst_det_sig_{:+.0f}%".format(self.sig_var*100)
            elif self.bkg_var != 0:
                self.reco_dir_name = "syst_det_bkg_{:+.0f}%".format(self.bkg_var*100)

        elif self.mixing_scheme == "CompleteExchange":
            self.reco_dir_name = "syst_mix_comp_exch"

        elif self.mixing_scheme == "AdiabaticMSW":
            if self.hierarchy == "normal":
                self.reco_dir_name = "syst_mix_MSW_NH"
            elif self.hierarchy == "inverted":
                self.reco_dir_name = "syst_mix_MSW_IH"

    def generate(self):
        """Simulates TS distribution for N=self.bkg_trials trials in batches of 10000 and saves the data.

        Args:
            filename (str, optional): Name of simulation output file. Defaults to None.
        """

        # lowest and highest frequency based on cutoff and Nyquist frequency
        self.freq_low = self.reco_para["freq_min"]
        self.freq_high = self.reco_para["freq_max"]

        # lowest and highest start time based on duration of signal and cropping from lack of padding in fourier transform
        self.time_low = self.reco_para["time_min"]
        self.time_high = self.reco_para["time_max"] - self.reco_para["duration"]

        # true frequency
        self.freq_true = np.zeros(self.trials) * u.Hz
        # true time
        self.time_true = np.zeros(self.trials) * u.ms
        
        self.freq_reco = {"null" : {"ic86" : np.zeros(self.trials, dtype=np.float64) * u.Hz, # reconstructed frequency for each detector
                                    "gen2" : np.zeros(self.trials, dtype=np.float64) * u.Hz, 
                                    "wls": np.zeros(self.trials, dtype=np.float64) * u.Hz},
                          "signal" : {"ic86" : np.zeros(self.trials, dtype=np.float64) * u.Hz,
                                      "gen2" : np.zeros(self.trials, dtype=np.float64) * u.Hz, 
                                      "wls": np.zeros(self.trials, dtype=np.float64) * u.Hz}}
        
        self.time_reco = {"null" : {"ic86" : np.zeros(self.trials, dtype=np.float64) * u.ms, # reconstructed time for each detector
                                    "gen2" : np.zeros(self.trials, dtype=np.float64) * u.ms, 
                                    "wls": np.zeros(self.trials, dtype=np.float64) * u.ms},
                          "signal" : {"ic86" : np.zeros(self.trials, dtype=np.float64) * u.ms,
                                      "gen2" : np.zeros(self.trials, dtype=np.float64) * u.ms, 
                                      "wls": np.zeros(self.trials, dtype=np.float64) * u.ms}}

        self.batch = 10000

        for r in tqdm(range(int(self.trials/self.batch))): # loop over batches
            # time_start and time_end used for template
            self.time_start = np.random.uniform(low = self.time_low.to(u.s).value, high = self.time_high.to(u.s).value, size = self.batch) * u.s
            self.time_end = self.time_start + self.reco_para["duration"].to(u.s)
            
            # true time = centre time = start time + duration/2
            self.t_true = self.time_start + self.reco_para["duration"].to(u.s)/2
            self.time_true[r*self.batch:(r+1)*self.batch] = self.t_true
        
            # true frequency
            self.f_true = np.random.uniform(low = self.freq_low.to(1/u.s).value, high = self.freq_high.to(1/u.s).value, size = self.batch) * u.Hz
            self.freq_true[r*self.batch:(r+1)*self.batch] = self.f_true

            temp_para = {"model": self.model,
                         "frequency": self.f_true, 
                         "amplitude": self.reco_para["ampl"], #in percent of max value
                         "time_start": self.time_start,
                         "time_end": self.time_end,
                         "duration": self.reco_para["duration"].to(u.s).value,
                         "position": self.reco_para["position"]}

            # Initialize reconstruction class and run analysis
            reco = Reconstruction(self.sim, res_dt = self.sim._res_dt, distance=self.distance, temp_para=temp_para)
            reco.run(mode = self.ft_mode, ft_para = self.ft_para, 
                    sig_var = self.sig_var, bkg_var = self.bkg_var, 
                    trials = self.batch, 
                    model = "generic", smoothing = False)
            
            for hypo in ["null", "signal"]: # loop over hypothesis
                for det in ["ic86", "gen2", "wls"]: # loop over detector
                    self.freq_reco[hypo][det][r*self.batch:(r+1)*self.batch] = reco.freq_reco[hypo][det] * u.Hz
                    self.time_reco[hypo][det][r*self.batch:(r+1)*self.batch] = reco.time_reco[hypo][det] * u.ms
        

        if self.verbose and 0:
            plot_reco(self, bins = 100, hypo = "signal", det = "ic86")
            plot_reco(self, bins = 100, hypo = "signal", det = "gen2")
            plot_reco(self, bins = 100, hypo = "signal", det = "wls")

            plot_reco(self, bins = 100, hypo = "null", det = "ic86")
            plot_reco(self, bins = 100, hypo = "null", det = "gen2")
            plot_reco(self, bins = 100, hypo = "null", det = "wls")

            #plot_reco_bias(self, hypo = "signal", det = "ic86")
            #plot_reco_bias(self, hypo = "signal", det = "gen2")
            #plot_reco_bias(self, hypo = "signal", det = "wls")

            #plot_reco_bias(self, hypo = "null", det = "ic86")
            #plot_reco_bias(self, hypo = "null", det = "gen2")
            #plot_reco_bias(self, hypo = "null", det = "wls")

        return
    
    def save(self):
        
        freq_true = strip_units(self.freq_true)
        freq_reco = strip_units(self.freq_reco)
        time_true = strip_units(self.time_true)
        time_reco = strip_units(self.time_reco)

        hypo = ["null", "signal"]
        det = ["ic86", "gen2", "wls"]


        filename = "./files/reco/{}/{}/RECO_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_ampl_{:.1f}%_mix_{}_hier_{}_trials_{:1.0e}_dist_{:.1f}kpc.npz".format(
                    self.ft_mode, self.reco_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
                    self.ft_mode, self.reco_para["duration"].value, self.reco_para["ampl"][0]*100, 
                    self.mixing_scheme, self.hierarchy, self.trials, self.distance.value)

        np.savez(file = filename, 
                    trials = self.trials, 
                    hypo = hypo,
                    det = det,
                    freq_true = freq_true,
                    freq_reco = freq_reco,
                    time_true = time_true,
                    time_reco = time_reco)
        return

    
    def load(self):
        """Load data.

        Args:
            filename (str): Filename.
        """

        filename = "./files/reco/{}/{}/RECO_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_ampl_{:.1f}%_mix_{}_hier_{}_trials_{:1.0e}_dist_{:.1f}kpc.npz".format(
                    self.ft_mode, self.reco_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
                    self.ft_mode, self.reco_para["duration"].value, self.reco_para["ampl"][0]*100, 
                    self.mixing_scheme, self.hierarchy, self.trials, self.distance.value)

        data = np.load(filename, allow_pickle=True)

        self.trials = data["trials"]

        self.freq_reco = {"null" : {"ic86" : np.zeros(self.trials, dtype=np.float64) * u.Hz, # reconstructed frequency for each detector
                                    "gen2" : np.zeros(self.trials, dtype=np.float64) * u.Hz, 
                                    "wls": np.zeros(self.trials, dtype=np.float64) * u.Hz},
                          "signal" : {"ic86" : np.zeros(self.trials, dtype=np.float64) * u.Hz,
                                      "gen2" : np.zeros(self.trials, dtype=np.float64) * u.Hz, 
                                      "wls": np.zeros(self.trials, dtype=np.float64) * u.Hz}}
        
        self.time_reco = {"null" : {"ic86" : np.zeros(self.trials, dtype=np.float64) * u.ms, # reconstructed time for each detector
                                    "gen2" : np.zeros(self.trials, dtype=np.float64) * u.ms, 
                                    "wls": np.zeros(self.trials, dtype=np.float64) * u.ms},
                          "signal" : {"ic86" : np.zeros(self.trials, dtype=np.float64) * u.ms,
                                      "gen2" : np.zeros(self.trials, dtype=np.float64) * u.ms, 
                                      "wls": np.zeros(self.trials, dtype=np.float64) * u.ms}}

        # add units
        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                self.freq_reco[hypo][det] = data["freq_reco"].item()[hypo][det] * u.Hz
                self.time_reco[hypo][det] = data["time_reco"].item()[hypo][det] * u.ms
        
        self.freq_true = data["freq_true"] * u.Hz
        self.time_true = data["time_true"] * u.ms
    
        return
    
    def stats(self, ampl_range, dist_range, mode = "load", axis = "distance"):

        self.ampl_range = ampl_range
        self.dist_range = dist_range
        
        if mode == "generate" and axis == "distance":

            for a, ampl in enumerate(self.ampl_range):
                print("Amplitude: {}%".format(np.round(ampl * 100, 1)))
                self.set_amplitude(ampl)

                self.freq_stat = {"null" : {"ic86" : np.zeros((self.dist_range.size, 3), dtype=np.float64) * u.Hz, # reconstructed frequency for each detector
                                            "gen2" : np.zeros((self.dist_range.size, 3), dtype=np.float64) * u.Hz, 
                                            "wls": np.zeros((self.dist_range.size, 3), dtype=np.float64) * u.Hz},
                                  "signal" : {"ic86" : np.zeros((self.dist_range.size, 3), dtype=np.float64) * u.Hz,
                                              "gen2" : np.zeros((self.dist_range.size, 3), dtype=np.float64) * u.Hz, 
                                              "wls": np.zeros((self.dist_range.size, 3), dtype=np.float64) * u.Hz}}

                self.time_stat = {"null" : {"ic86" : np.zeros((self.dist_range.size, 3), dtype=np.float64) * u.ms, # reconstructed frequency for each detector
                                            "gen2" : np.zeros((self.dist_range.size, 3), dtype=np.float64) * u.ms, 
                                            "wls": np.zeros((self.dist_range.size, 3), dtype=np.float64) * u.ms},
                                  "signal" : {"ic86" : np.zeros((self.dist_range.size, 3), dtype=np.float64) * u.ms,
                                              "gen2" : np.zeros((self.dist_range.size, 3), dtype=np.float64) * u.ms, 
                                              "wls": np.zeros((self.dist_range.size, 3), dtype=np.float64) * u.ms}}
                
                filename = "./files/reco/{}/{}/STAT_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_ampl_{:.1f}%_mix_{}_hier_{}_trials_{:1.0e}.npz".format(
                    self.ft_mode, self.reco_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
                    self.ft_mode, self.reco_para["duration"].value, self.reco_para["ampl"][0]*100, 
                    self.mixing_scheme, self.hierarchy, self.trials)


                for d, dist in enumerate(self.dist_range):
                    print("Distance: {}".format(dist))
                    self.set_distance(dist)
                    self.load()

                    for hypo in ["null", "signal"]: # loop over hypothesis
                        for det in ["ic86", "gen2", "wls"]: # loop over detector

                            rel_freq = self.freq_true-self.freq_reco[hypo][det]
                            rel_time = self.time_true-self.time_reco[hypo][det]

                            self.freq_stat[hypo][det][d] = np.array([np.median(rel_freq).value, np.quantile(rel_freq, 0.16).value, np.quantile(rel_freq, 0.84).value]) * u.Hz
                            self.time_stat[hypo][det][d] = np.array([np.median(rel_time).value, np.quantile(rel_time, 0.16).value, np.quantile(rel_time, 0.84).value]) * u.ms

                freq_stat = strip_units(self.freq_stat)
                time_stat = strip_units(self.time_stat)

                np.savez(file = filename, 
                        dist = self.dist_range.value, 
                        hypo = hypo,
                        det = det,
                        freq_stat = freq_stat,
                        time_stat = time_stat)
                
        elif mode == "generate" and axis == "amplitude":

            for d, dist in enumerate(self.dist_range):
                print("Distance: {}%".format(dist))
                self.set_distance(dist)

                self.freq_stat = {"null" : {"ic86" : np.zeros((self.ampl_range.size, 3), dtype=np.float64) * u.Hz, # reconstructed frequency for each detector
                                            "gen2" : np.zeros((self.ampl_range.size, 3), dtype=np.float64) * u.Hz, 
                                            "wls": np.zeros((self.ampl_range.size, 3), dtype=np.float64) * u.Hz},
                                  "signal" : {"ic86" : np.zeros((self.ampl_range.size, 3), dtype=np.float64) * u.Hz,
                                              "gen2" : np.zeros((self.ampl_range.size, 3), dtype=np.float64) * u.Hz, 
                                              "wls": np.zeros((self.ampl_range.size, 3), dtype=np.float64) * u.Hz}}

                self.time_stat = {"null" : {"ic86" : np.zeros((self.ampl_range.size, 3), dtype=np.float64) * u.ms, # reconstructed frequency for each detector
                                            "gen2" : np.zeros((self.ampl_range.size, 3), dtype=np.float64) * u.ms, 
                                            "wls": np.zeros((self.ampl_range.size, 3), dtype=np.float64) * u.ms},
                                  "signal" : {"ic86" : np.zeros((self.ampl_range.size, 3), dtype=np.float64) * u.ms,
                                              "gen2" : np.zeros((self.ampl_range.size, 3), dtype=np.float64) * u.ms, 
                                              "wls": np.zeros((self.ampl_range.size, 3), dtype=np.float64) * u.ms}}
                
                filename = "./files/reco/{}/{}/STAT_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_dist_{:.1f}kpc_mix_{}_hier_{}_trials_{:1.0e}.npz".format(
                    self.ft_mode, self.reco_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
                    self.ft_mode, self.reco_para["duration"].value, self.distance.value, 
                    self.mixing_scheme, self.hierarchy, self.trials)


                for a, ampl in enumerate(self.ampl_range):
                    print("Amplitude: {}%".format(np.round(ampl * 100, 1)))
                    self.set_amplitude(ampl)
                    self.load()

                    for hypo in ["null", "signal"]: # loop over hypothesis
                        for det in ["ic86", "gen2", "wls"]: # loop over detector

                            rel_freq = self.freq_true-self.freq_reco[hypo][det]
                            rel_time = self.time_true-self.time_reco[hypo][det]

                            self.freq_stat[hypo][det][a] = np.array([np.median(rel_freq).value, np.quantile(rel_freq, 0.16).value, np.quantile(rel_freq, 0.84).value]) * u.Hz
                            self.time_stat[hypo][det][a] = np.array([np.median(rel_time).value, np.quantile(rel_time, 0.16).value, np.quantile(rel_time, 0.84).value]) * u.ms

                freq_stat = strip_units(self.freq_stat)
                time_stat = strip_units(self.time_stat)

                np.savez(file = filename, 
                        dist = self.dist_range.value, 
                        hypo = hypo,
                        det = det,
                        freq_stat = freq_stat,
                        time_stat = time_stat)
        
        elif mode == "load":

            self.freq_stat = {"null" : {"ic86" : np.zeros((self.ampl_range.size, self.dist_range.size, 3), dtype=np.float64) * u.Hz, # reconstructed frequency for each detector
                                        "gen2" : np.zeros((self.ampl_range.size, self.dist_range.size, 3), dtype=np.float64) * u.Hz, 
                                        "wls": np.zeros((self.ampl_range.size, self.dist_range.size, 3), dtype=np.float64) * u.Hz},
                              "signal" : {"ic86" : np.zeros((self.ampl_range.size, self.dist_range.size, 3), dtype=np.float64) * u.Hz,
                                          "gen2" : np.zeros((self.ampl_range.size, self.dist_range.size, 3), dtype=np.float64) * u.Hz, 
                                          "wls": np.zeros((self.ampl_range.size, self.dist_range.size, 3), dtype=np.float64) * u.Hz}}

            self.time_stat = {"null" : {"ic86" : np.zeros((self.ampl_range.size, self.dist_range.size, 3), dtype=np.float64) * u.ms, # reconstructed frequency for each detector
                                        "gen2" : np.zeros((self.ampl_range.size, self.dist_range.size, 3), dtype=np.float64) * u.ms, 
                                        "wls": np.zeros((self.ampl_range.size, self.dist_range.size, 3), dtype=np.float64) * u.ms},
                              "signal" : {"ic86" : np.zeros((self.ampl_range.size, self.dist_range.size, 3), dtype=np.float64) * u.ms,
                                          "gen2" : np.zeros((self.ampl_range.size, self.dist_range.size, 3), dtype=np.float64) * u.ms, 
                                          "wls": np.zeros((self.ampl_range.size, self.dist_range.size, 3), dtype=np.float64) * u.ms}}

            if axis == "distance":
                for a, ampl in enumerate(self.ampl_range):
                    print("Amplitude: {}%".format(ampl*100))
                    self.set_amplitude(ampl)
                
                    filename = "./files/reco/{}/{}/STAT_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_ampl_{:.1f}%_mix_{}_hier_{}_trials_{:1.0e}.npz".format(
                                self.ft_mode, self.reco_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
                                self.ft_mode, self.reco_para["duration"].value, self.reco_para["ampl"][0]*100, 
                                self.mixing_scheme, self.hierarchy, self.trials)
                    data = np.load(filename, allow_pickle=True)


                    # add units
                    for hypo in ["null", "signal"]: # loop over hypothesis
                        for det in ["ic86", "gen2", "wls"]: # loop over detector
                            self.freq_stat[hypo][det][a] = data["freq_stat"].item()[hypo][det] * u.Hz
                            self.time_stat[hypo][det][a] = data["time_stat"].item()[hypo][det] * u.ms
                
                    if self.verbose:
                        plot_reco_horizon(self, ampl = a, hypo="null")
                        plot_reco_horizon(self, ampl = a, hypo="signal")

            elif axis == "amplitude":
                for d, dist in enumerate(self.dist_range):
                    print("Distance: {}%".format(dist))
                    self.set_distance(dist)
                
                    filename = "./files/reco/{}/{}/STAT_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_dist_{:.1f}kpc_mix_{}_hier_{}_trials_{:1.0e}.npz".format(
                                self.ft_mode, self.reco_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
                                self.ft_mode, self.reco_para["duration"].value, self.distance.value, 
                                self.mixing_scheme, self.hierarchy, self.trials)
                    data = np.load(filename, allow_pickle=True)


                    # add units
                    for hypo in ["null", "signal"]: # loop over hypothesis
                        for det in ["ic86", "gen2", "wls"]: # loop over detector
                            self.freq_stat[hypo][det][:,d] = data["freq_stat"].item()[hypo][det] * u.Hz
                            self.time_stat[hypo][det][:,d] = data["time_stat"].item()[hypo][det] * u.ms

        return
    
    def horizon(self, freq_thresh, time_thresh, axis = "distance"):

        self.freq_thresh = freq_thresh
        self.time_thresh = time_thresh

        self.freq_diff = {"null" : {"ic86" : np.zeros((self.ampl_range.size, self.dist_range.size), dtype=np.float64) * u.kpc, # reconstructed frequency for each detector
                                            "gen2" : np.zeros((self.ampl_range.size, self.dist_range.size), dtype=np.float64) * u.kpc, 
                                            "wls": np.zeros((self.ampl_range.size, self.dist_range.size), dtype=np.float64) * u.kpc},
                          "signal" : {"ic86" : np.zeros((self.ampl_range.size, self.dist_range.size), dtype=np.float64) * u.kpc,
                                      "gen2" : np.zeros((self.ampl_range.size, self.dist_range.size), dtype=np.float64) * u.kpc, 
                                      "wls": np.zeros((self.ampl_range.size, self.dist_range.size), dtype=np.float64) * u.kpc}}

        self.time_diff = {"null" : {"ic86" : np.zeros((self.ampl_range.size, self.dist_range.size), dtype=np.float64) * u.kpc, # reconstructed frequency for each detector
                                    "gen2" : np.zeros((self.ampl_range.size, self.dist_range.size), dtype=np.float64) * u.kpc, 
                                    "wls": np.zeros((self.ampl_range.size, self.dist_range.size), dtype=np.float64) * u.kpc},
                          "signal" : {"ic86" : np.zeros((self.ampl_range.size, self.dist_range.size), dtype=np.float64) * u.kpc,
                                      "gen2" : np.zeros((self.ampl_range.size, self.dist_range.size), dtype=np.float64) * u.kpc, 
                                      "wls": np.zeros((self.ampl_range.size, self.dist_range.size), dtype=np.float64) * u.kpc}}

        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
           
                self.freq_diff[hypo][det] = np.abs(self.freq_stat[hypo][det][:,:,2]-self.freq_stat[hypo][det][:,:,1])
                self.time_diff[hypo][det] = np.abs(self.time_stat[hypo][det][:,:,2]-self.time_stat[hypo][det][:,:,1])

        if axis == "distance":
            if self.verbose:
                for a, ampl in enumerate(self.ampl_range):

                    plot_reco_horizon_diff(self, ampl = a, hypo = "signal")
                    plot_reco_horizon_diff(self, ampl = a, hypo = "null")

            self.freq_hori = resolution_horizon(self.ampl_range, self.dist_range, self.freq_diff, self.freq_thresh)
            self.time_hori = resolution_horizon(self.ampl_range, self.dist_range, self.time_diff, self.time_thresh)

            plot_reco_horizon_amplitude(self, hypo = "null")
            plot_reco_horizon_amplitude(self, hypo = "signal")

        elif axis == "amplitude":

            plot_reco_at_distance(self, hypo = "signal")

        return
    
    def reco_at_significance(self, scan_para):

        ampl_range, amplitude, sig_trials, bkg_trials, time_start, time_end, sigma = scan_para

        self.ampl_range = ampl_range

        self.freq_reco_sig = {"ic86" : np.zeros((len(sigma), amplitude.size, 3), dtype=np.float64) * u.Hz, # reconstructed frequency at significance for each detector
                              "gen2" : np.zeros((len(sigma), amplitude.size, 3), dtype=np.float64) * u.Hz, 
                              "wls": np.zeros((len(sigma), amplitude.size, 3), dtype=np.float64) * u.Hz}

        self.time_reco_sig = {"ic86" : np.zeros((len(sigma), amplitude.size, 3), dtype=np.float64) * u.ms, # reconstructed time at significance for each detector
                              "gen2" : np.zeros((len(sigma), amplitude.size, 3), dtype=np.float64) * u.ms, 
                              "wls": np.zeros((len(sigma), amplitude.size, 3), dtype=np.float64) * u.ms}                  

        # filename for detection horizon data
        filename = "./files/scan/{}/{}/SIGN_FAVG_model_{}_{:.0f}_mode_{}_time_{:.0f}ms-{:.0f}ms_mix_{}_hier_{}_sig_var_{:.0f}%_bkg_var_{:.0f}%_sig_trials_{:1.0e}_bkg_trials_{:1.0e}_ampl_{:.1f}-{:.1f}%.npz".format(
                self.ft_mode, self.reco_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
                self.ft_mode, time_start.value, time_end.value, self.mixing_scheme, self.hierarchy,
                self.sig_var * 100, self.bkg_var * 100, sig_trials, bkg_trials, self.ampl_range[0]*100, self.ampl_range[-1]*100)

        print("Filename detection horizon: {}".format(filename))

        data = np.load(filename)

        for det in ["ic86", "gen2", "wls"]: # loop over detector
            print("Detector: {}".format(det))

            for s, sig in enumerate(sigma): # loop over significance level
                print("Significance: {} sigma".format(sig))

                for a, ampl in enumerate(amplitude): # loop over amplitude of model

                    print("Amplitude: {}%".format(ampl*100))
                    self.set_amplitude(ampl)

                    mean, std, q16, q84 = data[det][s][ampl_range == ampl].flatten() # read in horizon from SIGN_FAVG file
                    horizon = [mean, q16, q84] * u.kpc # compute resolution at mean and quantiles
                    if self.verbose: print("{} sigma detection horizon {}: {:.3f}+{:.3f}-{:.3f} kpc".format(sig, det, mean, q84-mean, mean-q16))

                    for d, dist in enumerate(horizon):
                        
                        if np.isnan(dist): # if detection horizon is nan e.g. for (low ampl, 5sigma, ic86)
                            self.freq_reco_sig[det][s,a,d] = np.nan 
                            self.time_reco_sig[det][s,a,d] = np.nan
                            continue

                        self.set_distance(dist)
                        print("Distance: {:.3f}".format(self.distance))
                        self.generate() # run reconstruction

                        rel_freq = self.freq_true-self.freq_reco["signal"][det] # take only detect that is on horizon
                        rel_time = self.time_true-self.time_reco["signal"][det]

                        self.freq_reco_sig[det][s,a,d] = np.quantile(rel_freq, 0.84) - np.quantile(rel_freq, 0.16) 
                        self.time_reco_sig[det][s,a,d] = np.quantile(rel_time, 0.84) - np.quantile(rel_time, 0.16)

        filename = "./files/reco/{}/{}/RECO_SIG_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_ampl_{:.1f}%_mix_{}_hier_{}_trials_{:1.0e}.npz".format(
                    self.ft_mode, self.reco_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
                    self.ft_mode, self.reco_para["duration"].value, amplitude[0] * 100,
                    self.mixing_scheme, self.hierarchy, self.trials)

        np.savez(file = filename, 
                 ampl = amplitude,
                 det = det,
                 sigma = sigma,
                 freq_reco_sig = self.freq_reco_sig,
                 time_reco_sig = self.time_reco_sig)
        
        return
            
    def bootstrap(self, filename, rep_trials = 10000, repetitions = 100):

        self.rep_trials = rep_trials
        self.repetitions = repetitions

        print("BOOTSTRAPPING: Trials: {}, Repetitions: {}".format(rep_trials, repetitions))
        # dict holds for all hypo and det for each repetition the median, 16% and 84% quantiles
        self.boot_freq = {"null" : {"ic86": np.zeros((3, self.repetitions)), "gen2": np.zeros((3, self.repetitions)), "wls": np.zeros((3, self.repetitions))}, 
                          "signal" : {"ic86": np.zeros((3, self.repetitions)), "gen2": np.zeros((3, self.repetitions)), "wls": np.zeros((3, self.repetitions))}} # empty dictionary
        self.boot_time = {"null" : {"ic86": np.zeros((3, self.repetitions)), "gen2": np.zeros((3, self.repetitions)), "wls": np.zeros((3, self.repetitions))}, 
                          "signal" : {"ic86": np.zeros((3, self.repetitions)), "gen2": np.zeros((3, self.repetitions)), "wls": np.zeros((3, self.repetitions))}}
        
        self.load(filename)
        samples = self.rep_trials * self.repetitions # sample rep_trials x repetitions
        index = np.arange(self.trials) # list with index positions until rep_trials x repetitions

        # sample indices, indices can be double
        ind = np.random.choice(index, size = samples, replace = True)

        # sample corresponding true frequency and time and reshape
        freq_true = self.freq_true[ind].reshape(self.repetitions, self.rep_trials)
        time_true = self.time_true[ind].reshape(self.repetitions, self.rep_trials)

        # for the reco frequency and time the values are detector and hypothesis dependent
        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                
                # sample reco frequency and time and reshape
                freq_reco = self.freq_reco[hypo][det][ind].reshape(self.repetitions, self.rep_trials)
                time_reco = self.time_reco[hypo][det][ind].reshape(self.repetitions, self.rep_trials)

                # compute median, 16% and 84% quantile for each repetition of the true fequency and time
                self.boot_freq[hypo][det] = np.array([np.mean(freq_reco-freq_true, axis = 1).value, np.quantile(freq_reco-freq_true, 0.16, axis = 1).value, np.quantile(freq_reco-freq_true, 0.84, axis = 1).value]) * u.Hz
                self.boot_time[hypo][det] = np.array([np.mean(time_reco-time_true, axis = 1).value, np.quantile(time_reco-time_true, 0.16, axis = 1).value, np.quantile(time_reco-time_true, 0.84, axis = 1).value]) * u.ms

        plot_reco_boot(self, hypo = "signal")
        plot_reco_boot(self, hypo = "null")

        return