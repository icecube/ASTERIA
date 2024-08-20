import os
import sys
from tqdm import tqdm
os.environ['ASTERIA'] = '/home/jakob/software/ASTERIA/ASTERIA'
from asteria.simulation import Simulation
from null_hypothesis import *
import scipy.stats as stats
import matplotlib.pyplot as plt 
from helper import *

# Class generating and resorting data for background trials
class Background_Trials():

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
        self.res_dt = self.para["res_dt"]
        self.ft_mode = self.para["ft_mode"]
        self.ft_para = self.para["ft_para"]
        self.bkg_trials = self.para["bkg_trials"]
        self.bkg_bins = self.para["bkg_bins"]
        self.sig_var = self.para["sig_var"]
        self.bkg_var = self.para["bkg_var"]

        self._file = os.path.dirname(os.path.abspath(__file__))

        np.random.seed(0)

    def get_dir_name(self):
        # select correct directory for systemics
        if self.mixing_scheme == "NoTransformation":
            self.bkg_dir_name = "default"
            if self.sig_var != 0:
                self.bkg_dir_name = "syst_det_sig_{:+.0f}%".format(self.sig_var*100)
            elif self.bkg_var != 0:
                self.bkg_dir_name = "syst_det_bkg_{:+.0f}%".format(self.bkg_var*100)

        elif self.mixing_scheme == "CompleteExchange":
            self.bkg_dir_name = "syst_mix_comp_exch"

        elif self.mixing_scheme == "AdiabaticMSW":
            if self.hierarchy == "normal":
                self.bkg_dir_name = "syst_mix_MSW_NH"
            elif self.hierarchy == "inverted":
                self.bkg_dir_name = "syst_mix_MSW_IH"

    def generate(self, filename = None):
        """Simulates TS distribution for N=self.bkg_trials trials in batches of 10000 and saves the data.

        Args:
            filename (str, optional): Name of simulation output file. Defaults to None.
        """
        
        self.get_dir_name()
        print(self.bkg_dir_name)

        # filename for simulation output
        filename = self._file + "/files/background/{}/{}/HIST_model_{}_{:.0f}_mode_{}_mix_{}_hier_{}_sig_var_{:+.0f}%_bkg_var_{:+.0f}%_bkg_trials_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.npz".format(
            self.ft_mode, self.bkg_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
            self.ft_mode, self.mixing_scheme, self.hierarchy,
            self.sig_var * 100, self.bkg_var * 100,
            self.bkg_trials, self.bkg_bins, self.distance.value)

        # number of maximum trials, number of repetitions needed to fill bkg_trials
        self.max_trials = 10000 # size of batches
        self.repetitions = np.round(self.bkg_trials/self.max_trials).astype(int) # number of repetitions with size batches

        self.ts_binned = {"ic86" : np.array([np.zeros(self.bkg_bins, dtype=np.float64), np.zeros(self.bkg_bins, dtype=np.float64)]), # histogram data (x,y) for each subdetector
                     "gen2" : np.array([np.zeros(self.bkg_bins, dtype=np.float64), np.zeros(self.bkg_bins, dtype=np.float64)]), 
                     "wls": np.array([np.zeros(self.bkg_bins, dtype=np.float64), np.zeros(self.bkg_bins, dtype=np.float64)])}
        
        bounds = {"ic86": None, "gen2": None, "wls": None}
        
        for r in tqdm(range(self.repetitions)): # loop over batches
            # Initialize null hypothsis class and run analysis
            nlh = Null_Hypothesis(self.sim, res_dt = self.res_dt, distance=self.distance)
            nlh.run(mode = self.ft_mode, ft_para = self.ft_para, 
                    sig_var = self.sig_var, bkg_var = self.bkg_var, 
                    bkg_trials = self.max_trials, 
                    model = "generic", smoothing = False)

            for det in ["ic86", "gen2", "wls"]: # loop over subdetectors
                if r == 0: bkg_min, bkg_max = nlh.ts[det].min(), nlh.ts[det].max()

                bounds[det] = (0.1 * bkg_min, 1.9 * bkg_max)
                hist_y, hist_bins = np.histogram(nlh.ts[det] , bins = self.bkg_bins, range = bounds[det], density=True)

                if r == 0: # x values are always the same
                    hist_x = (hist_bins[1:]+hist_bins[:-1])/2
                    self.ts_binned[det][0] = hist_x
                self.ts_binned[det][1] += hist_y # add y value for each batch to previous
        
        for det in ["ic86", "gen2", "wls"]: # loop over subdetectors
            self.ts_binned[det][1] *= 1/(np.sum(self.ts_binned[det][1]) * (self.ts_binned[det][0][1] - self.ts_binned[det][0][0])) # normalize sum of histgram  

        np.savez(file = filename, 
                 ic86 = self.ts_binned["ic86"],
                 gen2 = self.ts_binned["gen2"],
                 wls = self.ts_binned["wls"])
            
        return
    
    def load(self, filename):
        """Load data.

        Args:
            filename (str): Filename.
        """
        data = np.load(filename, allow_pickle=True)
        
        self.repetitions = data["reps"]
        self.max_trials = data["trials"]
        self.ts_binned = data

        return

    def quantiles(self, distance_range):
               
        qdict = {"ic86": [], "gen2": [], "wls": []}

        for dist in distance_range: # loop over all distances
            print("Distance: {}".format(dist))
            # filename of simulation output
            filename_in = self._file + "/files/background/{}/{}/HIST_model_{}_{:.0f}_mode_{}_mix_{}_hier_{}_sig_var_{:+.0f}%_bkg_var_{:+.0f}%_bkg_trials_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.npz".format(
                self.ft_mode, self.bkg_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
                self.ft_mode, self.mixing_scheme, self.hierarchy,
                self.sig_var * 100, self.bkg_var * 100,
                self.bkg_trials, self.bkg_bins, dist.value)
            data = np.load(filename_in)
            
            for det in ["ic86", "gen2", "wls"]: # loop over detectors
                perc = [0.5, 0.16, 0.84]
                quant = quantiles_histogram(data[det], perc)
                
                qdict[det].append(quant)
        
        for det in ["ic86", "gen2", "wls"]: # loop over detectors
            qdict[det] = np.array(qdict[det])
        
        # save npz files
        filename_out = self._file + "/files/background/{}/{}/QUAN_model_{}_{:.0f}_mode_{}_mix_{}_hier_{}_sig_var_{:+.0f}%_bkg_var_{:+.0f}%_bkg_trials_{:1.0e}_bins_{:1.0e}.npz".format(
            self.ft_mode, self.bkg_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
            self.ft_mode, self.mixing_scheme, self.hierarchy,
            self.sig_var * 100, self.bkg_var * 100,
            self.bkg_trials, self.bkg_bins)
        
        np.savez(file = filename_out, 
                 dist = distance_range.value, 
                 ic86 = qdict["ic86"],
                 gen2 = qdict["gen2"],
                 wls = qdict["wls"]) 
        
        return