import os
from tqdm import tqdm

from null_hypothesis import *
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
        self.ft_para = self.para["ft_para"]
        self.bkg_trials = self.para["bkg_trials"]
        self.bkg_bins = self.para["bkg_bins"]

        self.get_dir_name()
        self._file = os.path.dirname(os.path.abspath(__file__))
        self.dir_path = os.path.join(self._file, f"../files/background/{self.model['name']}/{self.bkg_dir_name}/")

        # create directory if it does not already exist
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
            print(f"Creating directory for bkg trials in {self.dir_path}...")

        np.random.seed(0)

    def get_dir_name(self):
        # select correct directory for systemics
        if self.mixing_scheme == "NoTransformation":
            self.bkg_dir_name = "default"
            
        elif self.mixing_scheme == "CompleteExchange":
            self.bkg_dir_name = "mix_comp_exch"

        elif self.mixing_scheme == "AdiabaticMSW":
            if self.hierarchy == "normal":
                self.bkg_dir_name = "mix_MSW_NH"
            elif self.hierarchy == "inverted":
                self.bkg_dir_name = "mix_MSW_IH"

        # model
        if self.model['param']['progenitor_mass'] != 20 * u.solMass:
            progenitor_mass = self.model['param']['progenitor_mass'].value
            direction = self.model['param']['direction']
            self.bkg_dir_name = f"mod_{progenitor_mass}Msol_{direction}"

        # analysis cut
        if self.ft_para['freq_win'][1] == 85*u.Hz:
            if self.ft_para['time_win'][0] == 150*u.ms:
                self.bkg_dir_name = "ana_fcut_tcut"
            else:
                self.bkg_dir_name = "ana_fcut"
        else:
            if self.ft_para['time_win'][0] == 150*u.ms:
                self.bkg_dir_name = "ana_tcut"

    def generate(self, filename = None):
        """Simulates TS distribution for N=self.bkg_trials trials in batches of 10000 and saves the data.

        Args:
            filename (str, optional): Name of simulation output file. Defaults to None.
        """

        # filename for simulation output
        filename = os.path.join(self.dir_path, "HIST_model_{}_{:.0f}_mix_{}_hier_{}_bkg_trials_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.npz".format(
            self.model["name"], 
            self.model["param"]["progenitor_mass"].value, 
            self.mixing_scheme, 
            self.hierarchy,
            self.bkg_trials, 
            self.bkg_bins, 
            self.distance.value))

        # number of maximum trials, number of repetitions needed to fill bkg_trials
        self.max_trials = 10000 # size of batches
        self.repetitions = np.round(self.bkg_trials/self.max_trials).astype(int) # number of repetitions with size batches

        self.ts_binned = {"ic86" : np.array([np.zeros(self.bkg_bins, dtype=np.float64), np.zeros(self.bkg_bins, dtype=np.float64)]), # histogram data (x,y) for each subdetector
                     "gen2" : np.array([np.zeros(self.bkg_bins, dtype=np.float64), np.zeros(self.bkg_bins, dtype=np.float64)]), 
                     "wls": np.array([np.zeros(self.bkg_bins, dtype=np.float64), np.zeros(self.bkg_bins, dtype=np.float64)])}
        
        bounds = {"ic86": None, "gen2": None, "wls": None}
        
        for r in tqdm(range(self.repetitions)): # loop over batches
            # Initialize null hypothsis class and run analysis
            nlh = Null_Hypothesis(self.sim, res_dt = self.sim._res_dt, distance=self.distance)
            nlh.run(ft_para = self.ft_para, 
                    bkg_trials = self.max_trials)

            for det in ["ic86", "gen2", "wls"]: # loop over subdetectors
                if r == 0: 
                    bkg_min, bkg_max = nlh.ts[det].min(), nlh.ts[det].max()
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

        if not os.path.exists(self.dir_path):
            raise FileNotFoundError("Directory does not exist. Run generate function first.")

        for dist in distance_range: # loop over all distances
            print("Distance: {}".format(dist))
            # filename of simulation output
            filename_in = os.path.join(self.dir_path, "HIST_model_{}_{:.0f}_mix_{}_hier_{}_bkg_trials_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.npz".format(
            self.model["name"], 
            self.model["param"]["progenitor_mass"].value, 
            self.mixing_scheme, 
            self.hierarchy,
            self.bkg_trials, 
            self.bkg_bins, 
            dist.value))

            data = np.load(filename_in)
            
            for det in ["ic86", "gen2", "wls"]: # loop over detectors
                perc = [0.5, 0.16, 0.84]
                quant = quantiles_histogram(data[det], perc)
                
                qdict[det].append(quant)
        
        for det in ["ic86", "gen2", "wls"]: # loop over detectors
            qdict[det] = np.array(qdict[det])
        
        # save npz files
        filename_out = os.path.join(self.dir_path, "QUAN_model_{}_{:.0f}_mix_{}_hier_{}_bkg_trials_{:1.0e}_bins_{:1.0e}.npz".format(
            self.model["name"], 
            self.model["param"]["progenitor_mass"].value, 
            self.mixing_scheme, 
            self.hierarchy,
            self.bkg_trials, 
            self.bkg_bins))
        
        np.savez(file = filename_out, 
                 dist = distance_range.value, 
                 ic86 = qdict["ic86"],
                 gen2 = qdict["gen2"],
                 wls = qdict["wls"]) 
        
        return