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
                 ana_para,
                 bkg_trials,
                 mode,
                 output = None,
                 bkg_bins = None,
                 verbose = None):
        """Class responsible for simulating the background TS distribution. Methods allow to manipulate the data such as histogramming, 
           fitting and quantiles of the distribution.

        Args:
            sim (asteria.simulation.Simulation): ASTERIA Simulation Class
            ana_para (dict): Dictionary containing all analysis parameters.
            bkg_trials (int): Number of background trials
            mode (str): Use full data (mode = "data") or histgram data to reduce storage (mode = "hist").
            output (int): Needed if simulation is split into several, shorter runs with different outputs.
            bkg_bins (int, optional): Number of bakcground bins if mode = "hist". Defaults to None.
            verbose (bool, optional): Verbose level. Defaults to None.

        Raises:
            ValueError: Valid values for mode are "data" and  "hist""
        """
        if mode != "data" and mode != "hist":
            raise ValueError('{} mode does not exist. Choose from "data" and "hist".'.format(mode))
        
        self.sim = sim
        self.ana_para = ana_para
        self.bkg_trials = bkg_trials
        self.mode = mode
        self.output = output
        self.bkg_bins = bkg_bins
        self.verbose = verbose

        self._file = os.path.dirname(os.path.abspath(__file__))

        np.random.seed(output)

    def generate_data(self, filename = None, load_bounds = None):
        """Simulates TS distribution for N=self.bkg_trials trials in batches of 10000. Finally, calls a method to reshape and save the data.

        Args:
            filename (str, optional): Name of simulation output file. Defaults to None.
            load_bounds (bool, optional): Use bounds if file exists. Defaults to None
        """
        print("DATA GENERATION -- SAMPLES {}".format(self.bkg_trials))

        if load_bounds:
            # load TS bounds data        
            bkg_bounds = np.load(self._file + "/files/background/hist/BOUNDS_model_{}_{:.0f}_mode_{}_samples_1e+08.npz".format(self.ana_para["model"]["name"], self.ana_para["model"]["param"]["progenitor_mass"].value, self.ana_para["mode"], self.bkg_trials))
            # interpolate TS bounds
            bkg_bounds_inter = interpolate_bounds(bkg_bounds, self.ana_para["distance"].value)

        # filename for simulation output
        filename = self._file + "/files/background/hist/HIST_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.npz".format(self.ana_para["model"]["name"], self.ana_para["model"]["param"]["progenitor_mass"].value, self.ana_para["mode"], self.bkg_trials, self.bkg_bins, self.ana_para["distance"].value)

        if self.output is not None:
            filename = os.path.splitext(filename)[0] + "_output_{}.npz".format(self.output)

        # number of maximum trials, number of repetitions needed to fill bkg_trials
        self.max_trials = 10000 # size of batches
        self.repetitions = np.round(self.bkg_trials/self.max_trials).astype(int) # number of repetitions with size batches

        self.ts_binned = {"ic86" : np.array([np.zeros(self.bkg_bins, dtype=np.float64), np.zeros(self.bkg_bins, dtype=np.float64)]), # histogram data (x,y) for each subdetector
                     "gen2" : np.array([np.zeros(self.bkg_bins, dtype=np.float64), np.zeros(self.bkg_bins, dtype=np.float64)]), 
                     "wls": np.array([np.zeros(self.bkg_bins, dtype=np.float64), np.zeros(self.bkg_bins, dtype=np.float64)])}
        
        bounds = {"ic86": None, "gen2": None, "wls": None}
        
        for r in tqdm(range(self.repetitions)): # loop over batches
            # Initialize null hypothsis class and run analysis
            nlh = Null_Hypothesis(self.sim, res_dt = self.ana_para["res_dt"], distance=self.ana_para["distance"])
            nlh.run(mode = self.ana_para["mode"], ft_para = self.ana_para["ft_para"], bkg_trials = self.max_trials, model = "generic", smoothing = False)

            for det in ["ic86", "gen2", "wls"]: # loop over subdetectors
                if load_bounds and r == 0: 
                    bkg_min, bkg_max = bkg_bounds_inter[det] # interpolated bounds for subdetector
                    bounds[det] = (0.1 * bkg_min, 1.9 * bkg_max)
                elif load_bounds is not True and r == 0: 
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
            
    def load_data(self, filename):
        """Load data.

        Args:
            filename (str): Filename.
        """
        data = np.load(filename, allow_pickle=True)
        data_new = {}
        for key in ["ic86", "gen2", "wls"]:
            data_new[key] = data[key]
        
        self.repetitions = data["reps"]
        self.max_trials = data["trials"]
        self.ts = data_new

    def combine_data(self, num_output, filebase = None, save = False):
        """Combine data or histogram.

        Args:
            num_output (int): Number of output files.
            filebase (str): Filename base. Defaults to None.
            save (bool, optional): Save combined file. Defaults to False.
        """

        if filebase is None:
            if self.mode == "data":
                filebase = self._file + "/files/background/generate/GENERATE_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_distance_{:.1f}kpc".format(self.ana_para["model"]["name"], self.ana_para["model"]["param"]["progenitor_mass"].value, self.ana_para["mode"], self.bkg_trials, self.ana_para["distance"].value)
            elif self.mode == "hist":
                filebase = self._file + "/files/background/hist/HIST_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc".format(self.ana_para["model"]["name"], self.ana_para["model"]["param"]["progenitor_mass"].value, self.ana_para["mode"], self.bkg_trials, self.bkg_bins, self.ana_para["distance"].value)

        if self.mode == "data":
            data_new = {"ic86": [], "gen2": [], "wls": []}

            for o in range(num_output):
                filename = filebase + "_output_{}.npz".format(o)

                data = np.load(filename, allow_pickle=True)
                
                for det in ["ic86", "gen2", "wls"]:
                    data_new[det].append(data[det])
            
            for det in ["ic86", "gen2", "wls"]:
                data_new[det] = np.array(data_new[det]).flatten() 
                
        elif self.mode == "hist":
            data_new = {"ic86": np.zeros(self.bkg_bins), "gen2": np.zeros(self.bkg_bins), "wls": np.zeros(self.bkg_bins)}

            for o in range(num_output):
                filename = filebase + "_output_{}.npz".format(o)

                data = np.load(filename, allow_pickle=True)
                
                for det in ["ic86", "gen2", "wls"]:
                    data_new[det] += data[det][1] # sum y-values of histgram

            for det in ["ic86", "gen2", "wls"]:
                data_new[det] *= 1/(np.sum(data_new[det]) * (data[det][0][1]-data[det][0][0])) # normalize sum of histgram  

        self.repetitions = data["reps"]
        self.max_trials = data["trials"]
        self.num_output = num_output
        self.bkg_trials = self.num_output * self.max_trials * self.repetitions

        if save:
            if self.mode == "data":
                filename = self._file + "/files/background/generate/GENERATE_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_distance_{:.1f}kpc.npz".format(self.ana_para["model"]["name"], self.ana_para["model"]["param"]["progenitor_mass"].value, self.ana_para["mode"], self.bkg_trials, self.ana_para["distance"].value)
            elif self.mode == "hist":
                filename = self._file + "/files/background/hist/HIST_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.npz".format(self.ana_para["model"]["name"], self.ana_para["model"]["param"]["progenitor_mass"].value, self.ana_para["mode"], self.bkg_trials, self.bkg_bins, self.ana_para["distance"].value)

            np.savez(file = filename, 
                    reps = self.repetitions * self.num_output, 
                    trials = self.max_trials, 
                    ic86 = data_new["ic86"],
                    gen2 = data_new["gen2"],
                    wls = data_new["wls"])
        
    def ts_fit(self, distance_range, bkg_distr, verbose = None):
        """Fit background distribution to TS data for all data specified in distance_range. As opposed to the method ts_binned_fit
        this method uses the scipt.stats.fit() method which applies a maximum likelihood fit to the input data.

        Args:
            distance_range (np.ndarray): Distance range
            bkg_distr (str, optional): Name of statistical distribution.
            verbose (str, optional): Verbose. Defaults to None.
        """

        distr = get_distribution_by_name(bkg_distr)

        model = self.ana_para["model"]
        mode = self.ana_para["mode"]
        pdict = {"ic86": [], "gen2": [], "wls": []}
        
        path_to_folder = self._file + "/files/background/" # directory to save fit parameters and quantiles

        for dist in distance_range: # loop over all distances
            print("Distance: {}".format(dist))
            for det in ["ic86", "gen2", "wls"]: # loop over detector

                self.load_data(path_to_folder+"generate/GENERATE_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_distance_{:.1f}kpc.npz"
                               .format(model["name"], model["param"]["progenitor_mass"].value, mode, self.bkg_trials, dist.value))

                params = distr.fit(self.ts[det])

                if verbose is not None:
                    fig, ax = plt.subplots(1,1)
                    x = np.linspace(self.ts[det].min(), self.ts[det].max(), 1000)
                    ax.hist(self.ts[det], bins = int(np.sqrt(self.bkg_trials)), density=True, histtype = "step", color = "C0")
                    ax.plot(x, distr(*params).pdf(x), color = "k", ls = "--")
                    ax.set_xlabel("TS value")
                    ax.set_ylabel("Normalized Counts")
                    ax.set_yscale("log")
                    plt.tight_layout()
                    if verbose: plt.show()
                    plt.savefig("./plots/background/TS_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_distance_{:.1f}kpc_det_{}_norm.pdf"
                        .format(model["name"], model["param"]["progenitor_mass"].value, mode, self.bkg_trials, dist.value, det))

                pdict[det].append(params)

        for det in ["ic86", "gen2", "wls"]: # transform list into np.array
            pdict[det] = np.array(pdict[det])
        
        # save npz files
        pfilename = path_to_folder+"fit/FIT_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_{}.npz".format(model["name"], model["param"]["progenitor_mass"].value, mode, self.bkg_trials, str(distr.name))
        np.savez(file = pfilename, 
                 dist = distance_range.value, 
                 ic86 = pdict["ic86"],
                 gen2 = pdict["gen2"],
                 wls = pdict["wls"])
        
        return
        
    def ts_quant(self, distance_range):

        model = self.ana_para["model"]
        mode = self.ana_para["mode"]
        qdict = {"ic86": [], "gen2": [], "wls": []}
        
        path_to_folder = self._file + "/files/background/" # directory to save fit parameters and quantiles

        for dist in distance_range: # loop over all distances
            print("Distance: {}".format(dist))
            for det in ["ic86", "gen2", "wls"]: # loop over detector            

                self.load_data(path_to_folder+"generate/GENERATE_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_distance_{:.1f}kpc.npz"
                               .format(model["name"], model["param"]["progenitor_mass"].value, mode, self.bkg_trials, dist.value))

                quants = np.array([np.median(self.ts[det]), np.quantile(self.ts[det], 0.16), np.quantile(self.ts[det], 0.84)])

                qdict[det].append(quants)

        for det in ["ic86", "gen2", "wls"]: # transform list into np.array
            qdict[det] = np.array(qdict[det])

        # save npz files
        qfilename = path_to_folder+"quantile/QUANTILE_model_{}_{:.0f}_mode_{}_samples_{:1.0e}.npz".format(model["name"], model["param"]["progenitor_mass"].value, mode, self.bkg_trials)
        np.savez(file = qfilename, 
                 dist = distance_range.value, 
                 ic86 = qdict["ic86"],
                 gen2 = qdict["gen2"],
                 wls = qdict["wls"])
    
    def ts_binned(self, bkg_bins = 1000):
        model = self.ana_para["model"]
        mode = self.ana_para["mode"]

        self.ts_binned = {"ic86": None, "gen2": None, "wls": None} # empty dictionary

        for det in ["ic86", "gen2", "wls"]: # loop over detectors
            # histogram TS distribution
            hist_y, hist_bins = np.histogram(self.ts[det] , bins = bkg_bins, range=(self.ts[det].min() , self.ts[det].max()), density=True)
            hist_x = (hist_bins[1:]+hist_bins[:-1])/2
            self.ts_binned[det] = np.array([hist_x, hist_y])

        # save histogrammed TS distribution
        path_to_folder = self._file + "/files/background/" # directory to save fit parameters and quantiles
        filename = path_to_folder+"hist/HIST_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.npz".format(model["name"], model["param"]["progenitor_mass"].value, mode, self.bkg_trials, bkg_bins, self.ana_para["distance"].value)

        np.savez(file = filename, 
                 ic86 = self.ts_binned["ic86"],
                 gen2 = self.ts_binned["gen2"],
                 wls = self.ts_binned["wls"])
        
        return
    
    def ts_binned_fit(self, distance_range, bkg_distr, bkg_trials, bkg_bins, log_scale = None, verbose = None):
        model = self.ana_para["model"]
        mode = self.ana_para["mode"]
        path_to_folder = self._file + "/files/background/" # directory to save fit parameters and quantiles
        
        distr = get_distribution_by_name(bkg_distr)

        pdict = {"ic86": [], "gen2": [], "wls": []}

        for dist in distance_range: # loop over all distances
            print("Distance: {}".format(dist))
            data = np.load(path_to_folder+"hist/HIST_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.npz"
                           .format(model["name"], model["param"]["progenitor_mass"].value, mode, bkg_trials, bkg_bins, dist.value))
            
            for det in ["ic86", "gen2", "wls"]: # loop over detectors
                fit = fit_distribution(distr, data[det], log_scale)

                pdict[det].append(fit.x)

                if verbose is not None and ((det == "ic86" and dist.value == 1) or (det == "ic86" and dist.value%10 == 0)):
                    fig, ax = plt.subplots(1,1)
                    x = np.linspace(data[det].min(), data[det].max(), 1000)
                    ax.step(*data[det], color = "C0")
                    ax.plot(x, distr(*fit.x).pdf(x), color = "k", ls = "--")
                    ax.set_xlabel("TS value")
                    ax.set_ylabel("Normalized Counts")
                    ax.set_yscale("log")
                    ax.set_ylim(1E-10, 1E-2)
                    plt.tight_layout()
                    if verbose: plt.show()
                           
        for det in ["ic86", "gen2", "wls"]: # transform list into np.array
            pdict[det] = np.array(pdict[det])

        # save npz files
        pfilename = path_to_folder+"fit/FIT_HIST_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_{}.npz".format(model["name"], model["param"]["progenitor_mass"].value, mode, self.bkg_trials, str(distr.name))
        np.savez(file = pfilename, 
                 dist = distance_range.value, 
                 ic86 = pdict["ic86"],
                 gen2 = pdict["gen2"],
                 wls = pdict["wls"])

    def ts_binned_quant(self, distance_range, bkg_trials, bkg_bins):
        model = self.ana_para["model"]
        mode = self.ana_para["mode"]
        path_to_folder = self._file + "/files/background/" # directory to save fit parameters and quantiles
        
        qdict = {"ic86": [], "gen2": [], "wls": []}

        for dist in distance_range: # loop over all distances
            print("Distance: {}".format(dist))
            data = np.load(path_to_folder+"hist/HIST_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.npz"
                           .format(model["name"], model["param"]["progenitor_mass"].value, mode, bkg_trials, bkg_bins, dist.value))
            
            for det in ["ic86", "gen2", "wls"]: # loop over detectors
                perc = [0.5, 0.16, 0.84]
                quant = quantiles_histogram(data[det], perc)
                
                qdict[det].append(quant)
        
        for det in ["ic86", "gen2", "wls"]: # loop over detectors
            qdict[det] = np.array(qdict[det])

        # save npz files
        qfilename = path_to_folder+"quantile/QUANTILE_model_{}_{:.0f}_mode_{}_samples_{:1.0e}.npz".format(model["name"], model["param"]["progenitor_mass"].value, mode, bkg_trials)
        np.savez(file = qfilename, 
                 dist = distance_range.value, 
                 ic86 = qdict["ic86"],
                 gen2 = qdict["gen2"],
                 wls = qdict["wls"]) 

    def ts_bounds(self, distance_range):
        model = self.ana_para["model"]
        mode = self.ana_para["mode"]
        path_to_folder = self._file + "/files/background/" # directory to save fit parameters and quantiles
        
        mdict = {"ic86": [], "gen2": [], "wls": []}

        for dist in distance_range: # loop over all distances
            print("Distance: {}".format(dist))
            data = self.load_data(path_to_folder+"generate/GENERATE_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_distance_{:.1f}kpc.npz"
                               .format(model["name"], model["param"]["progenitor_mass"].value, mode, self.bkg_trials, dist.value))
            
            for det in ["ic86", "gen2", "wls"]: # loop over detectors
                mdict[det](data[det].min(), data[det].max())

                           
        for det in ["ic86", "gen2", "wls"]: # transform list into np.array
            mdict[det] = np.array(mdict[det]).reshape(len(distance_range, 2))

        # save npz files
        pfilename = path_to_folder+"hist/BOUNDS_model_{}_{:.0f}_mode_{}_samples_{:1.0e}.npz".format(model["name"], model["param"]["progenitor_mass"].value, mode, self.bkg_trials)
        np.savez(file = pfilename, 
                 dist = distance_range.value, 
                 ic86 = mdict["ic86"],
                 gen2 = mdict["gen2"],
                 wls = mdict["wls"])

    def ts_to_pvalue(self, bkg_bins = 1000):

        self.pvalue = {"ic86": None, "gen2": None, "wls": None}

        for det in ["ic86", "gen2", "wls"]: # loop over detectors
            pv = []
            # compute p-value for range of TS values
            ts_range = np.linspace(self.ts[det].min(), self.ts[det].max(), num = bkg_bins, endpoint=True)
            for tss in tqdm(ts_range):
                pv.append(np.sum(self.ts[det] > tss)/len(self.ts[det]))

            self.pvalue[det] = np.array(pv) 
    
        return 