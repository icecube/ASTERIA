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
                 output,
                 verbose = None):
        
        self.sim = sim
        self.ana_para = ana_para
        self.bkg_trials = bkg_trials
        self.output = output
        self.verbose = verbose

        self._file = os.path.dirname(os.path.abspath(__file__))

        np.random.seed(output)

    def generate_data(self, bkg_bins, filename = None):
        print("DATA GENERATION -- SAMPLES {}".format(self.bkg_trials))

        if not os.path.isdir("./files"):
            os.mkdir("./files")
        if filename is None:
            path_to_folder = self._file + "/files/background/"
            #filename = path_to_folder+"generate/GENERATE_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_distance_{:.1f}kpc_output_{}.npz".format(self.ana_para["model"]["name"], self.ana_para["model"]["param"]["progenitor_mass"].value, self.ana_para["mode"], self.bkg_trials, self.ana_para["distance"].value, self.output)
            filename = path_to_folder+"hist/HIST_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.npz".format(self.ana_para["model"]["name"], self.ana_para["model"]["param"]["progenitor_mass"].value, self.ana_para["mode"], self.bkg_trials, bkg_bins, self.ana_para["distance"].value)

        self.max_trials = 10000 # size of batches
        self.repetitions = np.round(self.bkg_trials/self.max_trials).astype(int)

        ts = []
        for r in tqdm(range(self.repetitions)):
            # Initialize analysis class and run analysis
            ana = Null_Hypothesis(self.sim, res_dt = self.ana_para["res_dt"], distance=self.ana_para["distance"], trials = self.max_trials)
            ana.run(mode = self.ana_para["mode"], ft_para = self.ana_para["ft_para"], model = "generic", smoothing = False)
            ts.append(ana.ts)

        self.ts = ts
        #self.reshape_and_save(self.ts, filename)
        self.reshape_and_bin(self.ts, bkg_bins, filename)


    def reshape_and_save(self, item, filename):
        data = {"ic86" : [], "gen2" : [], "wls": []}

        #quantiles = [0.5, 0.16, 0.84]
        for det in ["ic86", "gen2", "wls"]:
            dd = []
            #for q in np.arange(len(quantiles)):
            for r in range(self.repetitions):
                d = item[r][det]
                dd.append(d)
            data[det] = np.array(dd, dtype=float).reshape(self.repetitions*self.max_trials)

        np.savez(file = filename, 
                    reps = self.repetitions, 
                    trials = self.max_trials, 
                    ic86 = data["ic86"],
                    gen2 = data["gen2"],
                    wls = data["wls"])
        return data
    
    def reshape_and_bin(self, item, bkg_bins, filename):

        # load TS bounds data        
        bkg_bounds = np.load(self._file + "/files/background/hist/BOUNDS_model_{}_{:.0f}_mode_{}_samples_1e+08.npz".format(self.ana_para["model"]["name"], self.ana_para["model"]["param"]["progenitor_mass"].value, self.ana_para["mode"], self.bkg_trials))
        # interpolate TS bounds
        bkg_inter_bounds = interpolate_bounds(bkg_bounds, self.ana_para["distance"].value)

        # 1) reshape data
        data = {"ic86" : [], "gen2" : [], "wls": []}
        for det in ["ic86", "gen2", "wls"]:
            dd = []
            #for q in np.arange(len(quantiles)):
            for r in range(self.repetitions):
                d = item[r][det]
                dd.append(d)
            data[det] = np.array(dd, dtype=float).reshape(self.repetitions*self.max_trials)

        # 2) bin data in histograms to reduce storage, density = False, normalization is done after datasets are combined
        self.ts_binned = {"ic86": None, "gen2": None, "wls": None} # empty dictionary
        for det in ["ic86", "gen2", "wls"]: # loop over detectors
            # histogram TS distribution
            bkg_min, bkg_max = bkg_inter_bounds[det] # load interpolated bounds
            bkg_min = 0.5 * bkg_min # safety margin of 50 %
            bkg_max = 1.5 * bkg_max
            hist_y, hist_bins = np.histogram(data[det] , bins = bkg_bins, range = (bkg_min , bkg_max), density=True)
            hist_x = (hist_bins[1:]+hist_bins[:-1])/2
            self.ts_binned[det] = np.array([hist_x, hist_y])

        np.savez(file = filename, 
                 ic86 = self.ts_binned["ic86"],
                 gen2 = self.ts_binned["gen2"],
                 wls = self.ts_binned["wls"])
        
        return
    
    def load_data(self, filename):
        data = np.load(filename, allow_pickle=True)
        data_new = {}
        for key in ["ic86", "gen2", "wls"]:
            data_new[key] = data[key]
        
        self.repetitions = data["reps"]
        self.max_trials = data["trials"]
        self.ts = data_new

    def combine_data(self, num_output, filebase = None, save = False):

        if filebase is None:
            filebase = "./files/background/generate/GENERATE_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_distance_{:.1f}kpc".format(self.ana_para["model"]["name"], self.ana_para["model"]["param"]["progenitor_mass"].value, self.ana_para["mode"], self.bkg_trials, self.ana_para["distance"].value)

        data_new = {"ic86": [], "gen2": [], "wls": []}

        for i in range(num_output):
            filename = filebase + "_output_{}.npz".format(i)

            data = np.load(filename, allow_pickle=True)

            for det in ["ic86", "gen2", "wls"]:
                data_new[det].append(data[det])
        
        for det in ["ic86", "gen2", "wls"]:
            data_new[det] = np.array(data_new[det]).flatten() 

        self.repetitions = data["reps"]
        self.max_trials = data["trials"]
        self.num_output = num_output
        self.bkg_trials = self.num_output * self.max_trials * self.repetitions
        self.ts = data_new

        if save:
            filename = "./files/background/generate/GENERATE_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_distance_{:.1f}kpc.npz".format(self.ana_para["model"]["name"], self.ana_para["model"]["param"]["progenitor_mass"].value, self.ana_para["mode"], self.bkg_trials, self.ana_para["distance"].value)

            np.savez(file = filename, 
                    reps = self.repetitions * self.num_output, 
                    trials = self.max_trials, 
                    ic86 = data_new["ic86"],
                    gen2 = data_new["gen2"],
                    wls = data_new["wls"])
        return data

        
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
    
    def ts_min_max(self, distance_range):
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
