import os
import sys
from tqdm import tqdm
os.environ['ASTERIA'] = '/home/jakob/software/ASTERIA/ASTERIA'
from asteria.simulation import Simulation
from null_hypothesis import *
from scipy.stats import lognorm
import matplotlib.pyplot as plt


# Class generating and resorting data for background trials
class Background_Trials():

    def __init__(self,
                 sim,
                 ana_para,
                 samples,
                 output,
                 verbose = None):
        
        self.sim = sim
        self.ana_para = ana_para
        self.samples = samples
        self.output = output
        self.verbose = verbose

        np.random.seed(output)

    def generate_data(self, filename = None):
        print("DATA GENERATION -- SAMPLES {}".format(self.samples))

        if not os.path.isdir("./files"):
            os.mkdir("./files")
        if filename is None:
            path_to_folder = "./files/background/"
            filename = path_to_folder+"/GENERATE_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_distance_{:.0f}kpc_output_{}.npz".format(self.ana_para["model"]["name"], self.ana_para["model"]["param"]["progenitor_mass"].value, self.ana_para["mode"], self.samples, self.ana_para["distance"].value, self.output)

        self.max_trials = 10000 # size of batches
        self.repetitions = np.round(self.samples/self.max_trials).astype(int)

        ts = []
        for r in tqdm(range(self.repetitions)):
            # Initialize analysis class and run analysis
            ana = Null_Hypothesis(self.sim, res_dt = self.ana_para["res_dt"], distance=self.ana_para["distance"], trials = self.max_trials)
            ana.run(mode = self.ana_para["mode"], ft_para = self.ana_para["ft_para"], model = "generic", smoothing = False)
            ts.append(ana.ts)

        self.ts = ts
        self.reshape_and_save(self.ts, filename)


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
    
    def load_data(self, filename):
        data = np.load(filename, allow_pickle=True)
        data_new = {}
        for key in ["ic86", "gen2", "wls"]:
            data_new[key] = data[key]
        
        self.repetitions = data["reps"]
        self.max_trials = data["trials"]
        self.ts = data_new

    def ts_fit_quant(self, distance_range, distribution = None, verbose = None):

        if distribution is None or distribution == "lognorm":
            distr = lognorm
        elif distribution == "skewnorm":
            distr = skewnorm
        else:
            raise ValueError('{} not supported. Choose from "lognorm" and "skewnorm"'.format(distribution)) 


        model = self.ana_para["model"]
        mode = self.ana_para["mode"]
        pdict = {"ic86": None, "gen2": None, "wls": None}
        qdict = {"ic86": None, "gen2": None, "wls": None}
        
        path_to_folder = "./files/background/" # directory to save fit parameters and quantiles

        for det in ["ic86", "gen2", "wls"]: # loop over detector

            parameters = []
            quantiles = []

            for dist in distance_range: # loop over all distances

                print("Distance: {}". format(dist))
                self.load_data(path_to_folder+"GENERATE_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_distance_{:.0f}kpc.npz"
                               .format(model["name"], model["param"]["progenitor_mass"].value, mode, self.samples, dist.value))

                params = distr.fit(self.ts[det])
                quants = np.array([np.median(self.ts[det]), np.quantile(self.ts[det], 0.16), np.quantile(self.ts[det], 0.84)])

                if verbose is not None:
                    fig, ax = plt.subplots(1,1)
                    x = np.linspace(self.ts[det].min(), self.ts[det].max(), 1000)
                    ax.hist(self.ts[det], bins = int(np.sqrt(self.samples)), density=True, histtype = "step", color = "C0")
                    ax.plot(x, distr(*params).pdf(x), color = "k", ls = "--")
                    ax.set_xlabel("TS value")
                    ax.set_ylabel("Normalized Counts")
                    ax.set_yscale("log")
                    plt.tight_layout()
                    if verbose: plt.show()
                    plt.savefig("./plots/background/TS_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_distance_{:.0f}kpc_det_{}_norm.pdf"
                        .format(model["name"], model["param"]["progenitor_mass"].value, mode, self.samples, dist.value, det))

                parameters.append(params)
                quantiles.append(quants)

            parameters = np.array(parameters)
            quantiles = np.array(quantiles)

            # save txt files
            pdata = np.hstack((distance_range.value.reshape(-1,1), parameters))
            qdata = np.hstack((distance_range.value.reshape(-1,1), quantiles))

            pcolname = "Distance [kpc] \t Shape \t Loc \t Scale" # ["Distance [kpc]", "Shape", "Loc", "Scale"]
            qcolname = "Distance [kpc] \t 50% \t 16% \t 84%" #["Distance [kpc]", "50%", "16%", "84%"]

            pfilename = path_to_folder+"FIT_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_det_{}_{}.txt".format(model["name"], model["param"]["progenitor_mass"].value, mode, self.samples, det, str(distr.name))
            qfilename = path_to_folder+"QUANTILE_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_det_{}.txt".format(model["name"], model["param"]["progenitor_mass"].value, mode, self.samples, det)

            np.savetxt(pfilename, pdata, fmt = "%.1f %.5e %.5e %.5e", delimiter='\t', header = pcolname, comments="")
            np.savetxt(qfilename, qdata, fmt = "%.1f %.5e %.5e %.5e", delimiter='\t', header = qcolname, comments="")

            # save data in dictionaries
            pdict[det] = parameters
            qdict[det] = quantiles

        pfilename = path_to_folder+"FIT_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_{}.npz".format(model["name"], model["param"]["progenitor_mass"].value, mode, self.samples, str(distr.name))
        qfilename = path_to_folder+"QUANTILE_model_{}_{:.0f}_mode_{}_samples_{:1.0e}.npz".format(model["name"], model["param"]["progenitor_mass"].value, mode, self.samples)

        # save npz files
        np.savez(file = pfilename, 
                    dist = distance_range.value, 
                    ic86 = pdict["ic86"],
                    gen2 = pdict["gen2"],
                    wls = pdict["wls"])

        np.savez(file = qfilename, 
                    dist = distance_range.value, 
                    ic86 = qdict["ic86"],
                    gen2 = qdict["gen2"],
                    wls = qdict["wls"])
        
    def ts_binned(self, bins = 1000):
        model = self.ana_para["model"]
        mode = self.ana_para["mode"]

        self.ts_binned = {"ic86": None, "gen2": None, "wls": None} # empty dictionary

        for det in ["ic86", "gen2", "wls"]: # loop over detectors
            # histogram TS distribution
            hist_y, hist_bins = np.histogram(self.ts[det] , bins = bins, range=(self.ts[det].min() , self.ts[det].max()), density=True)
            hist_x = (hist_bins[1:]+hist_bins[:-1])/2
            self.ts_binned[det] = np.array([hist_x, hist_y])

        # save histogrammed TS distribution
        path_to_folder = "./files/background/" # directory to save fit parameters and quantiles
        filename = path_to_folder+"MAPPING_model_{}_{:.0f}_mode_{}_samples_{:1.0e}_bins_{}_distance_{:.0f}kpc.npz".format(model["name"], model["param"]["progenitor_mass"].value, mode, self.samples, bins, self.ana_para["distance"].value)

        np.savez(file = filename, 
                 ic86 = self.ts_binned["ic86"],
                 gen2 = self.ts_binned["gen2"],
                 wls = self.ts_binned["wls"])
        
        return

    def ts_to_pvalue(self, bins = 1000):

        self.pvalue = {"ic86": None, "gen2": None, "wls": None}

        for det in ["ic86", "gen2", "wls"]: # loop over detectors
            pv = []
            # compute p-value for range of TS values
            ts_range = np.linspace(self.ts[det].min(), self.ts[det].max(), num = bins, endpoint=True)
            for tss in tqdm(ts_range):
                pv.append(np.sum(self.ts[det] > tss)/len(self.ts[det]))

            self.pvalue[det] = np.array(pv) 
    
        return 
