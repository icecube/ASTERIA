import numpy as np
from analysis import *
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from scipy.stats import lognorm, skewnorm

class Bootstrapping():

    def __init__(self,
                 sim,
                 ana_para,
                 verbose = None):
        
        self.sim = sim
        self.ana_para = ana_para
        self.verbose = verbose

    def generate_data(self, samples, filename = None):
        print("DATA GENERATION -- SAMPLES {}".format(samples))
        if filename is None:
            filename = "./files/bootstrapping/bootstrapping_generate_{:1.0e}_distance_{:.0f}kpc.npz".format(int(samples), self.ana_para["distance"].value)

        self.max_trials = 10000 # size of batches
        self.repetitions = np.round(samples/self.max_trials).astype(int)

        ts = []
        for r in tqdm(range(self.repetitions)):
            # Initialize analysis class and run analysis
            ana = Analysis(self.sim, res_dt = self.ana_para["res_dt"], distance=self.ana_para["distance"], trials = self.max_trials, temp_para=self.ana_para["temp_para"])
            ana.run(mode = self.ana_para["mode"], ft_para = self.ana_para["ft_para"], model = "generic")
            ts.append(ana.ts)

        self.ts = ts
        self.reshape_and_save(self.ts, filename)


    def reshape_and_save(self, item, filename):
        data = {"ic86" : {"null": [], "signal": []}, "gen2" : {"null": [], "signal": []}, "wls": {"null": [], "signal": []}}

        #quantiles = [0.5, 0.16, 0.84]
        hypothesis = ["null", "signal"]
        for det in ["ic86", "gen2", "wls"]:
            for hypo in hypothesis:
                dd = []
                #for q in np.arange(len(quantiles)):
                for r in range(self.repetitions):
                    d = item[r][hypo][det]
                    dd.append(d)
                data[det][hypo] = np.array(dd, dtype=float).reshape(self.repetitions*self.max_trials)

        np.savez(file = filename, 
                    reps = self.repetitions, 
                    trials = self.max_trials, 
                    hypo = hypothesis,
                    ic86 = data["ic86"],
                    gen2 = data["gen2"],
                    wls = data["wls"])
        return data


    def load_data(self, filename):
        ts = np.load(filename, allow_pickle=True)
        tts = {}
        for key in ["ic86", "gen2", "wls"]:
            tts[key] = ts[key].item()
        
        self.repetitions = ts["reps"]
        self.max_trials = ts["trials"]
        self.ts = tts


    def run_same(self, trials = 10000, repetitions = 100, distribution = None):
        """_summary_

        Args:
            trials (int, optional): _description_. Defaults to 10000.
            repetitions (int, optional): _description_. Defaults to 100.
            mode (str, optional): _description_. Defaults to "generate".
            distribution (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        
        samples = trials * repetitions
        self.zscore = {"ic86": [], "gen2": [], "wls": []}

        if distribution is None or distribution == "lognorm":
            distr = lognorm
        elif distribution == "skewnorm":
            distr = skewnorm
        else:
            raise ValueError('{} not supported. Choose from "lognorm" and "skewnorm"'.format(distribution)) 

        #loop over detector
        for det in ["ic86", "gen2", "wls"]:
            if self.verbose: print("Detector: {}".format(det))

            # sample trials of TS values from signal and null hypothesis, values can be re-picked
            ts_nul = np.random.choice(self.ts[det]["null"], size = samples, replace = True).reshape(repetitions, trials)
            ts_sig = np.random.choice(self.ts[det]["signal"], size = samples, replace = True).reshape(repetitions, trials)
            data = [ts_nul, ts_sig]
            
            for r in tqdm(range(repetitions)):
                # calculate median, 16% and 84% quantiles of sampled signal TS distribution
                ts_sig_stat = np.array([np.median(ts_sig[r]), 
                                        np.quantile(ts_sig[r], 0.16), 
                                        np.quantile(ts_sig[r], 0.84)])

                # fit of null hypothesis
                ts_nul_fit = distr(*distr.fit(ts_nul[r]))
                
                # get p value and Z score
                for i in range(3):
                    p = ts_nul_fit.sf(ts_sig_stat[i])
                    zz = norm.isf(p/2)
                    self.zscore[det].append(zz)

            self.zscore[det] = np.array(self.zscore[det]).reshape(repetitions, 3)

    def run_full(self, trials = 10000, repetitions = 100, distribution = None):
        
        if distribution is None or distribution == "lognorm":
            distr = lognorm
        elif distribution == "skewnorm":
            distr = skewnorm
        else:
            raise ValueError('{} not supported. Choose from "lognorm" and "skewnorm"'.format(distribution)) 

        samples = trials * repetitions
        self.zscore = {"ic86": [], "gen2": [], "wls": []}

        #loop over detector
        for det in ["ic86", "gen2", "wls"]:
            if self.verbose: print("Detector: {}".format(det))

            # bkg fit on full 1E6 bkg TS trials
            ts_nul_fit = distr(*distr.fit(self.ts[det]["null"]))

            ts_sig = np.random.choice(self.ts[det]["signal"], size = samples, replace = True).reshape(repetitions, trials)
             
            for r in tqdm(range(repetitions)):

                ts_sig_stat = np.array([np.median(ts_sig[r]), 
                                        np.quantile(ts_sig[r], 0.16), 
                                        np.quantile(ts_sig[r], 0.84)])
                    
                # get p value and Z score
                for i in range(3):
                    p = ts_nul_fit.sf(ts_sig_stat[i])
                    zz = norm.isf(p/2)
                    self.zscore[det].append(zz)

            self.zscore[det] = np.array(self.zscore[det]).reshape(repetitions, 3)
