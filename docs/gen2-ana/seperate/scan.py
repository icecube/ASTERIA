from asteria.simulation import Simulation
from analysis import *
from signal_hypothesis import *
from scipy.optimize import minimize, brentq
from helper import *
from tqdm import tqdm

from plthelper import plot_significance

def loss_dist_range_interpolate(dist, ana_para, sigma, det):
    sim, res_dt, trials, temp_para, mode, ft_para, bkg_distr = ana_para
    ana = Analysis(sim, res_dt = res_dt, distance = dist*u.kpc, trials = trials, temp_para = temp_para)
    ana.run(mode = mode, ft_para = ft_para, distribution = bkg_distr, model = "generic")
    if det == "ic86":
        ind = 1
    elif det == "wls":
        ind = 2
    loss = (ana.zscore[det][ind] - sigma)
    return loss

def loss_dist_range_fit(dist, ana_para, sigma, det):
    sim, res_dt, trials, temp_para, mode, ft_para, bkg_distr = ana_para
    ana = Analysis(sim, res_dt = res_dt, distance = dist*u.kpc, trials = trials, temp_para = temp_para)
    ana.run(mode = mode, ft_para = ft_para, distribution = bkg_distr, model = "generic")
    loss = (ana.zscore[det][0] - sigma)
    return loss

class Scan():
    
    def __init__(self,
                 sim,
                 scan_para,
                 ft_mode, 
                 ft_para,
                 sig_trials,
                 bkg_distr,
                 bkg_trials,
                 bkg_bins,
                 fit_hist,
                 verbose = None):
        
        self.sim = sim
        self.scan_para = scan_para
        self.ft_mode = ft_mode
        self.ft_para = ft_para
        self.sig_trials = sig_trials
        self.bkg_distr = bkg_distr
        self.bkg_trials = bkg_trials
        self.bkg_bins = bkg_bins
        self.fit_hist = fit_hist
        self.verbose = verbose

    def bias(self, bias_freq, bias_ampl, bias_reps = 100):

        self.bias_freq = bias_freq
        self.bias_ampl = bias_ampl
        self.bias_reps = bias_reps

        self.sigma = self.scan_para["sigma"]

        self.dist = [{} for r in range(self.bias_reps)]
        self.perc = [{} for r in range(self.bias_reps)]

        # template dictionary uses fixed ampl, freq and scan_para values
        temp_para = {"frequency": self.bias_freq, 
                     "amplitude": self.bias_ampl, #in percent of max value
                     "time_start": self.scan_para["time_start"],
                     "time_end": self.scan_para["time_end"],
                     "position": self.scan_para["position"]}
        
        if self.verbose is not None: print("Frequency: {}, Amplitude: {} % ".format(bias_freq, bias_ampl*100))

        for r in tqdm(np.arange(self.bias_reps)): # loop over repetition

            # 1) Find the distance bounds for which the significance falls from > 5 sigma to < 3 sigma for all subdetectors
            # This is done to avoid scanning unnecessarily many distances for which the significance is either far too high
            # or far too low.
            trials = 1000 # use 1/10 of trials used in later distance scan to increase speed of convergence
            ana_para = [self.sim, self.sim._res_dt, trials, temp_para, self.ft_mode, self.ft_para, self.bkg_distr]

            # arguments for the minimizer
            # The minimizer is defined above and initializes the Analysis class.
            args_dist_low = (ana_para, 5.5, "ic86")
            args_dist_high = (ana_para, 2.5, "wls")

            # returns lower bound distance
            root_low = brentq(loss_dist_range_interpolate, a = 1, b = 100, args = args_dist_low, xtol = 1e-2)
            dist_low = root_low * u.kpc
            # returns higher bound distance
            root_high = brentq(loss_dist_range_interpolate, a = 1, b = 100, args = args_dist_high, xtol = 1e-2)
            dist_high = root_high * u.kpc
            if self.verbose == "debug": print("Distance range: {:.1f} - {:.1f}".format(dist_low, dist_high))

            # 2) Distance scan

            # distance search range
            #dist_range = np.arange(np.floor(dist_low.value), np.ceil(dist_high.value)+1, 1) * u.kpc

            d_low, d_high = np.floor(dist_low.value), np.ceil(dist_high.value)

            # If difference between d_low and d_high is small (< 10 kpc) 5 steps should be enough, else 10 steps are used
            if d_high-d_low < 10:
                num_steps = int(d_high-d_low)
            else:
                num_steps = int(d_high-d_low) #10

            self.dist_range = np.linspace(d_low, d_high, num_steps, dtype = int) * u.kpc

            # initialize signal hypothesis instance for distance scan
            sgh = Signal_Hypothesis(self.sim, res_dt = self.sim._res_dt, 
                                    distance = dist_low, temp_para = temp_para)
            
            # returns z-score and ts statistics for all distances and all subdetectors
            Zscore, Ts_stat = sgh.dist_scan(self.dist_range, mode = self.ft_mode, ft_para = self.ft_para, 
                                            sig_trials = self.sig_trials, bkg_distr = self.bkg_distr, 
                                            bkg_trials = self.bkg_trials, bkg_bins = self.bkg_bins, fit_hist = self.fit_hist,
                                            model = "generic", verbose = self.verbose)              
            
            self.Zscore = Zscore
            self.Ts_stat = Ts_stat

            if self.verbose == "debug":
                import matplotlib.pyplot as plt
                plot_significance(self.dist_range, self.Zscore, self.Ts_stat)
                plt.show()

            # 3) Calculate the 3 (5) sigma significance via interpolation of the distance scan data
            self.quantiles = [0.5, 0.16, 0.84]
            dist, perc = significance_horizon(self.dist_range, self.Zscore, self.sigma)

            # save data
            self.dist[r] = dist
            self.perc[r] = perc

        return


    def run_interpolate(self):
        """The parameter scan is designed in five steps:
        1) A loop over all amplitudes.
        2) A loop over all frequencies.
        For each amplitude, frequency value the following operations are done
        3) Find the distance bounds for which the significance falls from > 5 sigma to < 3 sigma for all subdetectors
        4) The actual distance scan using 10 steps within the bounds established in 3)
        5) The so obtained significances are interpolated to compute the 3 and 5 sigma detection horizon.
        """
        
        self.ampl_range = self.scan_para["ampl_range"]
        self.freq_range = self.scan_para["freq_range"]
        self.sigma = self.scan_para["sigma"]

        # create empty 2D list with each entry being a dictionary
        self.dist = [[{} for f in range(len(self.freq_range))] for a in range(len(self.ampl_range))]
        self.perc = [[{} for f in range(len(self.freq_range))] for a in range(len(self.ampl_range))]

        for a, ampl in enumerate(self.ampl_range): # loop over scan amplitude
            for f, freq in enumerate(self.freq_range): # loop over scan frequency
                    
                if self.verbose is not None: print("Frequency: {}, Amplitude: {} % ".format(freq, ampl*100))

                # template dictionary uses loop ampl, freq and scan_para values
                temp_para = {"frequency": freq, 
                            "amplitude": ampl, #in percent of max value
                            "time_start": self.scan_para["time_start"],
                            "time_end": self.scan_para["time_end"],
                            "position": self.scan_para["position"]}
                

                # 1) Find the distance bounds for which the significance falls from > 5 sigma to < 3 sigma for all subdetectors
                # This is done to avoid scanning unnecessarily many distances for which the significance is either far too high
                # or far too low.
                trials = 1000 # use 1/10 of trials used in later distance scan to increase speed of convergence
                ana_para = [self.sim, self.sim._res_dt, trials, temp_para, self.ft_mode, self.ft_para, self.bkg_distr]

                # arguments for the minimizer
                # The minimizer is defined above and initializes the Analysis class.
                # To avoid extrapolation issues with our interpolator, 
                # we search for significances slightly above (below) 5 (3) sigma.
                # It turns out that for limits [2.5, 5.5] with a limited statistics of 1,000 trials there are cases in which the 
                # e.g. the IC86 significance for 10,000 trials is slightly below 5 sigma. Therefore we increase the range to [2,6].
                args_dist_low = (ana_para, 5, "ic86")
                args_dist_high = (ana_para, 3, "wls")

                if 0:#self.verbose == "debug":
                    import matplotlib.pyplot as plt
                    dd = np.linspace(5,30,50)
                    ls_low, ls_high = [], []
                    for d in dd:
                        ls_low.append(loss_dist_range_interpolate(d, *args_dist_low))
                        ls_high.append(loss_dist_range_interpolate(d, *args_dist_high))
                    fig, ax = plt.subplots(1,1)
                    ax.plot(dd, ls_low)
                    ax.plot(dd, ls_high)
                    ax.set_xlabel("Distance [kpc]")
                    ax.set_ylabel("Loss")
                    plt.show()    

                # returns lower bound distance
                root_low = brentq(loss_dist_range_interpolate, a = 1, b = 100, args = args_dist_low, xtol = 1e-2)
                dist_low = root_low * u.kpc
                # returns higher bound distance
                root_high = brentq(loss_dist_range_interpolate, a = 1, b = 100, args = args_dist_high, xtol = 1e-2)
                dist_high = root_high * u.kpc
                if self.verbose == "debug": print("Distance range: {:.1f} - {:.1f}".format(dist_low, dist_high))

                # 2) Distance scan

                # distance search range
                #dist_range = np.arange(np.floor(dist_low.value), np.ceil(dist_high.value)+1, 1) * u.kpc

                spkpc = 5 # steps per kpc

                d_low = np.floor(dist_low.value * spkpc) / spkpc
                d_high = np.ceil(dist_high.value * spkpc) / spkpc

                #self.dist_range = np.linspace(d_low, d_high, num_steps, dtype = int) * u.kpc
                self.dist_range = np.arange(d_low, d_high + 1/spkpc, spkpc) * u.kpc

                # initialize signal hypothesis instance for distance scan
                sgh = Signal_Hypothesis(self.sim, res_dt = self.sim._res_dt, 
                                        distance = dist_low, temp_para = temp_para)
            
                # returns z-score and ts statistics for all distances and all subdetectors
                Zscore, Ts_stat = sgh.dist_scan(self.dist_range, mode = self.ft_mode, ft_para = self.ft_para, 
                                                sig_trials = self.sig_trials,
                                                bkg_distr = self.bkg_distr, bkg_trials= self.bkg_trials, 
                                                model = "generic", verbose = self.verbose)              

                self.Zscore = Zscore
                self.Ts_stat = Ts_stat

                if self.verbose == "debug":
                    import matplotlib.pyplot as plt
                    plot_significance(self.dist_range, self.Zscore, self.Ts_stat)
                    plt.show()

                # 3) Calculate the 3 (5) sigma significance via interpolation of the distance scan data
                self.quantiles = [0.5, 0.16, 0.84]
                dist, perc = significance_horizon(self.dist_range, self.Zscore, self.sigma)

                # save data
                self.dist[a][f] = dist
                self.perc[a][f] = perc

                if self.verbose == "debug":
                    print("3sig distance horizon IC86: {:.1f}".format(dist["ic86"][0][0]))
                    print("3sig distance horizon Gen2: {:.1f}".format(dist["gen2"][0][0]))
                    print("3sig distance horizon Gen2+WLS: {:.1f}".format(dist["wls"][0][0]))

                    print("5sig distance horizon IC86: {:.1f}".format(dist["ic86"][1][0]))
                    print("5sig distance horizon Gen2: {:.1f}".format(dist["gen2"][1][0]))
                    print("5sig distance horizon Gen2+WLS: {:.1f}".format(dist["wls"][1][0]))

    def run_fit(self):

        self.ampl_range = self.scan_para["ampl_range"]
        self.freq_range = self.scan_para["freq_range"]
        self.sigma = self.scan_para["sigma"]

        # create empty 2D list with each entry being a dictionary
        self.dist = [[{} for f in range(len(self.freq_range))] for a in range(len(self.ampl_range))]
        self.perc = [[{} for f in range(len(self.freq_range))] for a in range(len(self.ampl_range))]

        for a, ampl in enumerate(self.ampl_range): # loop over scan amplitude
            for f, freq in enumerate(self.freq_range): # loop over scan frequency

                if self.verbose is not None: print("Frequency: {}, Amplitude: {} % ".format(freq, ampl*100))

                # template dictionary uses loop ampl, freq and scan_para values
                temp_para = {"frequency": freq, 
                            "amplitude": ampl, #in percent of max value
                            "time_start": self.scan_para["time_start"],
                            "time_end": self.scan_para["time_end"],
                            "position": self.scan_para["position"]}
                
                trials = 10000
                ana_para = [self.sim, self.sim._res_dt, trials, temp_para, self.ft_mode, self.ft_para]

                confidence_level = [3,5]
                sigma = [str(cl) + "sig" for cl in confidence_level] 

                dist = {key : {"ic86": None, "gen2": None, "wls": None} for key in sigma} # empty dictionary

                for i, sig in enumerate(sigma):
                    for det in ["ic86", "gen2", "wls"]:
                        print(sig, det)
                        args = (ana_para, confidence_level[i], det)
                        root = brentq(loss_dist_range_fit, a = 1, b = 100, args = args, xtol = 1e-2)
                        dist[sig][det] = root * u.kpc

                self.dist[a][f] = dist

                if self.verbose == "debug":
                    print("3sig distance horizon IC86: {:.1f}".format(dist["3sig"]["ic86"]))
                    print("3sig distance horizon Gen2: {:.1f}".format(dist["3sig"]["gen2"]))
                    print("3sig distance horizon Gen2+WLS: {:.1f}".format(dist["3sig"]["wls"]))

                    print("5sig distance horizon IC86: {:.1f}".format(dist["5sig"]["ic86"]))
                    print("5sig distance horizon Gen2: {:.1f}".format(dist["5sig"]["gen2"]))
                    print("5sig distance horizon Gen2+WLS: {:.1f}".format(dist["5sig"]["wls"]))

    def reshape_data(self, item, filename):

        data = {"ic86": [], "gen2": [], "wls": []}

        for det in ["ic86", "gen2", "wls"]:
            dd = []
            for s in np.arange(len(self.sigma)):
                for a in np.arange(len(self.ampl_range)):
                    for f in np.arange(len(self.freq_range)):
                        if item[a][f] != {}:
                            for q in np.arange(len(self.quantiles)):
                                d = item[a][f][det][s][q].value
                                dd.append(d)
                        else:
                            # Handle the case where the key is not found
                            dd.append(0)
            data[det] = np.array(dd, dtype=float).reshape(len(self.sigma),
                                                          len(self.ampl_range), 
                                                          len(self.freq_range),
                                                          len(self.quantiles))
        self.data = data

        np.savez(file = filename, 
                 ampl = self.ampl_range, 
                 freq = self.freq_range, 
                 sig = self.sigma, 
                 quan = self.quantiles,
                 ic86 = self.data["ic86"],
                 gen2 = self.data["gen2"],
                 wls = self.data["wls"])

        return data