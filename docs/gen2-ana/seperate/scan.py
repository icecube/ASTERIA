from tqdm import tqdm
from analysis import *
from signal_hypothesis import *
from scipy.optimize import minimize, brentq
from helper import *

from plthelper import plot_significance, plot_resolution

def loss_dist_range_interpolate(dist, ana_para, sigma, det, quant):
    sim, res_dt, trials, temp_para, mode, ft_para, bkg_distr = ana_para
    ana = Analysis(sim, res_dt = res_dt, distance = dist*u.kpc, temp_para = temp_para)
    ana.run(mode = mode, ft_para = ft_para, trials = trials, bkg_distr = bkg_distr, model = "generic")
    loss = (ana.zscore[det][quant] - sigma)
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

        self.ampl_range = self.scan_para["ampl_range"]
        self.freq_range = self.scan_para["freq_range"]
        self.sigma = self.scan_para["sigma"]
        self.quantiles = [0.5, 0.16, 0.84]

    def run_interpolate(self):
        """The parameter scan is designed in five steps:
        1) A loop over all amplitudes.
        2) A loop over all frequencies.
        For each amplitude, frequency value the following operations are done
        3) Find the distance bounds for which the significance falls from > 5 sigma to < 3 sigma for all subdetectors
        4) The actual distance scan using 10 steps within the bounds established in 3)
        5) The so obtained significances are interpolated to compute the 3 and 5 sigma detection horizon.
        """

        # create empty 2D list with each entry being a dictionary
        self.dist = [[{} for f in range((self.freq_range).size)] for a in range((self.ampl_range).size)]
        self.perc = [[{} for f in range((self.freq_range).size)] for a in range((self.ampl_range).size)]
        self.fres = [[{} for f in range((self.freq_range).size)] for a in range((self.ampl_range).size)]
        if self.ft_mode == "STF": self.tres = [[{} for f in range((self.freq_range).size)] for a in range((self.ampl_range).size)]

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
                args_dist_low = (ana_para, 5, "ic86", 1)
                args_dist_high = (ana_para, 3, "wls", 2)

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

                skip_next = False
                try:
                    # returns lower bound distance
                    root_low = brentq(loss_dist_range_interpolate, a = 0.1, b = 100, args = args_dist_low, xtol = 1e-2)
                    dist_low = root_low * u.kpc
                except ValueError:
                    print("IC86 boundaries failed")
                    try:
                        arg = list(args_dist_low) # if IC86 is too low, try Gen2
                        arg[2] = "gen2"
                        args_dist_low = list(arg)
                        root_low = brentq(loss_dist_range_interpolate, a = 0.1, b = 100, args = args_dist_low, xtol = 1e-2)
                        dist_low = root_low * u.kpc
                    except ValueError:
                        print("Gen2 boundaries failed")
                        try:
                            arg = list(args_dist_low) # if Gen2 is too low, try WLS
                            arg[2] = "wls"
                            args_dist_low = list(arg)
                            root_low = brentq(loss_dist_range_interpolate, a = 0.1, b = 100, args = args_dist_low, xtol = 1e-2)
                            dist_low = root_low * u.kpc
                        except ValueError:
                            print("WLS boundaries failed")
                            self.dist[a][f] = {"ic86": [np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])], "gen2": [np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])], "wls": [np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])]} 
                            self.perc[a][f] = {"ic86": [np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])], "gen2": [np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])], "wls": [np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])]} 
                            self.fres[a][f] = {"ic86": [np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])], "gen2": [np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])], "wls": [np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])]}                             
                            skip_next = True

                if skip_next:
                    continue

                # returns higher bound distance
                root_high = brentq(loss_dist_range_interpolate, a = 0.1, b = 100, args = args_dist_high, xtol = 1e-2)
                dist_high = root_high * u.kpc
                if self.verbose == "debug": print("Distance estimate: {:.3f} - {:.3f}".format(dist_low, dist_high))

                # 2) Distance scan

                # distance search range
                #dist_range = np.arange(np.floor(dist_low.value), np.ceil(dist_high.value)+1, 1) * u.kpc

                spkpc = 5 # steps per kpc
                mar_low, mar_high = 5, 2 # extra margin to lower/higher side in kpc
                d_low = np.max((np.floor(dist_low.value * spkpc) / spkpc - mar_low, 0.2)) # max between dist_low and 0.2 kpc (lowest sim bkg)
                d_high = np.min((np.ceil(dist_high.value * spkpc) / spkpc + mar_high, 60)) # min between dist_high and 60 kpc (highest sim bkg)

                if self.verbose == "debug": print("Distance scan range: {:.1f} - {:.1f}".format(d_low, d_high))

                #self.dist_range = np.linspace(d_low, d_high, num_steps, dtype = int) * u.kpc
                self.dist_range = np.round(np.arange(d_low, d_high + 1/spkpc, 1/spkpc), 1) * u.kpc

                # initialize signal hypothesis instance for distance scan
                sgh = Signal_Hypothesis(self.sim, res_dt = self.sim._res_dt, 
                                        distance = dist_low, temp_para = temp_para)
            
                # returns z-score and ts statistics for all distances and all subdetectors
                sgh_out = sgh.dist_scan(self.dist_range, mode = self.ft_mode, ft_para = self.ft_para, 
                                                sig_trials = self.sig_trials, bkg_distr = self.bkg_distr, 
                                                bkg_trials = self.bkg_trials, bkg_bins = self.bkg_bins, fit_hist = self.fit_hist,
                                                model = "generic", verbose = self.verbose) 
                
                if self.ft_mode == "STF": Pvalue, Zscore, Ts_stat, Freq_stat, Time_stat = sgh_out
                elif self.ft_mode == "FFT": Pvalue, Zscore, Ts_stat, Freq_stat = sgh_out

                self.Pvalue = Pvalue
                self.Zscore = Zscore
                self.Ts_stat = Ts_stat
                self.Freq_stat = Freq_stat
                if self.ft_mode == "STF": self.Time_stat = Time_stat

                if self.verbose is not None:
                    import matplotlib.pyplot as plt

                    plot_significance(self.dist_range, self.Zscore, self.Ts_stat)
                    rel_file = "/plots/scan/SIG_model_Sukhbold_2015_27_mode_{}_time_{:.0f}ms-{:.0f}ms_bkg_trials_{:.0e}_sig_trials_{:.0e}_ampl_{:.1f}%_freq_{:.0f}Hz.pdf".format(self.ft_mode, self.scan_para["time_start"].value, self.scan_para["time_end"].value, self.bkg_trials, self.sig_trials, ampl * 100, freq.value)
                    abs_file = os.path.dirname(os.path.abspath(__file__)) + rel_file
                    plt.savefig(abs_file)
                    plt.close()

                    plot_resolution(self.dist_range, self.Freq_stat, self.Zscore)
                    rel_file = "/plots/scan/FRES_model_Sukhbold_2015_27_mode_{}_time_{:.0f}ms-{:.0f}ms_bkg_trials_{:.0e}_sig_trials_{:.0e}_ampl_{:.1f}%_freq_{:.0f}Hz.pdf".format(self.ft_mode, self.scan_para["time_start"].value, self.scan_para["time_end"].value, self.bkg_trials, self.sig_trials, ampl * 100, freq.value)
                    abs_file = os.path.dirname(os.path.abspath(__file__)) + rel_file
                    plt.savefig(abs_file)
                    plt.close()

                    if self.ft_mode == "STF":

                        plot_resolution(self.dist_range, self.Time_stat, self.Zscore)
                        rel_file = "/plots/scan/TRES_model_Sukhbold_2015_27_mode_{}_time_{:.0f}ms-{:.0f}ms_bkg_trials_{:.0e}_sig_trials_{:.0e}_ampl_{:.1f}%_freq_{:.0f}Hz.pdf".format(self.ft_mode, self.scan_para["time_start"].value, self.scan_para["time_end"].value, self.bkg_trials, self.sig_trials, ampl * 100, freq.value)
                        abs_file = os.path.dirname(os.path.abspath(__file__)) + rel_file
                        plt.savefig(abs_file)
                        plt.close()

                # 3) Calculate the 3 (5) sigma significance via interpolation of the distance scan data
                dist, perc = significance_horizon(self.dist_range, self.Zscore, self.sigma)
                fres = resolution_at_horizon(self.dist_range, self.Freq_stat, dist, self.sigma)
                if self.ft_mode == "STF": tres = resolution_at_horizon(self.dist_range, self.Time_stat, dist, self.sigma)

                # save data
                self.dist[a][f] = dist
                self.perc[a][f] = perc
                self.fres[a][f] = fres
                if self.ft_mode == "STF": self.tres[a][f] = tres

                if self.verbose is not None:
                    print("3sig distance horizon IC86: {:.1f} - {:.1f} + {:.1f}".format(dist["ic86"][0][0], dist["ic86"][0][0]-dist["ic86"][0][1], dist["ic86"][0][2]-dist["ic86"][0][0]))
                    print("3sig distance horizon Gen2: {:.1f} - {:.1f} + {:.1f}".format(dist["gen2"][0][0], dist["gen2"][0][0]-dist["gen2"][0][1], dist["gen2"][0][2]-dist["gen2"][0][0]))
                    print("3sig distance horizon Gen2+WLS: {:.1f} - {:.1f} + {:.1f}".format(dist["wls"][0][0], dist["wls"][0][0]-dist["wls"][0][1], dist["wls"][0][2]-dist["wls"][0][0]))

                    print("5sig distance horizon IC86: {:.1f} - {:.1f} + {:.1f}".format(dist["ic86"][1][0], dist["ic86"][1][0]-dist["ic86"][1][1], dist["ic86"][1][2]-dist["ic86"][1][0]))
                    print("5sig distance horizon Gen2: {:.1f} - {:.1f} + {:.1f}".format(dist["gen2"][1][0], dist["gen2"][1][0]-dist["gen2"][1][1], dist["gen2"][1][2]-dist["gen2"][1][0]))
                    print("5sig distance horizon Gen2+WLS: {:.1f} - {:.1f} + {:.1f}".format(dist["wls"][1][0], dist["wls"][1][0]-dist["wls"][1][1], dist["wls"][1][2]-dist["wls"][1][0]))

    def reshape_data(self, item):

        data = {"ic86": [], "gen2": [], "wls": []}

        for det in ["ic86", "gen2", "wls"]:
            dd = []
            for s in np.arange(len(self.sigma)):
                for a in np.arange(len(self.ampl_range)):
                    for f in np.arange(len(self.freq_range)):
                        if item[a][f] != {}:
                            for q in np.arange(len(self.quantiles)):
                                if isinstance(item[a][f][det][s][q], u.Quantity):
                                    d = item[a][f][det][s][q].value
                                else:
                                    d = item[a][f][det][s][q]
                                dd.append(d)
                        else:
                            # Handle the case where the key is not found
                            dd.append(0)
            data[det] = np.array(dd, dtype=float).reshape(len(self.sigma),
                                                          len(self.ampl_range), 
                                                          len(self.freq_range),
                                                          len(self.quantiles))
        return data

    def save(self, filename):
        
        if self.ft_mode == "FFT":
            np.savez(file = filename, 
                     ampl = self.ampl_range, 
                     freq = self.freq_range, 
                     sig = self.sigma, 
                     quan = self.quantiles,
                     dist_ic86 = self.dist["ic86"],
                     dist_gen2 = self.dist["gen2"],
                     dist_wls = self.dist["wls"],
                     fres_ic86 = self.fres["ic86"],
                     fres_gen2 = self.fres["gen2"],
                     fres_wls = self.fres["wls"])
            
        elif self.ft_mode == "STF":
            np.savez(file = filename, 
                     ampl = self.ampl_range, 
                     freq = self.freq_range, 
                     sig = self.sigma, 
                     quan = self.quantiles,
                     dist_ic86 = self.dist["ic86"],
                     dist_gen2 = self.dist["gen2"],
                     dist_wls = self.dist["wls"],
                     fres_ic86 = self.fres["ic86"],
                     fres_gen2 = self.fres["gen2"],
                     fres_wls = self.fres["wls"],
                     tres_ic86 = self.tres["ic86"],
                     tres_gen2 = self.tres["gen2"],
                     tres_wls = self.tres["wls"])
        return
            
    def combine(self, filebase, ampl_range, item):
        print(item)
        self.ampl_range = ampl_range

        ic86 = [[{} for f in range((self.freq_range).size)] for a in range((self.ampl_range).size)]
        gen2 = [[{} for f in range((self.freq_range).size)] for a in range((self.ampl_range).size)]
        wls = [[{} for f in range((self.freq_range).size)] for a in range((self.ampl_range).size)]
        
        for a, ampl in enumerate(ampl_range):
            print("Amplitude: {}%".format(ampl*100))

            filename = filebase +"_ampl_{:.1f}%.npz".format(ampl*100)
            data = np.load(filename, allow_pickle = True)
            ic86[a] = data[item + "_ic86"]
            gen2[a] = data[item + "_gen2"]
            wls[a] = data[item + "_wls"]

        ic86 = np.transpose(np.squeeze(ic86), (1,2,0,3))
        gen2 = np.transpose(np.squeeze(gen2), (1,2,0,3))
        wls = np.transpose(np.squeeze(wls), (1,2,0,3))

        # indicator, SIGN saves the significance horizon, FRES the frequency resolution etc.
        if item == "dist":
            indic = "SIGN"
        elif item == "fres":
            indic = "FRES"
        elif item == "tres":
            indic = "TRES"

        filename = filebase.replace("SCAN",indic) +"_ampl_{:.1f}-{:.1f}%.npz".format(ampl_range[0]*100, ampl_range[-1]*100)
        
        np.savez(file = filename, 
                    ampl = self.ampl_range, 
                    freq = self.freq_range, 
                    sig = self.sigma, 
                    quan = self.quantiles,
                    ic86 = ic86,
                    gen2 = gen2,
                    wls = wls)
        return