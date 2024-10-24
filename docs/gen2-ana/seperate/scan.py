from analysis import *
from signal_hypothesis import *
from scipy.optimize import brentq
from helper import *

from plthelper import plot_significance

def loss_dist_range_interpolate(dist, ana_para, sigma, det, quant):
    sim, res_dt, trials, temp_para, mode, ft_para, sig_var, bkg_var = ana_para
    ana = Analysis(sim, res_dt = res_dt, distance = dist*u.kpc, temp_para = temp_para)
    ana.run(mode = mode, ft_para = ft_para, sig_var = sig_var, bkg_var = bkg_var, trials = trials, model = "generic")
    loss = (ana.zscore[det][quant] - sigma)
    return loss

class Scan():
    
    def __init__(self,
                 sim,
                 para,
                 verbose = None):
        
        self.sim = sim
        self.para = para
        self.verbose = verbose

        # read in keywords of para
        self.model = self.para["model"]
        self.hierarchy = self.para["hierarchy"]
        self.mixing_scheme = self.para["mixing_scheme"]
        self.scan_para = self.para["scan_para"]
        self.ft_mode = self.para["ft_mode"]
        self.ft_para = self.para["ft_para"]
        self.sig_trials = self.para["sig_trials"]
        self.bkg_trials = self.para["bkg_trials"]
        self.bkg_bins = self.para["bkg_bins"]
        self.sig_var = self.para["sig_var"]
        self.bkg_var = self.para["bkg_var"]

        # read in kewords of scan_para
        self.ampl_range = self.scan_para["ampl_range"]
        self.freq_range = self.scan_para["freq_range"]
        self.sigma = self.scan_para["sigma"]
        self.quantiles = [0.5, 0.16, 0.84]

        self._file = os.path.dirname(os.path.abspath(__file__))
        np.random.seed(0)

    def get_dir_name(self):
        
        self.mixing_scheme = self.sim.mixing_scheme
        if self.sim.hierarchy.name == "NORMAL":
            self.hierarchy = "normal"
        elif self.sim.hierarchy.name == "INVERTED":
            self.hierarchy = "inverted"

        # select correct directory for systemics
        if self.mixing_scheme == "NoTransformation":
            self.bkg_dir_name = "default"

            if self.temp_para["time_start"] == 150 * u.ms and self.temp_para["time_end"] == 300 * u.ms:
                self.scan_dir_name = "default"
            else:
                self.scan_dir_name = "syst_time_{:.0f}_{:.0f}ms".format(self.temp_para["time_start"].value, self.temp_para["time_end"].value)

            if self.sig_var != 0:
                self.bkg_dir_name = "syst_det_sig_{:+.0f}%".format(self.sig_var*100)
                self.scan_dir_name = self.bkg_dir_name
            elif self.bkg_var != 0:
                self.bkg_dir_name = "syst_det_bkg_{:+.0f}%".format(self.bkg_var*100)
                self.scan_dir_name = self.bkg_dir_name

        elif self.mixing_scheme == "CompleteExchange":
            self.bkg_dir_name = "syst_mix_comp_exch"
            self.scan_dir_name = self.bkg_dir_name

        elif self.mixing_scheme == "AdiabaticMSW":
            if self.hierarchy == "normal":
                self.bkg_dir_name = "syst_mix_MSW_NH"
                self.scan_dir_name = self.bkg_dir_name

            elif self.hierarchy == "inverted":
                self.bkg_dir_name = "syst_mix_MSW_IH"
                self.scan_dir_name = self.bkg_dir_name

    def run(self):
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

        for a, ampl in enumerate(self.ampl_range): # loop over scan amplitude
            for f, freq in enumerate(self.freq_range): # loop over scan frequency
                    
                if self.verbose is not None: print("Frequency: {}, Amplitude: {} % ".format(freq, ampl*100))

                # template dictionary uses loop ampl, freq and scan_para values
                self.temp_para = {"model": self.model,
                                  "frequency": freq, 
                                  "amplitude": ampl, #in percent of max value
                                  "time_start": self.scan_para["time_start"],
                                  "time_end": self.scan_para["time_end"],
                                  "position": self.scan_para["position"]}
                
                self.get_dir_name() # get scan_dir_name and bkg_dir_name

                # 1) Find the distance bounds for which the significance falls from > 5 sigma to < 3 sigma for all subdetectors
                # This is done to avoid scanning unnecessarily many distances for which the significance is either far too high
                # or far too low.
                trials = 1000 # use 1/10 of trials used in later distance scan to increase speed of convergence
                ana_para = [self.sim, self.sim._res_dt, trials, self.temp_para, self.ft_mode, self.ft_para, self.sig_var, self.bkg_var]

                # arguments for the minimizer
                # The minimizer is defined above and initializes the Analysis class.
                args_dist_low = (ana_para, 5, "ic86", 1)
                args_dist_high = (ana_para, 3, "wls", 2)
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
                        args_dist_low = tuple(arg)
                        root_low = brentq(loss_dist_range_interpolate, a = 0.1, b = 100, args = args_dist_low, xtol = 1e-2)
                        dist_low = root_low * u.kpc
                    except ValueError:
                        print("Gen2 boundaries failed")
                        try:
                            arg = list(args_dist_low) # if Gen2 is too low, try WLS
                            arg[2] = "wls"
                            args_dist_low = tuple(arg)
                            root_low = brentq(loss_dist_range_interpolate, a = 0.1, b = 100, args = args_dist_low, xtol = 1e-2)
                            dist_low = root_low * u.kpc
                        except ValueError:
                            print("WLS boundaries failed")
                            self.dist[a][f] = {"ic86": [np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])], "gen2": [np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])], "wls": [np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])]} 
                            self.perc[a][f] = {"ic86": [np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])], "gen2": [np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])], "wls": [np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])]} 
                            skip_next = True

                if skip_next:
                    continue

                # returns higher bound distance
                root_high = brentq(loss_dist_range_interpolate, a = 0.1, b = 100, args = args_dist_high, xtol = 1e-2)
                dist_high = root_high * u.kpc
                if self.verbose == "debug": print("Distance estimate: {:.3f} - {:.3f}".format(dist_low, dist_high))

                # 2) Distance scan

                # distance search range

                spkpc = 5 # steps per kpc
                mar_low, mar_high = 5, 2 # extra margin to lower/higher side in kpc
                d_low = np.max((np.floor(dist_low.value * spkpc) / spkpc - mar_low, 0.2)) # max between dist_low and 0.2 kpc (lowest sim bkg)
                d_high = np.min((np.ceil(dist_high.value * spkpc) / spkpc + mar_high, 60)) # min between dist_high and 60 kpc (highest sim bkg)

                if self.verbose == "debug": print("Distance scan range: {:.1f} - {:.1f}".format(d_low, d_high))

                self.dist_range = np.round(np.arange(d_low, d_high + 1/spkpc, 1/spkpc), 1) * u.kpc

                # initialize signal hypothesis instance for distance scan
                sgh = Signal_Hypothesis(self.sim, res_dt = self.sim._res_dt, 
                                        distance = dist_low, temp_para = self.temp_para)
            
                # returns z-score and ts statistics for all distances and all subdetectors
                sgh_out = sgh.dist_scan(self.dist_range, mode = self.ft_mode, ft_para = self.ft_para, 
                                                sig_var = self.sig_var, bkg_var = self.bkg_var,
                                                sig_trials = self.sig_trials, bkg_trials = self.bkg_trials, bkg_bins = self.bkg_bins,
                                                model = "generic", verbose = self.verbose) 
                
                Pvalue, Zscore, Ts_stat = sgh_out

                self.Pvalue = Pvalue
                self.Zscore = Zscore
                self.Ts_stat = Ts_stat

                if self.verbose is not None:
                    
                    plot_significance(self)
               
                # 3) Calculate the 3 (5) sigma significance via interpolation of the distance scan data
                dist, perc = significance_horizon(self.dist_range, self.Zscore, self.sigma)

                # save data
                self.dist[a][f] = dist
                self.perc[a][f] = perc

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
        
        np.savez(file = filename, 
                 ampl = self.ampl_range, 
                 freq = self.freq_range, 
                 sig = self.sigma, 
                 quan = self.quantiles,
                 dist_ic86 = self.dist["ic86"],
                 dist_gen2 = self.dist["gen2"],
                 dist_wls = self.dist["wls"])
        return
            
    def combine(self, filebase, ampl_range, item):
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

        filename = filebase.replace("SCAN","SIGN") +"_ampl_{:.1f}-{:.1f}%.npz".format(ampl_range[0]*100, ampl_range[-1]*100)
        
        np.savez(file = filename, 
                    ampl = self.ampl_range, 
                    freq = self.freq_range, 
                    sig = self.sigma, 
                    quan = self.quantiles,
                    ic86 = ic86,
                    gen2 = gen2,
                    wls = wls)
        return
    
    def frequency_average(self, filename, ampl_range):
        self.ampl_range = ampl_range
        
        data = np.load(filename)

        ic86 = np.zeros((len(self.sigma), self.ampl_range.size, 4))
        gen2 = np.zeros((len(self.sigma), self.ampl_range.size, 4))
        wls = np.zeros((len(self.sigma), self.ampl_range.size, 4))

        for s, sig in enumerate(self.sigma):
            for i, det in enumerate(["ic86", "gen2", "wls"]):

                mean, std = np.nanmean(np.abs(data[det][s,:,:,0]), axis = 0), np.nanstd(np.abs(data[det][s,:,:,0]), axis = 0)
                q16, q84 = np.nanmean(np.abs(data[det][s,:,:,1]), axis = 0), np.nanmean(np.abs(data[det][s,:,:,2]), axis = 0)

                vals = np.array([mean, std, q16, q84]).T # transverse from (4, len(ampl_range)) to (len(ampl_range), 4)

                if det == "ic86":
                    ic86[s,:,:] = vals
                elif det == "gen2":
                    gen2[s,:,:] = vals
                elif det == "wls":
                    wls[s,:,:] = vals

        filename = filename.replace("SIGN","SIGN_FAVG")

        np.savez(file = filename, 
                    ampl = self.ampl_range, 
                    freq = self.freq_range, 
                    sig = self.sigma, 
                    quan = self.quantiles,
                    ic86 = ic86,
                    gen2 = gen2,
                    wls = wls)

        return