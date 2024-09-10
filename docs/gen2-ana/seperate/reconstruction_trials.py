import os
import sys
from tqdm import tqdm
os.environ['ASTERIA'] = '/home/jakob/software/ASTERIA/ASTERIA'
from asteria.simulation import Simulation
from reconstruction import *
import scipy.stats as stats
import matplotlib.pyplot as plt 
from helper import *

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

        # filename for simulation output
        filename = self._file + "/files/reconstruction/{}/{}/RECO_model_{}_{:.0f}_mode_{}_mix_{}_hier_{}_sig_var_{:+.0f}%_bkg_var_{:+.0f}%_trials_{:1.0e}_distance_{:.1f}kpc.npz".format(
            self.ft_mode, self.bkg_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
            self.ft_mode, self.mixing_scheme, self.hierarchy,
            self.sig_var * 100, self.bkg_var * 100,
            self.trials, self.distance.value)

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
        

        if 0:
            fs = 10
            det = "wls"
            bins = 20
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            fig, ax = plt.subplots(1,3, figsize = (10,4))

            ax[0].scatter(self.time_true, self.time_reco[det], s=10, alpha=0.1, color="C0")
            ax[0].plot(np.linspace(0,1000,100), np.linspace(0,1000,100), "k--")
            ax[0].set_xlabel(r"$t_{true}$ [ms]", fontsize = fs)
            ax[0].set_ylabel(r"$t_{reco}$ [ms]", fontsize = fs)
            ax[0].tick_params(labelsize = fs)

            # add marginal plots to ax[0]
            divider0 = make_axes_locatable(ax[0])
            ax0_top = divider0.append_axes("top", 0.4, pad=0.3, sharex=ax[0])

            # histogram for reco time above the scatter plot
            ax0_top.hist(self.time_true, bins = bins, density=True, color='grey', orientation='vertical', align = "left")
            ax0_top.set_ylabel("%", fontsize = fs)

            ax[1].scatter(self.freq_true, self.freq_reco[det], s=10, alpha=0.1, color="C0")
            ax[1].plot(np.linspace(0,500,100), np.linspace(0,500,100), "k--")
            ax[1].set_xlabel(r"$f_{true}$ [Hz]", fontsize = fs)
            ax[1].set_ylabel(r"$f_{reco}$ [Hz]", fontsize = fs)
            ax[1].tick_params(labelsize = fs)

            # add marginal plots to ax[1]
            divider1 = make_axes_locatable(ax[1])
            ax1_top = divider1.append_axes("top", 0.4, pad=0.3, sharex=ax[1])

            # histogram for reco time above the scatter plot
            ax1_top.hist(self.freq_true, bins = bins, density=True, color='grey', orientation='vertical', align = "left")
            ax1_top.tick_params(labelsize = fs)

            ax[2].scatter(self.time_true-self.time_reco[det], self.freq_true-self.freq_reco[det], s=10, alpha=0.1, color="C0")
            ax[2].axvline(0, color = "k", ls = "--")
            ax[2].axhline(0, color = "k", ls = "--")
            ax[2].set_xlabel(r"$t_{true} - t_{reco}$ [ms]", fontsize = fs)
            ax[2].set_ylabel(r"$f_{true} - f_{reco}$ [Hz]", fontsize = fs)
            ax[2].tick_params(labelsize = fs)

            # add marginal plots to ax[2]
            divider2 = make_axes_locatable(ax[2])
            ax2_top = divider2.append_axes("top", 0.4, pad=0.3, sharex=ax[2])
            ax2_right = divider2.append_axes("right", 0.4, pad=0.3, sharey=ax[2])

            # histogram for reco time above the scatter plot
            ax2_top.hist((self.time_true - self.time_reco[det]), bins = bins, density=True, color='grey', orientation='vertical', align = "left")
            ax2_top.axvline(np.median(self.time_true - self.time_reco[det]).value, color="C0", ls = "--", lw = 2, label = r"$\langle t_{true} - t_{reco} \rangle$")
            ax2_top.tick_params(labelsize = fs)

            # histogram for reco freq above the scatter plot
            ax2_right.hist((self.freq_true - self.freq_reco[det]), bins = bins, density=True, color='grey', orientation='horizontal', align = "left")
            ax2_right.axhline(np.median(self.freq_true - self.freq_reco[det]).value, color="C0", ls = "--", lw = 2, label = r"$\langle f_{true} - f_{reco} \rangle$")
            ax2_right.tick_params(labelsize = fs)

            plt.show()

        return
    
    def save(self, filename):
        
        freq_true = strip_units(self.freq_true)
        freq_reco = strip_units(self.freq_reco)
        time_true = strip_units(self.time_true)
        time_reco = strip_units(self.time_reco)

        hypo = ["null", "signal"]
        det = ["ic86", "gen2", "wls"]

        np.savez(file = filename, 
                    trials = self.trials, 
                    hypo = hypo,
                    det = det,
                    freq_true = freq_true,
                    freq_reco = freq_reco,
                    time_true = time_true,
                    time_reco = time_reco)
        return

    
    def load(self, filename):
        """Load data.

        Args:
            filename (str): Filename.
        """

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
    
    def bootstrap(self, filename, trials = 10000, repetitions = 100):

        # dict holds for all hypo and det for each repetition the median, 16% and 84% quantiles
        self.boot_freq = {"null" : {"ic86": np.zeros((repetitions, 3)), "gen2": np.zeros((repetitions, 3)), "wls": np.zeros((repetitions, 3))}, 
                          "signal" : {"ic86": np.zeros((repetitions, 3)), "gen2": np.zeros((repetitions, 3)), "wls": np.zeros((repetitions, 3))}} # empty dictionary
        self.boot_time = {"null" : {"ic86": np.zeros((repetitions, 3)), "gen2": np.zeros((repetitions, 3)), "wls": np.zeros((repetitions, 3))}, 
                          "signal" : {"ic86": np.zeros((repetitions, 3)), "gen2": np.zeros((repetitions, 3)), "wls": np.zeros((repetitions, 3))}}
        
        self.load(filename)
        samples = trials * repetitions # sample trials x repetitions
        index = np.arange(self.trials) # list with index positions until trials x repetitions

        # sample indices, indices can be double
        ind = np.random.choice(index, size = samples, replace = True)

        # sample corresponding true frequency and time and reshape
        freq_true = self.freq_true[ind].reshape(repetitions, trials)
        time_true = self.time_true[ind].reshape(repetitions, trials)

        # for the reco frequency and time the values are detector and hypothesis dependent
        for hypo in ["null", "signal"]: # loop over hypothesis
            for det in ["ic86", "gen2", "wls"]: # loop over detector
                
                # sample reco frequency and time and reshape
                freq_reco = self.freq_reco[hypo][det][ind].reshape(repetitions, trials)
                time_reco = self.time_reco[hypo][det][ind].reshape(repetitions, trials)

                # compute median, 16% and 84% quantile for each repetition of the true fequency and time
                for r in range(repetitions):
                    self.boot_freq[hypo][det][r] = np.array([np.mean(freq_reco[r]-freq_true).value, np.quantile(freq_reco[r]-freq_true, 0.16).value, np.quantile(freq_reco[r]-freq_true, 0.84).value]) * u.Hz
                    self.boot_time[hypo][det][r] = np.array([np.mean(time_reco[r]-time_true).value, np.quantile(time_reco[r]-time_true, 0.16).value, np.quantile(time_reco[r]-time_true, 0.84).value]) * u.ms

        if 1:
            fig, ax = plt.subplots(3, 2, figsize=(12, 10))

            ax = ax.T.ravel()

            dets = ["ic86", "gen2", "wls"]
            for i in range(6):

                j = i % 3
                det = dets[j]

                if i > 3: item = self.boot_time["signal"]
                else: item = self.boot_freq["signal"]

                rel_error = np.std(item[det], axis = 0)/np.mean(item[det], axis = 0)

                n1, _, _ = ax[i].hist(item[det][:, 1], bins=20, density=True, histtype="step", color="C1", label="16%")
                n0, _, _ = ax[i].hist(item[det][:, 0], bins=20, density=True, histtype="step", color="C0", label="50%")
                n2, _, _ = ax[i].hist(item[det][:, 2], bins=20, density=True, histtype="step", color="C2", label="84%")

                ax[i].text(item[det][:, 1].mean(), n1.max()/2, s = "{:1.1E}".format(rel_error[1]), color = "C1", fontsize = 10, weight = "bold", ha = "center")
                ax[i].text(item[det][:, 0].mean(), n0.max()/2, s = "{:1.1E}".format(rel_error[0]), color = "C0", fontsize = 10, weight = "bold", ha = "center")
                ax[i].text(item[det][:, 2].mean(), n2.max()/2, s = "{:1.1E}".format(rel_error[2]), color = "C2", fontsize = 10, weight = "bold", ha = "center")

                ax[i].text(0.15, 0.9, s = det, transform=ax[i].transAxes, fontsize=14, va='top', 
                        bbox= dict(boxstyle='round', facecolor='white', alpha=0.5))
                ax[i].tick_params(labelsize = 14)


            # Create a common legend for all subplots
            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=14, ncols = 3)

            # add a big axis, hide frame
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel("Mean Significance", fontsize = 14)
            plt.ylabel("Normalized Counts", fontsize = 14)

            plt.tight_layout()
            plt.show()
