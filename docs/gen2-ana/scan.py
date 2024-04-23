from asteria.simulation import Simulation
from analysis import *
from scipy.optimize import minimize
from helper import *

from plthelper import plot_significance

def loss_dist_range(dist, ana_para, sigma, det):
    sim, res_dt, trials, temp_para, mode, ft_para = ana_para
    ana = Analysis(sim, res_dt = res_dt, distance = dist*u.kpc, trials = trials, temp_para = temp_para)
    ana.run(mode = mode, ft_para = ft_para, model = "generic")
    loss = np.sqrt((ana.zscore[det][0] - sigma)**2)
    return loss

class Scan():
    
    def __init__(self,
                 sim,
                 scan_para,
                 ft_mode, 
                 ft_para,
                 trials,
                 verbose = False):
        
        self.sim = sim
        self.scan_para = scan_para
        self.ft_mode = ft_mode
        self.ft_para = ft_para
        self.trials = trials
        self.verbose = verbose

    def run(self):
        """The parameter scan is designed in five steps:
        1) A loop over all amplitudes.
        2) A loop over all frequencies.
        For each amplitude, frequency value the following operations are done
        3) Find the distance bounds for which the significance falls from > 5 sigma to < 3 sigma for all subdetectors
        4) The actual distance scan using 10 steps within the bounds established in 3)
        5) The so obtained significances are interpolated to compute the 3 and 5 sigma detection horizon.
        """
        
        ampl_range = self.scan_para["ampl_range"]
        freq_range = self.scan_para["freq_range"]

        # create empty 2D list with each entry being a dictionary
        self.dist = [[{} for f in range(len(freq_range))] for a in range(len(ampl_range))]
        self.perc = [[{} for f in range(len(freq_range))] for a in range(len(ampl_range))]

        for a, ampl in enumerate(ampl_range): # loop over scan amplitude
            for f, freq in enumerate(freq_range): # loop over scan frequency
                    
                    print("Frequency: {}, Amplitude: {} % ".format(freq, ampl))

                    # template dictionary uses loop ampl, freq and scan_para values
                    temp_para = {"frequency": freq, 
                                "amplitude": ampl, #in percent of max value
                                "time_start": self.scan_para["time_start"],
                                "time_end": self.scan_para["time_end"],
                                "position": self.scan_para["position"]}
                    

                    # 1) Find the distance bounds for which the significance falls from > 5 sigma to < 3 sigma for all subdetectors
                    # This is done to avoid scanning unnecessarily many distances for which the significance is either far too high
                    # or far too low.
                    trials = 1000
                    ana_para = [self.sim, self.sim._res_dt, trials, temp_para, self.ft_mode, self.ft_para]

                    # arguments for the minimizer
                    # The minimizer is defined above and initializes the Analysis class.
                    # To avoid extrapolation issues with our interpolator, 
                    # we search for significances slightly above (below) 5 (3) sigma
                    args_dist_low = (ana_para, 5.5, "ic86")
                    args_dist_high = (ana_para, 2.5, "wls")

                    # returns lower bound distance
                    res_dist_low = minimize(loss_dist_range, x0 = 10, args=args_dist_low, tol = 1e-1, method = "Nelder-Mead")
                    dist_low = res_dist_low.x[0] * u.kpc
                    
                    # returns higher bound distance
                    res_dist_high = minimize(loss_dist_range, x0 = 10, args=args_dist_high, tol = 1e-1, method = "Nelder-Mead")
                    dist_high = res_dist_high.x[0] * u.kpc

                    if self.verbose: print("Distance range: {} - {}".format(dist_low, dist_high))

                    # 2) Distance scan

                    # distance search range
                    dist_range = np.linspace(dist_low, dist_high, 10, endpoint=True)

                    # initialize new Analysis instance for distance scan
                    ana = Analysis(self.sim, res_dt = self.sim._res_dt, 
                                    distance = dist_low, trials = self.trials, 
                                    temp_para = temp_para)
                    
                    # returns z-score and ts statistics for all distances and all subdetectors
                    Zscore, Ts_stat = ana.dist_scan(dist_range, mode = self.ft_mode, ft_para = self.ft_para, model = "generic", verbose = self.verbose)
                    
                    if self.verbose:
                        import matplotlib.pyplot as plt
                        plot_significance(dist_range, Zscore, Ts_stat)
                        plt.show()

                    # 3) Calculate the 3 (5) sigma significance via interpolation of the distance scan data
                    dist, perc = significance_horizon(dist_range, Zscore, confidence_level = [3,5])

                    # save data
                    self.dist[a][f] = dist
                    self.perc[a][f] = perc

                    if self.verbose:
                        print("3sig distance horizon IC86: {}".format(dist["3sig"]["ic86"][0]))
                        print("3sig distance horizon Gen2: {}".format(dist["3sig"]["gen2"][0]))
                        print("3sig distance horizon Gen2+WLS: {}".format(dist["3sig"]["wls"][0]))

                        print("5sig distance horizon IC86: {}".format(dist["5sig"]["ic86"][0]))
                        print("5sig distance horizon Gen2: {}".format(dist["5sig"]["gen2"][0]))
                        print("5sig distance horizon Gen2+WLS: {}".format(dist["5sig"]["wls"][0]))
