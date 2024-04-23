from asteria.simulation import Simulation
from analysis import *
from scipy.optimize import minimize

def loss_dist_range(dist, ana_para, sigma, det):
    sim, res_dt, trials, temp_para, mode, ft_para = ana_para
    ana = Analysis(sim, res_dt = res_dt, distance = dist*u.kpc, trials = trials, temp_para = temp_para)
    ana.run(mode = mode, ft_para = ft_para, model = "generic")
    loss = (ana.zscore[det][0] - sigma)**2
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

        for freq in self.scan_para["freq_range"]: # loop over scan frequency
            for ampl in self.scan_para["ampl_range"]: # loop over scan amplitude
                    
                    if self.verbose: print("Frequency: {}, Amplitude: {} % ".format(freq, ampl))

                    temp_para = {"frequency": freq, 
                                "amplitude": ampl, #in percent of max value
                                "time_start": self.scan_para["time_start"],
                                "time_end": self.scan_para["time_end"],
                                "position": self.scan_para["position"]}
                    

                    trials = 1000
                    ana_para = [self.sim, self.sim._res_dt, trials, temp_para, self.ft_mode, self.ft_para]

                    args_dist_low = (ana_para, 5.5, "ic86")
                    args_dist_high = (ana_para, 2.5, "wls")

                    res_dist_low = minimize(loss_dist_range, x0 = 10, args=args_dist_low, tol = 1e-1, method = "Nelder-Mead")
                    dist_low = res_dist_low.x[0] * u.kpc

                    res_dist_high = minimize(loss_dist_range, x0 = 10, args=args_dist_high, tol = 1e-1, method = "Nelder-Mead")
                    dist_high = res_dist_high.x[0] * u.kpc

                    if self.verbose: print("Distance range: {} - {}".format(dist_low, dist_high))
        
                    dist_range = np.linspace(dist_low, dist_high, 10, endpoint=True)

                    ana = Analysis(self.sim, res_dt = self.sim._res_dt, 
                                    distance = dist_low, trials = self.trials, 
                                    temp_para = temp_para)
                    
                    Zscore, Ts_stat = ana.dist_scan(dist_range, mode = self.ft_mode, ft_para = self.ft_para, model = "generic", verbose = self.verbose)

                    print(Zscore, Ts_stat)
