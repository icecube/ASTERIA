import os
import sys
from tqdm import tqdm
os.environ['ASTERIA'] = '/home/jakob/software/ASTERIA/ASTERIA'
from asteria.simulation import Simulation
from background_trials import *


# Class generating and resorting data for background trials
class BGT():

    def __init__(self,
                 sim,
                 ana_para,
                 verbose = None):
        
        self.sim = sim
        self.ana_para = ana_para
        self.verbose = verbose

    def generate_data(self, samples, filename = None):
        print("DATA GENERATION -- SAMPLES {}".format(samples))

        if not os.path.isdir("./files"):
            os.mkdir("./files")
        if filename is None:
            filename = "./files/background_{}_generate_{:1.0e}_distance_{:.0f}kpc.npz".format(self.ana_para["mode"], int(samples), self.ana_para["distance"].value)

        self.max_trials = 10000 # size of batches
        self.repetitions = np.round(samples/self.max_trials).astype(int)

        ts = []
        for r in tqdm(range(self.repetitions)):
            # Initialize analysis class and run analysis
            ana = Background_Trials(self.sim, res_dt = self.ana_para["res_dt"], distance=self.ana_para["distance"], trials = self.max_trials)
            ana.run(mode = self.ana_para["mode"], ft_para = self.ana_para["ft_para"], model = "generic")
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

# parsed arguments: distance and sample size
distance = float(sys.argv[1]) * u.kpc
samples = int(sys.argv[2])

print(distance, samples)

print("SETTING UP SIMULATION AT {}".format(distance))

############################################################
######################SIMULATION SETUP######################
############################################################

# detector scope
add_wls = True
detector_scope = "Gen2"

# time resolution
sim_dt = 1 * u.ms
res_dt = 1 * u.ms

# SN model
model = {'name': 'Sukhbold_2015',
         'param':{
             'progenitor_mass': 27*u.Msun, 
             'eos': 'LS220'}
        }

sim = Simulation(model=model,
                 distance=distance, 
                 res_dt=res_dt,
                 Emin=0*u.MeV, Emax=100*u.MeV, dE=1*u.MeV,
                 tmin=0.000*u.s, tmax=1*u.s, dt=sim_dt,
                 hierarchy = 'normal',
                 mixing_scheme = 'NoTransformation',
                 detector_scope = detector_scope,
                 add_wls = add_wls)
sim.run()

############################################################
#######################ANALYSIS SETUP#######################
############################################################

time_win = [0, 0.35] * u.s # time independent
freq_res = 1 * u.Hz 
freq_win = [75, 1E6] * u.Hz # freq independent
hanning = False

mode = "FFT"
fft_para = {"time_res": res_dt, 
             "time_win": time_win,
             "freq_res": freq_res,
             "freq_win": freq_win,
             "hanning": hanning}

ana_para = {"distance": distance,
            "res_dt": res_dt,
            "mode": mode,
            "ft_para": fft_para}

############################################################
#####################BACKGROUND TRIALS######################
############################################################

bgt = BGT(sim, ana_para=ana_para, verbose=True)
bgt.generate_data(samples=samples)
