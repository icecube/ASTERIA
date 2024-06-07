import os
import sys
from tqdm import tqdm
os.environ['ASTERIA'] = '/home/jakob/software/ASTERIA/ASTERIA'
from asteria.simulation import Simulation
from background_trials import *
import matplotlib.pyplot as plt


# parsed arguments: distance and sample size
distance = float(sys.argv[1]) * u.kpc
bkg_trials = int(sys.argv[2])
output = int(sys.argv[3])

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

ft_mode = "FFT"
fft_para = {"time_res": res_dt, 
            "time_win": time_win,
            "freq_res": freq_res,
            "freq_win": freq_win,
            "hanning": hanning}

ana_para = {"model": model,
            "distance": distance,
            "res_dt": res_dt,
            "mode": ft_mode,
            "ft_para": fft_para}

bkg_min = 0
bkg_max = int(1E6)
bkg_bins = int(2E7)

############################################################
#####################BACKGROUND TRIALS######################
############################################################

bgt = Background_Trials(sim, ana_para=ana_para, bkg_trials = bkg_trials, output = output, verbose=True)
bgt.generate_data(bkg_min, bkg_max, bkg_bins)