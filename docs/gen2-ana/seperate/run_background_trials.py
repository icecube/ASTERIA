import os
import sys
from tqdm import tqdm
os.environ['ASTERIA'] = '/home/jakob/software/ASTERIA/ASTERIA'
from asteria.simulation import Simulation
from background_trials import *
import matplotlib.pyplot as plt


# parsed arguments: distance and sample size
ind_dist = int(sys.argv[1])
bkg_trials = int(sys.argv[2])
output = None #int(sys.argv[3])

dist_min, dist_max, dist_step = 1, 60, 0.2
dist_range = np.arange(dist_min, dist_max + dist_step, dist_step) * u.kpc
dist_range = np.round(dist_range, 1)
distance = dist_range[ind_dist]

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

ft_mode = "STF"

### FFT ###
if ft_mode == "FFT":
    time_win = [0, 0.35] * u.s # time independent
    freq_res = 1 * u.Hz 
    freq_win = [75, 1E6] * u.Hz # freq independent
    hanning = False
    ft_mode = "FFT"

    ft_para = {"time_res": res_dt, 
                "time_win": time_win,
                "freq_res": freq_res,
                "freq_win": freq_win,
                "hanning": hanning}

### STF ###
elif ft_mode == "STF":
    hann_len = 100*u.ms # length of Hann window
    hann_res = 5*u.Hz # relates to frequency resolution from hanning, mfft = freq_sam/freq_sam
    hann_hop = 20*u.ms # offset by which Hann window is slided over signal
    freq_sam = (1/res_dt).to(u.Hz) # = 1/time_res
    time_int = True
    time_win = [0, 100] * u.s # time independent
    freq_win = [50, 1E6] * u.Hz # 
    ft_mode = "STF"

    ft_para = {"hann_len": hann_len,
                "hann_res": hann_res,
                "hann_hop": hann_hop, 
                "freq_sam": freq_sam,
                "time_int": time_int,
                "time_win": time_win,
                "freq_win": freq_win}

ana_para = {"model": model,
            "distance": distance,
            "res_dt": res_dt,
            "mode": ft_mode,
            "ft_para": ft_para}

bkg_bins = int(2E4)
mode = "hist"
############################################################
#####################BACKGROUND TRIALS######################
############################################################

bgt = Background_Trials(sim, ana_para = ana_para, bkg_trials = bkg_trials, mode = mode, output = output, bkg_bins = bkg_bins, verbose = True)
bgt.generate_data(filename = None, load_bounds = None)