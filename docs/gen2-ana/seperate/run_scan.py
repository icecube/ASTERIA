import os
import sys
from tqdm import tqdm
os.environ['ASTERIA'] = '/home/jakob/software/ASTERIA/ASTERIA'
from asteria.simulation import Simulation
from scan import *
import matplotlib.pyplot as plt


# parsed arguments: distance and sample size
ind_ampl = int(sys.argv[1])
sig_trials = int(sys.argv[2])

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

# initial SN distance
dist_ini = 5 * u.kpc

sim = Simulation(model=model,
                distance=dist_ini, 
                res_dt=res_dt,
                Emin=0*u.MeV, Emax=100*u.MeV, dE=1*u.MeV,
                tmin=0.000*u.s, tmax=0.999*u.s, dt=sim_dt,
                hierarchy = 'normal',
                mixing_scheme = 'NoTransformation',
                detector_scope = detector_scope,
                add_wls = add_wls)
sim.run()

############################################################
#######################TEMPLATE SETUP#######################
############################################################

ampl_range = np.array([2.5, 5, 7.5, 10, 15, 20, 25, 30, 35, 40, 45, 50])/100
amplitude = np.array([ampl_range[ind_ampl]])
freq_range = np.arange(80,100,10) * u.Hz

time_start = 150*u.ms
time_end = 300*u.ms
position = "center"

sigma = [3,5]

scan_para = {"freq_range": freq_range, 
             "ampl_range": amplitude, #in percent of max value
             "time_start": time_start,
             "time_end": time_end,
             "position": position,
             "sigma": sigma}

############################################################
#########################SCAN SETUP#########################
############################################################

ft_mode = "FFT"

if ft_mode == "FFT":

    time_win = [0, 0.35] * u.s # time independent
    freq_res = 1 * u.Hz 
    freq_win = [75, 1E6] * u.Hz # freq independent
    hanning = False

    ft_para = {"time_res": res_dt, 
                "time_win": time_win,
                "freq_res": freq_res,
                "freq_win": freq_win,
                "hanning": hanning}
    
elif ft_mode == "STF":

    time_win = [0, 2] * u.s # time independent
    freq_win = [50, 1E6] * u.Hz # freq independent
    hann_len = 100*u.ms # length of Hann window
    hann_res = 5*u.Hz # relates to frequency resolution from hanning, mfft = freq_sam/freq_sam
    hann_hop = 20*u.ms # offset by which Hann window is slided over signal
    freq_sam = (1/res_dt).to(u.Hz) # = 1/time_res
    time_int = True

    ft_para = {"hann_len": hann_len,
               "hann_res": hann_res,
               "hann_hop": hann_hop, 
               "freq_sam": freq_sam,
               "time_int": time_int,
               "time_win": time_win,
               "freq_win": freq_win}

############################################################
#########################RUN SCAN###########################
############################################################

bkg_distr = "hist"
bkg_trials = int(1E8)
bkg_bins = int(2E4)
fit_hist = False

scan = Scan(sim, scan_para, ft_mode = "FFT", ft_para = ft_para, sig_trials = sig_trials, 
            bkg_distr = bkg_distr, bkg_trials = bkg_trials, bkg_bins = bkg_bins, fit_hist = False, verbose = "debug")
scan.run_interpolate()
scan.reshape_data(item = scan.dist)
scan.reshape_data(item = scan.fres)
if ft_mode == "STF": scan.reshape_data(item = scan.tres)

filename = "./files/scan/SCAN_model_{}_{:.0f}_mode_{}_time_{:.0f}ms-{:.0f}ms_bkg_trials_{:1.0e}_sig_trials_{:1.0e}_ampl_{:.1f}%.npz".format(model["name"], model["param"]["progenitor_mass"].value, ft_mode, time_start.value, time_end.value, bkg_trials, sig_trials, amplitude[0]*100)
scan.save(filename = filename)