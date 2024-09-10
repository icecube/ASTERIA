import os
import sys
from tqdm import tqdm
os.environ['ASTERIA'] = '/home/jakob/software/ASTERIA/ASTERIA'
from asteria.simulation import Simulation
from reconstruction_trials import *
import matplotlib.pyplot as plt

def get_dir_name(para):
    # select correct directory for systemics
    if para["mixing_scheme"] == "NoTransformation":
        reco_dir_name = "syst_time_{:.0f}ms".format(duration.value)
        if para["reco_para"]["duration"] == 150 * u.ms:
            reco_dir_name = "default"
        if para["sig_var"] != 0:
            reco_dir_name = "syst_det_sig_{:.0f}%".format(sig_var*100)
        elif para["bkg_var"] != 0:
            reco_dir_name = "syst_det_bkg_{:.0f}%".format(bkg_var*100)

    elif para["mixing_scheme"] == "CompleteExchange":
        reco_dir_name = "syst_mix_comp_exch"

    elif para["mixing_scheme"] == "AdiabaticMSW":
        if para["hierarchy"] == "normal":
            reco_dir_name = "syst_mix_MSW_NH"
        elif para["hierarchy"] == "inverted":
            reco_dir_name = "syst_mix_MSW_IH"

    return reco_dir_name

# parsed arguments: distance and sample size
ind_ampl = int(sys.argv[1])
ind_dist = int(sys.argv[2])
trials = int(sys.argv[3])

ft_mode = "STF"

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

# neutrino flavor mixing scheme and hierarchy
mixing_scheme = "NoTransformation"
hierarchy = "normal"

# detector signal and background count variation
sig_var = 0 # signal variation of +-10%
bkg_var = 0 # background variation of +-10%

sim = Simulation(model=model,
                distance=dist_ini, 
                res_dt=res_dt,
                Emin=0*u.MeV, Emax=100*u.MeV, dE=1*u.MeV,
                tmin=0.000*u.s, tmax=0.999*u.s, dt=sim_dt,
                hierarchy = hierarchy,
                mixing_scheme = mixing_scheme,
                detector_scope = detector_scope,
                add_wls = add_wls)
sim.run()

############################################################
#######################TEMPLATE SETUP#######################
############################################################

# select amplitude
ampl_range = np.array([2.5, 5, 7.5, 10, 15, 20, 25, 30, 35, 40, 45, 50])/100 
amplitude = np.array([ampl_range[ind_ampl]])


# select distance
dist_min, dist_max, dist_step = 0.2, 60, 0.2
dist_range = np.arange(dist_min, dist_max + dist_step, dist_step) * u.kpc
dist_range = np.round(dist_range, 1)
distance = dist_range[ind_dist]

freq_min = 50*u.Hz
freq_max = 500*u.Hz

time_min = 50*u.ms
time_max = 950*u.ms
duration = 150*u.ms


position = "center"

sigma = [3,5]

reco_para = {"freq_min": freq_min,
             "freq_max": freq_max, 
             "ampl": amplitude, #in percent of max value
             "time_min": time_min,
             "time_max": time_max,
             "duration": duration,
             "position": position,
             "sigma": sigma}

############################################################
#########################reco SETUP#########################
############################################################

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

    ft_para = {"hann_len": hann_len,
               "hann_res": hann_res,
               "hann_hop": hann_hop, 
               "freq_sam": freq_sam,
               "time_win": time_win,
               "freq_win": freq_win}

para = {"model": model,
        "hierarchy": hierarchy,
        "mixing_scheme": mixing_scheme,
        "distance": distance,
        "reco_para": reco_para,
        "ft_mode": ft_mode,
        "ft_para": ft_para,
        "trials": trials,
        "sig_var": sig_var,
        "bkg_var": bkg_var}

############################################################
#########################RUN RECO###########################
############################################################

print("RECO")
print("-------------------------")
print("distance: {}".format(distance))
print("amplitude: {} %".format(amplitude * 100))
print("min frequency: {}, max frequency: {}".format(freq_min, freq_max))
print("min time: {}, max time: {}, duration: {}".format(time_min, time_max, duration))
print("-------------------------")
print("trials: {}".format(trials))
print("fourier mode: {}".format(ft_mode))
print("signal variation: {}%".format(sig_var*100))
print("background variation: {}%".format(bkg_var*100))
print("mixing scheme: {}, hierarchy: {}".format(mixing_scheme, hierarchy))
print("-------------------------")

reco_dir_name = get_dir_name(para)

MODE = "BOOTSTRAP"

rct = Reconstruction_Trials(sim, para = para, verbose = "debug")


filename = "./files/bootstrapping/reco/{}/RECO_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_ampl_{:.0f}%_mix_{}_hier_{}_trials_{:1.0e}_dist_{:.1f}%.npz".format(
    ft_mode, model["name"], model["param"]["progenitor_mass"].value, 
    ft_mode, duration.value, amplitude[0]*100, mixing_scheme, hierarchy, trials, distance.value)

if MODE == "GENERATE":
    rct.generate()
    rct.save(filename = filename)

elif MODE == "BOOTSTRAP":
    rct.bootstrap(filename, trials = 1000, repetitions = 100)
