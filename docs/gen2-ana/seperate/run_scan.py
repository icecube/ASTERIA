import os
import sys
from tqdm import tqdm
os.environ['ASTERIA'] = '/home/jakob/software/ASTERIA/ASTERIA'
from asteria.simulation import Simulation
from scan import *
import matplotlib.pyplot as plt

def get_dir_name(para):
    # select correct directory for systemics
    if para["mixing_scheme"] == "NoTransformation":
        scan_dir_name = "syst_time_{:.0f}_{:.0f}ms".format(time_start.value, time_end.value)
        if para["scan_para"]["time_start"] == 150 * u.ms and para["scan_para"]["time_end"] == 300 * u.ms:
            scan_dir_name = "default"
        if para["sig_var"] != 0:
            scan_dir_name = "syst_det_sig_{:.0f}%".format(sig_var*100)
        elif para["bkg_var"] != 0:
            scan_dir_name = "syst_det_bkg_{:.0f}%".format(bkg_var*100)

    elif para["mixing_scheme"] == "CompleteExchange":
        scan_dir_name = "syst_mix_comp_exch"

    elif para["mixing_scheme"] == "AdiabaticMSW":
        if para["hierarchy"] == "normal":
            scan_dir_name = "syst_mix_MSW_NH"
        elif para["hierarchy"] == "inverted":
            scan_dir_name = "syst_mix_MSW_IH"

    return scan_dir_name

# parsed arguments: distance and sample size
ind_ampl = int(sys.argv[1])
sig_trials = int(sys.argv[2])

bkg_trials = int(1E8)
bkg_bins = int(2E4)

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

ampl_range = np.array([2.5, 5, 7.5, 10, 15, 20, 25, 30, 35, 40, 45, 50])/100 
amplitude = np.array([ampl_range[ind_ampl]])

if ft_mode == "FFT":
    freq_range = np.arange(80,410,10) * u.Hz
elif ft_mode == "STF":
    freq_range = np.arange(60,410,10) * u.Hz

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
        "scan_para": scan_para,
        "ft_mode": ft_mode,
        "ft_para": ft_para,
        "sig_trials": sig_trials,
        "bkg_trials": bkg_trials,
        "bkg_bins": bkg_bins,
        "sig_var": sig_var,
        "bkg_var": bkg_var}

############################################################
#########################RUN SCAN###########################
############################################################

print("SCAN")
print("-------------------------")
print("amplitude: {} %".format(amplitude * 100))
print("frequency: {} ".format(freq_range))
print("start time: {}, end time: {}".format(time_start, time_end))
print("-------------------------")
print("signal trials: {}, background trials: {}, background bins: {}".format(sig_trials, bkg_trials, bkg_bins))
print("fourier mode: {}".format(ft_mode))
print("signal variation: {}%".format(sig_var*100))
print("background variation: {}%".format(bkg_var*100))
print("mixing scheme: {}, hierarchy: {}".format(mixing_scheme, hierarchy))
print("-------------------------")

scan_dir_name = get_dir_name(para)
MODE = "COMBINE"

scan = Scan(sim, para = para, verbose = "debug")

if MODE == "RUN":
    # SCAN
    scan.run()
    scan.dist = scan.reshape_data(item = scan.dist)
    scan.fres = scan.reshape_data(item = scan.fres)
    if ft_mode == "STF": scan.tres = scan.reshape_data(item = scan.tres)

    filename = "./files/scan/{}/{}/SCAN_model_{}_{:.0f}_mode_{}_time_{:.0f}ms-{:.0f}ms_mix_{}_hier_{}_sig_var_{:.0f}%_bkg_var_{:.0f}%_sig_trials_{:1.0e}_bkg_trials_{:1.0e}_ampl_{:.1f}%.npz".format(
        ft_mode, scan_dir_name, model["name"], model["param"]["progenitor_mass"].value, 
        ft_mode, time_start.value, time_end.value, mixing_scheme, hierarchy,
        sig_var * 100, bkg_var * 100, sig_trials, bkg_trials, amplitude[0]*100)
    scan.save(filename = filename)

elif MODE == "COMBINE":
    # COMBINE DATA
    filebase = "./files/scan/{}/{}/SCAN_model_{}_{:.0f}_mode_{}_time_{:.0f}ms-{:.0f}ms_mix_{}_hier_{}_sig_var_{:.0f}%_bkg_var_{:.0f}%_sig_trials_{:1.0e}_bkg_trials_{:1.0e}".format(
        ft_mode, scan_dir_name, model["name"], model["param"]["progenitor_mass"].value, 
        ft_mode, time_start.value, time_end.value, mixing_scheme, hierarchy,
        sig_var * 100, bkg_var * 100, sig_trials, bkg_trials)
    scan.combine(filebase = filebase, ampl_range = ampl_range, item = "dist")
    scan.combine(filebase = filebase, ampl_range = ampl_range, item = "fres")
    if ft_mode == "STF": scan.combine(filebase = filebase, ampl_range = ampl_range, item = "tres")