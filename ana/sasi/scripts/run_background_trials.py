import os
import sys

env_path = os.environ['ASTERIA']
script_path = os.path.join(env_path, '../../ana/sasi/scripts')
sys.path.append(str(script_path))

from snewpy.models.ccsn import Tamborra_2014
from asteria.simulation import Simulation
from background_trials import *


# parsed arguments: distance and sample size
ind_dist = int(sys.argv[1])
scenario = int(sys.argv[2])
bkg_trials = int(sys.argv[3])
bkg_bins = int(2E4)

dist_min, dist_max, dist_step = 5.2, 45, 0.2
dist_range = np.arange(dist_min, dist_max + dist_step, dist_step) * u.kpc
dist_range = np.round(dist_range, 1)
distance = dist_range[ind_dist]


############################################################
########################SWITCH BOARD########################
############################################################

# default settings, switch board only changes selected

# detector scope
add_wls = True
detector_scope = "Gen2"

# time resolution
sim_dt = 1 * u.ms
res_dt = 1 * u.ms

#tam14 = Tamborra_2014(progenitor_mass=20*u.solMass, direction=1)
t0, t1 = 0.0065017*u.s, 0.33801*u.s

# SN model
model = {'name': 'Tamborra_2014',
        'param':{'progenitor_mass': 20*u.Msun, 'direction': 1}}

# neutrino flavor mixing scheme and hierarchy
mixing_scheme = "NoTransformation" #"NoTransformation", "CompleteExchange", "AdiabaticMSW"
hierarchy = "normal" #"normal", "inverted"


# fourier transfer parameters
time_win = [0, 1] * u.s # time independent
freq_res = 1 * u.Hz 
freq_win = [75, 1E6] * u.Hz # freq independent
freq_fil =80 * u.Hz # freq to filter out for _signal_model
hanning = False


match scenario:

    case 0: # default
        mixing_scheme = "NoTransformation"
        hierarchy = "normal"
    case 1: # ana, 75 Hz < f < 85 Hz
        freq_win = [75, 85] * u.Hz
    case 2: # ana, 150 ms < t < 300 ms
        time_win = [150, 300] * u.ms
    case 3: # ana, 75 Hz < f < 85 Hz & 150 ms < t < 300 ms
        freq_win = [75, 85] * u.Hz
        time_win = [150, 300] * u.ms
    case 4: # mix, CompleteExchange
        mixing_scheme = "CompleteExchange"
    case 5: # mix, AdiabaticMSW & NH
        mixing_scheme = "AdiabaticMSW"
        hierarchy = "normal"
    case 6: # mix, AdiabaticMSW & IH
        mixing_scheme = "AdiabaticMSW"
        hierarchy = "inverted"
    case 7: # mod, Tamborra 2014, 27M, d1
        tam14 = Tamborra_2014(progenitor_mass=27*u.solMass, direction=1)
        t0, t1 = 0.0105 * u.s, 0.55162 * u.s
        model = {'name': 'Tamborra_2014',
                'param':{'progenitor_mass': 27*u.Msun, 'direction': 1}}
    case 8: # mod, Tamborra 2014, 27M, d2
        tam14 = Tamborra_2014(progenitor_mass=27*u.solMass, direction=2)
        t0, t1 = 0.0105 * u.s, 0.55162 * u.s
        model = {'name': 'Tamborra_2014',
                'param':{'progenitor_mass': 27*u.Msun, 'direction': 2}}
    case 9: # mod, Tamborra 2014, 27M, d3
        tam14 = Tamborra_2014(progenitor_mass=27*u.solMass, direction=3)
        t0, t1 = 0.0105 * u.s, 0.55162 * u.s
        model = {'name': 'Tamborra_2014',
                'param':{'progenitor_mass': 27*u.Msun, 'direction': 3}}   

sim = Simulation(model=model,
                distance=distance, 
                res_dt=res_dt,
                Emin=0*u.MeV, Emax=100*u.MeV, dE=1*u.MeV,
                tmin=t0, tmax=t1, dt=sim_dt,
                hierarchy = hierarchy,
                mixing_scheme = mixing_scheme,
                detector_scope = detector_scope,
                add_wls = add_wls)
sim.run()

ft_para = {"time_res": res_dt, 
            "time_win": time_win,
            "freq_res": freq_res,
            "freq_win": freq_win,
            "freq_fil": freq_fil,
            "hanning": hanning}

para = {"model": model,
        "hierarchy": hierarchy,
        "mixing_scheme": mixing_scheme,
        "distance": distance,
        "ft_para": ft_para,
        "bkg_trials": bkg_trials,
        "bkg_bins": bkg_bins}

############################################################
#####################BACKGROUND TRIALS######################
############################################################

print("BACKGROUND TRIALS")
print("-------------------------")
print(f"model: {model['name']}, {model['param']['progenitor_mass']} M, dir={model['param']['direction']}")
print(f"model time: [{t0.to_value(u.s)},{t1.to_value(u.s)}] s")
print("mixing scheme: {}, hierarchy: {}".format(mixing_scheme, hierarchy))
print("background trials: {}, background bins: {}".format(bkg_trials, bkg_bins))
print(f"time cut: [{time_win[0].value},{time_win[1].value}] s")
print(f"freq cut: [{freq_win[0].value},{freq_win[1].value}] Hz")
print("distance: {}".format(distance))
print("-------------------------")

MODE = "QUANTILE"
bgt = Background_Trials(sim = sim, para = para, verbose = True)

if MODE == "GENERATE":
    bgt.generate(filename = None)
elif MODE == "QUANTILE":
    bgt.quantiles(distance_range = dist_range)