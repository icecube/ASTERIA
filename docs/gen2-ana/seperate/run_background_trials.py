import os
import sys
from astropy.table import Table
from asteria.simulation import Simulation
from background_trials import *


# parsed arguments: distance and sample size
ind_dist = int(sys.argv[1])
bkg_trials = int(sys.argv[2])
bkg_bins = int(2E4)

dist_min, dist_max, dist_step = 20, 100, 0.2
dist_range = np.arange(dist_min, dist_max + dist_step, dist_step) * u.kpc
dist_range = np.round(dist_range, 1)
distance = dist_range[ind_dist]

############################################################
######################SIMULATION SETUP######################
############################################################

# detector scope
add_wls = True
detector_scope = "Gen2"

# time resolution
sim_dt = 0.1 * u.ms
res_dt = 0.1 * u.ms


model_name = "RDF_1_2"

if model_name == "RDF_1_2":
    file_path = "~/.astropy/cache/snewpy/models/RDF_1_2/RDF_1_2.dat"
    smoothing_frequency = 50 * u.Hz

elif model_name == "RDF_1_7":
    file_path = "~/.astropy/cache/snewpy/models/RDF_1_7/RDF_1_7.dat"
    smoothing_frequency = 500 * u.Hz

tab = Table().read(file_path, format = "ascii")

# SN model
model = {
    'name': 'Analytic3Species',
    'param': {
        'filename': file_path
    }
}

# neutrino flavor mixing scheme and hierarchy
mixing_scheme = "NoTransformation" #"NoTransformation", "CompleteExchange", "AdiabaticMSW"
hierarchy = "normal" #"normal", "inverted"

sim = Simulation(model=model,
                distance=distance, 
                res_dt=res_dt,
                Emin=0*u.MeV, Emax=100*u.MeV, dE=1*u.MeV,
                tmin=tab['TIME'][0]*u.s, tmax=tab['TIME'][-1]*u.s, dt=sim_dt,
                hierarchy = hierarchy,
                mixing_scheme = mixing_scheme,
                detector_scope = detector_scope,
                add_wls = add_wls)
sim.run()

############################################################
#######################ANALYSIS SETUP#######################
############################################################

mode = "AMP"
time_win = [0, 10] * u.s

para = {"mode": mode,
        "model": model_name,
        "hierarchy": hierarchy,
        "mixing_scheme": mixing_scheme,
        "distance": distance,
        "time_win": time_win,
        "smoothing_frequency": smoothing_frequency,
        "bkg_trials": bkg_trials,
        "bkg_bins": bkg_bins}

############################################################
#####################BACKGROUND TRIALS######################
############################################################

print("BACKGROUND TRIALS")
print("-------------------------")
print("mode: {}".format(mode))
print("model: {}".format(model))
print("mixing scheme: {}, hierarchy: {}".format(mixing_scheme, hierarchy))
print("distance: {}".format(distance))
print("time window: {}".format(time_win))
print("smoothing frequency: {}".format(smoothing_frequency))
print("background trials: {}, background bins: {}".format(bkg_trials, bkg_bins))
print("-------------------------")

MODE = "GENERATE"
bgt = Background_Trials(sim = sim, para = para, verbose = True)

if MODE == "GENERATE":
    bgt.generate(filename = None)
elif MODE == "QUANTILE":
    bgt.quantiles(distance_range = dist_range)