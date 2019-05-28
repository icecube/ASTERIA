from asteria import config, source, detector, IO
from asteria.interactions import Interactions
from asteria.neutrino import Flavor

import astropy.units as u

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ROOT import TFile, TH1D

Z = [ 0.02, 0.004 ]
M = [ 13.0, 20.0, 30.0, 50.0 ]
t_rev = [100, 200, 300]
names = []
titles = []
for z in Z:
    for m in M:
        if m == 30.0 and z == 0.004:            
            names.append('nakazato-shen-BH-z0.004-s30.0')
            names.append('nakazato-LS220-BH-z0.004-s30.0')
        else:
            for t in t_rev:
                names.append('nakazato-shen-z{0}-t_rev{1}ms-s{2}'.format(z, t, m))

## Set neutrino energy and time binning
E_min = 0.1; E_max = 100.1; dE = 0.1;
Enu = np.arange(E_min, E_max, dE)

t_min = -1; t_max = 15; dt = 0.0001;
time = np.arange(t_min, t_max, dt)

rFlavors = ['nu_e', 'anti_nu_e', 'nu_x', 'anti_nu_x']

for name in names:                    
    conf = config.load_config('../data/config/{0}.yaml'.format(name))

    E_per_V = IO.load(conf, Interactions, Flavor, Enu, time)
    total_E_per_V = np.sum( abs(E_per_V), axis=0 ) 

    preprocessed_file = TFile('nakazato/ROOT/{}.root'.format(name), 'recreate')
    h_total_E_per_V = TH1D('total_photonic_energy_distance_1kpc', name, time.size, t_min, t_max )
    h_E_per_V = [TH1D('{}_photonic_energy_distance_1kpc'.format(nu), name, time.size, t_min, t_max ) for nu in rFlavors]

    for t, epv in zip(time, total_E_per_V):
        h_total_E_per_V.Fill(t+0.5*dt, epv)

    for nu, flavor in enumerate(Flavor):
        for t, epv in zip(time, E_per_V[nu]):
            h_E_per_V[nu].Fill(t+0.5*dt, epv)
        
    h_total_E_per_V.SetDirectory(preprocessed_file)
    preprocessed_file.Write()
    preprocessed_file.Close()
