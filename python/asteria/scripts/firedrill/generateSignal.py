
from asteria import config, source, detector, IO
from asteria.interactions import Interactions
from asteria.neutrino import Flavor, Ordering
from asteria.oscillation import SimpleMixing

import numpy as np
from numpy.lib import recfunctions as rfn

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

import astropy.units as u

from os import path

import xml.etree.ElementTree as ET

def set_mpl_format():
    # During plt.savefig(), the following error was thrown
    #
    #       ! LaTeX Error: File `type1ec.sty' not found.
    #
    # Fix (linux): Install apt install cm-super
    # Source: https://github.com/matplotlib/matplotlib/issues/16911
    # TODO: Fix missing font warning
    axes_style = {'grid': 'True',
                  'labelsize': '20',
                  'labelpad': '8.0'}

    xtick_style = {'direction': 'out',
                   'labelsize': '18.',
                   'major.size': '5.',
                   'major.width': '1.',
                   'minor.visible': 'True',
                   'minor.size': '2.5',
                   'minor.width': '1.'}

    ytick_style = {'direction': 'out',
                   'labelsize': '18.',
                   'major.size': '5',
                   'major.width': '1.',
                   'minor.visible': 'True',
                   'minor.size': '2.5',
                   'minor.width': '1.'}

    grid_style = {'alpha': '0.75'}
    legend_style = {'fontsize': '16'}
    font_syle = {'size': '20',
                 'family': 'sans-serif'}
    text_style = {'usetex': 'True'}
    # math_style =   {         'fontset' : 'cm' }
    figure_style = {'subplot.hspace': '0.05'}

    mpl.rc('font', **font_syle)
    mpl.rc('text', **text_style)
    # mpl.rc('mathtext', **math_style )
    mpl.rc('axes', **axes_style)
    mpl.rc('xtick', **xtick_style)
    mpl.rc('ytick', **ytick_style)
    mpl.rc('grid', **grid_style)
    mpl.rc('legend', **legend_style)
    mpl.rc('figure', **figure_style)

    # mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'cm'


def compute_photon_spectra(Flavors, Enu):
    photon_spectra = np.zeros(shape=(len(Flavors), Enu.size))

    for nu, flavor in enumerate(Flavors):
        for interaction in Interactions:
            xs = interaction.cross_section(flavor, Enu).to(u.m ** 2).value
            E_lep = interaction.mean_lepton_energy(flavor, Enu).value
            photon_scaling_factor = interaction.photon_scaling_factor(flavor).value

            photon_spectra[nu] += xs * E_lep * photon_scaling_factor

    photon_spectra *= u.m ** 2
    return photon_spectra


def dom_signal(time, total_E_per_V, dt, dom):
    deadtime = 0.25e-3  # s
    dc_rel_eff = 1.35

    i3_dom_bg_mu = 284.9
    i3_dom_bg_sig = 26.2

    if dom['type'] == b'dc':
        eps_i3 = 0.87 / (1 + deadtime * total_E_per_V * dc_rel_eff / dt)
    elif dom['type'] == b'i3':
        eps_i3 = 0.87 / (1 + deadtime * total_E_per_V / dt)

    return total_E_per_V * dom['effvol'] * eps_i3


def dom_hits(time, total_E_per_V, dt, dom):
    return np.random.poisson(dom_signal(time, total_E_per_V, dt, dom))


def rebin(var, data, old_binning, new_binning):
    step = int(new_binning / old_binning)
    new_size = int(data.size / step)

    rebinned_data = np.sum(np.split(data, new_size), axis=1)
    rebinned_var = var.value[int(0.5 * step)::step]

    return rebinned_var, rebinned_data


def generate_hits(dom, time, total_E_per_V, dt, new_dt):
    hit_list = []
    hits = dom_hits(time, total_E_per_V, dt, dom)
    hit_list.append(hits)

    cut = np.where(hits > 0)[0]
    if time[cut][-1] == time[-1]:
        cut = np.delete(cut, -1)
        # Prevents loop ahead from breaking if last time bin has a hit
        # This hit is dropped
        # TODO: Find a better way of doing this!!

    t_hit = []
    # Time by default is binned at 0.1ms, should be 25ns for final sim.
    for count, t0, tf in zip(hits[cut], time[cut].value, time[cut + 1].value):
        for hit in range(count):
            t_hit.append(int((t0 + np.random.uniform() * (tf - t0)) * 1e9))
    t_hit = np.asarray(t_hit)  # Maintain units ns
    return t_hit


def get_DOM_id_table():
    """ Gets Table generated in get_dom_ids.py
    TODO: Implement dom ID in ASTERIA DOM Table to avoid this!
    """
    dtypes = [('str', 'f4'),
              ('i', 'f4'),
              ('x', 'f8'),
              ('y', 'f8'),
              ('z', 'f8'),
              ('mbid', 'S15'),
              ('type', 'S5'),
              ('effvol', 'f8')]
    doms = np.genfromtxt('./data/full_dom_table.txt', dtype=dtypes)
    return doms


def main():
    set_mpl_format()
    E_min = 0.1;
    E_max = 100.1;
    dE = 0.1;
    Enu = np.arange(E_min, E_max, dE) * u.MeV
    mix = SimpleMixing(33.2)
    order = Ordering.normal

    t_min = -1;
    t_max = 1;
    dt = 0.0001;
    time = np.arange(-1e6, 1e6, 100) * 1e-6 * u.s

    conf = config.load_config('../ASTERIA/data/config/nakazato-shen-z0.02-t_rev300ms-s13.0.yaml')
    ccsn = source.initialize(conf)
    i3 = detector.initialize(conf)

    photon_spectra = compute_photon_spectra(Flavor, Enu)

    doms = i3.doms_table()
    i3effvol = doms['effvol'][doms['type'] == 'i3']
    dceffvol = doms['effvol'][doms['type'] == 'dc']

    photon_spectra = compute_photon_spectra(Flavor, Enu)

    print('Running Simulation for {0} with mass hierarchy {1}'.format(ccsn.name, order.name))

    ## Compute Energy deposition per volume
    E_per_V = np.zeros(shape=(len(Flavor), time.size))
    total_E_per_V = np.zeros(time.size)

    for nu, (flavor, spectrum) in enumerate(zip(Flavor, photon_spectra)):
        # assignment ot unitless pre-allocated array results in loss of units.
        E_per_V[nu] = abs(ccsn.photonic_energy_per_vol(time, Enu, flavor, spectrum, mix.normal_mixing))
        total_E_per_V += E_per_V[nu]

    i = 0
    n_strings = 86
    doms = get_DOM_id_table()
    for string in range(1, n_strings+1):
        filename = './data/hits/ichub{0:02d}/signal.dat'.format(string)
        if not os.path.exists('./data/hits/ichub{0:02d}'.format(string)):
            os.makedirs('./data/hits/ichub{0:02d}'.format(string))

        with open('./data/hits/ichub{0:02d}/signal.dat'.format(string), 'w') as f:
            for dom in doms[doms['str'] == string]:
                dom_hits = generate_hits(dom, time, total_E_per_V, dt, dt)

                for hit in dom_hits:
                    f.write('{0:<18d}{1:<18s}\n'.format(hit, str(dom['mbid'])[2:-1]))

                print('{0:<3d} hits accumulated in DOM {1:d}-{2:d} / {3}'.format(
                    dom_hits.size, int(dom['str']), int(dom['i']), str(dom['mbid'])[2:-1]
                ))

        dtypes = [('time', 'i'),
                  ('mbid', 'U12')]
        detector_hits = np.genfromtxt(filename, dtype=dtypes)
        detector_hits = np.sort(detector_hits, order='time')

        with open(filename, 'w') as f:
            for row in detector_hits:
                f.write('{0:<18d}{1:<18s}\n'.format(row['time'], str(row['mbid'])))

if __name__ == "__main__":
    # At time of Writing 04/14/20, must be run against ASTERIA/detailed-osc for fix to oscillated spectrum generation.
    # TODO: Merge changes from branch to master
    main()
