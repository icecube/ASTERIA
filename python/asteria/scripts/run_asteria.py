"""
asteria.scripts.run_asteria
===========================
"""
from __future__ import absolute_import, division, print_function

from asteria.config import load_config, Configuration
from asteria.source import initialize, Source
from asteria.neutrino import Flavor
from asteria.interactions import Interactions
import asteria.IO as io

import numpy as np
from astropy import units as u

import os
from argparse import ArgumentParser


def parse(options=None):
    """Parse command line options.
    """
    if not 'ASTERIA' in os.environ:
        raise SystemExit('Missing environment variable ASTERIA.')

    p = ArgumentParser(description='IceCube Fast CCSN Simulator')
    p.add_argument('-c', '--config', dest='config',
                   default='{}/data/config/default.yaml'.format(
                       os.environ['ASTERIA']),
                   help='YAML configuration file.')
    # 
    # CCSN source fluxes and spectra.
    #
    src = p.add_argument_group('CCSN source')
    src.add_argument('--srcname', dest='srcname', required=False,
                      help='Source name.')
    src.add_argument('--srcmodel', dest='srcmodel', required=False,
                     help='Source model description.')
    src.add_argument('--srcmass', dest='srcmass', type=float, required=False,
                     help='Progenitor mass [M_sun].')
    #
    # CCSN source: add mutually exclusive distance models.
    #
    dist = src.add_mutually_exclusive_group(required=False)
    dist.add_argument('--fixed-dist', dest='fixed_dist', nargs=2, type=float,
                      metavar=('DIST', 'DISTERR'),
                      help='Fixed distance and uncertainty [kpc].')
    dist.add_argument('--stellar-dist', dest='stellar_dist', nargs=3,
                      metavar=('PATH', 'ADDLMC', 'ADDSMC'),
                      help='Path to stellar radial density profile.')
    src.add_argument('--srcpath', dest='srcpath', required=False, 
                     help='Path to source file (relative to ASTERIA/data).')
    #
    # Output options.
    #
    out = p.add_argument_group('Output')
    out.add_argument('--outformat', dest='outformat', required=False,
                     help='Output file format, e.g., h5.')
    out.add_argument('--outfile', dest='outfile', required=False,
                     help='Path to output file.')

    if options is None:
        args = p.parse_args()
    else:
        args = p.parse_args(options)

    # Load configuration into a dictionary.
    # Overload the values using the command-line options.
    conf = load_config(args.config, dict)

    if args.srcname:
        conf['source']['name'] = args.srcname
    if args.srcmodel:
        conf['source']['model'] = args.srcmodel
    if args.srcpath:
        conf['source']['table']['path'] = args.srcpath
    if args.srcmass:
        conf['source']['progenitor']['mass'] = '{} M_sun'.format(args.srcmass)

    if args.fixed_dist:
        conf['source']['progenitor']['distance'] = {
            'model' : 'FixedDistance',
            'distance' : '{} kpc'.format(args.fixed_dist[0]),
            'uncertainty' : '{} kpc'.format(args.fixed_dist[1]),
        }
    elif args.stellar_dist:
        conf['source']['progenitor']['distance'] = {
            'model' : 'StellarDensity',
            'path' : '{}'.format(args.stellar_dist[0]),
            'add_LMC' : '{}'.format(args.stellar_dist[1]),
            'add_SMC' : '{}'.format(args.stellar_dist[2]),
        }
    if args.outformat:
        conf['IO']['table']['format'] = args.outformat
    if args.outfile:
        conf['IO']['table']['path'] = args.outfile

    # Make options available to the simulation.
    return Configuration(conf)


def main(args=None):
    if isinstance(args, (list, tuple, type(None))):
        conf = parse(args)

    # Create the CCSN source.
    ccsn = initialize(conf)

    # Prepare energy and time range.
    Enu  = np.arange(0.1, 100.1, 0.1) * u.MeV
    time = np.arange(-1, 15, 0.001) * u.s

    # Compute photons from charged particle interactions.
    ph_spec = np.zeros(shape=(len(Flavor), Enu.size))

    for nu, flavor in enumerate(Flavor):
        for interaction in Interactions:
            xs = interaction.cross_section(flavor, Enu).to('m**2').value
            E_lep = interaction.mean_lepton_energy(flavor, Enu).value
            scale = interaction.photon_scaling_factor(flavor).to('1/MeV').value

            # Photon spectra per flavor in units of m**2.
            ph_spec[nu] += xs * E_lep * scale
    ph_spec *= u.m**2

    # Compute signal per DOM.
    E_per_V = np.zeros(shape=(len(Flavor), time.size))
    signal_per_DOM = np.zeros_like(E_per_V)

    ic_dt = 0.002                     # 2 ms bins
    effvol = 0.1654 * u.m**3 / u.MeV  # simple estimate of DOM effective vol.

    for nu, (flavor, ph_spectrum) in enumerate(zip(Flavor, ph_spec)):
        E_per_V[nu] = ccsn.photonic_energy_per_vol(time, Enu, flavor, ph_spec)

    # Save simulation to file, scaling results to 1 kpc distance.
    E_per_V_1kpc = E_per_V * ccsn.progenitor_distance.to(u.kpc).value**2
    io.save(conf, Interactions, Flavor, Enu.value, time.value, E_per_V_1kpc)
