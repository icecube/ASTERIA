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

from importlib.resources import files
from argparse import ArgumentParser


def parse(options=None):
    """Parse command line options.
    """
    if not 'ASTERIA' in os.environ:
        raise SystemExit('Missing environment variable ASTERIA.')

    p = ArgumentParser(description='IceCube Fast CCSN Simulator')
    p.add_argument('-c', '--config', dest='config',
                   default=files('asteria.etc').joinpath('default.yaml'),
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

    # Define neutrino energy range.
    E_min = 0.1; E_max = 100.1; dE = 0.1;
    Enu = np.arange(E_min, E_max, dE) * u.MeV

    # Define post-bounce times at which to evaluate
    t_min = -1; t_max = 15; dt = 0.0001;
    time = np.arange(t_min, t_max, dt) * u.s

	# Compute charged particle spectra.
    photon_spectra = np.zeros(shape=(len(Flavor), Enu.size))

    for nu, flavor in enumerate(Flavor):
        for interaction in Interactions: 
            xs    = interaction.cross_section(flavor, Enu).to(u.m**2).value
            E_lep = interaction.mean_lepton_energy(flavor, Enu).value
            scale = interaction.photon_scaling_factor(flavor).to(1/u.MeV).value
            photon_spectra[nu] +=  xs * E_lep * scale # u.m**2 
    photon_spectra *= u.m**2

    # Compute signal per DOM.
    E_per_V = np.zeros( shape=(len(Flavor), time.size) )
    signal_per_DOM = np.zeros_like(E_per_V)
    
    icecube_dt = 2e-3 #s
    
    effvol = 0.1654 * u.m**3 / u.MeV #Simple estimation of IceCube DOM Eff. Vol.
    
    for nu, (flavor, ph_spec) in enumerate(zip(Flavor, photon_spectra)):
        E_per_V[nu] = ccsn.photonic_energy_per_vol(time, Enu, flavor, ph_spec)
        
    signal_per_DOM = effvol * E_per_V

    # Save simulation to file.
    E_per_V_1kpc =  E_per_V * ccsn.progenitor_distance.to(u.kpc).value**2
    io.save(conf, Interactions, Flavor, Enu.value, time.value, E_per_V_1kpc)

