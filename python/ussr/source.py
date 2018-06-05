# -*- coding: utf-8 -*-
"""CCSN neutrino sources.

This module encapsulates the basic parameters of neutrino fluxes from
supernovae as modeled in the CCSN literature. For each species of neutrino one
requires an estimate of the luminosity vs. time as well as the energy spectrum
of the neutrinos at any given time.
"""

from __future__ import print_function, division

from .neutrino import Flavor
from .config import parse_quantity

from astropy import units as u
from astropy.table import Table

from scipy.interpolate import InterpolatedUnivariateSpline


class Source:
    
    def __init__(self, name, model, progenitor_mass, progenitor_distance,
                 luminosity={}, mean_energy={}, pinch={}):

        self.name = name
        self.model = model
        self.progenitor_mass = progenitor_mass
        self.progenitor_distance = progenitor_distance
        self.luminosity = luminosity
        self.mean_energy = mean_energy
        self.pinch = pinch

    def get_luminosity(self, t, flavor=Flavor.nu_e_bar):
        """Return source luminosity at time t for a given flavor.

        Parameters
        ----------
        
        t : float
            Time relative to core bounce.
        flavor : :class:`ussr.neutrino.Flavor`
            Neutrino flavor.

        Returns
        -------
        luminosity : float
            Source luminosity (units of power).
        """
        return luminosity[flavor](t)

    def get_flux(self, t, flavor=Flavor.nu_e_bar):
        """Return source flux at time t for a given flavor.

        Parameters
        ----------
        
        t : float
            Time relative to core bounce.
        flavor : :class:`ussr.neutrino.Flavor`
            Neutrino flavor.

        Returns
        -------
        flux : float
            Source flux.
        """
        return luminosity[flavor](t) / self.progenitor_distance**2


def initialize(config):
    """Initialize a Source model from configuration parameters.

    Parameters
    ----------

    config : :class:`ussr.config.Configuration`
        Configuration parameters used to create a Source.

    Returns
    -------
    Source
        An initialized source model.
    """
    # Dictionary of L, <E>, and alpha versus time, keyed by neutrino flavor.
    luminosity, mean_energy, pinch = {}, {}, {}

    if config.source.table.format.lower() == 'fits':
        # Open FITS file, which contains a luminosity table and a pinching
        # parameter (alpha) and mean energy table.
        fitsfile = '/'.join([config.abs_base_path, config.source.table.path])
        sn_data_table = Table.read(fitsfile)

        t = sn_data_table['TIME'].to('s')

        # Loop over all flavors in the table:
        for flavor in Flavor:
            fl = flavor.name.upper()

            L = sn_data_table['L_{:s}'.format(fl)].to('erg/s')
            E = sn_data_table['E_{:s}'.format(fl)].to('MeV')
            alpha = sn_data_table['ALPHA_{:s}'.format(fl)]

            luminosity[flavor] = InterpolatedUnivariateSpline(t, L)
            mean_energy[flavor] = InterpolatedUnivariateSpline(t, E)
            pinch[flavor] = InterpolatedUnivariateSpline(t, alpha) 

    elif config.source.table.format.lower() == 'ascii':
        # ASCII will be supported! Promise, promise.
        raise ValueError('Unsupported format: "ASCII"')

    else:
        raise ValueError('Unknown format {}'.format(config.source.table.format))

    return Source(config.source.name,
                  config.source.model,
                  parse_quantity(config.source.progenitor.mass),
                  parse_quantity(config.source.progenitor.distance),
                  luminosity,
                  mean_energy,
                  pinch)
