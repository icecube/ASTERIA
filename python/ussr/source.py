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

import numpy as np
from scipy.special import loggamma, gdtr
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

        # Energy PDF function is assumed to be like a gamma function,
        # parameterized by mean energy and pinch parameter alpha. True for
        # nearly all CCSN models.
        self.energy_pdf = lambda a, Ea, E: \
            np.exp((1 + a) * np.log(1 + a) - loggamma(1 + a) + a * np.log(E) - \
                   (1 + a) * np.log(Ea) - (1 + a) * (E / Ea))

        # Energy CDF, useful for random energy sampling.
        self.energy_cdf = lambda a, Ea, E: gdtr(1., a + 1., (a + 1.) * (E / Ea))

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
        return self.luminosity[flavor](t) * (u.erg / u.s)

    def get_mean_energy(self, t, flavor=Flavor.nu_e_bar):
        """Return source mean energy at time t for a given flavor.

        Parameters
        ----------
        
        t : float
            Time relative to core bounce.
        flavor : :class:`ussr.neutrino.Flavor`
            Neutrino flavor.

        Returns
        -------
        mean_energy : float
            Source mean energy (units of energy).
        """
        return self.mean_energy[flavor](t) * u.MeV


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
        return self.luminosity[flavor](t) / self.mean_energy[flavor](t)  * (u.erg.to( u.MeV ) / u.s)
        #return self.luminosity[flavor](t) / self.progenitor_distance**2

    def energy_spectrum(self, t, E, flavor=Flavor.nu_e_bar):
        """Compute the PDF of the neutrino energy distribution at time t.

        Parameters
        ----------

        t : float
            Time relative to core bounce.
        flavor : :class:`ussr.neutrino.Flavor`
            Neutrino flavor.
        E : `numpy.ndarray`
            Sorted grid of neutrino energies to compute the energy PDF.

        Returns
        -------
        spectrum : `numpy.ndarray`
            Table of PDF values computed as a function of energy.
        """
        # Given t, get current average energy and pinch parameter.
        # Use simple 1D linear interpolation
        a = self.pinch[flavor](t)
        Ea = self.mean_energy[flavor](t)
        if a <= 0. or Ea <= 0.:
            return np.zeros_like(E)
        if E[0] == 0.:
            E[0] = 1e-10 * u.MeV
        return self.energy_pdf(a, Ea, E.value).real

    def sample_energies(self, t, E, n=1, flavor=Flavor.nu_e_bar):
        """Generate a random sample of neutrino energies at some time t for a
        particular neutrino flavor. The energies are generated via inverse
        transform sampling of the CDF of the neutrino energy distribution.

        Parameters
        ----------

        t : float
            Time relative to core bounce.
        E : `numpy.ndarray`
            Sorted grid of neutrino energies to compute the energy PDF.
        n : int
            Number of energy samples to produce.
        flavor : :class:`ussr.neutrino.Flavor`
            Neutrino flavor.

        Returns
        -------
        energies : `numpy.ndarray`
            Table of energies sampled from the energy spectrum.
        """
        cdf = self.energy_cdf(flavor, t, E)
        energies = np.zeros(n, dtype=float)

        # Generate a random number between 0 and 1 and compare to the CDF
        # of the neutrino energy distribution at time t
        u = np.random.uniform(n)
        j = np.searchsorted(cdf, u)

        # Linearly interpolate in the CDF to produce a random energy
        energies[j <= 0] = E[0].to('MeV').value
        energies[j >= len(E)-1] = E[-1].to('MeV').value

        cut = (0 < j) & (j < len(E)-1)
        j = j[cut]
        en = E[j] + (E[j+1] - E[j]) / (cdf[j+1] - cdf[j]) * (u[cut] - cdf[j])
        energies[cut] = en

        return energies


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
