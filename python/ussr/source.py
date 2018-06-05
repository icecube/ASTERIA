# -*- coding: utf-8 -*-
"""CCSN neutrino sources.

This module encapsulates the basic parameters of neutrino fluxes from
supernovae as modeled in the CCSN literature. For each species of neutrino one
requires an estimate of the luminosity vs. time as well as the energy spectrum
of the neutrinos at any given time.
"""

from __future__ import print_function, division

from .neutrino import Flavor

from astropy import units as u

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
