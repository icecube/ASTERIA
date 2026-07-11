# -*- coding: utf-8 -*-
"""Stellar mass distribution.

Use this model to produce radial distributions of stellar mass densities with
respect to the solar system.
"""

from __future__ import print_function, division

from abc import ABC, abstractmethod

import numpy as np
from astropy.io import fits
from astropy import units as u
from scipy.stats import norm
from scipy.interpolate import PchipInterpolator


class Distance(ABC):
    """Abstract base class for generating progenitor distance(s).
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate_distance(self, size=1):
        """Generate distance to a progenitor.

        Parameters
        ----------
        size : int
            Number of distances to generate.

        Returns
        -------
        distance : ndarray
            Distance(s) to CCSN progenitor.
        """
        pass

    @abstractmethod
    def __str__(self):
        """Express as a string."""
        pass


class FixedDistance(Distance):
    """Generate fixed distances for the progenitor."""

    def __init__(self, d, sigma=None):
        """Progenitor distance with some uncertainty.
        
        Parameters
        ----------
        d : :class:`astropy.units.quantity.Quantity`
            Distance to progenitor.
        sigma : :class:`astropy.units.quantity.Quantity`
            Gaussian uncertainty in distance.
        """
        super().__init__()
        self.dist = d
        self.sigma = sigma

    def generate_distance(self, size=1):
        """Generate distance to a progenitor.

        Parameters
        ----------
        size : int
            Number of distances to generate.

        Returns
        -------
        distance : ndarray
            Distance(s) to CCSN progenitor.
        """
        if self.sigma is not None:
            d = np.random.normal(self.dist.value, self.sigma.value, size)
        else:
            d = np.full(size, self.dist.value)

        return d * self.dist.unit

    def __str__(self):
        """Express as a string.

        Returns
        -------
        s : str
            String representation of FixedDistance.
        """
        s = 'Fixed Distance:\n- distance = {}'.format(self.dist)
        if self.sigma is not None:
            s += '\n- sigma    = {}'.format(self.sigma)
        return s


class StellarDensity(Distance):
    """Generate distances according to a Sun-centric radial mass density."""

    def __init__(self, distcdf_file, add_LMC=False, add_SMC=False, m_MW=2.012e10*u.Msun, m_LMC=2.7e9*u.Msun, m_SMC=3.1e8*u.Msun):
        """Progenitor distance with some uncertainty.
        
        Parameters
        ----------
        distcdf_file : str
            File with cumulative radial stellar mass distribution w.r.t. Sun.
        add_LMC : bool
            If true, add a Gaussian model of the LMC stellar distribution.
        add_SMC : bool
            If true, add a Gaussian model of the SMC stellar distribution.
        m_MW : Quantity
            Stellar mass of the Milky Way, excluding remnants. Default value is from Lian+ ApJL 37:990, 2025.
        m_LMC : Quantity
            Stellar mass of the LMC. Default value from van der Marel+ Proc.  IAU Symp 256 4:81, 2009 (arXiv:0809.4268).
        m_SMC : Quantity
            Stellar mass of the SMC. Default value from van der Marel+ Proc.  IAU Symp 256 4:81, 2009 (arXiv:0809.4268).
        """
        super().__init__()

        hdu = fits.open(distcdf_file)
        distunit = u.Unit(hdu['DIST'].header['BUNIT'])
        self._dist = hdu['DIST'].data
        self._cdf = hdu['CDF'].data
        self._name = hdu['CDF'].header['NAME']
        self._publication = hdu['CDF'].header['PUB']
        self._use_LMC = add_LMC
        self._use_SMC = add_SMC

        if add_LMC:
            #- Treat the LMC as a Gaussian blob centered 50 kpc from Earth
            r_LMC = (50*u.kpc).to(distunit).value
            sigma_LMC = (2.5*u.kpc).to(distunit).value

            mratio = float(m_LMC/m_MW)
            dist_LMC = np.linspace(r_LMC-5*sigma_LMC, r_LMC+5*sigma_LMC, 51)
            cdf_LMC = mratio * norm.cdf(dist_LMC, r_LMC, sigma_LMC)

            self._dist = np.append(self._dist, dist_LMC)
            self._cdf = np.append(self._cdf, 1. + cdf_LMC)

            #- Cut repeat entries and ensure monotonicity in the CDF
            self._dist, idx = np.unique(self._dist[self._dist.argsort(kind='stable')], return_index=True)
            self._cdf = self._cdf[idx] / np.max(self._cdf[idx])
        if add_SMC:
            #- Treat the LMC as a Gaussian blob centered 60 kpc from Earth
            r_SMC = (60*u.kpc).to(distunit).value
            sigma_SMC = (1.25*u.kpc).to(distunit).value

            mratio = float(m_SMC/m_MW)
            dist_SMC = np.linspace(r_SMC-5*sigma_SMC, r_SMC+5*sigma_SMC, 51)
            cdf_SMC = mratio * norm.cdf(dist_SMC, r_SMC, sigma_SMC)

            self._dist = np.append(self._dist, dist_SMC)
            self._cdf = np.append(self._cdf, 1. + cdf_SMC)

            #- Cut repeat entries and ensure monotonicity in the CDF
            self._dist, idx = np.unique(self._dist[self._dist.argsort(kind='stable')], return_index=True)
            self._cdf = self._cdf[idx] / np.max(self._cdf[idx])

        self._inv_cdf = PchipInterpolator(self._cdf, self._dist)

        #- Set distance in distance units
        self._dist *= distunit

    def generate_distance(self, size=1):
        """Generate distance to a progenitor.

        Parameters
        ----------
        size : int
            Number of distances to generate.

        Returns
        -------
        distance : ndarray
            Distance(s) to CCSN progenitor.
        """
        u = np.random.uniform(0.,1., size)
        return self._inv_cdf(u) * self._dist.unit

    def __str__(self):
        """Express as a string.

        Returns
        -------
        s : str
            String representation of StellarDensity.
        """
        s = 'Stellar density:\n'
        s += '- model   = {}\n'.format(self._name)
        s += '- paper   = {}\n'.format(self._publication)
        s += '- use LMC = {}\n'.format(self._use_LMC)
        s += '- use SMC = {}'.format(self._use_SMC)
        return s

    @property
    def distance(self):
        return self._dist

    @property
    def cdf(self):
        return self._cdf

    @property
    def name(self):
        return self._name

    @property
    def publiation(self):
        return self._publication

    @property
    def use_LMC(self):
        return self._use_LMC

    @property
    def use_SMC(self):
        return self._use_SMC
