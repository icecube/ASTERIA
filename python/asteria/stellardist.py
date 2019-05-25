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
    """Basic abstrct class to generate progenitor distance(s).
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def distance(self, size=1):
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

    def distance(self, size=1):
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
            d = np.full(size, self.dist.value) * self.dist.unit

        return d * self.dist.unit


class StellarDensity(Distance):
    """Generate distances according to a Sun-centric radial mass density."""

    def __init__(self, distcdf_file, add_LMC=False, add_SMC=False):
        """Progenitor distance with some uncertainty.
        
        Parameters
        ----------
        distcdf_file : str
            File with cumulative radial stellar mass distribution w.r.t. Sun.
        add_LMC : bool
            If true, add a Gaussian model of the LMC stellar distribution.
        add_SMC : bool
            If true, add a Gaussian model of the SMC stellar distribution.
        """
        super().__init__()

        hdu = fits.open(distcdf_file)
        distunit = u.Unit(hdu['DIST'].header['BUNIT'])
        self.dist = hdu['DIST'].data
        self.cdf = hdu['CDF'].data
        self.name = hdu['CDF'].header['NAME']
        self.publication = hdu['CDF'].header['PUB']

        # Virial mass of the MW is ~1.5e12 Msun; see L. Watkins et al., 
        # ApJ 873:118 (2019), arXiv:1804.11348 [astro-ph.GA].
        # Assume 5% of this is stored in stellar mass (educated guess).
        m_MW = 0.05 * 1.5e12

        if add_LMC:
            # Combined visible mass of LMC disk and neutral gas; see 
            # Kinematical Structure of the Magellanic System,
            # R. P. van der Marel, N. Kallivayalil, G. Besla, Proc. IAU Symp.
            # 256, 2009, 81-92, arXiv:0809.4268 [astro-ph].
            m_LMC = 3.2e9
            r_LMC = (50*u.kpc).to(distunit).value
            sigma_LMC = (2.5*u.kpc).to(distunit).value

            mratio = m_LMC/m_MW
            dist_LMC = np.linspace(r_LMC-5*sigma_LMC, r_LMC+5*sigma_LMC, 51)
            cdf_LMC = mratio * norm.cdf(dist_LMC, r_LMC, sigma_LMC)

            self.dist = np.append(self.dist, dist_LMC)
            self.cdf = np.append(self.cdf, 1. + cdf_LMC)
            self.cdf /= np.max(self.cdf)
        if add_SMC:
            # Total stellar mass of the SMC is estimated to be 3.1x10^8 Msun;
            # R. P. van der Marel, N. Kallivayalil, G. Besla, Proc. IAU Symp.
            # 256, 2009, 81-92, arXiv:0809.4268 [astro-ph].
            m_SMC = 3.1e8
            r_SMC = (60*u.kpc).to(distunit).value
            sigma_SMC = (1.25*u.kpc).to(distunit).value

            mratio = m_SMC/m_MW
            dist_SMC = np.linspace(r_SMC-5*sigma_SMC, r_SMC+5*sigma_SMC, 51)
            cdf_SMC = mratio * norm.cdf(dist_SMC, r_SMC, sigma_SMC)

            self.dist = np.append(self.dist, dist_SMC)
            self.cdf = np.append(self.cdf, 1. + cdf_SMC)
            self.cdf /= np.max(self.cdf)

        self._inv_cdf = PchipInterpolator(self.cdf, self.dist)

        # Set distance in distance units.
        self.dist *= distunit

    def distance(self, size=1):
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
        return self._inv_cdf(u) * self.dist.unit

