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

    def __init__(self, distcdf_file):
        """Progenitor distance with some uncertainty.
        
        Parameters
        ----------
        distcdf_file : str
            File with cumulative radial stellar mass distribution w.r.t. Sun.
        """
        super().__init__()

        hdu = fits.open(distcdf_file)
        self.dist = hdu['DIST'].data * u.Unit(hdu['DIST'].header['BUNIT'])
        self.cdf = hdu['CDF'].data
        self.name = hdu['CDF'].header['NAME']
        self.publication = hdu['CDF'].header['PUB']

        self._inv_cdf = PchipInterpolator(hdu['CDF'].data, hdu['DIST'].data)

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

