# -*- coding: utf-8 -*-
"""Module for Earth and stellar density models.
"""

import numpy as np
from astropy import units as u

from abc import ABC, abstractmethod


class Body(ABC):
    """Base class defining the interface for matter interactions."""
    
    def __init__(self):
        pass

    @abstractmethod
    def density(self, r):
        """Return density as a function of distance from the core."""
        pass

    @abstractmethod
    def y_e(self, r):
        """Return electron fraction as a function of distance from the core."""
        pass


class PREM(Body):
    """Preliminary Reference Earth Model: A. M. Dziewonski and D. L. Anderson,
    PEPI 25:297-356, 1981.
    """

    def __init__(self):
        # Radial bin edges, in units of km.
        self._rbins = np.asarray(
                [   0.0, 1221.5, 3480.0, 3630.0, 5701.0, 5771.0,
                 5971.0, 6151.0, 6346.6, 6356.0, 6368.0, 6371.0])
        
        # Density coefficients in units of kg/m3, kg/m4, kg/m5.
        self._dcoef = np.asarray(
                [[1.3088e4,  1.9110e-8, -2.1773e-10],
                 [1.2346e4,  1.3976e-4, -2.4123e-10],
                 [7.3067e3, -5.0007e-4,  0.0000],
                 [6.7823e3, -2.4441e-4, -3.0922e-11],
                 [5.3197e3, -2.3286e-4,  0.0000],
                 [1.1249e4, -1.2603e-3,  0.0000],
                 [7.1083e3, -5.9706e-4,  0.0000],
                 [2.6910e3,  1.0869e-4,  0.0000],
                 [2.9000e3,  0.0000,     0.0000],
                 [2.6000e3,  0.0000,     0.0000],
                 [1.0200e3,  0.0000,     0.0000],
                 [0.0000,    0.0000,     0.0000]])

        # Electron fraction values: inner/outer core and mantle.
        self._ye = np.asarray([ 0.4656, 0.4957 ])

    def density(self, r):
        """Return density as a function of distance from the core.

        Parameters
        ----------
        r : float or ndarray
            Radial distance from core, in length units (astropy).

        Returns
        -------
        rho : float or ndarray
            Density at radial position(s) r.
        """
        index = np.digitize(r.to('km').value, self._rbins) - 1
        if type(r.value) in [list, np.ndarray]:
            rho = []
            for i, x in enumerate(r.to('m').value):
                j = index[i]
                c0, c1, c2 = self._dcoef[j]
                rho.append(1e-3 * (c0 + c1*x + c2*x**2))
            return np.asarray(rho) * u.g / u.cm**3
        else:
            x = r.to('m').value
            c0, c1, c2 = self._dcoef[index]
            return 1e-3 * (c0 + c1*x + c2*x**2) * u.g / u.cm**3
        
    def y_e(self, r):
        """Return electron fraction as a function of distance from the core.

        Parameters
        ----------
        r : float or ndarray
            Radial distance from core, in length units (astropy).

        Returns
        -------
        Y_e : float or ndarray
            Electron fraction at radial position(s) r.
        """
        idx = np.digitize(r.to('km').value, [0., 3480., 6371.]) - 1
        return self._ye[idx]


class SimpleEarth(Body):
    """A constant-density 13-layer approximation of the Preliminary Reference
    Earth Model: A. M. Dziewonski and D. L. Anderson, PEPI 25:297-356, 1981.
    """

    def __init__(self):
        # Radial bin edges, in units of km.
        self._rbins = np.asarray(
            [0.0, 1221.5, 1786.125, 2350.75 , 2915.375, 3480.0,
                  4220.3, 4960.7, 5701.0, 5771.0, 5971.0, 6151.0,
                  6346.0, 6356.0, 6368.0, 6371.0])

        # Constant density values, in units of g/cm3.
        self._rho = np.asarray(
            [12.9792, 12.0042, 11.5966, 11.0351, 10.3155,
              5.3828,  5.0073,  4.5988,  3.9840,  3.8496,
              3.4894,  3.3701,  2.9000,  2.6000,  1.0200, 0.0000])

        # Electron fraction values: inner/outer core and mantle.
        self._ye = np.asarray([ 0.4656, 0.4957 ])

    def density(self, r):
        """Return density as a function of distance from the core.

        Parameters
        ----------
        r : float or ndarray
            Radial distance from core, in length units (astropy).

        Returns
        -------
        rho : float or ndarray
            Density at radial position(s) r.
        """
        idx = np.digitize(r.to('km').value, self._rbins) - 1
        return self._rho[idx] * u.g / u.cm**3
        
    def y_e(self, r):
        """Return electron fraction as a function of distance from the core.

        Parameters
        ----------
        r : float or ndarray
            Radial distance from core, in length units (astropy).

        Returns
        -------
        Y_e : float or ndarray
            Electron fraction at radial position(s) r.
        """
        idx = np.digitize(r.to('km').value, [0., 3480., 6371.]) - 1
        return self._ye[idx]
