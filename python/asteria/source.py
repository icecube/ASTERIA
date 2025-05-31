# -*- coding: utf-8 -*-
"""CCSN neutrino sources.

Initialize a source using the models available in SNEWPY.

See either of these two resources:
- snewpy.readthedocs.io
- https://github.com/SNEWS2/snewpy
"""
try:
    from snewpy.models import _init_model as init_model
except ModuleNotFoundError:
    from .util import init_model as init_snewpy_model_from_param

try:
    from snewpy import model_path
except ImportError:
    from .util import model_path

from snewpy.neutrino import Flavor
from scipy.interpolate import PchipInterpolator
from numbers import Number
from scipy.special import loggamma, gdtr

import astropy.units as u
import numpy as np


class Source:

    def __init__(self, model, model_params=None):
        self.model = init_model(model, **model_params)
        self._interp_lum = {}
        self._interp_meanE = {}
        self._interp_pinch = {}

        for flavor in Flavor:
            t = self.model.time
            self._interp_lum.update({flavor: PchipInterpolator(t, self.model.luminosity[flavor], extrapolate=False)})
            self._interp_meanE.update({flavor: PchipInterpolator(t, self.model.meanE[flavor], extrapolate=False)})
            self._interp_pinch.update({flavor: PchipInterpolator(t, self.model.pinch[flavor], extrapolate=False)})

    def luminosity(self, t, flavor=Flavor.NU_E_BAR):
        """Return interpolated source luminosity at time t for a given flavor.

        Parameters
        ----------

        t : float
            Time relative to core bounce.
        flavor : :class:`asteria.neutrino.Flavor`
            Neutrino flavor.

        Returns
        -------
        luminosity : Astropy.units.quantity.Quantity
            Source luminosity (units of power).
        """
        return np.nan_to_num(self._interp_lum[flavor](t)) * (u.erg / u.s)

    def meanE(self, t, flavor=Flavor.NU_E_BAR):
        """Return interpolated source mean energy at time t for a given flavor.

        Parameters
        ----------

        t :
            Time relative to core bounce.
        flavor : :class:`asteria.neutrino.Flavor`
            Neutrino flavor.

        Returns
        -------
        mean_energy : Astropy.units.quantity.Quantity
            Source mean energy (units of energy).
        """
        # TODO Checks for units/unitless inputs
        return np.nan_to_num(self._interp_meanE[flavor](t)) * u.MeV

    def alpha(self, t, flavor=Flavor.NU_E_BAR):
        """Return source pinching paramter alpha at time t for a given flavor.
        Parameters
        ----------

        t :
            Time relative to core bounce.
        flavor : :class:`asteria.neutrino.Flavor`
            Neutrino flavor.
        Returns
        -------
        pinch :
        Source pinching parameter (unitless).
        """
        # TODO Checks for units/unitless inputs
        return np.nan_to_num(self._interp_pinch[flavor](t))

    def flux(self, t, flavor=Flavor.NU_E_BAR):
        """Return source flux at time t for a given flavor.

        Parameters
        ----------

        t : Quantity
            Time relative to core bounce (units seconds).
        flavor : :class:`asteria.neutrino.Flavor`
            Neutrino flavor.

        Returns
        -------
        flux :
        Source number flux (unit-less, count of neutrinos).
        """
        L = self.luminosity(t, flavor).to(u.MeV / u.s).value
        meanE = self.meanE(t, flavor).value

        if isinstance(t, np.ndarray):
            _flux = np.divide(L, meanE, where=(meanE > 0), out=np.zeros(L.size))
        else:
            # TODO: Fix case where t is list, or non astropy quantity. This is a front-end function for some use cases
            if meanE > 0.:
                _flux = L / meanE
            else:
                _flux = 0

        return _flux / u.s

    @staticmethod
    def _energy_pdf(a, Ea, E):
        return np.exp((1 + a) * np.log(1 + a) - loggamma(1 + a) +
                      a * np.log(E) - (1 + a) * np.log(Ea) - (1 + a) * (E / Ea))

    @staticmethod
    def _energy_cdf(a, Ea, E):
        return gdtr(1., a + 1., (a + 1.) * (E / Ea))

    def energy_pdf(self, t, E, flavor=Flavor.NU_E_BAR, *, limit_size=True):
        _E = E.to(u.MeV).value
        if isinstance(E, np.ndarray):
            if _E[0] == 0:
                _E[0] = 1e-10
            if limit_size and E.size > 1e6:
                raise ValueError(
                    'Input argument size exceeded. Argument`E` is a np.ndarray with size {E.size}, which may '
                    'lead to large memory consumption while this function executes. To proceed, please reduce '
                    'the size of `E` or use keyword argument `limit_size=False`')
        if isinstance(t, np.ndarray):
            _a = self.alpha(t, flavor)
            _Ea = self.meanE(t, flavor).to(u.MeV).value

            _a[_a < 0] = 0
            # Vectorized function can lead to unregulated memory usage, better to define it only when needed
            _vec_energy_pdf = np.vectorize(self._energy_pdf, excluded=['E'], signature='(1,n),(1,n)->(m,n)')
            return _vec_energy_pdf(a=_a.reshape(1, -1), Ea=_Ea.reshape(1, -1), E=_E.reshape(-1, 1)).T
        elif isinstance(t, Number):
            _a = self.alpha(t, flavor)
            _Ea = self.meanE(t, flavor).to(u.MeV).value
            return self._energy_pdf(_a, _Ea, E)
        else:
            raise ValueError(f'Invalid argument types, argument `t` must be numbers or np.ndarray, Given ({type(t)})')
