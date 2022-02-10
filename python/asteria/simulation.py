# -*- coding: utf-8 -*-
"""CCSN neutrino sources.

This module encapsulates the basic parameters of neutrino fluxes from
supernovae as modeled in the CCSN literature. For each species of neutrino one
requires an estimate of the luminosity vs. time as well as the energy spectrum
of the neutrinos at any given time.
"""

from __future__ import print_function, division

from astropy import units as u
from snewpy.neutrino import Flavor, MassHierarchy
from snewpy import flavor_transformation as ft

import numpy as np

from .interactions import Interactions
from .source import Source


class Simulation:

    def __init__(self, config=None, *, model=None, distance=10 * u.kpc, flavors=None, hierarchy=None,
                 interactions=Interactions, mixing_scheme=None, mixing_angle=None, E=None, Emin=None, Emax=None,
                 dE=None, t=None, tmin=None, tmax=None, dt=None):
        self.param = {}
        if model and not config:

            if not E and None in (Emin, Emax, dE):
                raise ValueError("Missing or incomplete energy range definition. Use argument `E` or "
                                 "arguments `Emin`, `Emax`, `dE")
            elif not E and None not in (Emin, Emax, dE):
                _Emin = Emin.to(u.MeV).value
                _Emax = Emax.to(u.MeV).value
                _dE = dE.to(u.MeV).value
                E = np.arange(_Emin, _Emax + _dE, _dE) * u.MeV
            else:
                E = np.arange(0, 100, 1) * u.MeV

            if not t and None in (tmin, tmax, dt):
                raise ValueError("Missing or incomplete energy range definition. Use argument `t` or "
                                 "arguments `tmin`, `tmax`, `dt")
            elif not t and None not in (tmin, tmax, dt):
                _tmin = tmin.to(u.ms).value
                _tmax = tmax.to(u.ms).value
                _dt = dt.to(u.ms).value
                t = np.arange(_tmin, _tmax + _dt, _dt) * u.ms
                t = t.to(u.s)
            else:
                t = np.arange(-1, 1, 0.001) * u.s

            self.source = Source(model['name'], **model['param'])
            self.distance = distance
            self.energy = E
            self.time = t
            if flavors is None:
                self.flavors = Flavor
            else:
                self.flavors = flavors

            self.hierarchy = hierarchy
            if self.hierarchy:
                self.hierarchy = getattr(MassHierarchy, hierarchy.upper())
            else:
                self.hierarchy = MassHierarchy.NORMAL

            self.mixing_scheme = mixing_scheme
            self.mixing_angle = mixing_angle
            if mixing_scheme:
                self._mixing = getattr(ft, mixing_scheme)(mh=self.hierarchy)
            else:
                self._mixing = ft.NoTransformation()

            self.interactions = interactions
            self._E_per_V = None
            self._total_E_per_V = None
            self._photon_spectra = None
            self._create_paramdict(model, distance, flavors, hierarchy, interactions, mixing_scheme, mixing_angle, E, t)

        elif config is not None:
            raise NotImplementedError('Setup from config file is not currently implemented')
        else:
            raise ValueError('Missing required arguments. Use argument `config` or `model`.')

    def _create_paramdict(self, model=None, distance=10 * u.kpc, flavors=None, hierarchy=None,
                          interactions=Interactions, mixing_scheme=None, mixing_angle=None, E=None, t=None):
        self.param.update({
            'model': model,
            'distance': distance,
            'energy': E,
            'time': t,
            'flavors': flavors,
            'hierarchy': hierarchy,
            'mixing_scheme': mixing_scheme,
            'mixing_angle': mixing_angle,
            'interactions': interactions,
        })

    def run(self, load_simulation=False):
        """Simulates the photonic energy per volume in the IceCube Detector or loads an existing simulation

        :param load_simulation: indicates whether or not to attempt to load an existing simulation
        :type load_simulation: bool
        :return: None
        :rtype: None
        """
        if load_simulation:
            raise NotImplementedError('Simulation loading is not currently implemented')

        self.compute_photon_spectra()
        self.compute_energy_per_vol()
        return

    def compute_photon_spectra(self):
        """Computes the spectrum of photons produced by neutrino interactions in the IceCube Detector
            Data are stored in SimulationHandler.photon_spectra
        :return: None
        :rtype: None
        """
        self._photon_spectra = {}

        for flavor in self.flavors:
            result = np.zeros(self.energy.size)
            for interaction in self.interactions:
                xs = interaction.cross_section(flavor, self.energy).to(u.m ** 2).value
                E_lep = interaction.mean_lepton_energy(flavor, self.energy).value
                photon_scaling_factor = interaction.photon_scaling_factor(flavor).value
                result += xs * E_lep * photon_scaling_factor
            self._photon_spectra.update({flavor: result * (u.m * u.m)})

    def get_combined_spectrum(self, t, E, flavor, mixing):
        # TODO: Check that this function still works when p_surv and pc_osc are arrays
        # TODO: Simplify after adding neutrino oscillates_to property to SNEWPY
        # The cflavor "complementary flavor" is the flavor that the provided argument `flavor` oscillates to/from
        if flavor.is_neutrino:
            if flavor.is_electron:
                coeffs = mixing.prob_ee(t, E), mixing.prob_xe(t, E)
                cflavor = Flavor.NU_X
            else:
                coeffs = mixing.prob_xx(t, E), mixing.prob_ex(t, E)
                cflavor = Flavor.NU_E
        else:
            if flavor.is_electron:
                coeffs = mixing.prob_eebar(t, E), mixing.prob_xebar(t, E)
                cflavor = Flavor.NU_X_BAR
            else:
                coeffs = mixing.prob_xxbar(t, E), mixing.prob_exbar(t, E)
                cflavor = Flavor.NU_E_BAR

        nu_spectrum = np.zeros(shape=(t.size, E.size))
        for coeff, _flavor in zip(coeffs, (flavor, cflavor)):
            alpha = self.source.alpha(t, _flavor)
            meanE = self.source.meanE(t, _flavor).to(u.MeV).value

            alpha[alpha < 0] = 0
            cut = (alpha >= 0) & (meanE > 0)

            flux = self.source.flux(t[cut], _flavor).value.reshape(-1, 1)
            nu_spectrum[cut] += coeff * self.source.energy_pdf(t[cut], E, _flavor) * flux

        photon_spectrum = self._photon_spectra[flavor].to(u.m ** 2).value.reshape(1, -1)
        return nu_spectrum * photon_spectrum

    def compute_energy_per_vol(self, *, part_size=1000):
        """Compute the energy deposited in a cubic meter of ice by photons
        from SN neutrino interactions.

        Parameters
        ----------

        part_size : int
           Maximum number of time steps to compute at once. A temporary numpy array
           of size n x time.size is created and can be very memory inefficient.

        Returns
        -------
        E_per_V
           Energy per m**3 of ice deposited  by neutrinos of requested flavor
        """
        if self.time.size < 2:
            raise ValueError("Time array size <2, unable to compute energy per volume.")

        H2O_in_ice = 3.053e28  # 1 / u.m**3
        dist = self.distance.to(u.m).value  # m**2

        self._E_per_V = {}
        self._total_E_per_V = np.zeros(self.time.size)

        for flavor in self.flavors:
            print(f'Starting {flavor.name} simulation... {" "*(10-len(flavor.name))}', end='')

            # Perform core calculation on partitions in E to regulate memory usage in vectorized function
            result = np.zeros(self.time.size)
            idx = 0
            if part_size < self.time.size:
                while idx + part_size < self.time.size:
                    spectrum = self.get_combined_spectrum(self.time[idx:idx + part_size], self.energy, flavor,
                                                          self._mixing)
                    result[idx:idx + part_size] = np.trapz(spectrum, self.energy.value, axis=1)
                    idx += part_size
            spectrum = self.get_combined_spectrum(self.time[idx:], self.energy, flavor, self._mixing)
            result[idx:] = np.trapz(spectrum, self.energy.value, axis=1)
            # Add distance, density and time-binning scaling factors
            result *= H2O_in_ice / (4 * np.pi * dist ** 2) * np.ediff1d(self.time,
                                                                        to_end=(self.time[-1] - self.time[-2])).value
            if not flavor.is_electron:
                result *= 2
            self._E_per_V.update({flavor: result * (u.MeV / u.m / u.m / u.m)})
            self._total_E_per_V += result
            print('DONE')
        self._total_E_per_V *= (u.MeV / u.m / u.m / u.m)

    @property
    def E_per_V(self):
        return self._E_per_V if self._E_per_V else None

    @property
    def total_E_per_V(self):
        return self._total_E_per_V if self._total_E_per_V else None
