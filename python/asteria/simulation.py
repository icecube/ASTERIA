# -*- coding: utf-8 -*-
"""CCSN neutrino sources.

This module encapsulates the basic parameters of neutrino fluxes from
supernovae as modeled in the CCSN literature. For each species of neutrino one
requires an estimate of the luminosity vs. time as well as the energy spectrum
of the neutrinos at any given time.
"""

from __future__ import print_function, division

from astropy import units as u
from snewpy.neutrino import Flavor

import numpy as np
import configparser
from .interactions import Interactions
from .source import Source
from .util import energy_pdf, parts_by_index


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
            self.hierarch = hierarchy
            self.mixing_scheme = mixing_scheme
            self.mixing_angle = mixing_angle
            self.interactions = interactions

            self._create_paramdict(model, distance, flavors, hierarchy, interactions, mixing_scheme, mixing_angle, E, t)

        elif config is not None:
            with open(config) as f:
                configuration = configparser.ConfigParser()
                configuration.read_file(config)
                default = configuration['DEFAULT']
                model = configuration['MODEL']
                mixing = configuration['MIXING']
                energy = configuration['ENERGY']
                time = configuration['TIME']
                
                if 'min' and 'max' and 'step' in configuration['ENERGY'].keys():
                    _Emin = float(energy['min'])
                    _Emax = float(energy['max'])
                    _dE = float(energy['step'])
                    E = np.arange(_Emin, _Emax + _dE, _dE) * u.MeV
                else:
                    E = np.arange(0, 100, 1) * u.MeV
                    
                if 'min' and 'max' and 'step' in configuration['TIME'].keys():
                    _tmin = float(time['min'])
                    _tmax = float(time['max'])
                    _dt = float(time['step'])
                    f = u.s.to(u.ms)
                    t = np.arange(f * _tmin, f * _tmax + _dt, _dt) * u.ms
                else:
                    t = np.arange(-1000, 1000, 1) * u.ms
                    
#                 self.source = Source(model[default['model']], **model['param'])
                self.model = {'name': model['name'],
                              'param': {
                                  'progenitor_mass': model['progenitor_mass'] * u.Msun,
                                  'revival_time': model['revival_time'] * u.ms,
                                  'metallicity': model['metallicity'],
                                  'eos': model['eos']}
                              }
                self.distance = default['distance'] * u.kpc
                self.energy = E
                self.time = t
                if default['flavors'] is None:
                    self.flavors = Flavor
                else:
                    self.flavors = default['flavors']
                self.hierarch = default['hierarchy']
                self.mixing_scheme = mixing['scheme']
                self.mixing_angle = float(mixing['angle'])
                self.interactions = default['interactions']

                self._create_paramdict(model, distance, flavors, hierarchy, interactions, mixing_scheme, mixing_angle, E, t)
                
                self.__init__(**self.param)
            
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
        self.compute_energy_per_volume()
        return

    def compute_photon_spectra(self):
        """Computes the spectrum of photons produced by neutrino interactions in the IceCube Detector
            Data are stored in SimulationHandler.photon_spectra
        :return: None
        :rtype: None
        """
        photon_spectra = np.zeros(shape=(len(self.flavors), self.energy.size))

        for nu, flavor in enumerate(self.flavors):
            for interaction in Interactions:
                xs = interaction.cross_section(flavor, self.energy).to(u.m ** 2).value
                E_lep = interaction.mean_lepton_energy(flavor, self.energy).value
                photon_scaling_factor = interaction.photon_scaling_factor(flavor).value
                photon_spectra[nu] += xs * E_lep * photon_scaling_factor

        photon_spectra *= u.m ** 2
        self._photon_spectra = photon_spectra

    def compute_energy_per_volume(self):
        """Computes the photonic energy per volume in the IceCube Detector for each flavor of neutrino
            Data are stored in SimulationHandler.E_per_V
        :return: None
        :rtype: None
        """
        self.E_per_V = {}

        for flavor, photon_spectrum in zip(self.flavors, self._photon_spectra):
            self.E_per_V.update({flavor: self._photonic_energy_per_vol(self.time, self.energy, flavor, photon_spectrum,
                                                                       self.mixing_scheme)})


    def _photonic_energy_per_vol(self, time, E, flavor, photon_spectrum, mixing=None, limit=1000):
        """Compute the energy deposited in a cubic meter of ice by photons
        from SN neutrino interactions.

        Parameters
        ----------

        time : float (units s)
           Time relative to core bounce.
        E : `numpy.ndarray`
           Sorted grid of neutrino energies
        flavor : :class:`asteria.neutrino.Flavor`
           Neutrino flavor.
        photon_spectrum : `numpy.ndarray` (Units vary, m**2)
           Grid of the product of lepton cross section with lepton mean energy
           and lepton path length per MeV, sorted according to parameter E
        n : int
           Maximum number of time steps to compute at once. A temporary numpy array
           of size n x time.size is created and can be very memory inefficient.

        Returns
        -------
        E_per_V
           Energy per m**3 of ice deposited  by neutrinos of requested flavor
        """
        H2O_in_ice = 3.053e28  # 1 / u.m**3

        t = time.to(u.s).value
        Enu = E.to(u.MeV).value
        if Enu[0] == 0:
            Enu[0] = 1e-10 * u.MeV
        phot = photon_spectrum.to(u.m ** 2).value.reshape((-1, 1))  # m**2

        dist = self.distance.to(u.m).value  # m**2

        if mixing is None:
            def nu_spectrum(t, _E, flavor):
                _a = self.source.alpha(t, flavor)
                _Ea = self.source.meanE(t, flavor).to(u.MeV).value
                return energy_pdf(a=_a, Ea=_Ea, E=_E) * self.source.flux(t, flavor)
        else:
            raise NotImplementedError("Oscillation scenerios are currently unimplemented")

        print('Beginning {0} simulation... {1}'.format(flavor.name, ' ' * (10 - len(flavor.name))), end='')
        # The following two lines exploit the fact that astropy quantities will
        # always return a number when numpy size is called on them, even if it is 1.
        if time.size < 2:
            raise RuntimeError("Time array size <2, unable to compute energy per volume.")

        # Perform core calculation on partitions in E to regulate memory usage in vectorized function
        result = np.zeros(time.size)
        for i_part in parts_by_index(time, limit):  # Limits memory usage
            result[i_part] += np.trapz( nu_spectrum(time[i_part], Enu, flavor).value * phot, Enu, axis=0)
        # idx = 0
        # if limit < E.size:
        #     idc_split = np.arange(E.size, step=limit)
        #     for idx in idc_split[:-1]:
        #         _E = Enu[idx:idx + limit]
        #         _phot = phot[idx:idx + limit]
        #         result[:, idx:idx + limit] = np.trapz(nu_spectrum(t=t, E=_E, flavor=flavor).value * _phot, _E, axis=0)
        #
        # _E = Enu[idx:]
        # _phot = phot[idx:]
        # result[:, idx:idx + limit] = np.trapz(nu_spectrum(t=t, E=_E, flavor=flavor).value * _phot, _E, axis=0)
        result *= H2O_in_ice / (4 * np.pi * dist ** 2) * np.ediff1d(t, to_end=(t[-1] - t[-2]))
        if not flavor.is_electron:
            result *= 2
        print('Completed')
        return result * (u.MeV / u.m / u.m / u.m)
