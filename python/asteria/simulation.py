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
import configparser
import os

from .interactions import Interactions
from .source import Source
from .detector import Detector


class Simulation:

    def __init__(self, config=None, *, model=None, distance=10 * u.kpc, flavors=None, hierarchy=None,
                 interactions=Interactions, mixing_scheme=None, mixing_angle=None, E=None, Emin=None, Emax=None,
                 dE=None, t=None, tmin=None, tmax=None, dt=None, geomfile=None, effvolfile=None):
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
            elif not E:
                E = np.arange(0, 100, 1) * u.MeV

            if not t and None in (tmin, tmax, dt):
                raise ValueError("Missing or incomplete energy range definition. Use argument `t` or "
                                 "arguments `tmin`, `tmax`, `dt")
            elif not t and None not in (tmin, tmax, dt):
                _tmin = tmin.to(u.ms).value
                _tmax = tmax.to(u.ms).value
                _dt = dt.to(u.ms)
                t = np.arange(_tmin, _tmax + _dt.value, _dt.value) * u.ms
                t = t.to(u.s)
            elif t:
                _dt = np.ediff1d(t)[0]
            else:
                t = np.arange(-1, 1, 0.001) * u.s
                _dt = 1 * u.ms

            self.source = Source(model['name'], model['param'])
            self.distance = distance
            self.energy = E
            self.time = t
            self._sim_dt = _dt
            if flavors is None:
                self.flavors = Flavor
            else:
                self.flavors = flavors

            if not hierarchy or hierarchy.upper() == 'DEFAULT':
                self.hierarchy = MassHierarchy.NORMAL
            else:
                self.hierarchy = getattr(MassHierarchy, hierarchy.upper())

            self.mixing_scheme = mixing_scheme
            self.mixing_angle = mixing_angle
            if mixing_scheme:
                if mixing_scheme == 'NoTransformation':
                    self._mixing = getattr(ft, mixing_scheme)()
                else:
                    # TODO: Improve mixing name checking, this argument is case sensitive
                    self._mixing = getattr(ft, mixing_scheme)(mh=self.hierarchy)
            else:
                self._mixing = ft.NoTransformation()

            self.interactions = interactions
            self._E_per_V = None
            self._total_E_per_V = None
            self._photon_spectra = None
            self._create_paramdict(model, distance, flavors, hierarchy, interactions, mixing_scheme, mixing_angle, E, t)

            if not geomfile:
                self._geomfile = os.path.join(os.environ['ASTERIA'],
                                              'data/detector/Icecube_geometry.20110102.complete.txt')
            else:
                self._geomfile = geomfile

            if not effvolfile:
                self._effvolfile = os.path.join(os.environ['ASTERIA'],
                                                'data/detector/effectivevolume_benedikt_AHA_normalDoms.txt')
            else:
                self._effvolfile = effvolfile

            self.detector = Detector(self._geomfile, self._effvolfile)

        elif config is not None:
            with open(config) as f:
                configuration = configparser.ConfigParser()
                configuration.read_file(f)
                basic = configuration['BASIC']
                model = configuration['MODEL']
                mixing = configuration['MIXING']
                energy = configuration['ENERGY']
                time = configuration['TIME']

                if 'min' and 'max' and 'step' in configuration['ENERGY'].keys():
                    _Emin = float(energy['min'])
                    _Emax = float(energy['max'])
                    _dE = float(energy['step'])
                    energy = np.arange(_Emin, _Emax + _dE, _dE) * u.MeV
                else:
                    energy = np.arange(0, 100, 1) * u.MeV

                if 'min' and 'max' and 'step' in configuration['TIME'].keys():
                    _tmin = float(time['min'])
                    _tmax = float(time['max'])
                    _dt = float(time['step'])
                    f = u.s.to(u.ms)
                    time = np.arange(f * _tmin, f * _tmax + _dt, _dt) * u.ms
                else:
                    time = np.arange(-1000, 1000, 1) * u.ms
                time = time.to(u.s)
#                 self.source = Source(model[default['model']], **model['param'])
                model_dict = {'name': model['name'],
                              'param': {}}
                for key, param in model.items():
                    if key != 'name':
                        _param = param.split()
                        if len(_param) > 1:
                            value = _param[0]
                            unit = u.Unit(_param[1])
                        else:
                            value = _param[0]
                            unit = None
                        if value.replace('.', '', 1).isnumeric():  # Replace allws detection of floats like '0.004'
                            value = float(value) if not float(value).is_integer() else int(value)
                            value *= unit if unit is not None else 1
                        model_dict['param'].update({key: value})

#                 self.source = Source(model['name'], **model['param'])
                dist = float(basic['distance']) * u.kpc

                if basic['flavors'] and basic['flavors'].upper() not in ('DEFAULT', 'ALL'):
                    flavors = basic['flavors']  # TODO Add str-to-flavor parser
                else:
                    flavors = Flavor

                if basic['interactions'] and basic['interactions'].upper() not in ('DEFAULT', 'ALL'):
                    interactions = basic['interactions']  # TODO Add str-to-interactions parser
                else:
                    interactions = Interactions

                self._create_paramdict(model_dict, dist, flavors, basic['hierarchy'], interactions, mixing['scheme'],
                                       float(mixing['angle']), energy, time)
                self.__init__(**self.param)
                
                if not geomfile:
                    self._geomfile = os.path.join(os.environ['ASTERIA'],
                                                  'data/detector/Icecube_geometry.20110102.complete.txt')
                else:
                    self._geomfile = geomfile

                if not effvolfile:
                    self._effvolfile = os.path.join(os.environ['ASTERIA'],
                                                    'data/detector/effectivevolume_benedikt_AHA_normalDoms.txt')
                else:
                    self._effvolfile = effvolfile

                self.detector = Detector(self._geomfile, self._effvolfile)

        else:
            raise ValueError('Missing required arguments. Use argument `config` or `model`.')

    def _create_paramdict(self, model=None, distance=10 * u.kpc, flavors=None, hierarchy=None,
                          interactions=Interactions, mixing_scheme=None, mixing_angle=None, E=None, t=None):
        self.param.update({
            'model': model,
            'distance': distance,
            'E': E,
            't': t,
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
                coeffs = mixing.prob_ee(t, E), mixing.prob_ex(t, E)
                cflavor = Flavor.NU_X
            else:
                coeffs = mixing.prob_xx(t, E), mixing.prob_xe(t, E)
                cflavor = Flavor.NU_E
        else:
            if flavor.is_electron:
                coeffs = mixing.prob_eebar(t, E), mixing.prob_exbar(t, E)
                cflavor = Flavor.NU_X_BAR
            else:
                coeffs = mixing.prob_xxbar(t, E), mixing.prob_xebar(t, E)
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
            print(f'Starting {flavor.name} simulation... {" " * (10 - len(flavor.name))}', end='')

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

    def avg_dom_signal(self, flavor):
        effvol = 0.1654 * u.m ** 3 / u.MeV  # Simple estimation of IceCube DOM Eff. Vol.
        return effvol * self._E_per_V[flavor]

    def detector_signal(self, dt=2 * u.ms, flavor=None, subdetector=None):
        """ Compute signal rates observed by detector
        Parameters
        ----------
        dt : Quantity
            Time binning for hit rates (must be a multiple of base dt used for simulation)
        flavor: snewpy.neutrino.Flavor
            Flavor for which to report signal, if None is provided, all-flavor signal is reported
        Notes
        -----
        "Signal" is defined to be the expected average hit rate in a bin
        """

        _dt = dt.to(u.s).value
        _t = self.time.to(u.s).value
        rebinfactor = int(_dt / self._sim_dt.to(u.s).value)
        total_E_per_V_binned = np.array([np.sum(part) for part in _get_partitions(self._total_E_per_V.value,
                                                                                  part_size=rebinfactor)])
        deadtime = self.detector.deadtime

        i3_effvol = self.detector.i3_effvol if subdetector != 'dc' else 0
        dc_effvol = self.detector.dc_effvol if subdetector != 'i3' else 0

        dc_rel_eff = self.detector.dc_rel_eff
        eps_i3 = 0.87 / (1 + deadtime * total_E_per_V_binned / _dt)
        eps_dc = 0.87 / (1 + deadtime * total_E_per_V_binned * dc_rel_eff / _dt)
        time_binned = np.array([part[0] for part in _get_partitions(_t, part_size=rebinfactor)])
        if flavor:
            E_per_V_binned = np.array([np.sum(part) for part in _get_partitions(self._E_per_V[flavor].value,
                                                                                part_size=rebinfactor)])
            return time_binned * u.s, E_per_V_binned * (i3_effvol * eps_i3 + dc_effvol * eps_dc)
        else:
            return time_binned * u.s, total_E_per_V_binned * (i3_effvol * eps_i3 + dc_effvol * eps_dc)

    def detector_hits(self, dt=2 * u.ms, flavor=None, subdetector=None,):
        """ Compute hit rates observed by detector
        Parameters
        ----------
        dt : Quantity
            Time binning for hit rates (must be a multiple of base dt used for simulation)
        flavor: snewpy.neutrino.Flavor
            Flavor for which to report signal, if None is provided, all-flavor signal is reported
        subdetector: None or str
            IceCube subdetector, must be None (Full Detector), 'i3' (IC80) or 'dc' (DeepCore)
        """
        time_binned, signal = self.detector_signal(dt, flavor, subdetector)
        # Possion-fluctuated
        return time_binned, np.random.poisson(signal)

    def detector_significance(self, dt=0.5 * u.s, *, by_subdetector=False):
        i3_dom_bg_var = self.detector.i3_dom_bg_sig ** 2 * dt.to(u.s).value
        dc_dom_bg_var = self.detector.dc_dom_bg_sig ** 2 * dt.to(u.s).value

        if by_subdetector:  # Use definition of delta_mu(_var) from SNDAQ
            time_binned, i3_hits_binned = self.detector_hits(dt, subdetector='i3')
            _, dc_hits_binned = self.detector_hits(dt, subdetector='dc')

            var_dmu = 1 / (self.detector.n_i3_doms / i3_dom_bg_var + self.detector.n_dc_doms / dc_dom_bg_var)
            dmu = var_dmu * (i3_hits_binned/i3_dom_bg_var + dc_hits_binned/dc_dom_bg_var)

            signi_binned = dmu/np.sqrt(var_dmu)
            return time_binned, signi_binned
        else:  # Use simple calculation from USSR
            detector_bg_var = (self.detector.n_i3_doms*i3_dom_bg_var + self.detector.n_dc_doms*dc_dom_bg_var)

            time_binned, hits_binned = self.detector_hits(dt)
            signi_binned = hits_binned/np.sqrt(detector_bg_var)
        return time_binned, signi_binned

    def trigger_significance(self, dt=0.5 * u.s, *, by_subdetector=False):
        return self.detector_significance(dt=dt, by_subdetector=by_subdetector)[1].max()

    def sample_significance(self, sample_size, dt=0.5*u.s, distance=10*u.kpc, by_subdetector=False):
        # TODO: This is a bit awkward, by adding distance scaling elsewhere, the conditional scaling here may be removed
        current_dist = self.distance.to(u.kpc).value

        if current_dist != distance.to(u.kpc).value:
            total_E_per_V = self._total_E_per_V
            self._total_E_per_V = self._total_E_per_V.value * (current_dist/distance.to(u.kpc).value)**2 * \
                                  (u.MeV / u.m / u.m / u.m)

            sample = np.array([self.trigger_significance(dt, by_subdetector=by_subdetector)
                               for _ in range(sample_size)])
            self._total_E_per_V = total_E_per_V
        else:
            sample = np.array([self.trigger_significance(dt, by_subdetector=by_subdetector)
                               for _ in range(sample_size)])
        return sample




def _get_partitions(*args, part_size=1000):
    if len(args) > 1:
        if not all(len(x) == len(args[0]) for x in args):
            raise ValueError(f'Inputs must have same size, given sizes ({", ".join((str(len(x)) for x in args))})')
    total_size = len(args[0])
    if part_size > total_size:
        yield tuple(x for x in args) if len(args) > 1 else args[0]
    else:
        idx = 0
        while idx + part_size < total_size:
            yield tuple(x[idx:idx + part_size] for x in args) if len(args) > 1 else args[0][idx:idx + part_size]
            idx += part_size
        yield tuple(x[idx:] for x in args) if len(args) > 1 else args[0][idx:]
