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
from math import ceil

import numpy as np
import configparser
import warnings
from importlib.resources import files
import abc
import os

from .interactions import Interactions
from .source import Source
from .detector import Detector


class Simulation:
    """ Top-level class for performing ASTERIA's core simulation routine, and handler for the resulting outputs
    """
    def __init__(self, config=None, *, model=None, distance=10 * u.kpc, res_dt=2 * u.ms, flavors=None, hierarchy=None,
                 interactions=Interactions(), mixing_scheme=None, mixing_angle=None, E=None, Emin=None, Emax=None,
                 dE=None, t=None, tmin=None, tmax=None, dt=None, geomfile=None, effvolfile=None):
        self.metadata = {key: str(val) for key, val in locals().items() if
                         val is not None and
                         key not in ['self', 'E', 't']}

        self.metadata.update({ 'interactions': ', '.join([_int().__class__.__name__ if isinstance(_int, abc.ABCMeta) else _int.__class__.__name__ for _int in interactions]) })

        self.param = {}
        if model and not config:

            self.metadata.update({'model': {'name': model['name'],
                                            'param': '; '.join([f"{key}, {val}" for key, val in model['param'].items()])}
                                })
            if E is None and None in (Emin, Emax, dE):
                raise ValueError("Missing or incomplete energy range definition. Use argument `E` or "
                                 "arguments `Emin`, `Emax`, `dE")
            elif E is None and None not in (Emin, Emax, dE):
                _Emin = Emin.to(u.MeV).value
                _Emax = Emax.to(u.MeV).value
                _dE = dE.to(u.MeV).value
                E = np.arange(_Emin, _Emax + _dE, _dE) * u.MeV
            elif E is None:
                E = np.arange(0, 100, 1) * u.MeV
                self.metadata.update({'Emin': 0 * u.MeV, 'Emax': 100 * u.MeV, 'dE': 1 * u.ms})

            if t is None and None in (tmin, tmax, dt):
                raise ValueError("Missing or incomplete energy range definition. Use argument `t` or "
                                 "arguments `tmin`, `tmax`, `dt")
            elif t is None and None not in (tmin, tmax, dt):
                _tmin = tmin.to(u.ms).value
                _tmax = tmax.to(u.ms).value
                _dt = dt.to(u.ms)
                t = np.arange(_tmin, _tmax + _dt.value, _dt.value) * u.ms
                t = t.to(u.s)
            elif t is not None:
                _dt = np.ediff1d(t)[0]
            else:
                t = np.arange(-1, 1, 0.001) * u.s
                _dt = 1 * u.ms
                self.metadata.update({'tmin': -1 * u.s, 'tmax': 1 * u.s, 'dt': 1 * u.ms})

            self.source = Source(model['name'], model['param'])
            self.distance = distance
            self.energy = E
            self.time = t
            self._sim_dt = _dt
            self._res_dt = res_dt
            self._res_offset = 0 * u.s  # TODO: Add config/arg option for this
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

            if interactions is None:
                self.interactions = Interactions()
            else:
                if isinstance(interactions, (tuple, list)):
                    self.interactions = Interactions(interactions)
                else:
                    self.interactions = Interactions([interactions])

            self._E_per_V = None
            self._total_E_per_V = None
            self._photon_spectra = None
            self._create_paramdict(model, distance, flavors, hierarchy, interactions, mixing_scheme, mixing_angle, E, t)

            if geomfile is None:
                self._geomfile = files('asteria.data').joinpath('detector/Icecube_geometry.20110102.complete.txt')
            else:
                self._geomfile = geomfile

            if effvolfile is None:
                self._effvolfile = files('asteria.data').joinpath('detector/effectivevolume_benedikt_AHA_normalDoms.txt')
            else:
                self._effvolfile = effvolfile

            self.detector = Detector(self._geomfile, self._effvolfile)
            self._eps_i3 = None
            self._eps_dc = None
            self._time_binned = None
            self._E_per_V_binned = None
            self._total_E_per_V_binned = None
            self._result_ready = False

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
                # self.source = Source(model[default['model']], **model['param'])
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

                # self.source = Source(model['name'], **model['param'])
                dist = float(basic['distance']) * u.kpc

                if basic['flavors'] and basic['flavors'].upper() not in ('DEFAULT', 'ALL'):
                    flavors = basic['flavors']  # TODO Add str-to-flavor parser
                else:
                    flavors = Flavor

                if basic['interactions'] and basic['interactions'].upper() not in ('DEFAULT', 'ALL'):
                    interactions = basic['interactions']  # TODO Add str-to-interactions parser
                else:
                    interactions = Interactions()

                self._create_paramdict(model_dict, dist, flavors, basic['hierarchy'], interactions, mixing['scheme'],
                                       float(mixing['angle']), energy, time)
                self.__init__(**self.param)

                if geomfile is None:
                    self._geomfile = files('asteria.data').joinpath('detector/Icecube_geometry.20110102.complete.txt')
                else:
                    self._geomfile = geomfile

                if effvolfile is None:
                    self._effvolfile = files('asteria.data').joinpath('detector/effectivevolume_benedikt_AHA_normalDoms.txt')
                else:
                    self._effvolfile = effvolfile

                self.detector = Detector(self._geomfile, self._effvolfile)

        else:
            raise ValueError('Missing required arguments. Use argument `config` or `model`.')

        self.scale_factor = None

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
        """Returns mixed neutrino spectrum as a function of time and energy arising from flavor oscillations

        Parameters
        ----------
        t : astropy.quantity.Quantity
            Array of times used to perform calculation
        E : astropy.quantity.Quantity
            Array of energies used to perform calculation
        flavor : snewpy.neutrino.Flavor
            Neutrino flavor used to perform calcuation, informs selection of oscillation probability
        mixing : snewpy.flavor_transformation.FlavorTransformation
            Mixing scheme used to perform calcuation

        Returns
        -------
        spectrum : np.ndarray
            Mixed neutrino spectrum as a 2D array with dim (time, energy)
        """
        nu_spectrum = self.source.model.get_transformed_spectra(t, E, mixing)[flavor].reshape(t.size, E.size)
        cut = (t < self.source.model.time[0]) | (self.source.model.time[-1] < t)
        # TODO: Apply a fix for this once a fix has been applied to SNEWPY
        nu_spectrum[cut] = 0
        photon_spectrum = self._photon_spectra[flavor].reshape(1, E.size)
        return nu_spectrum.to(1 / u.MeV / u.s).value * photon_spectrum.to(u.m ** 2).value

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

        if self.source.special_compat_mode:
            part_size = 1  # Done for certain SNEWPY Models until a fix has been applied
            # These models can only return spectra for 1 time per call.
            # TODO: Fix this once a fix has been applied to SNEWPY

        # Perform core calculation on partitions in E to regulate memory usage in vectorized function
        # Maximum usage is expected to be ~8MB
        for flavor in self.flavors:
            result = np.zeros(self.time.size)
            idx = 0

            # Perform integration over spectrum
            if part_size <= self.time.size:
                for idx in np.arange(0, self.time.size, part_size):
                    spectrum = self.get_combined_spectrum(self.time[idx:idx + part_size], self.energy, flavor,
                                                          self._mixing)
                    result[idx:idx + part_size] = np.trapz(spectrum, self.energy.value, axis=1)
            else:
                spectrum = self.get_combined_spectrum(self.time, self.energy, flavor, self._mixing)
                result = np.trapz(spectrum, self.energy.value, axis=1)

            result *= (
                H2O_in_ice *  # Target Molecule (H2O) density
                np.ediff1d(self.time, to_end=(self.time[-1] - self.time[-2])).value *  # Time bin scaling
                1 / (4 * np.pi * dist ** 2)  # Distance
            )

            if not flavor.is_electron:  # nu_x/nu_x_bar consist of nu_mu(_bar) & nu_tau(_bar), so double them
                # TODO: Double check that the models describe single flavor spectrum or multi-flavor spectrum
                result *= 2
            self._E_per_V.update({flavor: result * (u.MeV / u.m / u.m / u.m)})
            self._total_E_per_V += result

        self._total_E_per_V *= (u.MeV / u.m / u.m / u.m)
        self.rebin_result(dt=self._res_dt, force_rebin=True)

    @property
    def E_per_V(self):
        """Returns dictionary of photonic energy deposition vs time for each neutrino flavor.
        This property will return None if this Simulation instance has not yet been run.
        """
        return self._E_per_V if self._E_per_V else None

    @property
    def total_E_per_V(self):
        """Returns all-flavor photonic energy deposition vs time for each neutrino flavor.
        This property will return None if this Simulation instance has not yet been run.
        """
        # TODO: Compound statement is redundant, member is None at init and then changed only after running
        return self._total_E_per_V if self._total_E_per_V is not None else None

    def avg_dom_signal(self, dt=None, flavor=None):
        """Returns estimated signal in one DOM, computed using avg DOM effective volume
        This property will return None if this Simulation instance has not yet been run.

        Parameters
        ----------
        dt : astropy.quantity.Quantity or None
            Time binning used to report signal (e.g. 2 ms).
            If None is provided, this will return the avg signal in the binning used for the simulation
        flavor : snewpy.neutrino.Flavor or None
            Neutrino flavor for which signal is calculated.
            If None is provided, this will return the avg signal from all flavors.

        Returns
        -------
        avg_signal : numpy.ndarray
            Average signal observed in one DOM as a function of time
        """
        if not dt:
            dt = self._res_dt
        self.rebin_result(dt)

        if flavor is None:
            E_per_V = self._total_E_per_V_binned
        else:
            E_per_V = self._E_per_V_binned[flavor]

        effvol = 0.1654 * u.m ** 3 / u.MeV  # Simple estimation of IceCube DOM Eff. Vol.
        return effvol * E_per_V * (self.eps_dc + self.eps_i3)/2

    def rebin_result(self, dt, *, offset=0 * u.s, force_rebin=False):
        """Rebins the simulation results to a new time binning.

        Parameters
        ----------
        dt : astropy.quantity.Quantity
            New time binning, must be a multiple of the base binning used for the simulation.
        offset : astropy.quantity.Quantity
            Offset to apply to rebinned result in units s (or compatible)
        force_rebin : bool
            If True, perform the rebin operation, regardless of other circumstances.
            If False, only perform rebin if argument `dt` differs with current binning stored in `self._res_dt`

        Returns
        -------
        None
        """
        if self._E_per_V is None or self._total_E_per_V is None:
            raise RuntimeError("Simulation has not been executed yet, please use Simulation.run()")

        _dt = dt.to(u.s).value
        _offset = int(offset.to(u.us).value + 0.5)  # This is a guard against floating point errors
        is_same_rebin = _dt == self._res_dt.to(u.s).value and _offset == int(self._res_offset.to(u.us).value + 0.5)
        if not is_same_rebin or force_rebin:
            if _offset != 0:
                if _offset % int(self._sim_dt.to(u.us).value):
                    warnings.warn(f"Requested offset ({offset}) is not divisible by simulation binsize "
                                  f"{self._sim_dt}, offset will not be applied.")
                    _offset = 0
                if _offset > self.time[-1].to(u.us).value or _offset < self.time[0].to(u.us).value:
                    warnings.warn(f"Requested offset ({offset}) will shift signal onset beyond simulation time "
                                  f"[{self.time[0].to(u.s)}, {self.time[-1].to(u.s)}], offset will not be applied")
                    _offset = 0

            _t = self.time.to(u.s).value
            rebinfactor = int(_dt / self._sim_dt.to(u.s).value)  # TODO: Check behavior for case _res_dt % _sim_dt != 0
            offset_bins = int(_offset / self._sim_dt.to(u.us).value)

            self._time_binned = np.array([part[0] for part in _get_partitions(_t, part_size=rebinfactor)]) * u.s

            self._E_per_V_binned = {}
            self._total_E_per_V_binned = np.zeros_like(self._time_binned.value)

            for flavor in self.flavors:
                E_per_V = np.roll(self._E_per_V[flavor].value, offset_bins)
                if offset_bins < 0:
                    E_per_V[offset_bins:] = 0
                elif offset_bins > 0:
                    E_per_V[:offset_bins] = 0

                E_per_V_binned = np.array([np.sum(part) for part in _get_partitions(E_per_V, part_size=rebinfactor)])
                self._E_per_V_binned[flavor] = E_per_V_binned * (u.MeV / u.m / u.m / u.m)
                self._total_E_per_V_binned += E_per_V_binned
            self._total_E_per_V_binned *= (u.MeV / u.m / u.m / u.m)
            self._res_dt = _dt * u.s
            self._res_offset = _offset * u.us.to(u.s) * u.s
            self._eps_i3 = self._compute_deadtime_efficiency(domtype='i3')
            self._eps_dc = self._compute_deadtime_efficiency(domtype='dc')

    def scale_result(self, distance, force_rescale=False):
        """Rescales the simulation results to a progenitor distance.

        Parameters
        ----------
        distance : astropy.quantity.Quantity
            New progenitor, must be a multiple of the base binning used for the simulation.
        force_rescale : bool
            If True, perform the rebin operation, regardless of other circumstances.
            If False, only perform rebin if argument `dt` differs with current binning stored in `self._res_dt`

        Returns
        -------
        None
        """
        if self._E_per_V is None or self._total_E_per_V is None:
            raise RuntimeError("Simulation has not been executed yet, please use Simulation.run()")

        new_dist = distance.to(u.kpc).value
        current_dist = self.distance.to(u.kpc).value

        if new_dist != current_dist or force_rescale:
            scaling_factor = (current_dist / new_dist) ** 2
            for flavor in self.flavors:
                self._E_per_V[flavor] *= scaling_factor
                self._E_per_V_binned[flavor] *= scaling_factor
            self._total_E_per_V *= scaling_factor
            self._total_E_per_V_binned *= scaling_factor
            self.rebin_result(dt=self._res_dt, offset=self._res_offset, force_rebin=True)
            self.distance = new_dist * u.kpc

    def _compute_deadtime_efficiency(self, domtype='i3', *, dom_effvol=None):
        """Compute DOM deadtime efficiency factor (arises from 250 us artificial deadtime).
        From A&A 535, A109 (2011) [https://doi.org/10.1051/0004-6361/201117810]

        Parameters
        ----------
        domtype : str
            Type of IceCube DOM 'i3' is IC80 DOM, 'dc' is DeepCore DOM from the Simulation.detector member
            This argument is ignored if a specific DOM effective volume is provided.
        dom_effvol : float or np.ndarray, optional
            DOM effective volume measured in MeV / m**3 (but not stored with astropy units).
            This may either a float (for a single DOM) or an array of float (for a table of DOMs)

        Returns
        -------
        eps : float or np.ndarray
            DOM deadtime efficiency

        Notes
        -----
        This deadtime factor is calculated using the rate observed in 1s bins (hz), but the results stored
        in class members are not scaled to 1s after this function has been run.
        """
        if dom_effvol is None:  # If dom_effvol is provided, domtype argument is unused
            if domtype == 'i3':
                dom_effvol = self.detector.i3_dom_effvol
            elif domtype == 'dc':
                dom_effvol = self.detector.dc_dom_effvol  # dc effective vol already includes relative efficiency
            else:
                raise ValueError(f"Unknown domtype: {domtype}, expected ('i3', 'dc')")

        if isinstance(dom_effvol, np.ndarray):
            # Ensures proper np broadcasting
            dom_signal = self.total_E_per_V_binned.value.reshape(-1, 1) * dom_effvol.reshape(1, -1)
        else:
            # In SNDAQ this is calculated **with** poisson randomness
            dom_signal = dom_effvol * self.total_E_per_V_binned.value

        # TODO: Adjust this scaling based on the determined "proper" method for computing deadtime
        #   eps_dt = 0.87 / (1+ 250us * true_sn_rate) -- is true_sn_rate the rate in 500ms bins, 1s bins, etc?
        #   SNDAQ always uses 500ms
        # Convert scaling factor as if it is a 0.5s bin
        scaling_factor = 0.5/self._res_dt.to(u.s).value
        dom_signal *= scaling_factor
        return 0.87 / (1 + self.detector.deadtime * dom_signal)

    @property
    def eps_i3(self):
        """Deadtime efficiency for IC80 DOMs"""
        return self._eps_i3

    @property
    def eps_dc(self):
        """Deadtime efficiency for DeepCore DOMs"""
        return self._eps_dc

    @property
    def total_E_per_V_binned(self):
        """All-flavor photonic energy deposition in result time binning"""
        return self._total_E_per_V_binned

    @property
    def E_per_V_binned(self):
        """Flavor-keyed dictionary of photonic energy deposition in result time binning"""
        return self._E_per_V_binned

    @property
    def time_binned(self):
        """Leading bin edges of in result time binning"""
        return self._time_binned

    def detector_signal(self, dt=None, flavor=None, subdetector=None, offset=0*u.s):
        """Compute signal rates observed by detector

        Parameters
        ----------
        dt : Quantity
            Time binning for hit rates (must be a multiple of base dt used for simulation)
        flavor: snewpy.neutrino.Flavor
            Flavor for which to report signal, if None is provided, all-flavor signal is reported
        subdetector : None or str
            IceCube subdetector volume to use for effective volume. 'i3' for IC80, 'dc' for DeepCore, None for IC86
        offset : astropy.quantity.Quantity
            Offset to apply to rebinned result in units s (or compatible)

        Returns
        -------
        signal : numpy.ndarray
            Signal observed by the IceCube detector (or subdetector)

        Notes
        -----
        "Signal" is defined to be the expected average hit rate in a bin
        """
        self.rebin_result(dt, offset=offset)

        i3_total_effvol = self.detector.i3_total_effvol if subdetector != 'dc' else 0
        dc_total_effvol = self.detector.dc_total_effvol if subdetector != 'i3' else 0
        E_per_V = self.total_E_per_V_binned.value if flavor is None else self.E_per_V_binned[flavor].value

        return self.time_binned, E_per_V * (i3_total_effvol * self.eps_i3 + dc_total_effvol * self.eps_dc)

    def detector_hits(self, dt=0.5*u.ms, flavor=None, subdetector=None, offset=0*u.s):
        """Compute hit rates observed by detector

        Parameters
        ----------
        dt : Quantity
            Time binning for hit rates (must be a multiple of base dt used for simulation)
        flavor: snewpy.neutrino.Flavor
            Flavor for which to report signal, if None is provided, all-flavor signal is reported
        subdetector: None or str
            IceCube subdetector, must be None (Full Detector), 'i3' (IC80) or 'dc' (DeepCore)

        Returns
        -------
        hits : np.ndarray
            Hits observed by the IceCube detector (or subdetector) as a function of time
        """
        time_binned, signal = self.detector_signal(dt, flavor, subdetector, offset)
        # return time_binned, np.random.poisson(signal)

        return time_binned, np.random.normal(signal, np.sqrt(signal))

    def sample_significance(self, sample_size=1, dt=0.5*u.s, distance=10*u.kpc, offset=None, binnings=None,
                            use_random_offset=True, *, only_highest=True, debug_info=False, seeds=None):
        """Simulate and collects a sample of SNDAQ trigger, "significance", test statistics

        Parameters
        ----------
        sample_size : int
            Number of triggers to simulate
        dt : astropy.units.Quantity
            Size of smallest binning of neutrino lightcurve used in simulation.
        distance : astropy.units.Quantity
            Distance to SN progenitor used in simulation.
        offset : astropy.units.Quantity or None, default = 0 * astropy.units.s
            Time shift(s) applied to neutrino lightcurve (positive shifts lightcurve later).
            If an array of offsets are provided, it must have size equal to `sample_size`.
            The i-th offset corresponds to the i-th sample.
        binnings : astropy.units.Quantity or None, default = [0.5, 1.5, 4., 10.] * astropy.units.s
            Size of time binnings at which to calculate trigger test statistic

        use_random_offset : bool, optional
            If True, apply a random time offset from the range (0, 500ms) to onset of neutrino lightcurve.
                This will override the value of argument `offset`.
            If False, use argument `offset`
        only_highest : bool, optional
            If True, sample only the highest significance triggers across the binsizes in `binnings`
            If False, sample the trigger significances for each binsize in `binnings`
        debug_info : bool
            If True, return the offsets and seeds used during the simulation
        seeds : np.ndarray or None, optional
            Seeds used to obtain realizations of background rates

        Returns
        -------
        sample : np.ndarray
            Sample of simulated SN trigger significances.
        offsets : astropy.units.Quantity, optional
            Random time offsets on neutrino signal onset used during simulation (Only returned when `debug_info=True`)
        seeds : np.ndarray, optional
            Random seeds used to create background rate realizations (Only returned when `debug_info=True`)

        See Also
        --------
        asteria.simulation.Simulation.trigger_significance

        Notes
        -----
        The `offset` and `use_random_offset` arguments are motivated by uncertainty on timing og the signal
        onset as it arrives relative to the bin edges used by SNDAQ to form triggers.
        The signal lightcurve onset will align with a bin edge for the case `offset=0*u.s, use_random_offset=False`
        """
        self.scale_result(distance)

        if use_random_offset:
            offsets = np.random.randint(0, 500, size=sample_size) * u.ms
            seeds = os.urandom(sample_size)  # Sets random seed for realization of background in signi calc
        elif isinstance(offset, u.Quantity):
            if offset.size == 1:
                offsets = offset.to(u.s).value * np.ones(sample_size) * u.s
            else:
                offsets = offset
        else:
            offsets = np.zeros(sample_size) * u.s

        if seeds is None:
            seeds = [None] * sample_size

        sample = np.array([self.trigger_significance(dt=dt, offset=_offset, binnings=binnings, seed=seed)
                           for _offset, seed in zip(offsets, seeds)])
        if only_highest:
            sample = sample.max(axis=1)
        if debug_info:
            return sample, offsets, seeds
        return sample

    def trigger_significance(self, dt=0.5*u.s, binnings=[0.5, 1.5, 4, 10]*u.s, offset=0*u.s, *, seed=None):
        """Simulates one SNDAQ trigger "significance" test statistic for requested binnings

        Parameters
        ----------
        dt : astropy.units.Quantity
            Size of smallest binning of neutrino lightcurve used in simulation.
        offset : astropy.units.Quantity or None, default = 0 * astropy.units.s, optional
            Time shift(s) applied to neutrino lightcurve (positive shifts lightcurve later).
            If an array of offsets are provided, it must have size equal to `sample_size`.
            The i-th offset corresponds to the i-th sample.
        binnings : astropy.units.Quantity or None, default = [0.5, 1.5, 4., 10.] * astropy.units.s
            Size of time binnings at which to calculate trigger test statistic
            Unexpected behaviors may arise if the binnings are not cleanly divisible by argument `dt`
        seed : np.ndarray, optional
            Random seed used to create background rate realizations

        Returns
        -------
        xi : np.ndarray
            SNDAQ trigger significances corresponding to `binnings`

        Notes
        -----
        The `offset` and `use_random_offset` arguments are motivated by uncertainty on timing of the signal
        onset as it arrives relative to the bin edges used by SNDAQ to form triggers.
        The signal lightcurve onset will align with a bin edge for the case `offset=0*u.s, use_random_offset=False`

        This simulation is an approximation of the live calculation performed by SNDAQ. See arXiv:1108.0171 for more
        detail. The simulation proceeds as follows
            1 - Obtain a realization of IceCube's background Rate the base time binning `dt`
            2 - Rebin the background and signal rates to the search windows from `binnings`
            3 - Shift signal forward in increments of dt to mimic offset searches of SNDAQ (first iter has no offset)
            4 - Compute significance xi using max. LLH from arXiv:1108.0171
            5 - Compare current offset's significances to prior results, if a higher significance is found in the same
                binning overwrite that binning's prior result.
            6 - Repeat 3--5 for all offsets as appropriate for binning
            7 - Repeat 2--5 for all binnings in `binnings`

        """
        _, hits_i3 = self.detector_hits(dt=dt, offset=offset, subdetector='i3')
        _, hits_dc = self.detector_hits(dt=dt, offset=offset, subdetector='dc')
        xi = np.zeros(binnings.size)

        for idx_bin, binsize in enumerate(binnings):
            if seed is not None:
                np.random.seed(seed)

            rebin_factor = int(binsize.to(u.s).value / dt.to(u.s).value)
            n_bins = ceil(hits_i3.size/rebin_factor)

            bg_i3 = self.detector.i3_bg(dt=dt, size=hits_i3.size)
            bg_dc = self.detector.dc_bg(dt=dt, size=hits_i3.size)

            # Create a realization of background rate scaled up from dt to binsize
            bg_i3_binned = np.zeros(n_bins)
            bg_dc_binned = np.zeros(n_bins)
            for idx_time, (bg_i3_part, bg_dc_part) in enumerate(_get_partitions(bg_i3, bg_dc, part_size=rebin_factor)):
                bg_i3_binned[idx_time] = np.sum(bg_i3_part)
                bg_dc_binned[idx_time] = np.sum(bg_dc_part)

            # Compute *DOM* background rate variance
            # Background variance is not well estimated after the rebin, so use lower binning and upscale
            # This could be mitigated by extending background windows, at the cost of speed
            bg_i3_var_dom = rebin_factor * self.detector.i3_dom_bg(dt=dt, size=1000).var()
            bg_dc_var_dom = rebin_factor * self.detector.dc_dom_bg(dt=dt, size=1000).var()

            # If hits_i3.size / rebin_factor is not an integer, then the last bin in the rebinned rates will be partial
            # In this case, exclude it from the calculation of the background mean
            if bg_i3.size % rebin_factor != 0:
                idx_bg = bg_i3_binned.size - 1
            else:
                idx_bg = bg_i3_binned.size

            bg_i3_mean = bg_i3_binned[:idx_bg].mean()  # IC80 *subdetector* rate mean
            bg_dc_mean = bg_dc_binned[:idx_bg].mean()  # DeepCore *subdetector* rate mean

            # Compute xi with increments of 0.5s offsets, mimicking the offset searches of SNDAQ
            for idx_offset in range(rebin_factor):
                hits_i3_offset = np.roll(hits_i3, idx_offset)
                hits_dc_offset = np.roll(hits_dc, idx_offset)
                hits_i3_offset[:idx_offset] = 0
                hits_dc_offset[:idx_offset] = 0
                hits_i3_binned = np.zeros(n_bins)
                hits_dc_binned = np.zeros(n_bins)

                for idx_time, (hits_i3_part, hits_dc_part) in enumerate(_get_partitions(hits_i3_offset, hits_dc_offset,
                                                                                        part_size=rebin_factor)):
                    hits_i3_binned[idx_time] = np.sum(hits_i3_part)
                    hits_dc_binned[idx_time] = np.sum(hits_dc_part)

                var_dmu = 1/((self.detector.n_i3_doms/bg_i3_var_dom) +
                             (self.detector.n_dc_doms*self.detector.dc_rel_eff**2/bg_dc_var_dom))
                dmu = var_dmu * (
                        ((hits_i3_binned + bg_i3_binned - bg_i3_mean) / bg_i3_var_dom) +
                        ((hits_dc_binned + bg_dc_binned - bg_dc_mean) / bg_dc_var_dom))
                # Should rel. eff be considered here? It would have already been considered once during hit generation
                # It's unclear if SNDAQ applies this same factor, it does apply *a* factor that is somehow normalized
                #        ((hits_dc_binned + bg_dc_binned - bg_dc_mean) * self.detector.dc_rel_eff / bg_dc_var_dom))
                _xi = dmu/np.sqrt(var_dmu)

                xi[idx_bin] = np.max([xi[idx_bin], _xi.max()])
        return xi


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




