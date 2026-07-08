# -*- coding: utf-8 -*-
"""Top-level class for performing ASTERIA's core simulation routine, and handler for the resulting outputs."""

import os
import numpy as np
import warnings
import abc

from ast import literal_eval

from astropy import units as u
from snewpy.neutrino import Flavor, MassHierarchy, MixingParameters
from snewpy import flavor_transformation as ft
from math import ceil

from configparser import ConfigParser
from importlib import import_module
from importlib.resources import files

from .interactions import Interactions
from .detector import Detector

class Simulation:

    def __init__(self, configfile=None, *,
                 model=None,
                 interactions=Interactions(),
                 flavors=Flavor,
                 flavor_xform=None,
                 E=None,
                 t=None,
                 res_dt=2*u.ms,
                 res_offset=0*u.s,
                 distance=10*u.kpc,
                 detector_scope=None,
                 add_wls=None):
        """Initialize simulation and metadata.
        """

        #- Initialize from an argument list.
        if (model is not None) and (configfile is None):
            self.model = model
            self.interactions = interactions
            self.flavors = flavors
            self.xform = flavor_xform
            self.E = E
            self.t = t
            self.sim_dt = np.diff(self.t)[0]
            self.res_dt = res_dt
            self.res_offset = res_offset
            self.distance = distance
            self.detector_scope = detector_scope
            self.add_wls = add_wls

            self._E_per_V = None
            self._total_E_per_V = None
            self._photon_spectra = None

            self.detector = Detector(self.detector_scope)
            self._eps_i3 = None
            self._eps_dc = None
            self._eps_md = None
            self._eps_ws = None
            self._max_deadtime_eff_i3 = 0.884
            self._max_deadtime_eff_md = 0.958
            self._time_binned = None
            self._E_per_V_binned = None
            self._total_E_per_V_binned = None
            self._result_ready = False
            self.scale_factor = None
            
        #- Initialize from a config file
        elif configfile is not None:
            confdict = dict()
            with open(configfile, 'r') as f:
                conf = ConfigParser()
                conf.read_file(f)

                #- Read basic simulation settings
                basic = conf['BASIC']
                confdict.update(distance=u.Quantity(basic['distance']))

                Emin, Emax, dE = [u.Quantity(basic[n]) for n in ['Emin', 'Emax', 'dE']]
                nE = int((Emax - Emin) / dE) + 1
                confdict.update(E=np.linspace(Emin, Emax, nE))

                tmin, tmax, dt = [u.Quantity(basic[n]) for n in ['tmin', 'tmax', 'dt']]
                nt = int((tmax - tmin) / dt) + 1
                confdict.update(t=np.linspace(tmin, tmax, nt))

                dt_res, toff_res = [u.Quantity(basic[n]) for n in ['dt_res', 'toff_res']]
                confdict.update(res_dt=dt_res)
                confdict.update(res_offset=dt_res)

                #- Initialize a supernova neutrino source model
                model = conf['MODEL']
                module = import_module(model["type"])
                mclass = model['name']
                ModelClass = getattr(module, mclass)
                pars = {k: v if np.strings.isalpha(v) else u.Quantity(v) for k, v in model.items() if k not in ('type', 'name')}
                confdict.update(model=ModelClass(**pars))

                #- Set up the flavor transformations
                xform = None
                order = 'Normal'
                mix = MixingParameters(mass_order=order.upper())
                if 'XFORM' in conf:
                    xf = conf['XFORM']
                    order = xf['order'].upper()
                    vers = xf['mixing'] if 'mixing' in xf else 'NuFIT6.0'
                    mix = MixingParameters(mass_order=order, version=vers)
                    xformclass = xf['transformations']
                    xmodule = import_module('snewpy.flavor_transformation')
                    XFormClass = getattr(xmodule, xformclass)
                    xform = XFormClass() if xformclass == 'NoTransformation' else XFormClass(mixing_params=mix)

                confdict.update(flavor_xform=xform)

                #- Set up the detector scope
                detector = conf['DETECTOR']
                detscope = detector['detector_scope']
                add_wls = None
                if 'add_wls' in detector:
                    add_wls = literal_eval(detector['add_wls'])

                confdict.update(detector_scope=detscope)
                confdict.update(add_wls=add_wls)

                #- Initialize from an argument list!
                self.__init__(**confdict)

        #- Incorrect arguments to constructor
        else:
            raise ValueError('Missing required arguments. Use argument `config` or `model`.')

    def run(self):
        """Simulae the photonic energy per unit volume in IceCube.
        """
        self.compute_photon_spectra()
        self.compute_energy_per_vol()
        return

    def compute_photon_spectra(self):
        """Computes the spectrum of photons produced by neutrino interactions in IceCube, given a list of flavors, interactions, and an energy grid.

        Returns
        -------
        photon_spectra : Quantity
            Cross-sectional area per energy bin.
        """
        self._photon_spectra = {}

        for flavor in self.flavors:
            result = np.zeros_like(self.E.value) * u.m**2
            for i in self.interactions:
                xs = i.cross_section(flavor, self.E)
                E_lep = i.mean_lepton_energy(flavor, self.E)
                scale = i.photon_scaling_factor(flavor)
                result += xs * E_lep * scale
            self._photon_spectra[flavor] = result

    def compute_energy_per_vol(self):
        """Compute the energy deposited in a cubic meter of ice by photons from SN neutrino interactions.

        Returns
        -------
        E_per_V: dict
            Energy per volume of ice deposited  by neutrinos of requested flavor
        """
        if self.t.size < 2:
            raise ValueError("Time array size <2, unable to compute energy per volume.")

        H2O_in_ice = 3.053e28 / u.m**3
        dist = self.distance.to(u.m).value  # m**2

        self._E_per_V = {}
        self._total_E_per_V = np.zeros(self.t.size)

        #- Compute the flux for all flavors 
        d3f_dEdtdA = self.model.get_flux(self.t, self.E, self.distance, self.xform)
        d2f_dtdA = d3f_dEdtdA.integrate('energy')

        #- Compute total energy per volume per unit time for each flavor
        for flavor in self.flavors:
            d2f_dEdt = d3f_dEdtdA[flavor] * self._photon_spectra[flavor]
            df_dt = d2f_dEdt.integrate('energy')
            result = H2O_in_ice * df_dt.array[0,:,0]
            result *= np.ediff1d(self.t, to_end=(self.t[-1] - self.t[-2]))

            #- This is needed to harmonize the units of the updated calculation
            #  with the older version of ASTERIA, but it seems like we have an
            #  extraneous factor of MeV. To do: track this down!
            result *= u.MeV

            self._E_per_V[flavor] = result
            self._total_E_per_V += result.value

        self._total_E_per_V *= result.unit
        self.rebin_result(dt=self.res_dt, force_rebin=True)

    def detector_signal(self, dt=None, flavor=None, subdetector=None, offset=0*u.s):
        """Compute signal rates observed by detector

        Parameters
        ----------
        dt : Quantity
            Time binning for hit rates (must multiple of base dt for simulation)
        flavor: snewpy.neutrino.Flavor
            Flavor for which to report signal; if None, all-flavor is reported
        subdetector : None or str
            Subdetector volume; 'i3' for IC80, 'dc' for DeepCore, 'md' for mDOM 
            (if self.detector_scope == 'Gen2'), None for full IC86/Gen2 (depending on self.detector_scope)
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

        E_per_V = self.total_E_per_V_binned.value if flavor is None else self.E_per_V_binned[flavor].value

        if self.detector_scope == 'Gen2':
            if subdetector == 'i3':
                return self.time_binned, E_per_V * (self.detector.i3_total_effvol * self.eps_i3)
            elif subdetector == 'dc':
                return self.time_binned, E_per_V * (self.detector.dc_total_effvol * self.eps_dc)
            elif subdetector == 'md':
                return self.time_binned, E_per_V * (self.detector.md_total_effvol * self.eps_md)
            elif subdetector == 'ws':
                if self._add_wls:
                    return self.time_binned, E_per_V * (self.detector.ws_total_effvol * self.eps_ws)
                else:
                    raise ValueError(f"omtype = {subdetector} for add_wls = {self._add_wls} not allowed.")
            else:
                if self._add_wls:
                    return self.time_binned, E_per_V * (self.detector.i3_total_effvol * self.eps_i3 +
                                                        self.detector.dc_total_effvol * self.eps_dc +
                                                        self.detector.md_total_effvol * self.eps_md +
                                                        self.detector.ws_total_effvol * self.eps_ws)
                else:
                    return self.time_binned, E_per_V * (self.detector.i3_total_effvol * self.eps_i3 +
                                                        self.detector.dc_total_effvol * self.eps_dc +
                                                        self.detector.md_total_effvol * self.eps_md)
        else:
            if subdetector == 'md' or subdetector == 'ws':
                raise ValueError(f"Unknown omtype: {subdetector} for {self.detector_scope} detector scope")
            else:
                i3_total_effvol = self.detector.i3_total_effvol if subdetector != 'dc' else 0
                dc_total_effvol = self.detector.dc_total_effvol if subdetector != 'i3' else 0
                return self.time_binned, E_per_V * (i3_total_effvol * self.eps_i3 + dc_total_effvol * self.eps_dc)

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
        ----------
        avg_signal : numpy.ndarray
            Average signal observed in one DOM as a function of time
        """
        if dt is None:
            dt = self.res_dt
        self.rebin_result(dt)

        if flavor is None:
            E_per_V = self._total_E_per_V_binned
        else:
            E_per_V = self._E_per_V_binned[flavor]

        effvol_IC86 = 0.1654 * u.m ** 3 / u.MeV  # Simple estimation of IceCube DOM Eff. Vol.
        effvol_Gen2 = 0.4288 * u.m ** 3 / u.MeV  # Simple estimation of Gen2 mDOM Eff. Vol. (np.avg(effvol_table))

        if self.detector_scope == "Gen2":
            if self._add_wls:
                return effvol_IC86 * E_per_V * (self.eps_dc * self.detector.n_dc_doms + self.eps_i3 * self.detector.n_i3_doms) \
                        /(self.detector.n_dc_doms + self.detector.n_i3_doms) + effvol_Gen2 * E_per_V * self.eps_md
            #return effvol_IC86 * E_per_V * (self.eps_dc + self.eps_i3)/2 + effvol_Gen2 * E_per_V * self.eps_md

        else:
            return effvol_IC86 * E_per_V * (self.eps_dc * self.detector.n_dc_doms + self.eps_i3 * self.detector.n_i3_doms) \
                    /(self.detector.n_dc_doms + self.detector.n_i3_doms)
            #return effvol_IC86 * E_per_V * (self.eps_dc + self.eps_i3)/2

    def detector_hits(self, dt=0.5*u.ms, flavor=None, subdetector=None, offset=0*u.s, size=1):
        """Compute hit rates observed by detector

        Parameters
        ----------
        dt : Quantity
            Time binning for hit rates (must be a multiple of base dt used for simulation)
        flavor: snewpy.neutrino.Flavor
            Flavor for which to report signal, if None is provided, all-flavor signal is reported
        subdetector : None or str
            IceCube subdetector volume to use for effective volume. 'i3' for IC80, 'dc' for DeepCore, 'md' for mDOM
            (if self.detector_scope == 'Gen2'), None for full IC86/Gen2 (depending on self.detector_scope)
        size : int
            Number of random realizations of the hit rate.

        Returns
        -------
        hits : np.ndarray
            Hits observed by the IceCube detector (or subdetector) as a function of time
        """
        time_binned, signal = self.detector_signal(dt, flavor, subdetector, offset)
        # return time_binned, np.random.poisson(signal)
        detector_hits = np.random.normal(signal, np.sqrt(signal),size=(size,len(signal)))
        if size==1:
            return time_binned, detector_hits.reshape(-1)
        else:
            return time_binned, detector_hits

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

    def trigger_significance(self, dt=0.5 * u.s, binnings=[0.5, 1.5, 4, 10] * u.s, offset=0 * u.s, *, seed=None):
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
        dur_sim = self.time[-1] - self.time[0]
        if max(binnings) > dur_sim:
            warnings.warn(f"Simulation time range ({dur_sim}) is too short "
                          f"to generate xi using binning {max(binnings)}, unexpected behavior may occur.")

        # Switches to improve readability
        use_gen2 = self.detector_scope == 'Gen2'
        use_gen2_wls = use_gen2 and self._add_wls

        _, hits_i3 = self.detector_hits(dt=dt, offset=offset, subdetector='i3')
        _, hits_dc = self.detector_hits(dt=dt, offset=offset, subdetector='dc')

        # Preallocate and conditionally overwrite
        hits_md = np.zeros(hits_i3.size)
        hits_ws = np.zeros(hits_i3.size)
        if use_gen2:
            _, hits_md = self.detector_hits(dt=dt, offset=offset, subdetector='md')
        if use_gen2_wls:
            _, hits_ws = self.detector_hits(dt=dt, offset=offset, subdetector='ws')

        xi = np.zeros(binnings.size)

        # Create common background to use for comparisons
        n_bin_bg = int((600 * u.s / dt).value)  # mimics SNDAQ, i.e. 10 min for bg estimation
        bg_i3 = self.detector.i3_bg(dt=dt, size=n_bin_bg)
        bg_dc = self.detector.dc_bg(dt=dt, size=n_bin_bg)
        bg_md = self.detector.md_bg(dt=dt, size=n_bin_bg) if use_gen2 else np.zeros(hits_i3.size)
        bg_ws = self.detector.ws_bg(dt=dt, size=n_bin_bg) if use_gen2_wls else np.zeros(hits_i3.size)

        for idx_bin, binsize in enumerate(binnings):
            if seed is not None:
                np.random.seed(seed)

            rebin_factor = int(binsize.to(u.s).value / dt.to(u.s).value)

            # Compute *DOM* background rate variance in search window binsize
            bg_i3_var_dom = rebin_factor * self.detector.i3_dom_bg(dt=dt, size=n_bin_bg).var()
            bg_dc_var_dom = rebin_factor * self.detector.dc_dom_bg(dt=dt, size=n_bin_bg).var()
            bg_md_var_dom = rebin_factor * self.detector.md_dom_bg(dt=dt, size=n_bin_bg).var() if use_gen2 else None
            bg_ws_var_dom = rebin_factor * self.detector.ws_dom_bg(dt=dt, size=n_bin_bg).var() if use_gen2_wls else None

            # Compute *Subdetector* background rate mean in search window binsize

            # If hits_i3.size % rebin_factor > 0, then rebinning will yeild a partial bin; exclude it
            n_bins = ceil(hits_i3.size / rebin_factor)
            if hits_i3.size % rebin_factor > 0:
                n_bins -= 1

            # Rebin background
            bg_i3_binned = np.zeros(n_bins)
            bg_dc_binned = np.zeros(n_bins)
            bg_md_binned = np.zeros(n_bins)
            bg_ws_binned = np.zeros(n_bins)

            idx_parts = [p for p in _get_partitions(np.arange(hits_i3.size), part_size=rebin_factor)][:n_bins]
            for idx_time, idx_part in enumerate(idx_parts):
                bg_i3_binned[idx_time] = bg_i3[idx_part].sum()
                bg_dc_binned[idx_time] = bg_dc[idx_part].sum()
                bg_md_binned[idx_time] = bg_md[idx_part].sum()
                bg_ws_binned[idx_time] = bg_ws[idx_part].sum()

            bg_i3_mean = bg_i3_binned.mean()
            bg_dc_mean = bg_dc_binned.mean()
            bg_md_mean = bg_md_binned.mean() if use_gen2 else 0
            bg_ws_mean = bg_ws_binned.mean() if use_gen2_wls else 0

            # Compute var_dmu, which depends on only background estimation (later used in xi calc)
            # Compute as 1 / var_dmu, to streamline sums, then invert to obtain var_dmu
            inv_var_dmu = ((self.detector.n_i3_doms / bg_i3_var_dom) +
                           (self.detector.n_dc_doms * self.detector.dc_rel_eff ** 2 / bg_dc_var_dom))
            inv_var_dmu += self.detector.n_md / bg_md_var_dom if use_gen2 else 0
            inv_var_dmu += self.detector.n_ws * self.detector.ws_rel_eff ** 2 / bg_ws_var_dom if use_gen2_wls else 0
            # TODO Jakob: Is rel. efficiency factor needed here or is it implicitly included?
            var_dmu = 1 / inv_var_dmu

            # Apply offsets to signal to mimic SNDAQ offset searches
            for idx_offset in range(rebin_factor):
                hits_i3_offset = np.roll(hits_i3, idx_offset)
                hits_i3_offset[:idx_offset] = 0

                hits_dc_offset = np.roll(hits_dc, idx_offset)
                hits_dc_offset[:idx_offset] = 0

                # Define and conditionally overwrite
                hits_md_offset = np.zeros(hits_i3.size)
                hits_ws_offset = np.zeros(hits_i3.size)
                if use_gen2:
                    hits_md_offset = np.roll(hits_md, idx_offset)
                    hits_md_offset[:idx_offset] = 0

                if use_gen2_wls:
                    hits_ws_offset = np.roll(hits_ws, idx_offset)
                    hits_ws_offset[:idx_offset] = 0

                # Rebin hits post-offset
                hits_i3_binned = np.zeros(n_bins)
                hits_dc_binned = np.zeros(n_bins)
                hits_md_binned = np.zeros(n_bins)
                hits_ws_binned = np.zeros(n_bins)

                for idx_time, idx_part in enumerate(idx_parts):
                    hits_i3_binned[idx_time] = np.sum(hits_i3_offset[idx_part])
                    hits_dc_binned[idx_time] = np.sum(hits_dc_offset[idx_part])
                    hits_md_binned[idx_time] = np.sum(hits_md_offset[idx_part])
                    hits_ws_binned[idx_time] = np.sum(hits_ws_offset[idx_part])

                # Compute xi
                dmu = var_dmu * (
                        ((hits_i3_binned + bg_i3_binned - bg_i3_mean) / bg_i3_var_dom) +
                        ((hits_dc_binned + bg_dc_binned - bg_dc_mean) / bg_dc_var_dom))
                dmu += var_dmu * ((hits_md_binned + bg_md_binned - bg_md_mean) / bg_md_var_dom) if use_gen2 else 0
                dmu += var_dmu * ((hits_ws_binned + bg_ws_binned - bg_ws_mean) / bg_ws_var_dom) if use_gen2_wls else 0

                _xi = dmu / np.sqrt(var_dmu)

                # Update xi in current binning if offset window provides better xi
                xi[idx_bin] = np.max([xi[idx_bin], _xi.max()])

        return xi

    def rebin_result(self, dt, *, offset=0 * u.s, force_rebin=False):
        """Rebins the simulation results to a new time binning.

        Parameters
        ----------
        dt : Quantity
            New time binning, must be a multiple of the simulation base binning
        offset : Quantity
            Offset to apply to rebinned result in units s (or compatible)
        force_rebin : bool
            If True, rebin, regardless of other circumstances.
            If False, only perform rebin if arg `dt` differs from `self.res_dt`

        Returns
        -------
        None
        """
        if self._E_per_V is None or self._total_E_per_V is None:
            raise RuntimeError("Simulation has not been executed yet, please use Simulation.run()")

        _dt = dt.to_value('s')
        _offset = int(offset.to(u.us).value + 0.5)  # This is a guard against floating point errors
        is_same_rebin = _dt == self.res_dt.to(u.s).value and _offset == int(self.res_offset.to(u.us).value + 0.5)
        if not is_same_rebin or force_rebin:
            if _offset != 0:
                if _offset % int(self.sim_dt.to(u.us).value):
                    warnings.warn(f"Requested offset ({offset}) is not divisible by simulation binsize "
                                  f"{self.sim_dt}, offset will not be applied.")
                    _offset = 0
                if _offset > self.t[-1].to(u.us).value or _offset < self.t[0].to(u.us).value:
                    warnings.warn(f"Requested offset ({offset}) will shift signal onset beyond simulation time "
                                  f"[{self.t[0].to(u.s)}, {self.t[-1].to(u.s)}], offset will not be applied")
                    _offset = 0

            _t = self.t.to(u.s).value
            # TODO: Check behavior for case res_dt % sim_dt != 0
            rebinfactor = int(np.rint(_dt / self.sim_dt.to_value('s')))
            offset_bins = int(_offset / self.sim_dt.to_value('us'))

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
            self.res_dt = _dt * u.s
            self.res_offset = _offset * u.us.to(u.s) * u.s
            self._eps_i3 = self._compute_deadtime_efficiency(omtype='i3')
            self._eps_dc = self._compute_deadtime_efficiency(omtype='dc')
            if self.detector_scope == 'Gen2':
                self._eps_md = self._compute_deadtime_efficiency(omtype='md')
                if self._add_wls:
                    self._eps_ws = self._eps_md # Assume the same dead time efficiency for WLS and mDOM as the readout happens in mDOM.

    def save_config(self, filename, overwrite=False):
        """Save simulation configuration to an INI file for future runs.

        Parameters
        ----------
        filename: str
            Output filename
        overwrite: bool
            Overwrite an existing output filename if argument is True
        """
        if os.path.exists(filename):
            if not overwrite:
                raise FileExistsError(f'File {filename} exists. To overwrite, set overwrite=True')

        with open(filename, 'w') as f:
            output = '\n'.join([ '[BASIC]',
                                f'distance: {self.distance}',
                                f'Emin: {self.E[0]}',
                                f'Emax: {self.E[-1]}',
                                f'dE: {np.diff(self.E)[0]}',
                                f'tmin: {self.t[0]}',
                                f'tmax: {self.t[-1]}',
                                f'dt: {np.diff(self.t)[0]}',
                                f'dt_res: {self.res_dt}',
                                f'toff_res: {self.res_offset}',
                                 '\n[MODEL]',
                                f'type: {self.model.__module__}',
                                f'name: {self.model.__class__.__name__}',
                              ] +
                               [f'{k.lower().replace(" ", "_")} : {v}' for k, v in self.model.metadata.items()] + 
                              [  '\n[XFORM]',
                                f'transformations: {self.xform.transforms.in_sn.__class__.__name__}',
                                f'order: {self.xform.mixing_params.mass_order.name}',
                                 '\n[DETECTOR]',
                                f'detector_scope: {self.detector_scope}',
                                f'add_wls: {self.add_wls}',
                              ])
            f.write(f'{output}\n')

    def scale_result(self, distance, force_rescale=False):
        """Rescales the simulation results to a progenitor distance.

        Parameters
        ----------
        distance : Quantity
            New distance
        force_rescale : bool
            If True, rescale operation regardless of other conditions 
            If False, rescale only if the new and old distances differ

        Returns
        -------
        None
        """
        if self._E_per_V is None or self._total_E_per_V is None:
            raise RuntimeError("Simulation has not been executed yet, please use Simulation.run()")

        new_dist = distance.to_value('kpc')
        current_dist = self.distance.to_value('kpc')

        if new_dist != current_dist or force_rescale:
            scaling_factor = (current_dist / new_dist) ** 2
            for flavor in self.flavors:
                self._E_per_V[flavor] *= scaling_factor
                self._E_per_V_binned[flavor] *= scaling_factor
            self._total_E_per_V *= scaling_factor
            self._total_E_per_V_binned *= scaling_factor
            self.rebin_result(dt=self.res_dt, offset=self.res_offset, force_rebin=True)
            self.distance = new_dist * u.kpc

    def _compute_deadtime_efficiency(self, omtype='i3', *, dom_effvol=None):
        """Compute DOM deadtime efficiency factor (arises from 250 us artificial deadtime).
        From A&A 535, A109 (2011) [https://doi.org/10.1051/0004-6361/201117810]

        Parameters
        ----------
        omtype : str
            Type of IceCube DOM 'i3' is IC80 DOM, 'dc' is DeepCore DOM or Gen2 mDOM 'md' from the
            Simulation.detector member. This argument is ignored if a specific DOM effective volume
            is provided.
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
        if dom_effvol is None:  # If dom_effvol is provided, omtype argument is unused
            if omtype == 'i3':
                dom_effvol = self.detector.i3_dom_effvol
                max_deadtime_eff = self._max_deadtime_eff_i3
            elif omtype == 'dc':
                dom_effvol = self.detector.dc_dom_effvol  # dc effective vol already includes relative efficiency
                max_deadtime_eff = self._max_deadtime_eff_i3 # dc_ref_eff in detector.py should already include difference in deadtime
            elif omtype == 'md':
                if self.detector_scope == 'Gen2':
                    if self._add_wls:
                        dom_effvol = self.detector.md_dom_effvol + self.detector.ws_dom_effvol # WSL component contributes to mDOM signal
                    else:
                        dom_effvol = self.detector.md_dom_effvol
                    max_deadtime_eff = self._max_deadtime_eff_md
                else:
                    raise ValueError(f"Unknown omtype: {omtype} for {self.detector_scope} detector scope")
            else:
                raise ValueError(f"Unknown omtype: {omtype}, expected ('i3', 'dc', 'md')")

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
        scaling_factor = 1/self.res_dt.to(u.s).value
        dom_signal *= scaling_factor
        return max_deadtime_eff / (1 + self.detector.deadtime * dom_signal)

    @property
    def E_per_V(self):
        """Returns dictionary of photonic energy deposition vs time for each neutrino flavor. This property will return None if this Simulation instance has not yet been run.
        """
        return self._E_per_V

    @property
    def total_E_per_V(self):
        """Returns all-flavor photonic energy deposition vs time for each neutrino flavor. This property will return None if this Simulation instance has not yet been run.
        """
        return self._total_E_per_V

    @property
    def eps_i3(self):
        """Deadtime efficiency for IC80 DOMs"""
        return self._eps_i3

    @property
    def eps_dc(self):
        """Deadtime efficiency for DeepCore DOMs"""
        return self._eps_dc

    @property
    def eps_md(self):
        """Deadtime efficiency for Gen2 mDOMs"""
        return self._eps_md

    @property
    def eps_ws(self):
        """Deadtime efficiency for Gen2 WLS tube"""
        return self._eps_ws

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

