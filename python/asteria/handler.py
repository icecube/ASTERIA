
from . import config, source, detector, IO
from .interactions import Interactions
from .neutrino import Flavor, Ordering
from .oscillation import SimpleMixing
from .config import parse_quantity

from abc import ABC, abstractmethod
import warnings
import numpy as np
import astropy.units as u

from numpy.lib import recfunctions as rfn

class SimulationHandler:
    """Handling object for fast MC CCSN simulations using ASTERIA
    """

    def __init__(self, conf):
        self.conf = conf
        self.source = source.initialize(conf)
        self.detector = detector.initialize(conf)

        # Set neutrino flavor config
        if conf.simulation.flavors is None:
            self.flavors = Flavor
            self.conf.simulation.flavors = [flavor.name for flavor in Flavor]
        else:
            self.flavors = Flavor(conf.simulation.flavors)
            self.conf.simulation.flavors = [flavor.name for flavor in self.flavors]

        # Set neutrino interaction config
        if conf.simulation.interactions is None:
            self.interactions = Interactions
            self.conf.simulation.interactions = [interaction.name for interaction in Interactions]
        else:
            self.interactions = Interactions(conf.simulation.interactions)
            self.conf.simulation.interactions = [interaction.name for interaction in self.interactions]

        # Set neutrino mass hierarchy and mixing scheme (Oscillation method from asteria.oscillation)
        # TODO: Streamline error checking, perhaps move to ordering class. enable ordering class to take requests?
        self.hierarchy = None
        self.mixing = None
        if conf.simulation.hierarchy.lower() in ['none', 'no']:
            self.hierarchy = Ordering.none
            self.conf.simulation.hierarchy = Ordering.none.name
            self.conf.simulation.mixing.scheme = 'none'
            self.conf.simulation.mixing.angle = 'none'
        else:
            _mixing = None
            if conf.simulation.mixing.scheme is None:
                raise RuntimeError('Hierarchy provided but missing mixing scheme.')
            if conf.simulation.mixing.scheme.lower() in ['adiabatic-msw', 'default']:
                self.conf.simulation.mixing.scheme = 'adiabatic-msw'
                if conf.simulation.mixing.angle is None:
                    warnings.warn('No Mixing angle provided, using 33.2 deg')
                    _mixing = SimpleMixing(33.2)
                    self.conf.simulation.mixing.angle = '33.2 deg'
                else:
                    _mixing = SimpleMixing(parse_quantity(conf.simulation.mixing.angle).to(u.deg).value)
            else:
                raise RuntimeError('Unknown mixing scheme: {0}'.format(conf.simulation.mixing.scheme))

            if conf.simulation.hierarchy.lower() in ['normal', 'default']:
                self.hierarchy = Ordering.normal
                self.mixing = _mixing.normal_mixing
            if conf.simulation.hierarchy.lower() in ['inverted']:
                self.hierarchy = Ordering.inverted
                self.mixing = _mixing.inverted_mixing
            self.conf.simulation.hierarchy = self.hierarchy.name

        # TODO: Set Error checking to ensure ranges are defined
        # TODO: When writing to simulationHandler.conf, write the values for Enu and time fields as numbers not strings.
        #  This saves complexity and processing in the IO module IE IO.WriteBinning()
        # Set neutrino energy ranges
        _Emin = parse_quantity(conf.simulation.energy.min).to(u.eV).value
        _Emax = parse_quantity(conf.simulation.energy.max).to(u.eV).value
        _dE = parse_quantity(conf.simulation.energy.step).to(u.eV).value
        # Defining energy range in this way ensures uniform arange step size
        self.energy = np.arange(_Emin, _Emax + _dE, _dE) * u.MeV * u.eV.to(u.MeV)
        self._Emin = _Emin * u.MeV * u.eV.to(u.MeV)
        self._Emax = _Emax * u.MeV * u.eV.to(u.MeV)
        self._dE = _dE * u.MeV * u.eV.to(u.MeV)
        self.conf.simulation.energy.size = self.energy.size

        # Set simulation time range
        _tmin = parse_quantity(conf.simulation.time.min).to(u.ns).value
        _tmax = parse_quantity(conf.simulation.time.max).to(u.ns).value
        _dt = parse_quantity(conf.simulation.time.step).to(u.ns).value
        # Defining time range in this way ensures uniform arange step size
        self.time = np.arange(_tmin, _tmax + _dt, _dt) * u.s * u.ns.to(u.s)
        self._tmin = _tmin * u.s * u.ns.to(u.s)
        self._tmax = _tmax * u.s * u.ns.to(u.s)
        self. _dt = _dt * u.s * u.ns.to(u.s)
        self.conf.simulation.time.size = self.time.size

        self._photon_spectra = None
        self._E_per_V = None

    @property
    def tmin(self):
        """Supernova model time minimum, converted to s

        :return: tmin
        :rtype: astropy.units.quantity.Quantity
        """
        return self._tmin

    @property
    def tmax(self):
        """Supernova model time maximum, converted to s

        :return: tmax
        :rtype: astropy.units.quantity.Quantity
        """
        return self._tmax

    @property
    def dt(self):
        """Supernova model time step-size, converted to s

        :return: dt
        :rtype: astropy.units.quantity.Quantity
        """
        return self._dt

    @property
    def Emin(self):
        """Neutrino energy minimum, converted to MeV

        :return: Emin
        :rtype: astropy.units.quantity.Quantity
        """
        return self._Emin

    @property
    def Emax(self):
        """Neutrino energy maximum, converted to MeV

        :return: Emax
        :rtype: astropy.units.quantity.Quantity
        """
        return self._Emax

    @property
    def dE(self):
        """Neutrino energy step-size, converted to MeV

        :return: dE
        :rtype: astropy.units.quantity.Quantity
        """
        return self._dE

    @property
    def photon_spectra(self):
        if self._photon_spectra is None:
            raise RuntimeError("photon_spectra has not been computed yet, please run compute_photon_spectra(...)!")
        return self._photon_spectra

    @property
    def E_per_V(self):
        if self._E_per_V is None:
            raise RuntimeError("E_per_V has not been computed yet, please run compute_energy_per_volume(...)!")
        return self._E_per_V

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
        E_per_V = np.zeros(shape=(len(self.flavors), self.time.size))

        for nu, (flavor, photon_spectrum) in enumerate(zip(self.flavors, self.photon_spectra)):
            E_per_V[nu] = self.source.photonic_energy_per_vol(self.time, self.energy, flavor, photon_spectrum, self.mixing)

        self._E_per_V = E_per_V * u.MeV / u.m**3

        self._E_per_V = E_per_V

    def save(self, force_save=False):
        E_per_V_1kpc = self.E_per_V * self.source.progenitor_distance.to(u.kpc).value ** 2

        IO.save(self.conf, E_per_V_1kpc, force_save)

    def load(self):
        try:
            E_per_V = IO.load(self.conf)
            E_per_V /= self.source.progenitor_distance.to(u.kpc).value ** 2
            self._E_per_V = E_per_V

        except (FileNotFoundError, AttributeError, TypeError) as e:
            print(e)
