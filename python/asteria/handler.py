
from . import config, source, detector, IO
from .interactions import Interactions
from snewpy.neutrino import Flavor
from .neutrino import Ordering
from .oscillation import SimpleMixing
from .config import parse_quantity

import warnings
import numpy as np
import astropy.units as u


class SimulationHandler:
    """Handling object for fast MC CCSN simulations using ASTERIA

    :ivar conf: Asteria Configuration object, contains model, progenitor and simulation information
    :type conf: asteria.config.Configuration

    :ivar source: Asteria Source object, contains information on progenitor, neutrino flux, and neutrino spectrum
    :type source: asteria.source.Source

    :ivar detector: Asteria Detector object, contains informtion on IceCube Detector, DOM Efficiencies, etc
    :type detector: asteria.detector.Detector

    :ivar flavors: Enum of neutrino flavors, items contain helper methods for TeX, oscillation, comparisons, etc.
    :type flavors: asteria.neutrino._FlavorMeta

    :ivar interactions: Enum of neutrino interactions, items contain methods for cross sections, lepton energy, etc.
    :type interactions: asteria.interactions._InteractionsMeta

    :ivar hierarchy: Neutrino Mass hierarchy, see Enumeration in asteria.oscillations
    :type hierarchy: <enum 'Ordering'>

    :ivar mixing: Neutrino mixing scheme, currently only adiabatic-msw is supported
    :type mixing: method

    :ivar energy: Neutrino energy array, defines energy resolution at which to perform simulation. Units MeV.
    :type energy: numpy.ndarray of astropy.units.quantity.Quantity

    :return Emin, Emax, dE: Min., Max., and step-size of energy array respectively. Units MeV.
    :rtype Emin, Emax, dE: astropy.units.quantity.Quantity

    :ivar time: Time array, defines time resolution at which to perform simulation of CCSN response. Units s.
    :type time: numpy.ndarray of astropy.units.quantity.Quantity

    :return tmin, tmax, dt: Min., Max., and step-size of time array respectively. Units s.
    :rtype tmin, tmax, dt: astropy.units.quantity.Quantity

    :return photon_spectra: Spectrum of photons produced by neutrino interactions in IceCube detector
    :rtype photon_spectra: numpy.ndarray of astropy.units.quantity.Quantity

    :return E_per_V, total_E_per_V: Photonic energy deposition per volume due to CCSN neutrinos (flavor-wise, total)
    :rtype E_per_V, total_E_per_V: numpy.ndarray of astropy.units.quantity.Quantity

    :Example Usage:
    >>> conf = .config.load_config('../../data/config/default.yaml')
    >>> sim = SimulationHandler(conf)
    >>> sim.run()
    >>> sim.E_per_V
    """

    def __init__(self, conf):
        """Initializes simulation handler, performs minor error checking in input configuration

        :param conf: Configuration object holding information on source, model and simulation parameters
        :type conf: asteria.config.Configuration
        """
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
        """Spectrum of photons produced each neutrino flavor's interactions in the IceCube Detector

        :return: photon_spectra
        :rtype: numpy.ndarray of astropy.units.quantity.Quantity
        """
        if self._photon_spectra is None:
            raise RuntimeError("photon_spectra has not been computed yet, please run compute_photon_spectra(...)!")
        return self._photon_spectra

    @property
    def E_per_V(self):
        """Photonic energy deposition per volume for each flavor of neutrinos

        :return: E_per_V
        :rtype: numpy.ndarray of astropy.units.quantity.Quantity
        """
        if self._E_per_V is None:
            raise RuntimeError("E_per_V has not been computed yet, please run compute_energy_per_volume(...)!")
        return self._E_per_V

    @property
    def total_E_per_V(self):
        """Total photonic energy deposition per volume, summed across all flavors of neutrinos

        :return: total_E_per_V
        :rtype: numpy.ndarray of astropy.units.quantity.Quantity
        """
        return np.sum(self._E_per_V, axis=0)

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
        E_per_V = np.zeros(shape=(len(self.flavors), self.time.size))

        for nu, (flavor, photon_spectrum) in enumerate(zip(self.flavors, self.photon_spectra)):
            E_per_V[nu] = self.source.photonic_energy_per_vol(self.time, self.energy, flavor, photon_spectrum, self.mixing)

        self._E_per_V = E_per_V * u.MeV / u.m**3

    def run(self, load_simulation=True):
        """Simulates the photonic energy per volume in the IceCube Detector or loads an existing simulation

        :param load_simulation: indicates whether or not to attempt to load an existing simulation
        :type load_simulation: bool
        :return: None
        :rtype: None
        """
        if load_simulation:
            try:
                self.load()
                print('Simulation Loaded.')
                # If load simulation is successful, generate photon spectra so data member is accessible
                self.compute_photon_spectra()
                return
            except (FileNotFoundError, AttributeError) as e:
                print(e)

        print('Running Simulation...')
        self.compute_photon_spectra()
        self.compute_energy_per_volume()
        return

    def save(self, force=False):
        """Save Photonic energy deposition per volume scaled to a star at 1kpc
          Scales E_per_V to a progenitor at distance 1kpc and writes it to h5 file specified by self.conf.IO.table.path.
          This function is a wrapper for asteria.IO.save()

        :param force: indicates whether or not to overwrite an existing simulation if one is found
        :type force: bool
        :return: None
        :rtype: None
        """
        E_per_V_1kpc = self.E_per_V * self.source.progenitor_distance.to(u.kpc).value**2
        IO.save(self.conf, E_per_V_1kpc, force)

    def load(self):
        try:
            E_per_V = IO.load(self.conf)
            E_per_V /= self.source.progenitor_distance.to(u.kpc).value ** 2
            self._E_per_V = E_per_V * u.MeV / u.m**3

        # TypeError is included due to conf node values, FileNotFoundError and AttributeError indicate missing file/sim.
        except (FileNotFoundError, AttributeError, TypeError) as e:
            print(e)

    @property
    def conf_dict(self):
        """Returns configuration options used to save simulations in dictionary format

        :return: conf_dict: Dictionary with keys/value matching the configuration simulation node
        :rtype: dict
        """
        conf_dict = dict()
        conf_dict['flavors'] = [flavor.name for flavor in self.flavors]
        conf_dict['interactions'] = [interaction.name for interaction in self.interactions]
        conf_dict['hierarchy'] = self.hierarchy.name
        conf_dict['mixing'] = {
            'scheme': self.conf.simulation.mixing.scheme,
            'angle': self.conf.simulation.mixing.angle
        }
        conf_dict['energy'] = {
            'min': self.conf.simulation.energy.min,
            'max': self.conf.simulation.energy.max,
            'step': self.conf.simulation.energy.step,
            'size': self.conf.simulation.energy.size,
        }
        conf_dict['time'] = {
            'min': self.conf.simulation.time.min,
            'max': self.conf.simulation.time.max,
            'step': self.conf.simulation.time.step,
            'size': self.conf.simulation.time.size,
        }
        return conf_dict

    def print_config(self):
        """Prints configuration options used to save simulations in yaml-compatible format

        :return: None
        :rtype: None
        """
        for key, item in self.conf_dict.items():
            if isinstance(item, list):
                print('{0}:'.format(key))
                for subitem in item:
                    print('    - {0}'.format(subitem))
            elif isinstance(item, dict):
                print('{0}:'.format(key))
                for subkey, subitem in item.items():
                    print('    {0}: {1}'.format(subkey, subitem))
            else:
                print('{0}: {1}'.format(key, item))
