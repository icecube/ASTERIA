# -*- coding: utf-8 -*-
"""CCSN neutrino sources.

Initialize a source using the models available in SNEWPY.

See either of these two resources:
- snewpy.readthedocs.io
- https://github.com/SNEWS2/snewpy
"""
try:
    from snewpy.models.registry import init_model as init_snewpy_model_from_param
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
import os

import astropy.units as u
import numpy as np
import snewpy.models as snewpy_models

from .util import lookup_dict as _fname_lookup_dict
#
# def _get_modelpath(modelname, modelfile):
#     """Utility function to get the path to the model flux file.
#     If flux file is not present in ASTERIA/models/modelname, download it.
#
#     Parameters
#     ----------
#     modelname : str
#         Name of the model, e.g., Bollig_2016.
#     modelfile : str
#         Name of the corresponding flux file (omitting path).
#
#     Returns
#     -------
#     modfile : str
#         Full absolute path to the model file.
#     """
#     # Check modelname is in list of supported models.
#     if modelname in model_urls:
#         # Put model flux files into ASTERIA/models/modelname.
#         modeldir = '/'.join([os.environ['ASTERIA'], 'models'])
#         modfile = '/'.join([modeldir, modelname, modelfile])
#         if os.path.exists(modfile):
#             return modfile
#         else:
#             # Get model flux file if it's not found already.
#             get_models(modelname, download_dir=modeldir)
#
#         # Final check that the file was successfully downloaded.
#         if os.path.exists(modfile):
#             return modfile
#         raise RuntimeError(f'{modfile} does not exist.')
#     raise ValueError(f'Unrecognized model {modelname}.')
#
#
# def init_model(modelname, modelfile, *args):
#     """Given model name and file, initialize a model object.
#
#     Parameters
#     ----------
#     modelname : str
#         Name of the model, e.g., Bollig_2016.
#     modelfile : str
#         Name of the corresponding flux file (omitting path).
#
#     Returns
#     -------
#     model : snewpy.SupernovaModel
#         Instance of this particular supernova model.
#     """
#     # Get a valid path to a model flux file.
#     modfile = _get_modelpath(modelname, modelfile)
#
#     # Return a class instance of the modelname.
#     return getattr(sys.modules[__name__], modelname)(modfile)


def init_snewpy_model(model, model_params):
    try:
        fname = _fname_lookup_dict[model](**model_params)
    except KeyError:
        raise NotImplementedError(f'Unknown Model Requested, Allowed models are {tuple(_fname_lookup_dict.keys())}')
    # This is necessary to enable ASTERIA to use other SNEWPY models using the source_dec21
    if model == 'OConnor_2013':
        mass, EOS = fname  # In this instance fname is actually a tuple of params. Yes, this is problematic.
        return init_snewpy_model_from_param(model_name=model, base=os.path.join(model_path, f'{model}/'),
                                            mass=mass, eos=EOS)
    return init_snewpy_model_from_param(model_name=model, filename=os.path.join(model_path, model, fname))


class Source:

    def __init__(self, model, model_params=None):
        if model_params is None:
            model_params = {}
        if model == 'Nakazato_2013':
            self.model = init_snewpy_model_from_param(model, **model_params)
        else:
            self.model = init_snewpy_model(model, model_params)
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

# class Source:
#
#     def __init__(self, name=None, spectral_model=None, progenitor_mass=None, progenitor_distance=None, time=None,
#                  luminosity=None, mean_energy=None, pinch=None):
#         if any(p is None for p in (name, spectral_model, progenitor_mass, progenitor_distance,
#                                    time, luminosity, mean_energy, pinch)):
#             raise ValueError('Missing required arguments to initialize source')
#         self.name = name
#         self.model = spectral_model
#         self.progenitor_mass = progenitor_mass
#         self.progenitor_distance = progenitor_distance
#         self.time = time
#         self.luminosity = luminosity
#         self.mean_energy = mean_energy
#         self.pinch = pinch
#
#         # Energy PDF function is assumed to be like a gamma function,
#         # parameterized by mean energy and pinch parameter alpha. True for
#         # nearly all CCSN models.
#         self.energy_pdf = lambda a, Ea, E: \
#             np.exp((1 + a) * np.log(1 + a) - loggamma(1 + a) + a * np.log(E) - \
#                    (1 + a) * np.log(Ea) - (1 + a) * (E / Ea))
#
#         self.v_energy_pdf = np.vectorize(self.energy_pdf, excluded=['E'], signature='(1,n),(1,n)->(m,n)')
#
#         # Energy CDF, useful for random energy sampling.
#         self.energy_cdf = lambda a, Ea, E: \
#             gdtr(1., a + 1., (a + 1.) * (E / Ea))


#
#


#
#    def energy_spectrum(self, time, E, flavor=Flavor.NU_E_BAR):
#        """Compute the PDF of the neutrino energy distribution at time t.
#
#        Parameters
#        ----------
#
#        t : float
#            Time relative to core bounce.
#        flavor : :class:`asteria.neutrino.Flavor`
#            Neutrino flavor.
#        E : `numpy.ndarray`
#            Sorted grid of neutrino energies to compute the energy PDF.
#
#        Returns
#        -------
#        spectrum : `numpy.ndarray`
#            Table of PDF values computed as a function of energy.
#        """
#        # Given t, get current average energy and pinch parameter.
#        # Use simple 1D linear interpolation
#        t = time.to(u.s).value
#        Enu = E.to(u.MeV).value
#        if Enu[0] == 0.:
#            Enu[0] = 1e-10  # u.MeV
#        a = self.get_pinch_parameter(t, flavor)
#        Ea = self.get_mean_energy(t, flavor).to(u.MeV).value
#
#        if isinstance(t, (list, tuple, np.ndarray)):
#            # It is non-physical to have a<0 but some model files/interpolations still have this
#            a[a<0] = 0
#            cut = (a >= 0) & (Ea > 0)
#            E_pdf = np.zeros( (Enu.size, t.size), dtype = float )
#            E_pdf[:, cut] = self.v_energy_pdf( a[cut].reshape(1,-1), Ea[cut].reshape(1,-1), \
#                                               E=Enu.reshape(-1,1))
#            cut = (a < 0) & (Ea > 0)
#            E_pdf[:, cut] = self.v_energy_pdf(np.zeros_like(a[cut]).reshape(1, -1), Ea[cut].reshape(1, -1), \
#                                              E=Enu.reshape(-1, 1))
#            return E_pdf
#        else:
#            if Ea <= 0.:
#                return np.zeros_like(E)
#            elif a <= 0.:
#                return self.energy_pdf(0, Ea, E.value).real
#            else:
#                return self.energy_pdf(a, Ea, E.value).real
#
#    def sample_energies(self, t, E, n=1, flavor=Flavor.NU_E_BAR):
#        """Generate a random sample of neutrino energies at some time t for a
#        particular neutrino flavor. The energies are generated via inverse
#        transform sampling of the CDF of the neutrino energy distribution.
#
#        Parameters
#        ----------
#
#        t : float
#            Time relative to core bounce.
#        E : `numpy.ndarray`
#            Sorted grid of neutrino energies to compute the energy PDF.
#        n : int
#            Number of energy samples to produce.
#        flavor : :class:`asteria.neutrino.Flavor`
#            Neutrino flavor.
#
#        Returns
#        -------
#        energies : `numpy.ndarray`
#            Table of energies sampled from the energy spectrum.
#        """
#        cdf = self.energy_cdf(flavor, t, E)
#        energies = np.zeros(n, dtype=float)
#
#        # Generate a random number between 0 and 1 and compare to the CDF
#        # of the neutrino energy distribution at time t
#        u = np.random.uniform(n)
#        j = np.searchsorted(cdf, u)
#
#        # Linearly interpolate in the CDF to produce a random energy
#        energies[j <= 0] = E[0].to('MeV').value
#        energies[j >= len(E)-1] = E[-1].to('MeV').value
#
#        cut = (0 < j) & (j < len(E)-1)
#        j = j[cut]
#        en = E[j] + (E[j+1] - E[j]) / (cdf[j+1] - cdf[j]) * (u[cut] - cdf[j])
#        energies[cut] = en
#
#        return energies
#
#    def photonic_energy_per_vol(self, time, E, flavor, photon_spectrum, mixing=None, n=1000):
#        """Compute the energy deposited in a cubic meter of ice by photons
#        from SN neutrino interactions.
#
#        Parameters
#        ----------
#
#        time : float (units s)
#            Time relative to core bounce.
#        E : `numpy.ndarray`
#            Sorted grid of neutrino energies
#        flavor : :class:`asteria.neutrino.Flavor`
#            Neutrino flavor.
#        photon_spectrum : `numpy.ndarray` (Units vary, m**2)
#            Grid of the product of lepton cross section with lepton mean energy
#            and lepton path length per MeV, sorted according to parameter E
#        n : int
#            Maximum number of time steps to compute at once. A temporary numpy array
#            of size n x time.size is created and can be very memory inefficient.
#
#        Returns
#        -------
#        E_per_V
#            Energy per m**3 of ice deposited  by neutrinos of requested flavor
#        """
#        H2O_in_ice = 3.053e28 # 1 / u.m**3
#
#        t = time.to(u.s).value
#        Enu = E.to(u.MeV).value
#        if Enu[0] == 0:
#            Enu[0] = 1e-10 * u.MeV
#        phot = photon_spectrum.to(u.m**2).value.reshape((-1,1)) # m**2
#
#        dist = self.progenitor_distance.to(u.m).value # m**2
#        flux = self.get_flux( time, flavor ) # s**-1
#
#        if mixing is None:
#            def nu_spectrum(t, E, flavor):
#                return self.energy_spectrum(t, E, flavor) * self.get_flux(t, flavor)
#        else:
#            nu_spectrum = mixing(self)
#
#        print('Beginning {0} simulation... {1}'.format(flavor.name, ' '*(10-len(flavor.name))), end='')
#        # The following two lines exploit the fact that astropy quantities will
#        # always return a number when numpy size is called on them, even if it is 1.
#        E_per_V = np.zeros( time.size )
#        if time.size < 2:
#            raise RuntimeError("Time array size <2, unable to compute energy per volume.")
#        for i_part in self.parts_by_index(time, n): # Limits memory usage
#             E_per_V[i_part] += np.trapz( nu_spectrum(time[i_part], E, flavor).value * phot, Enu, axis=0)
#        E_per_V *= H2O_in_ice / ( 4 * np.pi * dist**2) * np.ediff1d(t, to_end=(t[-1] - t[-2]))
#        if not flavor.is_electron:
#            E_per_V *= 2
#        print('Completed')
#
#        return E_per_V * u.MeV / u.m**3
#
#


# def initialize(config):
#     """Initialize a Source model from configuration parameters.
#
#     Parameters
#     ----------
#
#     config : :class:`asteria.config.Configuration`
#        Configuration parameters used to create a Source.
#
#     Returns
#     -------
#     Source
#        An initialized source model.
#     """
#     # Dictionary of L, <E>, and alpha versus time, keyed by neutrino flavor.
#     luminosity, mean_energy, pinch = {}, {}, {}
#
#     if config.source.table.format.lower() == 'fits':
#         # Open FITS file, which contains a luminosity table and a pinching
#         # parameter (alpha) and mean energy table.
#         fitsfile = '/'.join([config.abs_base_path, config.source.table.path])
#         sn_data_table = Table.read(fitsfile)
#
#         time = sn_data_table['TIME'].to('s')
#
#         # Loop over all flavors in the table:
#         for flavor in Flavor:
#             fl = flavor.name.upper()
#             if any(fl in col for col in sn_data_table.keys()):
#
#                 L = sn_data_table['L_{:s}'.format(fl)].to('erg/s')
#                 E = sn_data_table['E_{:s}'.format(fl)].to('MeV')
#                 alpha = sn_data_table['ALPHA_{:s}'.format(fl)]
#
#             elif fl == 'NU_X_BAR':
#                 L = sn_data_table['L_NU_X'].to('erg/s')
#                 E = sn_data_table['E_NU_X'].to('MeV')
#                 alpha = sn_data_table['ALPHA_NU_X']
#
#             else:
#                 raise KeyError("""'{0}'""".format(fl))
#
#             luminosity[flavor] = PchipInterpolator(time, L, extrapolate=False)
#             mean_energy[flavor] = PchipInterpolator(time, E, extrapolate=False)
#             pinch[flavor] = PchipInterpolator(time, alpha, extrapolate=False)
#     elif config.source.table.format.lower() == 'ascii':
#         # ASCII will be supported! Promise, promise.
#         raise ValueError('Unsupported format: "ASCII"')
#     else:
#         raise ValueError('Unknown format {}'.format(config.source.table.format))
#
#     # Set up the distance model.
#     distance_model = None
#     dmtype = config.source.progenitor.distance.model
#     if dmtype == 'FixedDistance':
#         # FixedDistance model.
#         r = parse_quantity(config.source.progenitor.distance.distance)
#         dr = parse_quantity(config.source.progenitor.distance.uncertainty)
#         distance_model = FixedDistance(r, dr)
#     elif dmtype == 'StellarDensity':
#         # StellarDensity model, with options to add LMC and SMC.
#         fitsfile = '/'.join([config.abs_base_path,
#                              config.source.progenitor.distance.path])
#         lmc = parse_quantity(config.source.progenitor.distance.add_LMC)
#         smc = parse_quantity(config.source.progenitor.distance.add_SMC)
#         distance_model = StellarDensity(fitsfile, lmc, smc)
#     else:
#         raise ValueError('Unrecognized distance_model: {}'.format(dmtype))
#
#     return Source(config.source.name,
#                   config.source.model,
#                   parse_quantity(config.source.progenitor.mass),
#                   distance_model.distance(),
#                   time,
#                   luminosity,
#                   mean_energy,
#                   pinch)
