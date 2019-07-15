# -*- coding: utf-8 -*-
"""CCSN neutrino sources.

This module encapsulates the basic parameters of neutrino fluxes from
supernovae as modeled in the CCSN literature. For each species of neutrino one
requires an estimate of the luminosity vs. time as well as the energy spectrum
of the neutrinos at any given time.
"""

from __future__ import print_function, division

from .neutrino import Flavor
from .stellardist import FixedDistance, StellarDensity
from .config import parse_quantity

from astropy import units as u
from astropy.table import Table

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import loggamma, gdtr
from scipy.interpolate import PchipInterpolator


class Source:
    
    def __init__(self, name, 
                 spectral_model, progenitor_mass, progenitor_distance,
                 time={}, luminosity={}, mean_energy={}, pinch={}):

        self.name = name
        self.model = spectral_model
        self.progenitor_mass = progenitor_mass
        self.progenitor_distance = progenitor_distance
        self.time = time
        self.luminosity = luminosity
        self.mean_energy = mean_energy
        self.pinch = pinch

        # Energy PDF function is assumed to be like a gamma function,
        # parameterized by mean energy and pinch parameter alpha. True for
        # nearly all CCSN models.
        self.energy_pdf = lambda a, Ea, E: \
            np.exp((1 + a) * np.log(1 + a) - loggamma(1 + a) + a * np.log(E) - \
                   (1 + a) * np.log(Ea) - (1 + a) * (E / Ea))
                   
        self.v_energy_pdf = np.vectorize(self.energy_pdf, excluded=['E'], signature='(1,n),(1,n)->(m,n)' )

        # Energy CDF, useful for random energy sampling.
        self.energy_cdf = lambda a, Ea, E: \
            gdtr(1., a + 1., (a + 1.) * (E / Ea))

    def parts_by_index(self, x, n): 
        """Returns a list of size-n numpy arrays containing indices for the 
        elements of x, and one size-m array ( with m<n ) if there are remaining 
        elements of x.

        Returns
        -------
        i_part : list
            List of index partitions (partitions are numpy array).
        """
        nParts = x.size//n    
        i_part = [ np.arange( i*n, (i+1)*n ) for i in range(nParts) ]

        if len(i_part)*n != x.size:
            i_part += [ np.arange( len(i_part)*n, x.size ) ]
        
        return i_part
        
    def get_time(self):
        """Return source time as numpy array.

        Returns
        -------
        time : float
            Source time profile (units of s).
        """
        return self.time
    
    def get_luminosity(self, t, flavor=Flavor.nu_e_bar):
        """Return source luminosity at time t for a given flavor.

        Parameters
        ----------
        
        t : float
            Time relative to core bounce.
        flavor : :class:`asteria.neutrino.Flavor`
            Neutrino flavor.

        Returns
        -------
        luminosity : float
            Source luminosity (units of power).
        """
        return np.nan_to_num(self.luminosity[flavor](t)) * (u.erg / u.s)

    def get_mean_energy(self, t, flavor=Flavor.nu_e_bar):
        """Return source mean energy at time t for a given flavor.

        Parameters
        ----------
        
        t : float
            Time relative to core bounce.
        flavor : :class:`asteria.neutrino.Flavor`
            Neutrino flavor.

        Returns
        -------
        mean_energy : float
            Source mean energy (units of energy).
        """
        return np.nan_to_num(self.mean_energy[flavor](t)) * u.MeV

    def get_pinch_parameter(self, t, flavor=Flavor.nu_e_bar):
        """Return source pinching paramter alpha at time t for a given flavor.
        Parameters
        ----------
        
        t : float
            Time relative to core bounce.
        flavor : :class:`asteria.neutrino.Flavor`
            Neutrino flavor.
        Returns
        -------
        pinch : float
            Source pinching parameter (unitless).
        """
        return np.nan_to_num(self.pinch[flavor](t))

    def get_flux(self, time, flavor=Flavor.nu_e_bar):
        """Return source flux at time t for a given flavor.

        Parameters
        ----------
        
        t : float
            Time relative to core bounce (units seconds).
        flavor : :class:`asteria.neutrino.Flavor`
            Neutrino flavor.

        Returns
        -------
        flux : float
            Source number flux (unit-less, count of neutrinos).
        """      
        t = time.to(u.s).value
        luminosity  = self.get_luminosity(t, flavor).to( u.MeV/u.s ).value
        mean_energy = self.get_mean_energy(t, flavor).value
        
        # Where the mean energy is not zero, return rate in units neutrinos
        # per second, elsewhere, returns zero.
        rate = np.divide(luminosity, mean_energy, where=(mean_energy != 0),
                         out=np.zeros(luminosity.size))
        flux = np.ediff1d(t, to_end=(t[-1] - t[-2])) * rate
        
        return flux
			 
    def energy_spectrum(self, t, E, flavor=Flavor.nu_e_bar):
        """Compute the PDF of the neutrino energy distribution at time t.

        Parameters
        ----------

        t : float
            Time relative to core bounce.
        flavor : :class:`asteria.neutrino.Flavor`
            Neutrino flavor.
        E : `numpy.ndarray`
            Sorted grid of neutrino energies to compute the energy PDF.

        Returns
        -------
        spectrum : `numpy.ndarray`
            Table of PDF values computed as a function of energy.
        """
        # Given t, get current average energy and pinch parameter.
        # Use simple 1D linear interpolation
        a = self.get_pinch_parameter(t, flavor)
        Ea = self.get_mean_energy(t, flavor).to(u.MeV).value
        Enu = E.to(u.MeV).value

        if E[0] == 0.:
            E[0] = 1e-10  # u.MeV
        
        if isinstance(t, (list, tuple, np.ndarray)):
            cut = (a > 0) & (Ea > 0)
            E_pdf = np.zeros( (Enu.size, t.size), dtype = float )
            E_pdf[:, cut] = self.v_energy_pdf( a[cut].reshape(1,-1), Ea[cut].reshape(1,-1), \
                                               E =Enu.reshape(-1,1))
            return E_pdf
        else:
            if a <= 0. or Ea <= 0.:
                return np.zeros_like(E)
            
            return self.energy_pdf(a, Ea, E.value).real

    def sample_energies(self, t, E, n=1, flavor=Flavor.nu_e_bar):
        """Generate a random sample of neutrino energies at some time t for a
        particular neutrino flavor. The energies are generated via inverse
        transform sampling of the CDF of the neutrino energy distribution.

        Parameters
        ----------

        t : float
            Time relative to core bounce.
        E : `numpy.ndarray`
            Sorted grid of neutrino energies to compute the energy PDF.
        n : int
            Number of energy samples to produce.
        flavor : :class:`asteria.neutrino.Flavor`
            Neutrino flavor.

        Returns
        -------
        energies : `numpy.ndarray`
            Table of energies sampled from the energy spectrum.
        """
        cdf = self.energy_cdf(flavor, t, E)
        energies = np.zeros(n, dtype=float)

        # Generate a random number between 0 and 1 and compare to the CDF
        # of the neutrino energy distribution at time t
        u = np.random.uniform(n)
        j = np.searchsorted(cdf, u)

        # Linearly interpolate in the CDF to produce a random energy
        energies[j <= 0] = E[0].to('MeV').value
        energies[j >= len(E)-1] = E[-1].to('MeV').value

        cut = (0 < j) & (j < len(E)-1)
        j = j[cut]
        en = E[j] + (E[j+1] - E[j]) / (cdf[j+1] - cdf[j]) * (u[cut] - cdf[j])
        energies[cut] = en

        return energies
        
    def photonic_energy_per_vol(self, time, E, flavor, photon_spectrum, mixing=None, n=1000):
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
        H2O_in_ice = 3.053e28 # 1 / u.m**3
                
        t = time.to(u.s).value
        Enu = E.to(u.MeV)
        if Enu[0] == 0:
            Enu[0] = 1e-10 * u.MeV
        phot = photon_spectrum.to(u.m**2).value.reshape((-1,1)) # m**2
        
        dist = self.progenitor_distance.to(u.m).value # m**2
        flux = self.get_flux( time, flavor ) # Unitless
        
        if mixing is None:
            def nu_spectrum(t, E, flavor):
                return self.energy_spectrum(t, E, flavor) * self.get_flux(t, flavor)
        else:
            nu_spectrum = mixing( self, flavor )

        
        
        print('Beginning {0} simulation... {1}'.format(flavor.name, ' '*(10-len(flavor.name))), end='')
        # The following two lines exploit the fact that astropy quantities will
        # always return a number when numpy size is called on them, even if it is 1.
        E_per_V = np.zeros( time.size ) 
        for i_part in self.parts_by_index(time, n): # Limits memory usage
             E_per_V[i_part] += np.trapz( nu_spectrum(time[i_part], Enu, flavor) * phot, Enu.value, axis=0)
        E_per_V *= H2O_in_ice / ( 4 * np.pi * dist**2)
        if not flavor.is_electron:
            E_per_V *= 2
        print('Completed')
    
        return E_per_V * u.MeV / u.m**3


def initialize(config):
    """Initialize a Source model from configuration parameters.

    Parameters
    ----------

    config : :class:`asteria.config.Configuration`
        Configuration parameters used to create a Source.

    Returns
    -------
    Source
        An initialized source model.
    """
    # Dictionary of L, <E>, and alpha versus time, keyed by neutrino flavor.
    luminosity, mean_energy, pinch = {}, {}, {}

    if config.source.table.format.lower() == 'fits':
        # Open FITS file, which contains a luminosity table and a pinching
        # parameter (alpha) and mean energy table.
        fitsfile = '/'.join([config.abs_base_path, config.source.table.path])
        sn_data_table = Table.read(fitsfile)

        time = sn_data_table['TIME'].to('s')

        # Loop over all flavors in the table:
        for flavor in Flavor:
            fl = flavor.name.upper()
            if any( fl in col for col in sn_data_table.keys() ):

                L = sn_data_table['L_{:s}'.format(fl)].to('erg/s')
                E = sn_data_table['E_{:s}'.format(fl)].to('MeV')
                alpha = sn_data_table['ALPHA_{:s}'.format(fl)]

            elif fl == 'NU_X_BAR':
                L = sn_data_table['L_NU_X'].to('erg/s')
                E = sn_data_table['E_NU_X'].to('MeV')
                alpha = sn_data_table['ALPHA_NU_X'] 
            
            else:
                raise KeyError("""'{0}'""".format(fl)) 
                
            luminosity[flavor] = PchipInterpolator(time, L, extrapolate=False )
            mean_energy[flavor] = PchipInterpolator(time, E, extrapolate=False )
            pinch[flavor] = PchipInterpolator(time, alpha, extrapolate=False )  
    elif config.source.table.format.lower() == 'ascii':
        # ASCII will be supported! Promise, promise.
        raise ValueError('Unsupported format: "ASCII"')
    else:
        raise ValueError('Unknown format {}'.format(config.source.table.format))

    # Set up the distance model.
    distance_model = None
    dmtype = config.source.progenitor.distance.model
    if dmtype == 'FixedDistance':
        # FixedDistance model.
        r  = parse_quantity(config.source.progenitor.distance.distance)
        dr = parse_quantity(config.source.progenitor.distance.uncertainty)
        distance_model = FixedDistance(r, dr)
    elif dmtype == 'StellarDensity':
        # StellarDensity model, with options to add LMC and SMC.
        fitsfile = '/'.join([config.abs_base_path,
                             config.source.progenitor.distance.path])
        lmc = parse_quantity(config.source.progenitor.distance.add_LMC)
        smc = parse_quantity(config.source.progenitor.distance.add_SMC)
        distance_model = StellarDensity(fitsfile, lmc, smc)
    else:
        raise ValueError('Unrecognized distance_model: {}'.format(dmtype))

    return Source(config.source.name,
                  config.source.model,
                  parse_quantity(config.source.progenitor.mass),
                  distance_model.distance(),
                  time,
                  luminosity,
                  mean_energy,
                  pinch)
