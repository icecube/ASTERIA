# -*- coding: utf-8 -*-
"""CCSN spectral model I/O.

This module encapsulates the basic parameters of neutrino fluxes from
supernovae as modeled in the CCSN literature. For each species of neutrino one
requires an estimate of the luminosity vs. time as well as the energy spectrum
of the neutrinos at any given time.

The NuFlux class in this module contains the necessary access to both
luminosity and spectrum vs. time as a function of neutrino flavor. It is
expected that the model data are stored in FITS files.
"""

import numpy as np
from scipy.special import loggamma, gdtr

import astropy.units as u
from astropy.table import Table
from astropy.modeling.tabular import Tabular1D

from .neutrino import Flavor


class NuFlux:
    """Encapsulate the neutrino luminosity and energy PDFs of a CCSN model.
    """

    def __init__(self, fitsfile):
        """Read in CCSN model data from a FITS file.

        :param fitsfile: name of CCSN model file.
        """

        table = Table.read(fitsfile)

        self.t = table['TIME'].to('s')
        self.E, self.L, self.alpha = {}, {}, {}
        self.EvsT, self.LvsT, self.AvsT = {}, {}, {}

        for flavor in Flavor:
            fl = flavor.name.upper()

            self.E[flavor] = table['E_{:s}'.format(fl)].to('MeV')
            self.L[flavor] = table['L_{:s}'.format(fl)].to('erg/s')
            self.alpha[flavor] = table['ALPHA_{:s}'.format(fl)]

            self.EvsT[flavor] = Tabular1D(self.t, self.E[flavor])
            self.LvsT[flavor] = Tabular1D(self.t, self.L[flavor])
            self.AvsT[flavor] = Tabular1D(self.t, self.alpha[flavor])

        self.specPDF = lambda a, Ea, E: np.exp(
            (1 + a) * np.log(1 + a) - loggamma(1 + a) + a * np.log(E) - (1 + a) * np.log(Ea) - (1 + a) * (E / Ea))
        self.specCDF = lambda a, Ea, E: gdtr(1., a + 1., (a + 1.) * (E / Ea))

    def energy_spectrum_pdf(self, flavor, t, E):
        """Compute the PDF of the neutrino energy distribution at time t.

         :param flavor: neutrino flavor enum type.
         :param t: time w.r.t. core bounce.
         :param E: sorted array of neutrino energies to used to compute the neutrino energy PDF.
         :return pdf: table of PDF values computed at E.
        """
        # Given t, get current average energy and pinch parameter.
        # Use simple 1D linear interpolation
        a = self.AvsT[flavor](t)
        Ea = self.EvsT[flavor](t)
        if a == 0. or Ea == 0.:
            return np.zeros(len(E), dtype=float)
        if E[0] == 0.:
            E[0] = 1e-10 * u.MeV
        return self.specPDF(a, Ea, E.value).real

    def energy_spectrum_cdf(self, flavor, t, E):
        """Compute the CDF of the neutrino energy distribution at time t.

         :param flavor: neutrino flavor enum type.
         :param t: time w.r.t. core bounce.
         :param E: sorted array of neutrino energies to used to compute the neutrino energy CDF.
         :return cdf: table of CDF values computed at E.
        """
        # Given t, get current average energy and pinch parameter.
        # Use simple 1D linear interpolation
        a = self.AvsT[flavor](t)
        Ea = self.EvsT[flavor](t)
        if a == 0. or Ea == 0.:
            return np.zeros(len(E), dtype=float)
        return self.specCDF(a, Ea, E.value).real

    def sample_nu_energy(self, flavor, t, E, n=1):
        """Generate a random sample of neutrino energies at some time t for a
        particular neutrino flavor. The energies are generated via inverse
        transform sampling of the CDF of the neutrino energy distribution.

        :param flavor: neutrino flavor enum type.
        :param t: time w.r.t. core bounce.
        :param E: sorted array of neutrino energies to used to compute the neutrino energy CDF.
        :param n: number of energies to sample from the CDF.
        :return energies: list of n energies sampled from the CDF.
        """
        cdf = self.energy_spectrum_cdf(flavor, t, E)
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
        en = E[j] + (E[j + 1] - E[j]) / (cdf[j + 1] - cdf[j]) * (u[cut] - cdf[j])
        energies[cut] = en

        return energies
