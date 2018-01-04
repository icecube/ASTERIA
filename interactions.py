from abc import ABC, abstractmethod

import astropy.units as u
import astropy.constants as c
from astropy.modeling.tabular import Tabular1D
from astropy.io import ascii

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class Interaction(ABC):

    def __init__(self):
        # Define masses in rational units
        self.Mn = (c.m_n*c.c**2).to('MeV').value
        self.Mp = (c.m_p*c.c**2).to('MeV').value
        self.Me = (c.m_e*c.c**2).to('MeV').value

        super().__init__()

    @abstractmethod
    def cross_section(self, Enu):
        pass

    @abstractmethod
    def mean_lepton_energy(self, Enu):
        pass


class InvBeta(Interaction):

    def __init__(self):
        super().__init__()

        # Calculate IBD threshold
        self.Eth = self.Mn + self.Me - self.Mp
        self.delta = (self.Mn**2 - self.Mp**2 -self.Me**2) / (2*self.Mp)

    def cross_section(self, Enu):
        """Inverse beta decay cross section parameterization from Strumia and
        Vissani, Phys. Lett. B 564:42, 2003.

        :param Enu: neutrino energy with proper units. Can be an array.
        :return: Inverse beta cross section.
        """
        # Convert all units to MeV
        Enu = Enu.to('MeV').value

        # Calculate mean positron energy and momentum using the crappy estimate
        # from Strumia and Vissani eq. 25
        Ee = Enu - (self.Mn - self.Mp)
        pe = np.sqrt(Ee**2 - self.Me**2)

        # Handle possibility of list/array input
        if isinstance(Enu, (list, tuple, np.ndarray)):
            xs = np.zeros(len(Enu), dtype=float)
            cut = Enu > self.Eth
            xs[cut] = 1e-43 * pe[cut] * Ee[cut] * \
                      Enu[cut]**(-0.07056+0.02018*np.log(Enu[cut])-0.001953*np.log(Enu[cut])**3)
            return xs * u.cm**2
        # Handle float input
        else:
            if Enu <= self.Eth:
                return 0.*u.cm**2
            return 1e-43*u.cm**2 * pe * Ee * \
                   Enu**(-0.07056+0.02018*np.log(Enu)-0.001953*np.log(Enu)**3)

    def mean_lepton_energy(self, Enu):
        # Mean lepton energy from Abbasi et al., A&A 535:A109, 2011, p.6.
        # Could also use Strumia and Vissani eq. 16
        Elep = (Enu - self.delta)*(1. - Enu/(Enu + self.Mp))
        return Elep * u.MeV


class InvBetaTabular(Interaction):

    def __init__(self):
        super().__init__()

        # Calculate IBD threshold
        self.Eth = self.Mn + self.Me - self.Mp

        # Table of energy values [MeV]
        self.E = [1.806, 2.01, 2.25, 2.51, 2.8, 3.12, 3.48, 3.89, 4.33, 4.84,
                  5.4, 6.02, 6.72, 7.49, 8.36, 8.83, 9.85, 11.0, 12.3, 13.7,
                 15.3, 17.0, 19.0, 21.2, 23.6, 26.4, 29.4, 32.8, 36.6, 40.9,
                 43.2, 48.2, 53.7, 59.9, 66.9, 74.6, 83.2, 92.9, 104.0, 116.0,
                 129.0, 144.0, 160.0, 179.0, 200.0]

        # Table of cross sections [1e-41 cm2]
        self.x = [0., 0.00351, 0.00735, 0.0127, 0.0202, 0.0304, 0.044, 0.0619,
                  0.0854, 0.116, 0.155, 0.205, 0.269, 0.349, 0.451, 0.511,
                  0.654, 0.832, 1.05, 1.33, 1.67, 2.09, 2.61, 3.24,
                  4.01, 4.95, 6.08, 7.44, 9.08, 11.0, 12.1, 14.7,
                 17.6, 21.0, 25.0, 29.6, 34.8, 40.7, 47.3,
                 54.6, 62.7, 71.5, 81.0, 91.3, 102.0]

        self.XvsE = Tabular1D(self.E, self.x)

    def cross_section(self, Enu):
        """Tabular inverse beta decay cross section from Strumia and Vissani,
        Phys. Lett. B 564:42, 2003.

        :param Enu: neutrino energy with proper units. Can be an array.
        :return: Inverse beta cross section.
        """
        # Convert all units to MeV
        Enu = Enu.to('MeV').value

        if isinstance(Enu, (list, tuple, np.ndarray)):
            cut = Enu > self.Eth
            xs = np.zeros(len(Enu), dtype=float)
            xs[cut] = self.XvsE(Enu[cut]) * 1e-41 * u.cm**2
            return xs
        else:
            if Enu > self.Eth:
                return self.XvsE(Enu) * 1e-41 * u.cm**2
            return 0.

    def mean_lepton_energy(self, Enu):
        pass

fig, ax = plt.subplots(1,1, figsize=(9,6))

Enu = np.linspace(0., 200., 101) * u.MeV

for ibd in [InvBeta(), InvBetaTabular()]:
    xs = ibd.cross_section(Enu)
    ax.plot(Enu, xs/1e-41)

#ibd_parameterized = InvBeta()
#xs = ibd_parameterized.crossSection(Enu)
#ax.plot(Enu, xs/1e-41)
#
#ibd_tabular = InvBetaTabular()
#xs = ibd_tabular.crossSection(Enu)
#ax.plot(Enu, xs/1e-41, '.')

ax.set(xlabel=r'$E_{\nu}$ [MeV]',
       ylabel=r'$\sigma(\bar{\nu}_e p)$ [10$^{-41}$ cm$^2$]')
fig.tight_layout()

plt.show()
