from neutrino import Flavor

from abc import ABC, abstractmethod

import astropy.units as u
import astropy.constants as c
from scipy import interpolate

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class Interaction(ABC):

    def __init__(self):
        # Define masses in rational units
        self.Mn = (c.m_n * c.c ** 2).to('MeV').value
        self.Mp = (c.m_p * c.c ** 2).to('MeV').value
        self.Me = (c.m_e * c.c ** 2).to('MeV').value

        super().__init__()

    @abstractmethod
    def cross_section(self, flavor, e_nu):
        pass

    @abstractmethod
    def mean_lepton_energy(self, flavor, e_nu):
        pass


class InvBetaPar(Interaction):
    """Inverse beta decay parameterization from Strumia and Vissani,
    Phys. Lett. B 564:42, 2003."""

    def __init__(self):
        super().__init__()

        # Calculate IBD threshold
        self.Eth = self.Mn + self.Me - self.Mp
        self.delta = (self.Mn ** 2 - self.Mp ** 2 - self.Me ** 2) / (2 * self.Mp)

    def cross_section(self, flavor, e_nu):
        """Inverse beta decay cross section, Strumia and Vissani eq. 25.

        :param e_nu: neutrino energy with proper units. Can be an array.
        :return: Inverse beta cross section.
        """
        # Only works for electron antineutrinos
        if flavor != Flavor.nu_e_bar:
            if isinstance(e_nu, (list, tuple, np.ndarray)):
                return np.zeros(len(e_nu), dtype=float) * u.cm**2
            return 0. * u.cm**2

        # Convert all units to MeV
        Enu = e_nu.to('MeV').value

        # Calculate mean positron energy and momentum using the crappy estimate
        # from Strumia and Vissani eq. 25.
        Ee = Enu - (self.Mn - self.Mp)
        pe = np.sqrt(Ee ** 2 - self.Me ** 2)

        # Handle possibility of list/array input
        if isinstance(Enu, (list, tuple, np.ndarray)):
            xs = np.zeros(len(Enu), dtype=float)
            cut = Enu > self.Eth
            xs[cut] = 1e-43 * pe[cut] * Ee[cut] * \
                Enu[cut] ** (-0.07056 + 0.02018 * np.log(Enu[cut]) - 0.001953 * np.log(Enu[cut]) ** 3)
            return xs * u.cm ** 2
        # Handle float input
        else:
            if Enu <= self.Eth:
                return 0. * u.cm ** 2
            return 1e-43 * u.cm ** 2 * pe * Ee * \
                Enu ** (-0.07056 + 0.02018 * np.log(Enu) - 0.001953 * np.log(Enu) ** 3)

    def mean_lepton_energy(self, flavor, e_nu):
        """Mean lepton energy from Abbasi et al., A&A 535:A109, 2011, p.6.
        Could also use Strumia and Vissani eq. 16.
        """
        # Only works for electron antineutrinos
        if flavor != Flavor.nu_e_bar:
            if isinstance(e_nu, (list, tuple, np.ndarray)):
                return np.zeros(len(e_nu), dtype=float) * u.MeV
            return 0. * u.MeV

        Enu = e_nu.to('MeV').value

        # Handle possibility of list/array input
        if isinstance(Enu, (list, tuple, np.ndarray)):
            lep = np.zeros(len(Enu), dtype=float)
            cut = Enu > self.Eth
            lep[cut] = (Enu[cut] - self.delta) * (1. - Enu[cut] / (Enu[cut] + self.Mp))
            return lep * u.MeV
        else:
            if Enu > self.Eth:
                return (Enu - self.delta) * (1. - Enu / (Enu + self.Mp))
            return 0. * u.MeV


class InvBetaTab(Interaction):
    """Tabulated inverse beta decay computation by Strumia and Vissani,
    Phys. Lett. B 564:42, 2003."""

    def __init__(self):
        super().__init__()

        # Calculate IBD threshold
        self.Eth = self.Mn + self.Me - self.Mp
#        self.logEth = np.log10(self.Mn + self.Me - self.Mp)

        # Tabulated energy values [MeV], Strumia and Vissani Table 1
        self.E = [1.806, 2.01, 2.25, 2.51, 2.8, 3.12, 3.48, 3.89, 4.33, 4.84,
                  5.4, 6.02, 6.72, 7.49, 8.36, 8.83, 9.85, 11.0, 12.3, 13.7,
                  15.3, 17.0, 19.0, 21.2, 23.6, 26.4, 29.4, 32.8, 36.6, 40.9,
                  43.2, 48.2, 53.7, 59.9, 66.9, 74.6, 83.2, 92.9, 104.0, 116.0,
                  129.0, 144.0, 160.0, 179.0, 200.0]

        # Tabulated cross sections [1e-41 cm2], Strumia and Vissani Table 1
        self.x = [1e-20, 0.00351, 0.00735, 0.0127, 0.0202, 0.0304, 0.044, 0.0619,
                  0.0854, 0.116, 0.155, 0.205, 0.269, 0.349, 0.451, 0.511,
                  0.654, 0.832, 1.05, 1.33, 1.67, 2.09, 2.61, 3.24,
                  4.01, 4.95, 6.08, 7.44, 9.08, 11.0, 12.1, 14.7,
                  17.6, 21.0, 25.0, 29.6, 34.8, 40.7, 47.3,
                  54.6, 62.7, 71.5, 81.0, 91.3, 102.0]

        # Tabulated mean lepton energy [MeV], Strumia and Vissani Table 1
        self.lep = [1e-20, 0.719, 0.952, 1.21, 1.50, 1.82, 2.18, 2.58, 3.03, 3.52,
                    4.08, 4.69, 5.38, 6.15, 7.00, 7.46, 8.47, 9.58, 10.8, 12.2,
                    13.7, 15.5, 17.4, 19.5, 21.8, 24.4, 27.3, 30.5, 34.1, 38.0,
                    40.2, 44.8, 49.9, 55.6, 61.8, 68.8, 76.5, 85.0, 94.5, 105.,
                    117., 130., 144., 161., 179.]

        self.XvsE = interpolate.interp1d(self.E, self.x)
#        self.logxVslogE = interpolate.interp1d(np.log10(self.E), np.log10(self.x))
        self.lepVsE = interpolate.interp1d(self.E, self.lep)
#        self.loglepVslogE = interpolate.interp1d(np.log10(self.E), np.log10(self.lep))

    def cross_section(self, flavor, e_nu):
        """Tabulated inverse beta decay cross section from
        Strumia and Vissani, Table 1.

        :param e_nu: neutrino energy with proper units. Can be an array.
        :return: Inverse beta cross section.
        """
        # Only works for electron antineutrinos
        if flavor != Flavor.nu_e_bar:
            if isinstance(e_nu, (list, tuple, np.ndarray)):
                return np.zeros(len(e_nu), dtype=float) * u.cm**2
            return 0. * u.cm**2

        # Convert all units to MeV
        Enu = e_nu.to('MeV').value
#        logEnu = np.log10(e_nu.to('MeV').value)

#        if isinstance(logEnu, (list, tuple, np.ndarray)):
#            cut = logEnu > self.logEth
#            xs = np.zeros(len(logEnu), dtype=float)
#            xs[cut] = 10**(self.logxVslogE(logEnu[cut]) - 41)
        if isinstance(Enu, (list, tuple, np.ndarray)):
            cut = Enu > self.Eth
            xs = np.zeros(len(Enu), dtype=float)
            xs[cut] = self.XvsE(Enu[cut]) * 1e-41
            return xs * u.cm**2
        else:
            if Enu > self.Eth:
                return self.XvsE(Enu) * 1e-41 * u.cm ** 2
#            if logEnu > self.logEth:
#                return 10**(self.logxVslogE(logEnu) - 41) * u.cm ** 2
            return 0. * u.cm**2

    def mean_lepton_energy(self, flavor, e_nu):
        """Tabulated mean lepton energy after the interaction from
        Strumia and Vissani, Table 1.

        :param e_nu: neutrino energy with proper units. Can be an array.
        :return: Mean lepton energy after the interaction.
        """
        # Only works for electron antineutrinos
        if flavor != Flavor.nu_e_bar:
            if isinstance(e_nu, (list, tuple, np.ndarray)):
                return np.zeros(len(e_nu), dtype=float) * u.MeV
            return 0. * u.MeV

        # Perform calculation in MeV
        Enu = e_nu.to('MeV').value

        if isinstance(Enu, (list, tuple, np.ndarray)):
            lep = np.zeros(len(Enu), dtype=float)
            cut = Enu > self.Eth
            lep[cut] = self.lepVsE(Enu[cut])
            return lep * u.MeV
        else:
            if Enu > self.Eth:
                return self.lepVsE(Enu) * u.MeV
            return 0. * u.MeV
#        logEnu = np.log10(e_nu.to('MeV').value)
#
#        if isinstance(logEnu, (list, tuple, np.ndarray)):
#            loglep = np.zeros(len(logEnu), dtype=float)
#            cut = logEnu > self.logEth
#            loglep[cut] = self.loglepVslogE(logEnu[cut])
#            return 10**loglep * u.MeV
#        else:
#            if logEnu > self.logEth:
#                return 10**(self.loglepVslogE(logEnu)) * u.MeV
#            return 0. * u.MeV


fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True,
                         gridspec_kw={'height_ratios':[3,1], 'hspace':0})

xs_old = None
lep_old = None
enu = np.linspace(0., 200., 101) * u.MeV

for ibd, style, lab in zip([InvBetaPar(), InvBetaTab()],
                           ['-', '.'],
                           ['Parametric (Eq. 25)', 'Table 1']):
    xs = ibd.cross_section(Flavor.nu_e_bar, enu)
    lep = ibd.mean_lepton_energy(Flavor.nu_e_bar, enu)
    axes[0][0].plot(enu, xs/1e-41, style, label=lab)
    axes[0][1].plot(enu, lep, style, label=lab)

    # Plot residuals
    if xs_old is not None and lep_old is not None:
        cut = xs_old != 0.
        res = (xs[cut] - xs_old[cut])/xs_old[cut]
        axes[1][0].plot(enu[cut], 1e2*res, style)
        cut = lep_old != 0.
        res = (lep[cut] - lep_old[cut])/lep_old[cut]
        axes[1][1].plot(enu[cut], 1e2*res, style)
    xs_old = xs
    lep_old = lep

axes[0,0].set(ylabel=r'$\sigma(\bar{\nu}_e p)$ [10$^{-41}$ cm$^2$]')
axes[1,0].set(xlabel=r'$E_\nu$ [MeV]',
              ylabel=r'$\Delta\sigma/\sigma$ [%]',
              ylim=[-1.2,1.2])
axes[0,1].set(ylabel=r'$\langle E_e\rangle$ [MeV]')
axes[1,1].set(ylabel=r'$\Delta\langle E\rangle/\langle E\rangle$ [%]',
              ylim=[-5,10])

leg = axes[0,0].legend()
fig.suptitle(r'Inv. $\beta$ Models from Strumia and Vissani, PLB 564, 2003')
fig.subplots_adjust(left=0.075, right=0.95)

plt.show()

pass