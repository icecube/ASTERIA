from neutrino import Flavor

from abc import ABC, abstractmethod

import astropy.units as u
import astropy.constants as c
from scipy import interpolate

import numpy as np


class Interaction(ABC):

    def __init__(self):
        # Define masses in rational units
        self.Mn = (c.m_n * c.c ** 2).to('MeV').value
        self.Mp = (c.m_p * c.c ** 2).to('MeV').value
        self.Me = (c.m_e * c.c ** 2).to('MeV').value

        # Weak mixing angle... for large momentum transfer?
        self.sinw2 = 0.23122

        # Cherenkov threshold energy [MeV]
        self.Eckov = 0.783

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

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy with proper units. Can be an array.
        :returns: Inverse beta cross section.
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

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy.
        :returns: mean energy of the emitted lepton.
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

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy with proper units. Can be an array.
        :returns: Inverse beta cross section.
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

        :param flavor: neutrino flavor
        :param e_nu: neutrino energy with proper units. Can be an array.
        :returns: Mean lepton energy after the interaction.
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


class ElectronScatter(Interaction):
    """Cross sections for elastic neutrino-electron scattering.
    """

    def __init__(self):
        super().__init__()

    def cross_section(self, flavor, e_nu):
        """Cross section from Marciano and Parsa, J. Phys. G 29:2969, 2003.

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy.
        :returns: neutrino cross section.
        """
        # Convert all units to MeV
        Enu = e_nu.to('MeV').value

        # Define flavor-dependent parameters
        epsilons = [-self.sinw2, 0.]
        if flavor.is_electron:
            epsilons[1] = -0.5 - self.sinw2
        else:
            epsilons[1] =  0.5 - self.sinw2

        if flavor.is_neutrino:
            epsilon_p, epsilon_m = epsilons
        else:
            epsilon_m, epsilon_p = epsilons

        norm = 1.5e-44
        ymax = 1./(1 + self.Me/(2*Enu))
        xs = norm*Enu * (
            ymax**3 * epsilon_p**2/3.
            - ymax**2 * 0.5*(epsilon_p*epsilon_m*self.Me/Enu + 2*epsilon_p**2)
            + ymax * (epsilon_p**2 + epsilon_m**2)
        )

        return xs * u.cm**2

    def mean_lepton_energy(self, flavor, e_nu):
        pass


class Oxygen16CC(Interaction):
    """O16 charged current interaction, using estimates from Kolbe et al.,
    PRD 66:013007, 2002.
    """

    def __init__(self):
        super().__init__()

        # Interaction threshold energies, in MeV
        self.Eth_nu = 15.4
        self.Eth_nubar = 11.4

    def _xsfunc(self, E, pars):
        """Parametric fit function for the CC O16 interaction.
        See Appendix B.3 of Tomàs et al., PRD 68:093013, 2003.

        :param E: neutrino energy [MeV].
        :param pars: four fit parameters.
        :returns: cross section [cm^2].
        """
        a, b, c, d = pars
        return a * (E**b - c**b)**d

    def cross_section(self, flavor, e_nu):
        """Compute the CC cross section for oxygen-16.

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy.
        :returns: cross section.
        """
        if not flavor.is_electron:
            if isinstance(e_nu, (list, tuple, np.ndarray)):
                return np.zeros_like(e_nu) * u.cm**2
            return 0. * u.cm**2

        # Convert all units to MeV
        Enu = e_nu.to('MeV').value

        if flavor == Flavor.nu_e:
            if isinstance(e_nu, (list, tuple, np.ndarray)):
                xs = np.where(Enu <= self.Eth_nu, 0.,
                              self._xsfunc(Enu, (4.73e-40, 0.25, 15., 6.)))
            else:
                if Enu <= self.Eth_nu:
                    xs = 0.
                else:
                    xs = self._xsfunc(Enu, (4.73e-44, 0.25, 0.25, 15.))
            return xs * u.cm**2
        elif flavor == Flavor.nu_e_bar:
            if isinstance(e_nu, (list, tuple, np.ndarray)):
                xs = np.where(Enu <= 42.3293,
                              self._xsfunc(Enu, (2.11357e-40, 0.224172, 8.36303, 6.80079)),
                              self._xsfunc(Enu, (2.11357e-40, 0.260689, 16.7893, 4.23914)))
            else:
                if Enu <= self.Eth_nu:
                    xs = 0.
                else:
                    if Enu <= 42.3293:
                        xs = self._xsfunc(Enu, (2.11357e-40, 0.224172, 8.36303, 6.80079))
                    else:
                        xs = self._xsfunc(Enu, (2.11357e-40, 0.260689, 16.7893, 4.23914))
            return xs * u.cm**2
        else:
            if isinstance(e_nu, (list, tuple, np.ndarray)):
                return np.zeros_like(e_nu) * u.cm**2
            return 0. * u.cm**2

    def mean_lepton_energy(self, flavor, e_nu):
        """Compute mean energy of lepton emitted in CC interaction.

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy.
        :returns: lepton energy.
        """
        if not flavor.is_electron:
            if isinstance(e_nu, (list, tuple, np.ndarray)):
                return np.zeros_like(e_nu) * u.MeV
            return 0. * u.MeV

        # Convert all units to MeV
        Enu = e_nu.to('MeV').value

        # Compute correct minimum energy
        if flavor == Flavor.nu_e:
            e_thr = self.Eth_nu + self.Eckov
        else:
            e_thr = self.Eth_nubar + self.Eckov

        if isinstance(e_nu, (list, tuple, np.ndarray)):
            lep = np.where(Enu < e_thr, 0., Enu - e_thr)
        else:
            lep = 0. if Enu < e_thr else Enu - e_thr

        return lep * u.MeV


class Oxygen16NC(Interaction):
    """O16 neutral current interaction, using estimates from Kolbe et al.,
    PRD 66:013007, 2002.
    """

    def __init__(self):
        super().__init__()

        # Energy threshold for interaction [MeV]
        self.Eth = 12.

        # Relative contribution from photon production to NC cross section
        self.relative_photon_prob = 0.249 + 0.067
        self.mean_photon_energy = 4.7

        # Relative contribution from neutron production to NC cross section
        self.relative_neutron_prob = 0.187 + 0.067 + 0.085 + 0.007
        self.mean_neutron_energy = 0.95

        self._lepton_energy = self.relative_neutron_prob*self.mean_neutron_energy + \
                              self.relative_photon_prob*self.mean_photon_energy

    def cross_section(self, flavor, e_nu):
        """Calculate cross section.

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy.
        :returns: neutrino cross section.
        """
        # Convert all units to MeV
        Enu = e_nu.to('MeV').value

        if isinstance(Enu, (list, tuple, np.ndarray)):
            xs = np.where(Enu <= self.Eth, 0., 6.7e-40 * (Enu**0.208 - 8.**0.25)**6)
            # xs[cut] = 6.7e-40 * (Enu[cut]**0.208 - 8.**0.25)**6
        else:
            if Enu < self.Eth:
                return 0 * u.cm**2
            xs = 6.7e-40 * (Enu**0.208 - 8.**0.25)**6
        return xs * u.cm**2

    def mean_lepton_energy(self, flavor, e_nu):
        """Calculate mean lepton energy.

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy.
        :returns: mean energy of emitted lepton.
        """
        # Convert all units to MeV
        Enu = e_nu.to('MeV').value

        if isinstance(Enu, (list, tuple, np.ndarray)):
            lep = np.where(Enu <= self.Eth, 0., self._lepton_energy)
        else:
            lep = 0. if Enu < self.Eth else self._lepton_energy
        return lep * u.MeV
