# -*- coding: utf-8 -*-
""" Module for neutrino interaction cross sections.
"""

from abc import ABC, abstractmethod

import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy import interpolate

from snewpy.neutrino import Flavor


class Interaction(ABC):

    def __init__(self):
        # Define masses in rational units
        self.Mn = (c.m_n * c.c ** 2).to('MeV').value
        self.Mp = (c.m_p * c.c ** 2).to('MeV').value
        self.Me = (c.m_e * c.c ** 2).to('MeV').value

        # Weak mixing angle.
        self.sinw2 = 0.23122

        # Cherenkov threshold energy [MeV]
        self.e_ckov = 0.783
        
        # Move this to scale_to_H2O conditional blocks?
        self.H2O_in_ice = 3.053e28 # u.m**3
        
        # Energy of Compton electrons produced by 2.225 MeV gammas from
        # neutron capture. Evaluates to 0.95 MeV. Should be rechecked, as
        # there may be multiple Compton scattering.
        # self.e_compton = self.compton_electron_mean_energy(2.225)
        self.e_compton_n = 0.95
        
        # Electron path length per leptonic MeV above cherenkov threshold. [m/MeV]
        self.e_path_per_MeV = 0.580e-2 * u.m / u.MeV;
        # Positron path length per leptonic MeV above cherenkov threshold. [m/MeV]
        self.p_path_per_MeV = 0.577e-2 * u.m / u.MeV;
        
        # Scaling factor to account for the number of photons produced by
        # an electron which differs from a positron due to the slight 
        # difference in path length per MeV
        self.p2e_path_ratio = self.p_path_per_MeV.value / self.e_path_per_MeV.value ;
        
        # Number of photons per leptonic MeV. [1/MeV]
        self.photons_per_lepton_MeV = self.e_path_per_MeV * self.photons_per_path()
        
        super().__init__()

    @abstractmethod
    def cross_section(self, flavor, e_nu):
        pass

    @abstractmethod
    def mean_lepton_energy(self, flavor, e_nu):
        pass
        
    @abstractmethod
    def photon_scaling_factor(self, flavor):
        pass
        
    def compton_cherenkov_energy_dist(self, e_electron, e_photon):
        """Compute compton-cherenkov electron energy unnormalized distribution

        :param e_electron: electron energy.
        :param e_photon: photon energy.
        :returns: energy pdf.
        """

        # Convert all units to MeV
        e_ph = e_photon.to("MeV").value
        e_el = e_electron.to("MeV").value


        # Calculate unnormalized energy distribution
        e_fr = e_ph/self.Me

        pdf = e_ph**3 / (e_ph - e_el)**2
        pdf += e_ph * (2*e_fr + 1)
        pdf += e_fr**2 * (e_ph - e_el)
        pdf -= e_ph**2 * (2 + 2*e_fr-e_fr**2) / (e_ph - e_el)
        pdf *= 1./e_ph**5 * self.Me**3

        return pdf.value

    def compton_electron_mean_energy(self, e_photon):
        """ Compute Compton-Cherenkov mean electron energy

        :param e_photon: photon energy.
        :returns: mean electron energy.
        """
        # Convert all units into MeV
        e_ph = e_photon.to('MeV')

        # Cherenkov kinetic energy threshold
        e_th = self.e_ckov - self.Me
        e_max = 2. * e_ph**2 / (self.Me + 2*e_ph)
        if e_th >= e_max:
            return 0. * u.MeV

        # Define Compton-Cherenkov electron energy distribution
        nstep = 100000
        e_el = np.linspace(0., e_max, nstep+1)
        step_size = e_max/nstep
        e_pdf = self.compton_cherenkov_energy_dist(e_el, e_ph)

        # fraction of electron above energy threshold
        cut = e_el > e_th
        e_frac = np.trapz(e_pdf[cut], dx=step_size)
        e_frac /= np.trapz(e_pdf, dx=step_size)

        e_mean = (np.average(e_el[cut], weights=e_pdf[cut]) - e_th) * e_frac

        # Calculate electron mean energy in MeV
        return e_mean * u.MeV
        
    def photons_per_path(self, min_wavelength = 300e-9 * u.m, max_wavelength = 600e-9 * u.m ):
        """Compute the number of photons emitted per unit distance (m)
         of electron path length by integrating wavelength out of the 
         Frank-Tamm formula. Index of refraction was given by Eq. (6) 
         of Price & Bergstrom, AO 36:004181, 1997.
         

        :param min_wavelength: lower integral limit. Default = 300 nm
        :param max_wavelength: upper integral limit. Default = 600 nm
            Defaults taken from USSR.
        
        :returns: number of photons with wavelength between 
            integration limits per unit eletron path length.
        """
        min_lambda = min_wavelength.to(u.m).value
        max_lambda = max_wavelength.to(u.m).value
        
        # Parameters from Price & Bergstrom
        n_div = 700e-9 # u.m
        n_mul = 1.2956
        n_pow = -0.021735
        
        # Integral of Frank-Tamm Formula with substituted wavelength dependence.
        frank_tamm = lambda x: 2*np.pi*c.alpha.value * \
             (n_div**(2*n_pow) / (n_mul**2 * (1+2*n_pow)*x**(1+2*n_pow)) - 1/x)
        
        #photons_per_leptonic_MeV = self.e_length_above_Ech_per_MeV.value * \
        return (frank_tamm(max_lambda) - frank_tamm(min_lambda) )  / u.m 
        
        #return effvol_photon * photons_per_leptonic_MeV * self.H2O_in_ice/( 4*np.pi ) 
        #return  photons_per_leptonic_MeV * self.H2O_in_ice/( 4*np.pi ) 


class InvBetaPar(Interaction):
    """Inverse beta decay parameterization from Strumia and Vissani,
    Phys. Lett. B 564:42, 2003."""

    def __init__(self):
        super().__init__()

        # Calculate IBD threshold
        self.Eth = self.Mn + self.Me - self.Mp
        self.delta = (self.Mn ** 2 - self.Mp ** 2 - self.Me ** 2) / (2 * self.Mp)
        
        # Number of H atoms in H2O molecule.
        self.H_per_H2O = 2

    def cross_section(self, flavor, e_nu, scale_to_H2O=True):
        """Inverse beta decay cross section, Strumia and Vissani eq. 25.

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy with proper units. Can be an array.
        :param scale_to_H2O: indicator to scale cross section to H2O target.
        :returns: Inverse beta cross section.
        """
        # Only works for electron antineutrinos
        if flavor != Flavor.NU_E_BAR:
            if isinstance(e_nu, (list, tuple, np.ndarray)):
                return np.zeros(len(e_nu), dtype=float) * u.cm**2
            return 0. * u.cm**2

        # Convert all units to MeV
        Enu = e_nu.to('MeV').value
        

        # Calculate mean positron energy and momentum using the crappy estimate
        # from Strumia and Vissani eq. 25.
        Ee = Enu - (self.Mn - self.Mp)
        pe = np.sqrt(Ee ** 2 - self.Me ** 2, where=(Ee ** 2 - self.Me ** 2)>0, out=np.zeros_like(Ee))

        # Handle possibility of list/array input
        if isinstance(Enu, (list, tuple, np.ndarray)):
            xs = np.zeros(len(Enu), dtype=float)
            cut = Enu > self.Eth
            xs[cut] = 1e-43 * pe[cut] * Ee[cut] * \
                Enu[cut] ** (-0.07056 + 0.02018 * np.log(Enu[cut]) - 0.001953 * np.log(Enu[cut]) ** 3)

        # Handle float input
        else:
            if Enu <= self.Eth:
                return 0. * u.cm ** 2
                
            xs = 1e-43 * pe * Ee * \
                Enu ** (-0.07056 + 0.02018 * np.log(Enu) - 0.001953 * np.log(Enu) ** 3)
        
        if scale_to_H2O:
            return self.H_per_H2O * xs * u.cm**2
        return xs * u.cm**2
        
    def mean_lepton_energy(self, flavor, e_nu):
        """Mean lepton energy from Abbasi et al., A&A 535:A109, 2011, p.6.
        Could also use Strumia and Vissani eq. 16.

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy.
        :returns: mean energy of the emitted lepton.
        """
        # Only works for electron antineutrinos
        if flavor != Flavor.NU_E_BAR:
            if isinstance(e_nu, (list, tuple, np.ndarray)):
                return np.zeros(len(e_nu), dtype=float) * u.MeV
            return 0. * u.MeV

        Enu = e_nu.to('MeV').value

        # Handle possibility of list/array input
        if isinstance(Enu, (list, tuple, np.ndarray)):
            # Note: subtraction of energy lost to Cherenkov production and
            # Compton scattering is present in the USSR code, but it appears
            # to make the agreement with the Strumia and Vissani calculation
            # (the InvBetaTab class) worse.
            lep = np.where(Enu < self.Eth + self.e_ckov, 0.,
                           (Enu - self.delta) * (1. - Enu / (Enu + self.Mp)))# - self.e_ckov - self.e_compton_n)
            return lep * u.MeV
        else:
            if Enu > self.Eth:
                return (Enu - self.delta) * (1. - Enu / (Enu + self.Mp))# - self.e_ckov - self.e_compton_n
            return 0. * u.MeV
            
    def photon_scaling_factor(self, flavor):
        if flavor is not Flavor.NU_E_BAR:
            return self.photons_per_lepton_MeV
        else:
            return self.photons_per_lepton_MeV * self.p2e_path_ratio
        

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
        self.lepVsE = interpolate.interp1d(self.E, self.lep)
        
        # Number of H atoms in H2O molecule.
        self.H_per_H2O = 2

    def cross_section(self, flavor, e_nu, scale_to_H2O=True):
        """Tabulated inverse beta decay cross section from
        Strumia and Vissani, Table 1.

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy with proper units. Can be an array.
        :param scale_to_H2O: indicator to scale cross section to H2O target.
        :returns: Inverse beta cross section.
        """
        # Only works for electron antineutrinos
        if flavor != Flavor.NU_E_BAR:
            if isinstance(e_nu, (list, tuple, np.ndarray)):
                return np.zeros(len(e_nu), dtype=float) * u.cm**2
            return 0. * u.cm**2

        # Convert all units to MeV
        Enu = e_nu.to('MeV').value

        # Handle possibility of list input
        if isinstance(Enu, (list, tuple, np.ndarray)):
            cut = Enu > self.Eth
            xs = np.zeros(len(Enu), dtype=float)
            xs[cut] = self.XvsE(Enu[cut]) * 1e-41
            
        else:
            if Enu > self.Eth:
                xs = self.XvsE(Enu) * 1e-41 
            return 0. * u.cm**2
        
        if scale_to_H2O:
            return self.H_per_H2O * xs * u.cm**2
        return xs * u.cm**2
        
    def mean_lepton_energy(self, flavor, e_nu):
        """Tabulated mean lepton energy after the interaction from
        Strumia and Vissani, Table 1.

        :param flavor: neutrino flavor
        :param e_nu: neutrino energy with proper units. Can be an array.
        :returns: Mean lepton energy after the interaction.
        """
        # Only works for electron antineutrinos
        if flavor != Flavor.NU_E_BAR:
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
            
    def photon_scaling_factor(self, flavor):
        if flavor is not Flavor.NU_E_BAR:
            return self.photons_per_lepton_MeV
        else:
            return self.photons_per_lepton_MeV * self.p2e_path_ratio


class ElectronScatter(Interaction):
    """Cross sections for elastic neutrino-electron scattering.
       Note: Subtraction of Cherenkov threshold energy is not performed
    """

    def __init__(self):
        super().__init__()
        
        # Number of electrons in H2O.
        self.e_per_H2O = 10
        
    def _integrated_XSxE(self, params, E, y ):
        """Integrated product of differential cross section and final state lepton
        energy. """
        norm, eps_p, eps_m, y_ckov = params
        XSxE = norm* E**2 * (
              1/4. * y**4 * eps_p**2
            - 1/3. * y**3 * (eps_p*eps_m*self.Me/E + 2*eps_p**2 + eps_p**2 * y_ckov)
            + 1/2. * y**2 * (eps_p**2 + eps_m**2 + ( 2*eps_p**2 + eps_p*eps_m*self.Me/E) * y_ckov)
            -        y    * (eps_m**2 + eps_p**2 ) * y_ckov )
        return XSxE
    
    def cross_section(self, flavor, e_nu, scale_to_H2O=True):
        """Cross section from Marciano and Parsa, J. Phys. G 29:2969, 2003.

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy.
        :param scale_to_H2O: indicator to scale cross section to H2O target.
        :returns: neutrino cross section.
        """
        
        if e_nu[0] == 0.:
            e_nu[0] = 1e-10 * u.MeV
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

        # See (12) in source - Converted to MeV Units
        norm = 1.5e-44
        ymax = 1./(1 + self.Me/(2*Enu))
        
        # xs is definite integral over differential cross section on 0 to ymax.
        xs = norm*Enu * (
            ymax**3 * epsilon_p**2/3.
            - ymax**2 * (epsilon_p*epsilon_m*self.Me/Enu + 2*epsilon_p**2)/2
            + ymax * (epsilon_p**2 + epsilon_m**2)
        )

        if scale_to_H2O:
            return self.e_per_H2O * xs * u.cm**2
        return xs * u.cm**2

    def mean_lepton_energy(self, flavor, e_nu, scale_to_H2O=True):
        """Mean Lepton Energy from Marciano and Parsa, J. Phys. G 29:2969, 2003.

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy.
        :param scale_to_H2O: indicator to scale cross section to H2O target.
        NOTE: This "mean energy" is the integrated product of the differential
              cross section and lepton final-state energy. Scaling the cross
              section to H2O affects the "mean energy" as well.
        :returns: neutrino integrated product of lepton final-state energy with differential cross section.
        """
        # Convert all units to MeV
        Enu = e_nu.to('MeV').value

        if e_nu[0] == 0.:
            e_nu[0] = 1e-10 * u.MeV

        # Convert all units to MeV
        Enu = e_nu.to('MeV').value
        
        cut = Enu > (self.e_ckov - self.Me)

        # See (12) in source - units cm**2 / MeV
        norm = 1.5e-44         
        y_max = 1. / (1 + self.Me/(2*Enu[cut]))
        y_ckov = (self.e_ckov - self.Me) / Enu[cut]

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

        lep = np.zeros_like( Enu )
        params = (norm, epsilon_p, epsilon_m, y_ckov) 

        lep[cut] += self._integrated_XSxE(params, Enu[cut], y_max)
        lep[cut] -= self._integrated_XSxE(params, Enu[cut], y_ckov)    
        lep /= self.cross_section(flavor, e_nu).to(u.cm**2).value
        
        if scale_to_H2O:
            return self.e_per_H2O * lep * u.MeV 
        return lep * u.MeV 
        
    def photon_scaling_factor(self, flavor):
        return self.photons_per_lepton_MeV


class Oxygen16CC(Interaction):
    """O16 charged current interaction, using estimates from Kolbe et al.,
    PRD 66:013007, 2002.
    """

    def __init__(self):
        super().__init__()

        # Interaction threshold energies, in MeV
        self.Eth_nu = 15.4
        self.Eth_nubar = 11.4
        
        # Abundance of the oxygen-16 isotope: 99.76%
        self.O16frac = 0.99762
        
        # Number of oxygen atoms in H2O (For Consistency).
        self.O_per_H2O = 1

    def _xsfunc(self, E, pars):
        """Parametric fit function for the CC O16 interaction.
        See Appendix B.3 of Tomas et al., PRD 68:093013, 2003.

        :param E: neutrino energy [MeV].
        :param pars: four fit parameters.
        :returns: cross section [cm^2].
        """
        a, b, c, d = pars
        return a * (E**b - c**b)**d

    def cross_section(self, flavor, e_nu, scale_to_H2O=True):
        """Compute the CC cross section for oxygen-16.

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy.
        :param scale_to_H2O: indicator to scale cross section to H2O target.
        :returns: cross section.
        """
        # Convert all units to MeV
        Enu = e_nu.to('MeV').value
        
        if not flavor.is_electron:
            if isinstance(Enu, (list, tuple, np.ndarray)):
                return np.zeros_like(Enu) * u.cm**2
            return 0. * u.cm**2

        if flavor == Flavor.NU_E:
            if isinstance(Enu, (list, tuple, np.ndarray)):
                cut = Enu >= self.Eth_nu
                xs = np.zeros_like(Enu)
                xs[cut] = self._xsfunc(Enu[cut], (4.73e-40, 0.25, 15., 6.))
                # xs = np.where(Enu <= self.Eth_nu, 0.,
                #               self._xsfunc(Enu, (4.73e-40, 0.25, 15., 6.)))
            else:
                if Enu < self.Eth_nu:
                    xs = 0.
                else:
                    xs = self._xsfunc(Enu, (4.73e-40, 0.25, 15., 6.))

        elif flavor == Flavor.NU_E_BAR:
            if isinstance(Enu, (list, tuple, np.ndarray)):
                cut1 = np.logical_and(Enu >= self.Eth_nubar, Enu < 42.3293)
                cut2 = Enu >= 42.3293
                xs = np.zeros_like(Enu)
                xs[cut1] = self._xsfunc(Enu[cut1], (2.11357e-40, 0.224172, 8.36303, 6.80079))
                xs[cut2] = self._xsfunc(Enu[cut2], (2.11357e-40, 0.260689, 16.7893, 4.23914))
                # xs = np.where(Enu <= 42.3293,
                #               self._xsfunc(Enu, (2.11357e-40, 0.224172, 8.36303, 6.80079)),
                #               self._xsfunc(Enu, (2.11357e-40, 0.260689, 16.7893, 4.23914)))
            else:
                if Enu <= self.Eth_nu:
                    xs = 0.
                else:
                    if Enu <= 42.3293:
                        xs = self._xsfunc(Enu, (2.11357e-40, 0.224172, 8.36303, 6.80079))
                    else:
                        xs = self._xsfunc(Enu, (2.11357e-40, 0.260689, 16.7893, 4.23914))
            
        else:
            if isinstance(Enu, (list, tuple, np.ndarray)):
                return np.zeros_like(Enu) * u.cm**2
            return 0. * u.cm**2
        
        if scale_to_H2O:
            return self.O_per_H2O * self.O16frac * xs * u.cm**2
        return xs * u.cm**2
        
    def photon_scaling_factor(self, flavor):
        if flavor is not Flavor.NU_E_BAR:
            return self.photons_per_lepton_MeV
        else:
            return self.photons_per_lepton_MeV * self.p2e_path_ratio

    def mean_lepton_energy(self, flavor, e_nu):
        """Compute mean energy of lepton emitted in CC interaction.

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy.
        :returns: lepton energy.
        """
        # Convert all units to MeV
        Enu = e_nu.to('MeV').value
        
        if not flavor.is_electron:
            if isinstance(Enu, (list, tuple, np.ndarray)):
                return np.zeros_like(Enu) * u.MeV
            return 0. * u.MeV

        # Compute correct minimum energy
        if flavor == Flavor.NU_E:
            e_thr = self.Eth_nu + self.e_ckov
        else:
            e_thr = self.Eth_nubar + self.e_ckov

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

        # Energy threshold for interaction, all flavors [MeV]
        self.Eth = 12.

        # Relative contribution from photon production to NC cross section
        #  - 15^N + p + gamma: 24.9%
        #  - 15^O + n + gamma:  6.7%

        # The mean gamma-ray energy is 7.5 MeV, and the related Compton
        # electron will be 4.7 MeV.
        self.relative_photon_prob = 0.249 + 0.067
        self.e_compton_g = 4.7
        # self.e_compton_g = self.compton_electron_mean_energy(7.5)

        # Relative contribution from neutron production to NC cross section
        #  - 15^O(g.s.) + n:   18.7%
        #  - 15^O + n + gamma:  6.7%
        #  - 14^N + p + n:      8.5%
        #  - 11^B + n + 4^He:   0.7%

        # The mean energy is 2.225 Mev and the related Compton electron
        # will be 0.95 MeV (calculated in base class).
        self.relative_neutron_prob = 0.187 + 0.067 + 0.085 + 0.007

        self._lepton_energy = self.relative_neutron_prob * self.e_compton_n + \
                              self.relative_photon_prob * self.e_compton_g

        # Abundance of the oxygen-16 isotope: 99.76%
        self.O16frac = 0.99762
        
        # Number of oxygen atoms in H2O (For Consistency).
        self.O_per_H2O = 1
        
    def cross_section(self, flavor, e_nu, scale_to_H2O=True):
        """Calculate cross section.

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy.
        :param scale_to_H2O: indicator to scale cross section to H2O target.
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
        
        if scale_to_H2O:
            return self.O_per_H2O * self.O16frac * xs * u.cm**2
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
        
    def photon_scaling_factor(self, flavor):
        return self.photons_per_lepton_MeV


class Oxygen18(Interaction):
    """O18 CC interaction, using quadratic fit to cross section estimated from
    Kamiokande data from Haxton and Robertson, PRC 59:515, 1999.
    
    See page on Mainz Wiki, Neutrino cross sections on natural oxygen.
    """
    def __init__(self):
        super().__init__()

        # Energy threshold: 1.66 MeV
        self.e_th = 1.66

        # Abundance of the oxygen-18 isotope: 0.2%
        self.O18frac = 0.002
        
        # Number of oxygen atoms in H2O (For Consistency).
        self.O_per_H2O = 1
        
    def _xsfunc(self, E, pars):
        """Quadratic fit function for the CC O18 interaction.
        See Mainz Wiki, Neutrino cross section on natural oxygen.

        :param E: neutrino energy [MeV].
        :param pars: three fit parameters.
        NOTE: Parameter untis are cm**2/MeV**2, cm**2/MeV and cm**2.
        :returns: cross section [cm^2].
        """
        a, b, c = pars
        return a * E**2 + b * E + c

    def cross_section(self, flavor, e_nu, scale_to_H2O=True):
        """Oxygen-18 interaction cross section.

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy.
        :param scale_to_H2O: indicator to scale cross section to H2O target.
        :returns: cross section.
        """        
        # Convert all units to MeV
        Enu = e_nu.to('MeV').value
        
        # Only electron neutrinos interact with O18!
        if flavor != Flavor.NU_E:
            if isinstance(Enu, (list, tuple, np.ndarray)):
                return np.zeros_like(Enu) * u.cm**2
            return 0. * u.cm**2

        # Handle list input
        if isinstance(Enu, (list, tuple, np.ndarray)):
            xs = np.where(Enu < self.e_th, 0., self._xsfunc(Enu, [1.7e-46, 1.6e-45, -0.9e-45]) )

        else:
            if Enu < self.e_th:
                xs = 0.
            else:
                xs = self._xsfunc(Enu, [1.7e-46, 1.6e-45, -0.9e-45])
                
        if scale_to_H2O:
            # According to USSR, xs is obtained by scaling by a factor of
            # o18frac. This is not mentioned on the Mainz wiki. It is unclear
            # why it is divided by o18frac in the first place. Following this
            # logic, scaled xs appear unscaled, but is correct.
            return self.O_per_H2O * xs * u.cm**2
        return (xs/self.o18frac) * u.cm**2

    def mean_lepton_energy(self, flavor, e_nu):
        """Mean lepton energy produced by O18 interaction.

        :param flavor: neutrino flavor.
        :param e_nu: neutrino energy.
        :returns: mean energy.
        """
        # Convert all units to MeV
        Enu = e_nu.to('MeV').value
        
        # Only electron neutrinos interact with O18!
        if flavor != Flavor.NU_E:
            if isinstance(Enu, (list, tuple, np.ndarray)):
                return np.zeros_like(Enu) * u.MeV
            return 0. * u.MeV

        e_min = self.e_th + self.e_ckov
        if isinstance(Enu, (list, tuple, np.ndarray)):
            lep = np.where(Enu < e_min, 0., Enu - e_min)
        else:
            lep = 0.
            if Enu >= e_min:
                lep = Enu - e_min

        return lep * u.MeV
        
    def photon_scaling_factor(self, flavor):
        if flavor is not Flavor.NU_E_BAR:
            return self.photons_per_lepton_MeV
        else:
            return self.photons_per_lepton_MeV * self.p2e_path_ratio


class Interactions:
    """ Public-facing enumeration object for ASTERIA Interactions
    Iterating this object without instantiating it will yield a set of default
    asteria Interactions
    """

    _defaults = (
        ElectronScatter(),
        InvBetaPar(),
        Oxygen16CC(),
        Oxygen16NC(),
        Oxygen18()
    )

    def __init__(self, interactions=None):
        """Initialization for custom Interaction enumeration object.

        Parameters
        ----------
        interactions : list, tuple or numpy.ndarray of asteria.interactions.Interaction
        """
        if interactions is None:
            self._values = self._defaults
        else:
            if not isinstance(interactions, (list, tuple, np.ndarray)):
                raise ValueError("Invalid iterable for interactions, expected list,"
                                 f"tuple or numpy.ndarray, but received {type(interactions)}")

            elif len(interactions) == 0:
                raise ValueError("No interactions requested")

            elif len(set(interactions)) != len(interactions):
                raise ValueError("Duplicate interactions requested")

            elif any([isinstance(i, Interaction) for i in interactions]):
                raise ValueError("Interaction instance requested, please request interactions by class "
                                 "(i.e. ElectronScatter)")

            elif any(not issubclass(x, Interaction) for x in interactions):
                idx = np.where([not issubclass(x, Interaction) for x in interactions])[0][0]
                raise ValueError(f"Invalid interaction {interactions[idx]} at index {idx}, "
                                 "only Interaction objects derived from asteria.interactions.Interaction "
                                 "are supported")

            # Create tuple of requested interaction object instances. The above checks are intended to guard against
            # use of invalid Interaction types
            self._values = tuple([i() for i in interactions])


    def __iter__(self):
        """Iterate through a custom list of interactions
        """
        # Note: This will overwrite the metaclass definition only when an Interaction object is instantiated
        #       Otherwise, the default options defined in _InteractionsMeta will be used.
        for interaction in self._values:
            yield interaction

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return "\n".join(["Interactions: "]+[f"{i:>2d} - {x.__class__.__name__}" for i, x in enumerate(self)])
