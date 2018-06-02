""" Module for low energy neutrino interaction in ice, including inverse-beta
decay and neutrino electron scattering. """

import numpy
from astropy import units as u

import neutrino

# Constants
# mass constants
MASS_P = 938.272081358*u.MeV
MASS_N = 939.5654133*u.MeV
MASS_E = 0.510998946131*u.MeV
DELTA_NP = MASS_N-MASS_P # neutron-proton mass difference; unit MeV

# cherenkov threshold energy
CHERENKOV_ERG_THRESHOLD = 0.783*u.MeV

# coupling constants
FERMI_COUPLING = 1.16637876e-11/u.MeV**2

# cabbibo angle
CABBIBO_COS2 = 0.94984516
CABBIBO_SIN2 = 0.05015484

# weinberg angle
WEIGNBERG_COS2 = 0.768705
WEIGNBERG_SIN2 = 0.231295

# conversion factor
HBAR_C = 197.326978812e-13*u.MeV *u.cm

def compton_electron_prob(erg_e, erg_photon):
    """ Unnormalized energy distribution of electron in compton-scattering given photon
    energy and electron energy in MeV.
    Inputs:
    + erg_e: astropy.u.Quantity
        Electron energy in MeV.
    + erg_photon: astropy.u.Quantity
        Photon  energy in MeV.
    Outputs:
    + prob: float
        Probability of an electron of a given energy
    """
    erg_e.to("MeV")
    erg_photon.to("MeV")

    # energy of photon as a fraction of electron mass
    erg_frac = erg_photon/MASS_E
    # probability
    prob = erg_photon**3/(erg_photon-erg_e)**2
    prob += erg_photon*(2*erg_frac+1)
    prob += erg_frac**2*(erg_photon-erg_e)
    prob -= erg_photon**2*(2+2*erg_frac-erg_frac**2)/(erg_photon-erg_e)
    prob *= 1./erg_photon**5*MASS_E**3
    return prob.value

def compton_electron_mean_energy(erg_photon):
    """ Compute the mean energy in MeV of electrons given photon energy in MeV
    Inputs:
    + erg_photon: astropy.u.Quantity
        Photon  energy in MeV.
    Outputs:
    + mean_aerg: astropy.u.Quantity
        Mean energy of electron in MeV.
    """
    try:
        erg_photon = erg_photon.to('MeV')
    except AttributeError:
        print('Photon energy must have unit of energy.')

    # cherenkov kinetic energy threshold
    min_erg = CHERENKOV_ERG_THRESHOLD-MASS_E
    max_erg = 2.*erg_photon**2/(MASS_E+2*erg_photon)
    if min_erg >= max_erg:
        return 0*u.MeV

    # define energy distribution
    nstep = 100000
    erg_range = numpy.linspace(0., max_erg, nstep+1)
    erg_step = 1.*max_erg/nstep

    # calculate fraction of energy above the threshold energy
    erg_pdf = compton_electron_prob(erg_range, erg_photon)

    cut = erg_range > min_erg
    erg_frac = numpy.trapz(erg_pdf[cut], dx=erg_step)
    erg_frac /= numpy.trapz(erg_pdf, dx=erg_step)

    # calculate mean energy in MeV
    mean_erg = (numpy.average(erg_range[cut], weights=erg_pdf[cut])-min_erg)*erg_frac
    return mean_erg


class InverseBetaDecay(object):
    """ Class for the inverse beta decay interaction using parameterized
    formular from arXiv:astro-ph/0302055.
    """
    def __init__(self):
        """ Constructor to define class variables and constants """
        # energy threshold for inverse beta-decay
        # minimum energy 1.806 MeV, below minimum energy cross section is 0
        self._min_erg = 1.806*u.MeV

    def cross_section(self, erg_nu):
        """ Calculate the cross section in cm2 of inverse beta decay
        using parametrized equation from arXiv:astro-ph/0302055.
        This approximation agrees well with the full result for neutrino energy
        less than 300 MeV.
        Inputs:
        - erg_nu: astropy.u.Quantity
            Neutrino energy in MeV.
        Outputs:
        - cross_section: astropy.u.Quantity
            Return the full cross secton in cm2.
        """
        # convert neutrino energy unit into MeV
        try:
            erg_nu = erg_nu.to('MeV')
        except AttributeError:
            print('Neutrino energy must have unit of energy.')

        # check if the input is a scalar or array
        isarray = True
        if not hasattr(erg_nu.value, "__len__"):
            # if energy is less than threshold, cross section is 0 cm2
            if erg_nu <= self._min_erg:
                return 0*u.cm**2
            isarray = False

        # define parameters and cross section
        # energy and momentum of positron in MeV
        erg_pos = erg_nu-DELTA_NP
        if isarray:
            p_pos = numpy.sqrt(erg_pos**2-MASS_E**2,
                               out=numpy.zeros_like(erg_nu),
                               where=erg_pos > MASS_E)*u.MeV
        else:
            p_pos = erg_pos**2-MASS_E**2

        # calculate cross section in cm^2
        ln_nu = numpy.log(erg_nu.value) # ln(E_nu)
        cross_section = 1e-43*p_pos.value*erg_pos.value
        cross_section *= numpy.power(erg_nu.value,
                                     -0.07056+0.02018*ln_nu-0.001953*ln_nu**3)

        # if energy less than threshold, cross section is 0 cm2
        if isarray:
            cross_section[erg_nu <= self._min_erg] = 0
        return cross_section*u.cm**2

    def mean_leptonic_energy(self, erg_nu):
        """ Calculate the mean leptonic energy in MeV of inverse beta decay
        given neutrino energy in MeV.
        Inputs:
        - erg_nu: astropy.u.Quantity
            Neutrino energy in MeV.
        Outputs:
        - mean_erg_lep: astropy.u.Quantity
            Return the mean leptonic energy in MeV.
        """
        # minimum energy threshold
        min_erg = self._min_erg+CHERENKOV_ERG_THRESHOLD

        # compton-cherenkov energy
        neutron_mean_erg_e = 0.95*u.MeV

        # calculate mean leptonic energy in MeV
        delta = MASS_N**2-MASS_P**2-MASS_E**2
        mean_erg_lep = (erg_nu-delta)*(1.-erg_nu/(erg_nu+MASS_P))
        mean_erg_lep += neutron_mean_erg_e-CHERENKOV_ERG_THRESHOLD
        if hasattr(erg_nu, '__len__'):
            mean_erg_lep[erg_nu <= min_erg] = 0.*u.MeV
        elif erg_nu <= min_erg:
            mean_erg_lep = 0.*u.MeV

        return mean_erg_lep

    @property
    def min_erg(self):
        """ Return minimum energy threshold for neutrino """
        return self._min_erg

class NeutrinoElectronScatter(object):
    """ Class for elastic neutrino-electron scattering interaction.
    Implementing from USSR neutrino interaction function."""
    def __init__(self):
        pass

    def cross_section(self, erg_nu, flavor_nu):
        """ Calculate the cross section in cm2 of neutrino electron scattering
        assuming that transfer momentum is much less than mediator's mass (W, Z).
        Only neutrino electron and anti-neutrino electorn interacts.
        Inputs:
        - erg_nu: astropy.u.Quantity
            Neutrino energy in MeV.
        - flavor_nu: int or neutrino.Flavor
            Neutrino flavor. Can be an integer or an instance of neutrino.Flavor.
            Valid integer inputs:
                +1: nu_e     +2: nu_mu     +3: nu_tau
                -1: nu_e_bar -2: nu_mu_bar -3: nu_tau_bar
        Outputs:
        - cross_section: astropy.u.Quantity
            Return the full cross secton in cm2.
        """
        # convert neutrino energy unit into MeV
        try:
            erg_nu = erg_nu.to('MeV')
        except AttributeError:
            print('Neutrino energy must have unit of energy.')

        # define parameters
        # check for neutrino flavor
        epsilons = [-WEIGNBERG_SIN2, 0]
        if isinstance(flavor_nu, neutrino.Flavor):
            flavor_nu = flavor_nu.value
        if flavor_nu == 1 or flavor_nu == -1:
            epsilons[1] = -0.5-WEIGNBERG_SIN2
        elif (flavor_nu == 2
              or flavor_nu == 3
              or flavor_nu == -2
              or flavor_nu == -3):
            epsilons[1] = 0.5-WEIGNBERG_SIN2

        # check whether neutrino or anti-neutrino
        epsilon_p = epsilons[0]
        epsilon_m = epsilons[1]
        if flavor_nu < 0:
            epsilon_p = epsilons[1]
            epsilon_m = epsilons[0]

        # calculate cross section
        norm = 1.5e-44
        y_max = 1./(1+0.5*MASS_E/erg_nu)

        cross_section = norm*erg_nu.value
        cross_section *= (
            y_max**3*epsilon_p**2/3
            - y_max**2*(epsilon_p*epsilon_m*MASS_E/erg_nu+2*epsilon_p**2)/2
            + y_max*(epsilon_p**2+epsilon_m**2))

        return cross_section*u.cm**2

    def mean_leptonic_energy(self):
        pass

class NeutrinoOxygen16CC(object):
    """ Class for charged current neutrino 016 interaction. Implementing
    from USSR neutrino interaction function."""
    def __init__(self):
        """ Constructor set variable """
        # Minimum energy threshold of neutrino and anti-neutrino
        self._min_nu_erg = 15.4*u.MeV
        self._min_nubar_erg = 11.4*u.MeV
        self._nubar_threshold = 42.3293*u.MeV

    def _cross_section_par(self, erg_nu, *pars):
        """ General parameterized form of neutrino O16 CC interaction """
        return pars[0]*(erg_nu.value**pars[1]-pars[2]**pars[1])**pars[3]

    def cross_section(self, erg_nu, flavor_nu):
        """ Calculate the cross section in cm2 of charged current neutrino O16
        interaction using parameterized equations given neutrino energy in MeV
        and flavor. Only neutrino electron and anti-neutrino electorn interacts.
        Inputs:
        - erg_nu: astropy.u.Quantity
            Neutrino energy in MeV.
        - flavor_nu: int or neutrino.Flavor
            Neutrino flavor. Can be an integer or an instance of neutrino.Flavor.
            Valid integer inputs:
                +1: nu_e     +2: nu_mu     +3: nu_tau
                -1: nu_e_bar -2: nu_mu_bar -3: nu_tau_bar
        Outputs:
        - cross_section: astropy.u.Quantity
            Return the full cross secton in cm^2."""
        # convert neutrino energy unit into MeV
        try:
            erg_nu = erg_nu.to('MeV')
        except AttributeError:
            print('Neutrino energy must have unit of energy.')

        # convert Enum to scalar
        if isinstance(flavor_nu, neutrino.Flavor):
            flavor_nu = flavor_nu.value

        # check for neutrino flavor
        # parameter of cross section depended on flavor
        if flavor_nu == 1:
            cross_section = self._cross_section_par(
                erg_nu, 4.73e-40, 0.25, 15., 6)
            # check whether the input is a scalar or an array
            if hasattr(erg_nu.value, "__len__"):
                cross_section[erg_nu <= self._min_nu_erg] = 0.
            elif erg_nu <= self._min_nu_erg:
                cross_section = 0.
        elif flavor_nu == -1:
            if hasattr(erg_nu.value, "__len__"):
                cross_section = numpy.zeros_like(erg_nu.value)

                # the parameter of cross section of nubar_016 depends on energy
                i_low = numpy.logical_and(self._min_nubar_erg < erg_nu,
                                          erg_nu < self._nubar_threshold)
                i_high = erg_nu >= self._nubar_threshold

                cross_section[i_low] = self._cross_section_par(
                    erg_nu[i_low], 2.11357e-40, 0.224172, 8.36303, 6.80079)
                cross_section[i_high] = self._cross_section_par(
                    erg_nu[i_high], 2.11357e-40, 0.260689, 16.7893, 4.23914)
            else:
                if erg_nu <= self._min_nubar_erg:
                    cross_section = 0.
                elif erg_nu < self._nubar_threshold:
                    cross_section = self._cross_section_par(
                        erg_nu, 2.11357e-40, 0.224172, 8.36303, 6.80079)
                else:
                    cross_section = self._cross_section_par(
                        erg_nu, 2.11357e-40, 0.260689, 16.7893, 4.23914)
        else:
            # only nue and nue interacts
            cross_section = erg_nu.value-erg_nu.value

        # return cross section in u of cm2
        return cross_section*u.cm**2

    def mean_leptonic_energy(self, erg_nu, flavor_nu):
        """ Calculate the mean leptonic energy in MeV of charged current
        neutrino O16 interaction given neutrino energy in MeV and flavor.
        Only neutrino electron and anti-neutrin electorn interacts.
        Inputs:
        - erg_nu: astropy.u.Quantity
            Neutrino energy in MeV.
        - flavor_nu: int or neutrino.Flavor
            Neutrino flavor. Can be an integer or an instance of neutrino.Flavor.
            Valid integer inputs:
                +1: nu_e     +2: nu_mu     +3: nu_tau
                -1: nu_e_bar -2: nu_mu_bar -3: nu_tau_bar
        Outputs:
        - mean_erg_nu: astropy.u.Quantity
            Return the mean leptonic energy in MeV."""
        # convert neutrino energy unit into MeV
        try:
            erg_nu = erg_nu.to('MeV')
        except AttributeError:
            print('Neutrino energy must have unit of energy.')

        # convert Enum to scalar
        if isinstance(flavor_nu, neutrino.Flavor):
            flavor_nu = flavor_nu.value

        # calulate mean leptonic energy based on flavor
        if flavor_nu == 1 or flavor_nu == -1:
            # below minimum energy, mean lepton energy is 0 MeV
            if flavor_nu == 1:
                min_erg = self._min_nu_erg+CHERENKOV_ERG_THRESHOLD
            else:
                min_erg = self._min_nubar_erg+CHERENKOV_ERG_THRESHOLD
            mean_erg_lep = erg_nu-min_erg
            mean_erg_lep = numpy.minimum(mean_erg_lep, 0.*u.MeV)
        else:
            # only nue and nuebar interacts
            mean_erg_lep = erg_nu-erg_nu

        # return mean leptonic energy in u of MeV
        return mean_erg_lep

    @property
    def min_nu_erg(self):
        """ Return minimum energy threshold for neutrino """
        return self._min_nu_erg

    @property
    def min_nubar_erg(self):
        """ Return minimum energy threshold for anti-neutrino """
        return self._min_nubar_erg

    @property
    def nubar_threshold(self):
        """ Return the threshold energy of nuebar-016 interaction """
        return self._nubar_threshold


class NeutrinoOxygen16NC(object):
    """ Class for neutral current neutrino 016 interaction. Implementing from
    USSR neutrino interaction function. """
    def __init__(self):
        """ Constructor set variable """
        # Minimum energy threshold of neutrino
        self._min_erg = 12.*u.MeV

    def cross_section(self, erg_nu):
        """ Calculate the cross section in cm2 of neutral current neutrino O16
        interaction using parameterized equations given neutrino erg in MeV.
        Inputs:
        - erg_nu: astropy.u.Quantity
            Neutrino energy in MeV.
        Outputs:
        - cross_section: astropy.u.Quantity
            Return the full cross secton in cm^2.
        """
        # convert neutrino energy unit into MeV
        try:
            erg_nu = erg_nu.to('MeV')
        except AttributeError:
            print('Neutrino energy must have unit of energy.')

        # calculate cross section
        cross_section = 6.7e-40*(erg_nu.value**0.208-8.0**0.25)**6
        if hasattr(cross_section, '__len__'):
            cross_section[erg_nu <= self._min_erg] = 0
        elif erg_nu <= self._min_erg:
            cross_section = 0

        # return cross section in unit of cm^2
        return cross_section*u.cm**2


    def mean_leptonic_energy(self, erg_nu):
        """ Calculate the mean leptonic energy in MeV of charged current
        neutrino O16 interaction given neutrino energy in MeV.
        Inputs:
        - erg_nu: astropy.u.Quantity
            Neutrino energy in MeV.
        Outputs:
        - mean_erg_lep: astropy.u.Quantity
            Return the mean leptonic energy in MeV. Return 0 if energy is less
            than energy threshold."""
        # convert neutrino energy unit into MeV
        try:
            erg_nu = erg_nu.to('MeV')
        except AttributeError:
            print('Neutrino energy must have unit of energy.')

        # relative contribution from photon-production to NC cross section
        relative_photon_prob = 0.249+0.067
        photon_mean_erg_e = 4.7 # in MeV

        # relative contribution from neutron-production to NC cross section
        relative_neutron_prob = 0.187+0.067+0.085+0.007
        neutron_mean_erg_e = 0.95 # in MeV

        # calculate mean leptonic energy
        mean_erg_lep = relative_neutron_prob*neutron_mean_erg_e
        mean_erg_lep += relative_photon_prob*photon_mean_erg_e

        if hasattr(erg_nu, '__len__'):
            mean_erg_lep = numpy.where(erg_nu <= self._min_erg,
                                       0., mean_erg_lep)
        elif erg_nu <= self._min_erg:
            mean_erg_lep = 0.

        # return energy in MeV
        return mean_erg_lep*u.MeV

    @property
    def min_erg(self):
        """ Return minimum energy threshold for neutrino """
        return self._min_erg


class NeutrinoOxygen18(object):
    """ Class for the interaction of neutrino and O18. Implementing from
    USSR neutrino interaction function. """
    def __init__(self):
        """ Constructor sets up variables """
        self._O18_per_O = 0.002 # fraction of 018 isotope
        self._min_erg = 1.66*u.MeV  # min energy for interaction

    def cross_section(self, erg_nu, flavor_nu):
        """ Calculate the cross section in cm2 of neutrino-018 interaction
        given neutrino energy in MeV and neutrino flavor.
        Inputs:
        - erg_nu: astropy.u.Quantity
            Neutrino energy in MeV.
        - flavor_nu: int or neutrino.Flavor
            Neutrino flavor. Can be an integer or an instance of neutrino.Flavor.
            Valid integer inputs:
                +1: nu_e     +2: nu_mu     +3: nu_tau
                -1: nu_e_bar -2: nu_mu_bar -3: nu_tau_bar
        Outputs:
        - cross_section: astropy.u.Quantity
            Return the full cross secton in cm^2.
        """
        # convert neutrino energy unit into MeV
        try:
            erg_nu = erg_nu.to('MeV')
        except AttributeError:
            print('Neutrino energy must have unit of energy.')

        # only neutrino electron interacts
        if isinstance(flavor_nu, neutrino.Flavor):
            flavor_nu = flavor_nu.value
        if flavor_nu != 1:
            return (erg_nu-erg_nu).value*u.cm**2

        # calculate cross section for nue-018
        cross_section = 1.7e-50*erg_nu**2+1.6e-49*erg_nu-0.9e-49
        cross_section *= 1./self._O18_per_O
        if hasattr(erg_nu, '__len__'):
            cross_section[erg_nu <= self._min_erg] = 0.
        elif erg_nu <= self._min_erg:
            cross_section = 0.

        # return cross section in u of cm2
        return cross_section*u.cm**2

    def mean_leptonic_energy(self, erg_nu, flavor_nu):
        """ Calculate the mean leptonic energy in MeV of nu-018 interaction
        given neutrino energy in MeV and neutrino flavor.
        Inputs:
        - erg_nu: astropy.u.Quantity
            Neutrino energy in MeV.
        - flavor_nu: int or neutrino.Flavor
            Neutrino flavor. Can be an integer or an instance of neutrino.Flavor.
            Valid integer inputs:
                +1: nu_e     +2: nu_mu     +3: nu_tau
                -1: nu_e_bar -2: nu_mu_bar -3: nu_tau_bar
        Outputs:
        - cross_section: astropy.u.Quantity
            Return the full cross secton in cm^2."""
        # convert neutrino energy unit into MeV
        try:
            erg_nu = erg_nu.to('MeV')
        except AttributeError:
            print('Neutrino energy must have unit of energy.')

        # only neutrino electron interacts
        if isinstance(flavor_nu, neutrino.Flavor):
            flavor_nu = flavor_nu.value
        if flavor_nu != 1:
            return (erg_nu-erg_nu).value*u.cm**2

        # calculate mean leptonic energy in MeV
        min_erg = self._min_erg+CHERENKOV_ERG_THRESHOLD
        mean_erg_lep = erg_nu-min_erg
        mean_erg_lep = numpy.minimum(mean_erg_lep, 0.*u.MeV)

        return mean_erg_lep

    @property
    def min_erg(self):
        """ Return minimum energy threshold for neutrino """
        return self._min_erg

    @property
    def _O18_per_O(self):
        """ Return minimum energy threshold for neutrino """
        return self._O18_per_O
