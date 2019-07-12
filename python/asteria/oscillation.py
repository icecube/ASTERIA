# -*- coding: utf-8 -*-
"""Various implementations of neutrino flavor oscillation.
"""

import numpy as np
from enum import Enum

from astropy.constants import c, hbar


class SimpleMixing(object):
    """Class that does very simple SN neutrino flavor mixing. Based on
    calculation in SNOwGLoBES.
    """
    
    def __init__(self, t12):
        '''Initializes the mixing angle
        
        Parameters
        ----------
        t12: float 
             Mixing angle
        '''
        self.t12 = np.radians(t12)
        self.s2t12 = np.sin(self.t12)**2
        self.c2t12 = np.cos(self.t12)**2
        
    def normal_mixing(self, nu_list):
        """Performs flavor oscillations as per normal hierarchy.
        
        Parameters
        ----------
        nu_list : ndarray
            neutrino fluxes ordered by flavor (nu_e, nu_e_bar, nu_x, nu_x_bar)
        
        Returns
        -------
        nu_new = ndarray
            neutrino fluxes after mixing (nu_e, nu_e_bar, nu_x, nu_x_bar)
        """
        nu_e = nu_list[2]
        nu_x = [(a + b)/2 for a, b in zip(nu_list[0], nu_list[2])]
        nu_e_bar = [a*self.c2t12 + (b)*self.s2t12 for a, b in zip(nu_list[1], nu_list[3])]
        nu_x_bar = [((1.0-self.c2t12)*a + (1.0+self.c2t12)*b)/2 for a, b in zip(nu_list[1], nu_list[3])]
        nu_new = np.asarray([nu_e, nu_e_bar, nu_x, nu_x_bar])
        return nu_new
    
    def inverted_mixing(self, nu_list):
        """Performs flavor oscillations as per inverted hierarchy.
        
        Parameters
        ----------
        nu_list : ndarray 
            neutrino fluxes ordered by flavor (nu_e, nu_e_bar, nu_x, nu_x_bar)
        
        Returns
        -------
        nu_new = ndarray
            neutrino fluxes after mixing (nu_e, nu_e_bar, nu_x, nu_x_bar)
        """
        nu_e = [a*self.s2t12 + b*self.c2t12 for a, b in zip(nu_list[0], nu_list[2])]
        nu_x = [((1.0-self.s2t12)*a + (1.0+self.s2t12)*b)/2 for a, b in zip(nu_list[0], nu_list[2])]
        nu_e_bar = nu_list[3]
        nu_x_bar = [(a + b)/2 for a, b in zip(nu_list[1], nu_list[3])]
        nu_new = np.asarray([nu_e, nu_e_bar, nu_x, nu_x_bar])
        return nu_new


class USSRMixing(object):
    """Class that does very simple SN neutrino flavor mixing. Based on
    calculation in USSR.
    """

    def __init__(self):
        '''Initializes the PMNS Matrix (Taken from USSR- OUTDATED!!)

        Parameters
        ----------
        t12: float
             Mixing angle
        '''
        self.pmns = np.asarray( [[0.825,  0.546,  0.148],
                                [-0.490, 0.562,  0.665],
                                [0.280, -0.621,  0.732]])

    def normal_mixing(self, source, flavor):
        """Performs flavor oscillations as per normal hierarchy.

        Parameters
        ----------
        flavor : asteria.neutrino.flavor
            CCSN Neutrino flavor
            
        source : asteria.source
            

        Returns
        -------
        mixed_spectrum : function
            Function object for neutrino energy spectrum after mixing
            Arguments are the same as by asteria.source.energy_spectrum
        """
        p_surv = 0
        if flavor.is_neutrino:
            p_surv = self.pmns[0, 1]**2
        else:
            p_surv = self.pmns[0, 0]**2
            
        if not flavor.is_electron:
            p_surv = 0.5 * (1 + p_surv)
            
        def mixed_spectrum(t, E, flavor):
            
            req_spectrum = source.energy_spectrum(t, E, flavor)
            req_flux = source.get_flux(t, flavor)
            comp_spectrum = source.energy_spectrum(t, E, flavor.oscillates_to)
            comp_flux = source.get_flux(t, flavor.oscillates_to)
            
            return p_surv * req_spectrum * req_flux + (1-p_surv) * comp_spectrum * comp_flux
        return mixed_spectrum
    
    def inverted_mixing(self, flavor, spectrum):
        """Performs flavor oscillations as per inverted hierarchy.
        IN THE USSR IMPLEMENTATION THIS IS THE SAME AS NORMAL"""
        return self.normal_mixing(self, flavor, spectrum)
