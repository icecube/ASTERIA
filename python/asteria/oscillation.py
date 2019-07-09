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
        pmns = np.asarray( [[0.825,  0.546,  0.148],
                            [-0.490, 0.562,  0.665],
                            [0.280, -0.621,  0.732]])
        self.P_nu_e = pmns[0, 1] ** 2
        self.P_nu_e_bar = pmns[0, 0] ** 2
        self.P_nu_x = 0.5 * (1 + self.P_nu_e)
        self.P_nu_x_bar = 0.5 * (1 + self.P_nu_e_bar)

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

        nu_e = self.P_nu_e*nu_list[0] + (1-self.P_nu_e)*nu_list[2]
        nu_x = self.P_nu_x*nu_list[2] + (1-self.P_nu_x)*nu_list[0]
        nu_e_bar = self.P_nu_e_bar*nu_list[1] + (1-self.P_nu_e_bar)*nu_list[3]
        nu_x_bar = self.P_nu_x_bar*nu_list[3] + (1-self.P_nu_x_bar)*nu_list[1]
        nu_new = np.asarray([nu_e, nu_e_bar, nu_x, nu_x_bar])
        return nu_new

    def inverted_mixing(self, nu_list):
        """Performs flavor oscillations as per inverted hierarchy.
        IN THE USSR IMPLEMENTATION THIS IS THE SAME AS NORMAL"""
        return self.normal_mixing(nu_list)
