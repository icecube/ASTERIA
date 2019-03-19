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
        self.s2t12 = np.power(np.sin(t12), 2)
        self.c2t12 = np.power(np.cos(t12), 2)
        
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
        
        nu_e = [a + b for a, b in zip(nu_list[2], nu_list[3])]  
        nu_x = [(a + b + c)/2 for a, b, c in zip(nu_list[0], nu_list1[2], nu_list[3])]
        nu_e_bar = [a*self.c2t12 + (b+c)*self.s2t12 for a, b, c \
            in zip(nu_list[1], nu_list[2], nu_list[3])]
        nu_x_bar = [((1.0-self.c2t12)*a + (1.0+self.c2t12)*(b+c))/2 for a, b, c \
            in zip(nu_list[1], nu_list[2], nu_list[3])]
        nu_new = [nu_e, nu_e_bar, nu_x, nu_x_bar]
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
        nu_e = [a*self.s2t12 + b*self.c2t12 for a, b in zip(nu_list[0], nu_list[1])]   
        nu_x = [((1.0-self.s2t12)*a + (1.0+self.s2t12)*(b+c))/2 for a, b, c \
            in zip(nu_list[0], nu_list[2], nu_list[3])] 
        nu_e_bar = [a + b for a, b in zip(nu_list[2], nu_list[3])]
        nu_x_bar = [(a + b + c)/2 for a, b, c \
            in zip(nu_list[1], nu_list[2], nu_list[3])]
        nu_new = [nu_e, nu_e_bar, nu_x, nu_x_bar]
        return nu_new
