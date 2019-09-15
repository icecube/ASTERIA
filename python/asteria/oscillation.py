# -*- coding: utf-8 -*-
"""Various implementations of neutrino flavor oscillation.
"""

import numpy as np


class SimpleMixing(object):
    """Class that does very simple SN neutrino flavor mixing. Based on
    calculation in SNOwGLoBES from arXiv:1508.00785.
    """

    def __init__(self, t12):
        '''Initializes the mixing angle
        
        Parameters
        ----------
        t12: float 
             Mixing angle
        '''
        self.t12 = np.radians(t12)
        self.s2t12 = np.sin(self.t12) ** 2
        self.c2t12 = np.cos(self.t12) ** 2

    def normal_mixing(self, source):
        """Generates spectrum object under normal hierarchy mixing.
        
        Parameters
        ----------
        source : asteria.source

        Returns
        -------
        mixed_spectrum : function
            Function object for neutrino energy spectrum after mixing
            Arguments are the same as by asteria.source.energy_spectrum
        """

        def mixed_spectrum(t, E, flavor):

            if flavor.is_neutrino:
                if flavor.is_electron:
                    coeff = [0, 1]
                else:
                    coeff = [0.5, 0.5]
            else:
                if flavor.is_electron:
                    coeff = [self.c2t12, self.s2t12]
                else:
                    coeff = [0.5 * (1.0 + self.c2t12), 0.5 * self.s2t12 ]

            req_spectrum = source.energy_spectrum(t, E, flavor)
            req_flux = source.get_flux(t, flavor)
            comp_spectrum = source.energy_spectrum(t, E, flavor.oscillates_to)
            comp_flux = source.get_flux(t, flavor.oscillates_to)

            return coeff[0] * req_spectrum * req_flux + coeff[1] * comp_spectrum * comp_flux

        return mixed_spectrum

    def inverted_mixing(self, source):
        """Generates spectrum object under inverted hierarchy mixing.
        
        Parameters
        ----------
        source : asteria.source

        Returns
        -------
        mixed_spectrum : function
            Function object for neutrino energy spectrum after mixing
            Arguments are the same as by asteria.source.energy_spectrum
        """

        def mixed_spectrum(t, E, flavor):

            if flavor.is_neutrino:
                if flavor.is_electron:
                    coeff = [self.s2t12, self.c2t12]
                else:
                    coeff = [0.5 * (1 + self.s2t12), 0.5 * self.c2t12]
            else:
                if flavor.is_electron:
                    coeff = [0, 1]
                else:
                    coeff = [0.5, 0.5]

            req_spectrum = source.energy_spectrum(t, E, flavor)
            req_flux = source.get_flux(t, flavor)
            comp_spectrum = source.energy_spectrum(t, E, flavor.oscillates_to)
            comp_flux = source.get_flux(t, flavor.oscillates_to)

            return coeff[0] * req_spectrum * req_flux + coeff[1] * comp_spectrum * comp_flux

        return mixed_spectrum


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

    def normal_mixing(self, source):
        """Performs flavor oscillations as per normal hierarchy.

        Parameters
        ----------
        source : asteria.source


        Returns
        -------
        mixed_spectrum : function
            Function object for neutrino energy spectrum after mixing
            Arguments are the same as by asteria.source.energy_spectrum
        """
            
        def mixed_spectrum(t, E, flavor):
            if flavor.is_neutrino:
                p_surv = self.pmns[0, 1] ** 2
            else:
                p_surv = self.pmns[0, 0] ** 2

            if not flavor.is_electron:
                p_surv = 0.5 * (1 + p_surv)
            
            req_spectrum = source.energy_spectrum(t, E, flavor)
            req_flux = source.get_flux(t, flavor)
            comp_spectrum = source.energy_spectrum(t, E, flavor.oscillates_to)
            comp_flux = source.get_flux(t, flavor.oscillates_to)
            
            return p_surv * req_spectrum * req_flux + (1-p_surv) * comp_spectrum * comp_flux
        return mixed_spectrum
    
    def inverted_mixing(self, source):
        """Performs flavor oscillations as per inverted hierarchy.
        IN THE USSR IMPLEMENTATION THIS IS THE SAME AS NORMAL"""
        return self.normal_mixing(source)
