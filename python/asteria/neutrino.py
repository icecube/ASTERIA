# -*- coding: utf-8 -*-
"""Some basic neutrino physics.
"""

import numpy as np
from enum import Enum, EnumMeta, _EnumDict

from astropy.constants import c, hbar


class Ordering(Enum):
    """Neutrino mass ordering types.
    """
    normal = 1
    inverted = 2
    any = 3
    none = 4


class _FlavorMeta(EnumMeta):
    """ Internal Meta-class for Flavor enumeration object.
    Extends functionality of enum.Enum __new__ and __call__ methods.
    
    .. data:: _FlavorDict : dict
        Dictionary of CCSN model neutrino types.
    
    """
    _FlavorDict = {'nu_e'      :  1,
                   'nu_e_bar'  : -1,
                   'nu_x'      :  2,
                   'nu_x_bar'  : -2 }
               
    def __call__(cls, requests=None):
        """Given a dictionary of requests for CCSN neutrino flavors, returns an
        enumeration containing the flavors.           
            
           .. param :: cls : Flavor
              Called object. (Similar to 'self' keyword)
            
           .. param :: requests : dict 
              Dictionary of requested flavors
               - Keys with value True are initialized as enumeration members
               - Keys with value False and keys missing from requests are added to excluded. 
                
        """
        # Declare Meta-class _FlavorMeta for error-checking.
        metacls = cls.__class__         
        
        # If no requests have been made, raise an error.
        if requests is None or all( not val for val in requests.values() ):
            raise RuntimeError('No flavors requested. ')
        # If an unknown flavor is requested, rais an error.    
        elif any( key not in metacls._FlavorDict for key in requests):
            raise AttributeError('Unknown flavor(s) "{0}" Requested'.format(
                                 '", "'.join( set(requests)-set(metacls._FlavorDict)) ))
        # If requests does not have all boolean values, throw an error.  
        elif not all( isinstance( val, bool) for val in requests.values() ):
            bad_vals = [' {0} : {1}'.format(key, val) for key, val in 
                        requests.items() if not isinstance( val, bool) ]
            raise ValueError('Requests must be dictionary with bool values. '+
                              'Given... '+ ',\n\t{:59}'.format('').join(bad_vals))
        # Otherwise, create a new Enum object...
        
        
        # Exclude any missing flavors from the enumeration members  
        missing = {key: False for key in metacls._FlavorDict if 
                   key not in requests }
        requests.update( missing )
        
        # Sort requests according to metacls._FlavorDict
        requests = {key: requests[key] for key in metacls._FlavorDict }
        
        # Populate an _EnumDict with fields required for Enum creation.        
        bases = (Enum, )
        classdict = _EnumDict()
        fields = {'__doc__'               : cls.__doc__,
                  '__init__'              : cls.__init__,
                  '__module__'            : cls.__module__,
                  '__qualname__'          : 'Flavor',
                  '_generate_next_value_' : cls._generate_next_value_,
                  'to_tex'                : cls.to_tex,
                  'is_electron'           : cls.is_electron,
                  'is_neutrino'           : cls.is_neutrino,
                  'is_antineutrino'       : cls.is_antineutrino,
                  'requests'              : requests }
        classdict.update({ key : val for key, val in fields.items()})
        
        # Create and return an Enum object using _FlavorMeta.__new__      
        return metacls.__new__(metacls, 'Flavor', bases, classdict)
    
    def __new__(metacls, cls, bases, classdict): 
        """Returns an Enum object containing CCSN neutrino flavors.
    
            .. param:: metacls : class 
                Meta-class of new Enum object being created (_FlavorMeta).
                
            .. param:: cls : str
                String for name of new Enum object being created
                
            .. param:: bases : tuple
                Tuple of base classes ( enum.Enum,).  
                
            .. param:: classdict : _EnumDict
                Extended dictionary object (from package enum) for creating an Enum object.                
        """
        
        # Use default flavors
        if 'requests' not in classdict:
            classdict.update( {'requests' : { key: True for key in metacls._FlavorDict }} )
        
        for key, val in classdict['requests'].items():
            if val:
                # Add a member to the enumeration.
                classdict[key] = metacls._FlavorDict[key]
            else:
                # DO NOT add a member to the enumeration.
                classdict.update({ key : metacls._FlavorDict[key] })
            
        # Create and return an Enum object using Enum.__new__ method.
        return super().__new__(metacls, cls, bases, classdict)


class Flavor(Enum, metaclass=_FlavorMeta):
    """CCSN model neutrino types. 
    
    .. param (Optional):: requests : dict       
        Dictionary of requested neutrino flavors. Each key must be
        the string of an flavor name, values must be True/False. 
        The default is
        
        default = {'nu_e'      : True,
                   'nu_e_bar'  : True,
                   'nu_x'      : True,
                   'nu_x_bar'  : True }.
                   
    .. data :: nu_e
        Electron neutrino.

    .. data :: nu_e_bar
        Electron antineutrino.

    .. data :: nu_x
        Muon neutrino or tau neutrino.
        
    .. data :: nu_x_bar
        Muon antineutrino or tau antineutrino.
                   
    .. data :: requests : dict
        Dictionary of requests made for flavors (e.g. default).
        
    See also: _FlavorMeta Meta-class, which extends the
    functionality of this object and defines its type.        
    """        
     
    def to_tex(self):
        """LaTeX-comptable string representations of flavor.
        """
        if '_bar' in self.name:
            return r'$\overline{{\nu}}_{0}$'.format(self.name[3])
        return r'$\{0}$'.format(self.name)

    @property
    def is_electron(self):
        return self.value in (Flavor.nu_e.value, Flavor.nu_e_bar.value)

    @property
    def is_neutrino(self):
        return self.value in (Flavor.nu_e.value, Flavor.nu_x.value)

    @property
    def is_antineutrino(self):
        return self.value in (Flavor.nu_e_bar.value, Flavor.nu_x_bar.value)


class ValueCI(object):

    def __init__(self, val, ci_lo, ci_hi):
        """Initialize a value with potentially asymmetric error bars.
        Assume the CI refers to 68% region around the central value.

        Parameters
        ----------
        val : float
            Centra value.
        ci_lo : float
            Lower range of 68% C.I. around central value.
        ci_hi : float
            Upper range of 68% C.I. around central value.
        """
        self.value = val
        self.ci_lo = ci_lo
        self.ci_hi = ci_hi

    def get_random(self, n=1):
        """Randomly sample values from the distribution.
        If the distribution is asymmetric, treat it as a 2-sided Gaussian.

        Parameters
        ----------
        n : int
            Number of random draws.

        Returns
        -------
        Returns n random draws from a symmetric/asymmetric Gaussian about
        the central value.
        """
        if self.ci_lo == self.ci_hi:
            return np.random.normal(loc=self.value, scale=self.ci_lo, size=n)
        else:
            return np.random.normal(loc=self.value, scale=self.ci_lo, size=n)
        

class PMNS(object):

    def __init__(self):
        pass


#class Oscillation(object):
#
#    def __init__(self, theta12, theta23, theta13, deltaCP,
#                 deltaM2_21, deltaM2_32):
#
#        self._theta12 = theta12
#        self._theta23 = theta23
#        self._theta13 = theta13
#        self._deltaCP = deltaCP
#
#        self._deltaM2_21 = deltaM2_21
#        self._deltaM2_32 = deltaM2_32
#        m2_2 = np.fmax(self._deltaM2_21, self._deltaM2_32)
#        m1_2 = m2_2 - self._deltaM2_21
#        m3_2 = m2_2 + self._deltaM2_32
#        self._m2 = [m1_2, m2_2, m3_2]
#
#        c12 = np.cos(self._theta12)
#        s12 = np.sin(self._theta12)
#        c23 = np.cos(self._theta23)
#        s23 = np.sin(self._theta23)
#        c13 = np.cos(self._theta13)
#        s13 = np.sin(self._theta13)
#        pCP = np.cos(self._deltaCP) + 1j * np.sin(self._deltaCP)
#        mCP = np.cos(self._deltaCP) - 1j * np.sin(self._deltaCP)
#        self._PMNS = np.asarray([
#            [c12 * c13, s12 * c13, s13 * mCP],
#            [-s12 * c23 - c12 * s23 * s13 * pCP, c12 * c23 - s12 * s23 * s13 * pCP, s23 * c13],
#            [s12 * s23 - c12 * c23 * s13 * pCP, -c12 * s23 - s12 * c23 * s13 * pCP, c23 * c13]])
#
#    @property
#    def theta12(self):
#        return self._theta12
#
#    @property
#    def theta23(self):
#        return self._theta23
#
#    @property
#    def theta13(self):
#        return self._theta13
#
#    @property
#    def deltaCP(self):
#        return self._deltaCP
#
#    @property
#    def pmns(self):
#        return self._PMNS
#
#    def prob(self, nu_i, nu_f, L, E):
#        if nu_i in neutrinos:
#            if nu_f in antineutrinos:
#                return 0.  # nu->nu_bar is impossible
#            U = self.pmns
#        elif nu_i in antineutrinos:
#            if nu_f in neutrinos:
#                return 0.  # nu_bar->nu is impossible
#            U = self.pmns.conj()
#        else:
#            raise ValueError('Invalid initial neutrino state')
#
#        a = np.abs(nu_i.value) - 1
#        b = np.abs(nu_f.value) - 1
#        s = 0.
#
#        # Convert to natural units with hbar=c=1.
#        m2 = [M2.to('GeV**2').value for M2 in self._m2]
#        LGeV = (L / (hbar * c)).to('1/GeV').value
#        EGeV = E.to('GeV').value
#
#        # Neutrino oscillation probability amplitude:
#        for j in range(3):
#            uu = U[a][j].conjugate() * U[b][j]
#            phase = 0.5 * m2[j] * LGeV / EGeV
#            s += uu * np.exp(-1j * phase)
#
#        return np.abs(s) ** 2
