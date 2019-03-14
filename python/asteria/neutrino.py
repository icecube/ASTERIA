# -*- coding: utf-8 -*-
"""Some basic neutrino physics.
"""

import numpy as np
from enum import Enum

from astropy.constants import c, hbar


class Ordering(Enum):
    """Neutrino mass ordering types.
    """
    normal = 1
    inverted = 2
    any = 3
    none = 4


class Flavor(Enum):
    """CCSN model neutrino types.

    .. data:: nu_e
        Electron neutrino.

    .. data:: nu_e_bar
        Electron antineutrino.

    .. data:: nu_x
        Muon neutrino or tau neutrino.
        
    .. data:: nu_x_bar
        Muon antineutrino or tau antineutrino.
    """
    nu_e = 1,
    nu_e_bar = -1,
    nu_x = 2
    nu_x_bar = -2

    def to_tex(self):
        """LaTeX-comptable string representations of flavor.
        """
        if '_bar' in self.name:
            return r'$\overline{{\nu}}_{0}$'.format(self.name[3])
        return r'$\{0}$'.format(self.name)

    @property
    def is_electron(self):
        return self in (Flavor.nu_e, Flavor.nu_e_bar)

    @property
    def is_neutrino(self):
        return self in (Flavor.nu_e, Flavor.nu_x)

    @property
    def is_antineutrino(self):
        return self in (Flavor.nu_e_bar, Flavor.nu_x_bar)


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
