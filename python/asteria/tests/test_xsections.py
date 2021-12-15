from asteria import interactions
from snewpy.neutrino import Flavor

import astropy.units as u

import numpy as np

def test_ibd_xs():
    ibd = interactions.InvBetaTab()
    flavors = [Flavor.NU_E, Flavor.NU_E_BAR, Flavor.NU_X]
    Enu = np.asarray([10.])*u.MeV

    xs = ibd.cross_section(Flavor.NU_E_BAR, Enu)
    assert(abs(xs[0] - 1.35e-41*u.cm**2) / xs[0] < 0.01)
