from asteria import interactions
from asteria.neutrino import Flavor

import astropy.units as u

import numpy as np

def test_ibd_xs():
    ibd = interactions.InvBetaTab()
    flavors = [Flavor.nu_e, Flavor.nu_e_bar, Flavor.nu_x]
    Enu = np.asarray([10.])*u.MeV

    xs = ibd.cross_section(Flavor.nu_e_bar, Enu)
    assert(abs(xs[0] - 1.35e-41*u.cm**2) / xs[0] < 0.01)
