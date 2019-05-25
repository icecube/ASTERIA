from asteria.stellardist import FixedDistance, StellarDensity

import astropy.units as u

import numpy as np

def test_fixed_distance():
    fd = FixedDistance(10*u.kpc)
    d = fd.distance()[0]
    assert(d == 10*u.kpc)

    d = fd.distance(size=5)
    assert(len(d) == 5)

def test_stellar_density():
    np.random.seed(1)
    sd = StellarDensity('data/stellar/sn_radial_distrib_adams.fits')

    d = sd.distance()
    assert(np.abs(d.to('kpc').value - 8.853) < 1e-3)
