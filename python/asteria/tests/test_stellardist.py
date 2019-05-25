from asteria.stellardist import FixedDistance, StellarDensity

import astropy.units as u

import numpy as np

def test_fixed_distance():
    fd = FixedDistance(10*u.kpc)
    d = fd.distance()[0]
    assert(d == 10*u.kpc)

    d = fd.distance(size=5)
    assert(len(d) == 5)

#def test_prob_distance():
#    pd = ProbDistance('data/stellar/sn_radial_distrib_adams.fits')
#    d = pd.get_distance()
#    assert(d < 30*u.kpc)
#
#    d = pd.get_distance(size=5)
#    assert(len(d) == 5)
