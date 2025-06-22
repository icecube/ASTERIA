import unittest
from importlib.resources import files
import astropy.units as u
import numpy as np

from asteria.stellardist import FixedDistance, StellarDensity

class TestStellarDistributions(unittest.TestCase):

    def test_fixed_distance(self):
        fd = FixedDistance(10*u.kpc)
        d = fd.distance()[0]
        self.assertTrue(d == 10*u.kpc)

        d = fd.distance(size=5)
        self.assertTrue(len(d) == 5)

    def test_stellar_density(self):
        np.random.seed(1)
        stellar_file = files('asteria.data.stellar').joinpath('sn_radial_distrib_adams.fits')
        self.assertTrue(stellar_file.exists())

        sd = StellarDensity(stellar_file)

        d = sd.distance()
        self.assertTrue(np.abs(d.to('kpc').value - 8.853) < 1e-3)
