import unittest
import numpy as np

import astropy.units as u

from asteria.simulation import Simulation
from snewpy.neutrino import Flavor

import os

class TestSimulation(unittest.TestCase):

    def setUp(self):
        # Create a list of interactions for use later.
        self.basedir = os.environ['ASTERIA']

    def test_config_from_ini(self):
        # Initialize a simulation from an INI file.
        inifile = os.path.join(self.basedir, 'data/config/example.ini')
        sim = Simulation(config=inifile)
        sim.run()

        # Test a few of the simulation outputs.
        n = len(sim.time)
        self.assertTrue(n == 2001)
        t = np.arange(-1, 1.001, 0.001)
        self.assertTrue(np.allclose(sim.time.value, t))

        avg_dom_sig = {
            Flavor.NU_E: np.array([4.20202084e-06, 8.46115430e-03, 5.86603896e-04, 3.92918377e-04, 3.03062133e-04]),
            Flavor.NU_X: np.array([2.68005994e-05, 5.82466719e-04, 6.43659362e-05, 4.63354600e-05, 3.77328193e-05]),
            Flavor.NU_E_BAR: np.array([2.02507386e-08, 1.22391284e-01, 1.19597714e-02, 8.34539170e-03, 6.71534764e-03]),
            Flavor.NU_X_BAR: np.array([1.25463999e-10, 5.30875684e-04, 7.43147121e-05, 5.29821047e-05, 4.25364249e-05])
        }

        for fl in Flavor:
            ads = sim.avg_dom_signal(flavor=fl)[n//2::250]
            self.assertTrue(np.allclose(avg_dom_sig[fl], ads))

