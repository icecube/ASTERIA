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
        self.assertTrue(len(sim.time) == 2001)
        t = np.arange(-1, 1.001, 0.001)
        self.assertTrue(np.allclose(sim.time.value, t))

