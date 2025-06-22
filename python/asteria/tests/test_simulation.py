import unittest
import numpy as np

import astropy.units as u

from asteria.simulation import Simulation
from snewpy.neutrino import Flavor, MassHierarchy

from importlib.resources import files

class TestSimulation(unittest.TestCase):

    def test_config_from_ini(self):
        # Initialize a simulation from an INI file.
        inifile = files('asteria.etc').joinpath('example.ini')
        self.assertTrue(inifile.exists())

        sim = Simulation(config=inifile)

        # Test the configuration metadata.
        self.assertEqual(sim.metadata['model']['name'], 'Nakazato_2013')
        self.assertEqual(sim.metadata['model']['param'], 'progenitor_mass, 13.0 solMass; revival_time, 300.0 ms; metallicity, 0.02; eos, shen')
        self.assertEqual(sim.metadata['distance'], '10.0 kpc')
        self.assertEqual(sim.metadata['res_dt'], '2.0 ms')
        self.assertEqual(sim.metadata['interactions'], 'ElectronScatter, InvBetaPar, Oxygen16CC, Oxygen16NC, Oxygen18')
        self.assertEqual(sim.metadata['flavors'], "<enum 'Flavor'>")
        self.assertEqual(sim.metadata['hierarchy'], 'default')
        self.assertEqual(sim.metadata['mixing_scheme'], 'AdiabaticMSW')

        # Test the simulation object.
        self.assertTrue(np.allclose(sim.distance, 10*u.kpc))
        self.assertEqual(sim.hierarchy, MassHierarchy.NORMAL)
        self.assertEqual(sim.mixing_scheme, 'AdiabaticMSW')

        # Test a few of the simulation outputs.
        sim.run()

        n = len(sim.time)
        self.assertTrue(n == 2001)
        t = np.arange(-1, 1.001, 0.001)
        self.assertTrue(np.allclose(sim.time.value, t))

        avg_dom_sig = {
            Flavor.NU_E: np.array([4.26963191e-06, 8.53221226e-03, 5.95606673e-04, 3.99037307e-04, 3.07812661e-04]),
            Flavor.NU_X: np.array([2.72318245e-05, 5.87358355e-04, 6.53537785e-05, 4.70570435e-05, 3.83242849e-05]),
            Flavor.NU_E_BAR: np.array([2.05765757e-08, 1.23419143e-01, 1.21433214e-02, 8.47535474e-03, 6.82061135e-03]),
            Flavor.NU_X_BAR: np.array([1.27482731e-10, 5.35334052e-04, 7.54552411e-05, 5.38071966e-05, 4.32031874e-05])
        }

        for fl in Flavor:
            ads = sim.avg_dom_signal(flavor=fl)[n//2::250]
            self.assertTrue(np.allclose(avg_dom_sig[fl], ads))

