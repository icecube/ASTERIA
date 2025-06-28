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

        # Average DOM signal.
        avg_dom_sig = {
            Flavor.NU_E: np.array([4.26963191e-06, 8.53221226e-03, 5.95606673e-04, 3.99037307e-04, 3.07812661e-04]),
            Flavor.NU_X: np.array([2.72318245e-05, 5.87358355e-04, 6.53537785e-05, 4.70570435e-05, 3.83242849e-05]),
            Flavor.NU_E_BAR: np.array([2.05765757e-08, 1.23419143e-01, 1.21433214e-02, 8.47535474e-03, 6.82061135e-03]),
            Flavor.NU_X_BAR: np.array([1.27482731e-10, 5.35334052e-04, 7.54552411e-05, 5.38071966e-05, 4.32031874e-05])
        }

        for fl in Flavor:
            ads = sim.avg_dom_signal(flavor=fl)[n//2::250]
            self.assertTrue(np.allclose(avg_dom_sig[fl], ads))

        # Detector signal, broken down by subdtectors.
        dt = 2*u.ms

        signal_hits = {
            'i3' : np.array([1.55373919e-01, 6.56452122e+02, 6.34896629e+01, 4.42417801e+01, 3.55396726e+01]),
            'dc' : np.array([1.86439333e-02, 7.78671331e+01, 7.60985887e+00, 5.30461375e+00, 4.26188122e+00])
        }

        for subdet in ('i3', 'dc'):
            t, sig = sim.detector_signal(subdetector=subdet, dt=dt)
            n = len(t)
            self.assertTrue(np.allclose(sig[n//2::250], signal_hits[subdet]))

        # Check that invalid subdetectors raise an exception.
        with self.assertRaises(ValueError):
            # 'md' refers to mDOMs in Gen2.
            t, sig = sim.detector_signal(subdetector='md', dt=dt)

    def test_config_gen2_ini(self):
        # Initialize a simulation from an INI file.
        inifile = files('asteria.etc').joinpath('example-gen2.ini')
        self.assertTrue(inifile.exists())

        sim = Simulation(config=inifile)

        # Test a few of the simulation outputs.
        sim.run()

        # Detector signal, broken down by subdtectors.
        dt = 2*u.ms

        signal_hits = {
            'i3' : np.array([1.55359089e-01, 6.56390678e+02, 6.34836144e+01, 4.42375629e+01, 3.55362840e+01]),
            'dc' : np.array([1.86482824e-02, 7.78847417e+01, 7.61162873e+00, 5.30584859e+00, 4.26287374e+00]),
            'md' : np.array([9.10360348e-01, 3.72672464e+03, 3.70848101e+02, 2.58662274e+02, 2.07873086e+02])
        }

        for subdet in ('i3', 'dc', 'md'):
            t, sig = sim.detector_signal(subdetector=subdet, dt=dt)
            n = len(t)
            self.assertTrue(np.allclose(sig[n//2::250], signal_hits[subdet]))

        # Check that invalid subdetectors raise an exception.
        with self.assertRaises(ValueError):
            # 'ws' refers to mDOM + WLS in Gen2.
            t, sig = sim.detector_signal(subdetector='ws', dt=dt)
