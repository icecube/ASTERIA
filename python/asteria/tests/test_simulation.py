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

        sim = Simulation(configfile=inifile)

        # Test the configuration metadata.
        self.assertEqual(sim.model.__class__.__name__, 'Nakazato_2013')
        self.assertEqual(sim.model.metadata['Progenitor mass'], 13.0*u.Msun)
        self.assertEqual(sim.model.metadata['Revival time'], 300*u.ms)
        self.assertEqual(sim.model.metadata['Metallicity'].value, 0.004)
        self.assertEqual(sim.model.metadata['EOS'], 'shen')

        self.assertEqual(sim.distance, 10*u.kpc)
        self.assertEqual(sim.res_dt, 2.0*u.ms)
        self.assertEqual(str(sim.xform), 'AdiabaticMSW+NoVacuumTransformation+NoEarthMatter_NORMAL')
        self.assertEqual(str(sim.xform.mixing_params.mass_order), 'NMO')

        # Test a few of the simulation outputs.
        sim.run()

        n = len(sim.t)
        self.assertTrue(n == 2001)
        t = np.arange(-1, 1.001, 0.001)
        self.assertTrue(np.allclose(sim.t.value, t))

        # Average DOM signal.
        avg_dom_sig = {
            Flavor.NU_E: np.array([5.16889953e-06, 1.18123025e-03, 3.07186600e-04]),
            Flavor.NU_E_BAR: np.array([3.61184655e-08, 2.40719252e-02, 6.82728860e-03]),
            Flavor.NU_MU: np.array([1.55878943e-05, 6.66018627e-05, 1.96334114e-05]),
            Flavor.NU_MU_BAR: np.array([1.10412951e-10, 7.68557848e-05, 2.21263110e-05]),
            Flavor.NU_TAU: np.array([1.75533745e-05, 6.30614907e-05, 1.86384849e-05]),
            Flavor.NU_TAU_BAR: np.array([1.57766164e-10, 7.29366377e-05, 2.09731266e-05])
        }

        for fl in Flavor:
            ads = sim.avg_dom_signal(flavor=fl)[n//4::250]
            self.assertTrue(np.allclose(avg_dom_sig[fl], ads))

        # Detector signal, broken down by subdtectors.
        dt = 2*u.ms

        signal_hits = {
            'i3' : np.array([0., 0.18901162, 125.87067215, 35.56873207]),
            'dc' : np.array([0., 0.02268024,  15.07023892,  4.26536381])
        }

        for subdet in ('i3', 'dc'):
            t, sig = sim.detector_signal(subdetector=subdet, dt=dt)
            n = len(t)
            self.assertTrue(np.allclose(sig[n//4::250], signal_hits[subdet]))

        # Check that invalid subdetectors raise an exception.
        with self.assertRaises(ValueError):
            # 'md' refers to mDOMs in Gen2.
            t, sig = sim.detector_signal(subdetector='md', dt=dt)

#    def test_config_gen2_ini(self):
#        # Initialize a simulation from an INI file.
#        inifile = files('asteria.etc').joinpath('example-gen2.ini')
#        self.assertTrue(inifile.exists())
#
#        sim = Simulation(configfile=inifile)
#
#        # Test a few of the simulation outputs.
#        sim.run()
#
#        # Detector signal, broken down by subdtectors.
#        dt = 2*u.ms
#
#        signal_hits = {
#            'i3' : np.array([1.55359089e-01, 6.56390678e+02, 6.34836144e+01, 4.42375629e+01, 3.55362840e+01]),
#            'dc' : np.array([1.86482824e-02, 7.78847417e+01, 7.61162873e+00, 5.30584859e+00, 4.26287374e+00]),
#            'md' : np.array([9.10360348e-01, 3.72672464e+03, 3.70848101e+02, 2.58662274e+02, 2.07873086e+02])
#        }
#
#        for subdet in ('i3', 'dc', 'md'):
#            t, sig = sim.detector_signal(subdetector=subdet, dt=dt)
#            n = len(t)
#            self.assertTrue(np.allclose(sig[n//2::250], signal_hits[subdet]))
#
#        # Check that invalid subdetectors raise an exception.
#        with self.assertRaises(ValueError):
#            # 'ws' refers to mDOM + WLS in Gen2.
#            t, sig = sim.detector_signal(subdetector='ws', dt=dt)
