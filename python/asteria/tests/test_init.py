import asteria
import unittest
import matplotlib as mpl
from importlib.resources import files

class TestInitialization(unittest.TestCase):

    def test_version_exists(self):
        self.assertTrue(hasattr(asteria, '__version__'))

    def test_set_rcparams_default(self):
        # Set matplotlib rcParams from a default configuration.
        asteria.set_rcparams()

        self.assertTrue(mpl.rcParams['axes.grid'])
        self.assertEqual(mpl.rcParams['axes.labelsize'], 24)
        self.assertEqual(mpl.rcParams['axes.linewidth'], 0.75)
        self.assertEqual(mpl.rcParams['axes.labelpad'], 8.)

        self.assertEqual(mpl.rcParams['xaxis.labellocation'], 'right')
        self.assertEqual(mpl.rcParams['yaxis.labellocation'], 'top')

        self.assertEqual(mpl.rcParams['xtick.direction'], 'out')
        self.assertEqual(mpl.rcParams['xtick.labelsize'], 20.)
        self.assertEqual(mpl.rcParams['xtick.major.size'], 5.   )
        self.assertEqual(mpl.rcParams['xtick.major.width'], 1.)
        self.assertTrue(mpl.rcParams['xtick.minor.visible'])
        self.assertEqual(mpl.rcParams['xtick.minor.size'], 2.5)
        self.assertEqual(mpl.rcParams['xtick.minor.width'], 1.)

        self.assertEqual(mpl.rcParams['ytick.direction'], 'out')
        self.assertEqual(mpl.rcParams['ytick.labelsize'], 20.)
        self.assertEqual(mpl.rcParams['ytick.major.size'], 5   )
        self.assertEqual(mpl.rcParams['ytick.major.width'], 1.)
        self.assertTrue(mpl.rcParams['ytick.minor.visible'])
        self.assertEqual(mpl.rcParams['ytick.minor.size'], 2.5)
        self.assertEqual(mpl.rcParams['ytick.minor.width'], 1.)

        self.assertEqual(mpl.rcParams['lines.linewidth'], 2.)

        self.assertEqual(mpl.rcParams['grid.alpha'], 0.75)


        self.assertEqual(mpl.rcParams['figure.subplot.hspace'], 0.05)
        self.assertEqual(mpl.rcParams['figure.figsize'], [6., 7.])

        self.assertEqual(mpl.rcParams['legend.fontsize'], 16.)

        self.assertEqual(mpl.rcParams['font.size'], 18.)
        self.assertEqual(mpl.rcParams['font.family'], ['serif'])

    def test_set_rcparams_verbose(self):
        # Check that verbose output works.
        rcparams = asteria.set_rcparams(verbose=True)
        self.assertIsNotNone(rcparams)
        self.assertIsInstance(rcparams, mpl.RcParams)

    def test_set_rcparams_from_file(self):
        # Check manually set file.
        rcfile = files('asteria.etc').joinpath('asteria.rcParams')
        self.assertTrue(rcfile.exists())

        rcparams = asteria.set_rcparams(verbose=True)
        self.assertIsNotNone(rcparams)
        self.assertIsInstance(rcparams, mpl.RcParams)

        with self.assertRaises(FileNotFoundError):
            asteria.set_rcparams('does_not_exist.rcParams')
