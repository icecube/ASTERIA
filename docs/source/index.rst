.. ASTERIA documentation master file, created by
   sphinx-quickstart on Sat May 31 12:54:31 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ASTERIA: A Supernova TEst Routine for IceCube Analysis
======================================================

ASTERIA is a fast supernova neutrino simulation designed for the IceCube Neutrino Observatory. The original version, called the Unified Supernova Simulation Routine (USSR), was written in C++ by Thomas Kowarik and Gösta Kroll at Universität Mainz in 2011. This project began as a Python port of the original program.

The code uses estimates of the supernova neutrino luminosity from large-scale simulations of core-collapse supernovae to calculate photons in the IceCube detector. The calculation includes parameterizations of the most important interactions contributing to signal in the ice from core-collapse neutrinos:

* Inverse beta decay.
* Electron scattering.
* Charged-current interactions on <sup>16</sup>O.
* Neutral-current interactions on <sup>16</sup>O.
* Inelastic scattering on <sup>17/18</sup>O.

Details are available in R. Abbasi et al., `IceCube sensitivity for low-energy neutrinos from nearby supernovae <(https://arxiv.org/abs/1108.0171>`_, A&A 535:A109, 2011.

Access to supernova neutrino simulations is provided through the SNEWPY code, documented on `readthedocs <https://snewpy.readthedocs.io/en/stable/>`_ and `github <https://github.com/SNEWS2/snewpy>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
