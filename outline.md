---
title: 'ASTERIA: A Python package for fast Core-Collapse Supernova simulations in IceCube'

tags: 
  - Python
  - astronomy
  - astrophyiscs
  - particle physics
  - neutrinos

authors:
  - name: Spencer Griswold
    orcid: 
    afffiliation: 1

affiliations:
  - name: University of Rochester
    index: 1

date: 4 November 2020

bibliography: paper.bib

---

# Summary
_Describe the high-level functionality and purpose of the software for a diverse, non-specialist audience._

__Outline__
- ASTERIA is a package for performing fast simulations of the IceCube Neutrino Observatory's response to core collapse 
supernovae.
- Provides a suite of tools for handling physical attributes IE neutrino interactions, flavors and oscillations
- Easy-to-use interface for including a wide variety of CCSN models

# Statement of need
_Illustrate the research purpose of the software_

__Outline__
- Simulating CCSN in IceCube presents computational difficulty
  - Full MC simulations are expensive
  - Explosion simulations have many free parameters
  - Neutrino fundamental properties are uncertain/unknown
- ASTERIA addresses these issues.

__Scratch__

- Simulating CCSN in IceCube presents computation difficulty
  - Full Monte Carlo tracking neutrino interactions and photon production with full raytracing through a non-uniform 
detector medium is extremely expensive. Moreover, most of the photons are never detected.
  - There are many different explosion simulations and models which depend on free parameters like the progenitor mass, 
the equation of state of the proto-neutron star, etc. Many different models with explosions produced at different 
distances from Earth need to be considered to fully investigate the sensitivity to CCSN.
  - Unknown or highly uncertain fundamental properties of neutrinos can have a significant impact on the signal observed 
at Earth. For example, the neutrino mixing angles and mass hierarchy, as well as different neutrino interactions and 
cross sections of those interactions. These uncertainties also need to be explored systematically.
- ASTERIA is designed to address these issues in a computationally efficient manner. It’s not a full MC but is validated 
against more complete but expensive detector simulations.

# Package Structure and Workflow (Include?)
ASTERIA consists of a number of primary modules.

__Outline__
- CCSN Models (source.py)
- Interactions (interactions.py)
- Flavors & Oscillations (neutrino.py, oscillations.py)
- Numpy & primary calculation
- Performance improvements over C++/ROOT predecessor

__Scratch__

ASTERIA is Python-based framework for performing fast supernova monte carlo simulations of the IceCube Detector 
response. It relies heavily on the python NumPy, SciPy, and AstroPy packages. Given a model of the luminosity, flux and 
average neutrino energy as a function of time, ASTERIA will produce the average expected signal in an IceCube light 
sensor. The calculation of this signal is given by [SIGNAL EQUATION], performed over a number of steps. 