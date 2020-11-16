## Summary
ASTERIA is a python package for performing fast simulations of the IceCube Neutrino Observatory’s detector response to 
core collapse supernovae (CCSNe). ASTERIA takes a model of the flux and energy of neutrinos produced by a CCSN, 
propagates them through a variety of neutrino oscillation scenarios and neutrino interactions to obtain the number of 
photons observed in the IceCube detector. This package provides a suit of tools for handling the physical attributes of 
neutrinos such as neutrino oscillations, flavors, and interactions. It also provides an interface through which a 
variety of core collapse models may be used. The ASTERIA simulation suite is Object-Oriented in nature, making it easy 
to use. Through the use of Numpy-vectorized functions and simplifying assumptions about the detection medium, large 
numbers of results may be quickly obtained (give runtime description w/ chipset). Though ASTERIA has been developed to 
simulate the response of the IceCube neutrino Detector, its application extends to other experiments and to the handling
of astrophysical models.

## Statement of Need
Simulating CCSN in the icecube detector presents considerable computational difficulty. (i) The non-uniform medium of 
the IceCube detector makes full Monte carlo tracking of neutrino interactions and subsequent photon production extremely 
computationally expensive. Additionally, most photons are never detected. (ii) There are many different CCSN explosion 
models which depend on free parameters including the progenitor mass, and the equation of state of the stellar core. 
Many different models with explosions produced at different distances from Earth are required to fully investigate 
IceCube’s sensitivity to CCSN. (iii) Neutrinos have fundamental properties that are unknown or highly uncertain, such 
as the neutrino mixing angles and mass hierarchy, and these properties can have significant impacts on the signal 
observed at earth. These uncertainties must also be explored systematically.

ASTERIA was designed to address each of these concerns. (i) The ASTERIA simulation makes use of simplifying assumptions 
about the detector medium which significantly reduces the computational cost. Rather than propagate the photons produced
by neutrino interactions, a probability of observing a photon is computed using properties of the ice and of the 
modules of the IceCube detector. This method has been vetted against more robust simulations. (ii) ASTERIA provides an 
interface through which any model may be loaded into the simulation, provided it can be converted to the appropriate 
format. ASTERIA requires time profiles of the neutrino luminosity and mean energy produced by a CCSN. Given these 
quantities, the remaining required quantities to run the ASTERIA simulation can be extracted. Already, a variety of 
public models have been implemented in ASTERIA, including models by Nakazato, OConnor, and Suhkbold (Add citations). 
(iii) ASTERIA provides a number of objects that handle neutrino properties, and can be configured to suit the purpose 
of an analysis. These three design choices provide a tool for efficiently running a large number of varied CCSN 
simulations.

## Project structure and workflow
ASTERIA simulations are accessed through the top-level handler object. The handler object configured the other six 
components which interact with each other: config, source, Flavor, Interactions, SimpleMixing, and detector.

Config loads a configuration in yaml format, and stores paths to relevant data files including the model file, the 
simulation output file and a table describing the detector properties. Source loads the explosion model and provides 
access to quantities such as the neutrino luminosity, mean energy, flux, and energy spectrum. Flavor handles the four 
different neutrino flavors considered in CCSN simulations: nu_e, nu_e_bar, nu_x, and nu_x_bar. Nu_x refers to mu and tau
neutrinos collectively. These flavors are used to index different quantities used in other objects such as Source flux. 
Interactions provides the cross section and daughter particle energy spectrum for five types of neutrino interactions: 
inverse beta decay [cite], between anti-electron neutrinos and protons; elastic neutrino scattering on electrons [cite];
charged current interactions with O16 [cite]; neutral current interactions with O16 [cite]; and (v) O18 charged-current 
interaction [cite]. SimpleMixing provides a method for handling neutrino matter effects with the stellar medium 
[cite (Adiabatic MSW)] under the normal and inverted hierarchies. Each of these objects is called by the Source object 
which calculates the energy deposition in the IceCube detector by photons produced by the daughter particles of neutrino 
interactions. This is presented as the photonic energy per cubic meter of the detector using the following equation 
(to be added) where.. Is …

Detector contains information about the modules of the IceCube, and can be used to convert the photonic energy per 
volume to the number of photons observed the signal IceCube reports. These steps are handled by the handler object, 
based on the options provided in the configuration file loaded by config


## Conclusions
Asteria is a tool for handling a suite of CCSN explosion models, but it is not the only tool that exists to serve this 
purpose. More detailed GEANT simulations exist that are more robust and computationally expensive. The SNOwGLoBES tool 
also provides a fast simulation of detector responses for a variety of neutrino detectors, provided a description of the
effective volume of that detector is available. ASTERIA exists standalone aside from these tools as an intermediate 
simulation that is more customizable to handle the effective volume of Icecube, which is complex. Even though ASTERIA 
was developed for IceCube, the principle of the calculation may be applied to any other detector. Currently, ASTERIA 
has been used in several projects internal to IceCube including a supernova signal injection data challenge, and 
assessing the detectors’s sensitivity to Supernova using a bayesian blocks algorithm. Additionally, ASTERIA has been 
sed to aid development of community tools such as SNEWPy, a package in development by the SNEWS2 collaboration for 
handling various detector responses to CCSN. In the future, ASTERIA will be more closely integrated with tools such as 
SNEWPy.
