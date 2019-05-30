# ASTERIA: A Supernova TEst Routine for IceCube Analysis [![CircleCI](https://circleci.com/gh/IceCubeOpenSource/ASTERIA.svg?style=svg)](https://circleci.com/gh/IceCubeOpenSource/ASTERIA)

## Introduction

This is a fast supernova neutrino simulation designed for the IceCube Neutrino
Observatory. The original version was written in C++ by Thomas Kowarik and
Gösta Kroll at Universität Mainz in 2011, and has been updated by Lutz Köpke.
This project began as a Python port of the original program, the Unified 
Supernova Simulation Routine (USSR).

The code uses estimates of the supernova neutrino luminosity from large-scale
simulations of core-collapse supernovae to calculate photons in the IceCube
detector.

## Installation

ASTERIA can be installed by cloning the repository and running

```
python setup.py install
```

Alternatively, for rapid development the command

```
python setup.py develop
export ASTERIA=/path/to/asteria_folder
```

will install softlinks in your python path to the source in your git checkout.

## License

[BSD License](LICENSE.rst)

ASTERIA is free software licensed under a 3-clause BSD-style license.

Copyright (c) 2017-2019, the IceCube Collaboration.
