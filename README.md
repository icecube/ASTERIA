# Unified Supernova Simulation Routine (USSR)

## Introduction

This is a fast supernova neutrino simulation designed for the IceCube Neutrino
Observatory. The original version was written in C++ by Thomas Kowarik and
Gösta Kroll at Universität Mainz in 2011, and has been updated by Lutz Köpke.
This project began as a Python port of the original USSR.

The code uses estimates of the supernova neutrino luminosity from large-scale
simulations of core-collapse supernovae to calculate photons in the IceCube
detector.

## Installation

USSR can be installed by cloning the repository and running

```
python setup.py install
```

Alternatively, for rapid development the command

```
python setup.py develop
```

will install softlinks in your python path to the source in your git checkout.

## License

[BSD License](LICENSE)

USSR is free software licensed under a 3-clause BSD-style license.

Copyright (c) 2018, the IceCube Collaboration.
