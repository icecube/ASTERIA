# ASTERIA: A Supernova TEst Routine for IceCube Analysis 
## Introduction

This is a fast supernova neutrino simulation designed for the IceCube Neutrino
Observatory. The original version was written in C++ by Thomas Kowarik and
Gösta Kroll at Universität Mainz in 2011, and has been updated by Lutz Köpke.
This project began as a Python port of the original program, the Unified 
Supernova Simulation Routine (USSR).

The code uses estimates of the supernova neutrino luminosity from large-scale
simulations of core-collapse supernovae to calculate photons in the IceCube
detector.

## Access

ASTERIA can be cloned from this GitHub repository in the usual way. It also
pulls in a private submodule containing core-collapse supernova flux
calculations. To pull the submodule after cloning the repository, run

```
git submodule update --init --recursive
```

This only needs to be done the first time you clone ASTERIA. To update the
submodule in your working copy, run the command

```
git submodule update --recursive --remote
```

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

## Ignored Files

ASTERIA is configured in such a way that the following directories will be
automatically generated if they are missing, but their content will be ignored by git.

```
/path/to/asteria_folder/scratch
/path/to/asteria_folder/data/
```

`/scratch` is intended to be user work space.
`/data/processed` contains processed simulation files, which are potentially large.

## License

[BSD License](LICENSE.rst)

ASTERIA is free software licensed under a 3-clause BSD-style license.
