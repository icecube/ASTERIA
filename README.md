# ASTERIA: A Supernova TEst Routine for IceCube Analysis

[![tests](https://github.com/icecube/ASTERIA/actions/workflows/tests.yml/badge.svg)](https://github.com/icecube/ASTERIA/actions/workflows/tests.yml)

## Introduction

This is a fast supernova neutrino simulation designed for the IceCube Neutrino Observatory. The original version, called the Unified Supernova Simulation Routine (USSR), was written in C++ by Thomas Kowarik and
Gösta Kroll at Universität Mainz in 2011. This project began as a Python port of the original program.

The code uses estimates of the supernova neutrino luminosity from large-scale simulations of core-collapse supernovae to calculate photons in the IceCube detector.

## Access

ASTERIA can be cloned from this GitHub repository in the usual way by running
```
git clone https://github.com/icecube/ASTERIA.git
```

## Installation

ASTERIA can be installed by cloning the repository and running

```
cd /path/to/asteria
pip install . [--user]
```

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

[BSD 3-Clause License](LICENSE.rst)

ASTERIA is free software licensed under a 3-clause BSD-style license.
