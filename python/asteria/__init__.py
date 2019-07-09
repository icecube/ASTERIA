# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
=======
asteria
=======
Tools for IceCube supernova simulations, including neutrino luminosity and
energy spectra data files.
"""
from __future__ import absolute_import
from ._version import __version__
import os

base_path = os.environ['ASTERIA']
local_path = os.path.realpath(__file__)

# Test for environment variable ASTERIA
if base_path not in local_path:
    raise OSError('Environment variable ASTERIA not set properly')

# Test for existence of scratch directory
if not os.path.isdir(base_path + '/scratch'):
    os.mkdir(base_path + '/scratch')

# Test for existence of processed directory
if not os.path.isdir(base_path + '/data/processed'):
    os.mkdir(base_path + '/data/processed')