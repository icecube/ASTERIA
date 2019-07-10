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


src_path = os.path.realpath(__path__[0])
base_path = os.sep.join(src_path.split(os.sep)[:-2])

# Test for existence of scratch directory
scratch_path = os.path.join(base_path, 'scratch')
if not os.path.isdir(scratch_path):
    os.mkdir(scratch_path)

# Test for existence of processed directory
processed_path = os.path.join(base_path, 'data', 'processed')
if not os.path.isdir(processed_path):
    os.mkdir(processed_path)
