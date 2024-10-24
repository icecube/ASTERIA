#!/usr/bin/env python
#
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
#
# Standard imports
#
from glob import glob
import os
import re
import sys
#
from distutils.command.sdist import sdist as DistutilsSdist
from setuptools import setup, find_packages
#
from python.asteria._git import get_version, SetVersion
#
# Begin setup
#
setup_keywords = dict()
#
setup_keywords['name'] = 'asteria'
setup_keywords['description'] = 'IceCube supernova simulation package'
setup_keywords['author'] = 'IceCube Collaboration'
setup_keywords['author_email'] = 'sn-wg@icecube.wisc.edu'
setup_keywords['license'] = 'BSD'
setup_keywords['url'] = 'https://github.com/IceCubeOpenSource/ASTERIA'
setup_keywords['version'] = get_version()
#
# Use README.md as a long_description.
#
setup_keywords['long_description'] = ''
if os.path.exists('README.md'):
    with open('README.md') as readme:
        setup_keywords['long_description'] = readme.read()
#
# Set other keywords for the setup function.
#
if os.path.isdir('bin'):
    # Treat everything in bin as a script to be installed.
    setup_keywords['scripts'] = \
    [fname for fname in glob(os.path.join('bin', '*'))]
setup_keywords['provides'] = [setup_keywords['name']]
setup_keywords['requires'] = ['Python (>2.7.0)']  # TODO: Require Python >=3.7, cleanup following lines too
setup_keywords['zip_safe'] = False
setup_keywords['use_2to3'] = False
setup_keywords['packages'] = find_packages('python')
setup_keywords['package_dir'] = {'': 'python'}
setup_keywords['cmdclass'] = {'version': SetVersion, 'sdist': DistutilsSdist}
# setup_keywords['test_suite']='nose2.collector.collector'
# setup_keywords['tests_require']=['nose2', 'nose2[coverage_plugin]>=0.6.5']

requires = []
with open('requirements.txt', 'r') as f:
    for line in f:
        if line.strip():
            requires.append(line.strip())
setup_keywords['install_requires'] = requires
#
# Internal data directories.
#
setup_keywords['data_files'] = [('asteria/data/config', glob('data/config/*')),
                                ('asteria/data/spectra', glob('data/spectra/*'))]
#
# Run setup command.
#
setup(**setup_keywords)
