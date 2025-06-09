# Simulation management class adapted from https://github.com/desihub/specsim.
# The specsim project is distributed under a 3-clause BSD style license:
#
# Copyright (c) 2015, Specsim Developers <dkirkby@uci.edu>
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the Specsim Team nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
"""Manage simulation configuration data.

Configuration data are normally loaded from a yaml file. Some standard
configurations are included with this package and can be loaded by name,
for example:

    >>> test_config = load_config('test')

Otherwise any filename with extension .yaml can be loaded::

    my_config = load_config('path/my_config.yaml')

Configuration data is accessed using attribute notation to specify a
sequence of keys:

    >>> test_config.name
    'Test Simulation'
    >>> test_config.source.name
    'Sukhbold 9.6Msun progenitor with SFHo equation of state'
"""

import os
import re
import yaml

import astropy.units
import astropy.constants
import astropy.utils.data

from importlib.resources import readers, files

# Extract a number from a string with optional leading and
# trailing whitespace.
_float_pattern = re.compile(r'\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*')


def parse_quantity(quantity, dimensions=None):
    """Parse a string containing a numeric value with optional units.

    The result is a :class:`Quantity <astropy.units.Quantity` object even
    when units are not present. Optional units are interpreted by
    :class:`astropy.units.Unit`. Some valid examples::

        1.23
        1.23um
        123 um / arcsec
        1 electron/adu

    Used by :meth:`Configuration.get_constants`.

    Parameters
    ----------
    quantity : str or astropy.units.Quantity
        String to parse.  If a quantity is provided, it is checked against
        the expected dimensions and passed through.
    dimensions : str or astropy.units.Unit or None
        The units of the input quantity are expected to have the same
        dimensions as these units, if not None.  Raises a ValueError if
        the input quantity is not convertible to dimensions.

    Returns
    -------
    astropy.units.Quantity
        If dimensions is not None, the returned quantity will be converted
        to its units.

    Raises
    ------
    ValueError
        Unable to parse quantity.
    """
    if not isinstance(quantity, astropy.units.Quantity):
        # Look for a valid number starting the string.
        found_number = _float_pattern.match(quantity)
        if not found_number:
            raise ValueError('Unable to parse quantity.')
        value = float(found_number.group(1))
        unit = quantity[found_number.end():]
        quantity = astropy.units.Quantity(value, unit)
    if dimensions is not None:
        try:
            if not isinstance(dimensions, astropy.units.Unit):
                dimensions = astropy.units.Unit(dimensions)
            quantity = quantity.to(dimensions)
        except (ValueError, astropy.units.UnitConversionError):
            raise ValueError('Quantity "{0}" is not convertible to {1}.'
                             .format(quantity, dimensions))
    return quantity


class Node(object):
    """A single node of a configuration data structure.
    """
    def __init__(self, value, path=[]):
        self._assign('_value', value)
        self._assign('_path', path)

    def keys(self):
        return self._value.keys()

    def _assign(self, name, value):
        # Bypass our __setattr__
        super(Node, self).__setattr__(name, value)

    def __str__(self):
        return '.'.join(self._path)

    def __getattr__(self, name):
        # This method is only called when self.name fails.
        child_path = self._path[:]
        child_path.append(name)
        if name in self._value:
            child_value = self._value[name]
            if isinstance(child_value, dict):
                return Node(child_value, child_path)
            else:
                # Return the actual value for leaf nodes.
                return child_value
        else:
            raise AttributeError(
                'No such config node: {0}'.format('.'.join(child_path)))

    def __setattr__(self, name, value):
        # This method is always triggered by self.name = ...
        child_path = self._path[:]
        child_path.append(name)
        if name in self._value:
            child_value = self._value[name]
            if isinstance(child_value, dict):
                raise AttributeError(
                    'Cannot assign to non-leaf config node: {0}'
                    .format('.'.join(child_path)))
            else:
                self._value[name] = value
        else:
            raise AttributeError(
                'No such config node: {0}'.format('.'.join(child_path)))


class Configuration(Node):
    """Configuration parameters container and utilities.

    This class specifies the required top-level keys and delegates the
    interpretation and validation of their values to other functions.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters, normally obtained by parsing
        a YAML file with :func:`load`.

    Raises
    ------
    ValueError
        Missing required top-level configuration key.
    """

    def __init__(self, config):

        Node.__init__(self, config)
        self.update()

    def update(self):
        """Update this configuration.
        """
        # Use environment variables to interpolate {NAME} in the base path.
        base_path = self.base_path
        if base_path == '<PACKAGE_DATA>':
            self._assign('abs_base_path', files('asteria.data'))
        else:
            try:
                self._assign('abs_base_path', readers.MultiplexedPath(base_path.format(**os.environ)))
            except KeyError as e:
                raise ValueError('Environment variable not set: {0}.'.format(e))

    def __str__(self):

        settings = []

        def _traverse(cfg, indent=0, outstr=''):
            for key in cfg.keys():
                val = cfg.__getattr__(key)
                space = ''.join(['    ']*indent)
                if type(val) != Node:
                    settings.append('{}{} : {}'.format(space, key, val))
                else:
                    settings.append('{}{} :'.format(space, key))
                    _traverse(val, indent + 1)
        _traverse(self)

        return '\n'.join(settings)


def load_config(name, config_type=Configuration):
    """Load configuration data from a YAML file.

    Valid configuration files are YAML files containing no custom types, no
    sequences (lists), and with all mapping (dict) keys being valid python
    identifiers.

    Parameters
    ----------
    name : str
        Name of the configuration to load, which can either be a pre-defined
        name or else the name of a yaml file (with extension .yaml) to load.
        Pre-defined names are mapped to corresponding files in this package's
        data/config/ directory.

    Returns
    -------
    Configuration
        Initialized configuration object.

    Raises
    ------
    ValueError
        File name has wrong extension or does not exist.
    RuntimeError
        Configuration data failed a validation test.
    """
    base_name, extension = os.path.splitext(name)
    if extension not in ('', '.yaml'):
        raise ValueError('Config file must have .yaml extension.')
    if extension:
        file_name = name
    else:
        file_name = astropy.utils.data._find_pkg_data_path(
            'data/config/{0}.yaml'.format(name))
    if not os.path.isfile(file_name):
        raise ValueError('No such config file "{0}".'.format(file_name))

    # Validate that all mapping keys are valid python identifiers.
    valid_key = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*\Z')
    with open(file_name) as f:
        next_value_is_key = False
        for token in yaml.scan(f):
            # if isinstance(
            #     token,
            #     (yaml.BlockSequenceStartToken, yaml.FlowSequenceStartToken)):
            #     raise RuntimeError('Config sequences not implemented yet.')
            if next_value_is_key:
                if not isinstance(token, yaml.ScalarToken):
                    raise RuntimeError(
                        'Invalid config key type: {0}'.format(token))
                if not valid_key.match(token.value):
                    raise RuntimeError(
                        'Invalid config key name: {0}'.format(token.value))
            next_value_is_key = isinstance(token, yaml.KeyToken)

    with open(file_name) as f:
        conf = config_type(yaml.safe_load(f))
        # Enforces type of data member
        # TODO: Add warning/error checking for missing config or IO members
        #  This includes the simulations and IO fields. They must exist even if unspecified.
        #  - Perhaps read from default if they are missing?
        if isinstance(conf.simulation.interactions, str):
            temp = conf.simulation.interactions.replace(' ', '').split(',')
            if len(temp) > 1:
                conf.simulation.interactions = temp
            else:
                conf.simulation.interactions = temp[0]

        if isinstance(conf.simulation.flavors, str):
            temp = conf.simulation.flavors.replace(' ', '').split(',')
            if len(temp) > 1:
                conf.simulation.flavors = temp
            else:
                conf.simulation.flavors = temp[0]
        return conf
