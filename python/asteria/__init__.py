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


def set_rcparams(rcParams_file=None, *, verbose=False):
    """Updates Matplotlib rcParams with format specifiers from file

    Parameters
    ----------
    rcParams_file : str
        Path to custom rcParams file, Default value is
    verbose : bool
        If true, the rcParams

    Returns
    -------

    """
    import matplotlib as mpl
    if not rcParams_file:  # Load ASTERIA default
        import os
        rcParams_file = os.path.join(os.environ['ASTERIA'], 'asteria.rcParam')
    rcParams = mpl.rc_params_from_file(rcParams_file, fail_on_error=True)
    mpl.rcParams.update(rcParams)
    return rcParams if verbose else None
