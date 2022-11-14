# -*- coding: utf-8 -*-
""" Module to simulate IceCube detector """

from __future__ import print_function, division

from astropy import units as u
from astropy.table import Table

import numpy as np
from scipy.interpolate import PchipInterpolator

class Detector:

    """ Class for IceCube detector """

    def __init__(self, doms_table, effvol_table, max_height=1900,
                 f_dc_str=81, dc_rel_eff=1.35):

        # Read in and sort effective volume table
        effvol = np.genfromtxt(effvol_table)
        self._effvol_table = Table(effvol, names=['z', 'effvol'], dtype=['f8', 'f8'],
                                    meta={'Name': 'Effective_Volume'})
        self._effvol_table.sort('z')

        # Read in doms table from file
        doms = np.genfromtxt(doms_table)
        doms = doms[doms[:, -1] <= max_height]

        # Doms type: 'i3' (normal doms) and 'dc' (deep cores)
        doms_type = np.where(doms[:, 0] < f_dc_str, 'i3', 'dc').reshape(-1, 1)

        # Doms effective volume DeepCore and normal doms
        doms_effvol = self.effvol(doms[:, -1]).reshape(-1, 1)
        doms_effvol[doms_type == 'dc'] = doms_effvol[doms_type == 'dc']*dc_rel_eff

        # Create doms table
        self._doms_table = Table(np.hstack((doms, doms_effvol, doms_type)),
                                 names=['str', 'i', 'x', 'y', 'z', 'effvol', 'type'],
                                 dtype=['f4', 'f4', 'f8', 'f8', 'f8', 'f8', 'S2'],
                                 meta={'Name': 'DomsTable'})
        self.n_i3_doms = np.sum(self._doms_table['type'] == 'i3')
        self.n_dc_doms = np.sum(self._doms_table['type'] == 'dc')

        # Total effective volume:
        self._i3_effvol = np.sum(self._doms_table['effvol'][self._doms_table['type'] == 'i3'])
        self._dc_effvol = np.sum(self._doms_table['effvol'][self._doms_table['type'] == 'dc'])

        # DOM Artificial deadtime
        self.deadtime = 0.25e-3  # s
        # Relative efficiency of dc DOMs compared to i3 DOMs
        self.dc_rel_eff = 1.35

        # Background rate and sigma for i3 DOMs (hits / s)
        self._i3_dom_bg_mu = 284.9
        self._i3_dom_bg_sig = 26.2

        # Background rate and sigma for dc DOMs
        self._dc_dom_bg_mu = 358.9
        self._dc_dom_bg_sig = 36.0

    @property
    def i3_dom_bg_mu(self):
        return self._i3_dom_bg_mu

    @property
    def i3_dom_bg_sig(self):
        return self._i3_dom_bg_sig

    @property
    def dc_dom_bg_mu(self):
        return self._dc_dom_bg_mu

    @property
    def dc_dom_bg_sig(self):
        return self._dc_dom_bg_sig

    def set_i3_background(self, mu=284.9, sig=26.2):
        self._i3_dom_bg_mu = mu
        self._i3_dom_bg_sig = sig

    def set_dc_background(self, mu=358.9, sig=36.0):
        self._dc_dom_bg_mu = mu
        self._dc_dom_bg_sig = sig

    def i3_dom_bg(self, dt=0.5*u.s, size=1):
        return np.random.normal(loc=self.i3_dom_bg_mu * dt.to(u.s).value,
                                scale=self.i3_dom_bg_sig * np.sqrt(dt.to(u.s).value),
                                size=size)

    def dc_dom_bg(self, dt=0.5*u.s, size=1):
        return np.random.normal(loc=self.dc_dom_bg_mu * dt.to(u.s).value,
                                scale=self.dc_dom_bg_sig * np.sqrt(dt.to(u.s).value),
                                size=size)

    def i3_bg(self, dt=0.5*u.s, size=1):
        return np.random.normal(loc=self.i3_dom_bg_mu * dt.to(u.s).value * self.n_i3_doms,
                                scale=self.i3_dom_bg_sig * np.sqrt(dt.to(u.s).value * self.n_i3_doms),
                                size=size)

    def dc_bg(self, dt=0.5*u.s, size=1):
        return np.random.normal(loc=self.dc_dom_bg_mu * dt.to(u.s).value * self.n_dc_doms,
                                scale=self.dc_dom_bg_sig * np.sqrt(dt.to(u.s).value * self.n_dc_doms),
                                size=size)
    @property
    def i3_total_effvol(self):
        return self._i3_effvol

    @property
    def dc_total_effvol(self):
        return self._dc_effvol

    @property
    def i3_dom_effvol(self):
        return self.i3_total_effvol / self.n_i3_doms

    @property
    def dc_dom_effvol(self):
        return self.dc_total_effvol / self.n_dc_doms

    def effvol(self, depth):
        """ Interpolate table to to get effective volumne
        Inputs:
        - depth: float, list, tuple, ndarray
            Depth to evaluate effective volumne
        Outputs:
        - vol: float, list, tuple, ndarray
            Effective volume at depth """
        vol = PchipInterpolator(self._effvol_table['z'], self._effvol_table['effvol'])(depth)
        if isinstance(depth, (list, tuple, np.ndarray)):
            return vol
        # Avoid 0-dimensional array
        return float(vol)

    @property
    def effvol_table(self):
        """ Return a copy of the effective volume table """
        return self._effvol_table

    @property
    def doms_table(self, dom_type=None):
        """ Return a copy of the doms table given type
        Inputs:
        + type: str (default=None)
            If None, return full table. Else return the doms with the input type.
            Type must be "dc" or "i3". """
        if dom_type is None:
            return self._doms_table
        elif dom_type == 'dc' or dom_type == 'i3':
            return self._doms_table[self._doms_table['type'] == dom_type]
        else:
            raise ValueError('Type must be either "dc" or "i3".')


def initialize(config):
    """Initialize a Detector model from configuration parameters.

    Parameters
    ----------

    config : :class:`asteria.config.Configuration`
        Configuration parameters used to create a Detector.

    Returns
    -------
    Detector
        An initialized detector model.
    """

    geomfile = '/'.join([config.abs_base_path,
                         config.detector.geometry.table.path])
    effvfile = '/'.join([config.abs_base_path,
                         config.detector.effvol.table.path])

    return Detector(geomfile, effvfile)

"""
def main():
    Test main
    doms_table_fname = "../Icecube_geometry.20110102.complete.txt"
    effvol_table_fname = "../effvol/effectivevolume_benedikt_AHA_normalDoms.txt"
    icecube = Detector(doms_table_fname, effvol_table_fname)
    effvol = icecube.effvol_table()
    i3_doms = icecube.doms_table('i3')
    dc_doms = icecube.doms_table('dc')

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, figsize=(8, 5))

    # ax.scatter(i3_doms['x'], i3_doms['y'], color='k', marker='o', label='i3_string')
    # ax.scatter(dc_doms['x'], dc_doms['y'], color='r', marker='o', label='dc_string')
    # ax.set(xlabel='X-axis [m]', ylabel='Y-axis [m]')

    depth = np.arange(-500., 500., 5.)
    ax.plot(effvol['z'], effvol['effvol'], 'ko', label='Effective Vol')
    ax.plot(depth, icecube.effvol(depth), 'b--', label='Interpolated Effective Vol')
    ax.set(xlabel='Depth [m]', ylabel='Effective Vol')

    ax.legend()
    fig.tight_layout()

    plt.savefig('effvol.png', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()"""
