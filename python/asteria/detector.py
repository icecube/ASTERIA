# -*- coding: utf-8 -*-
""" Module to simulate IceCube detector """

from __future__ import print_function, division

from .config import parse_quantity

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
                                    meta={'Name': 'Effetive_Volume'})
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

        # Total effective volumne:
        self._i3_effvol = np.sum(self._doms_table['effvol'][self._doms_table['type'] == 'i3'])
        self._dc_effvol = np.sum(self._doms_table['effvol'][self._doms_table['type'] == 'dc'])

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

    def effvol_table(self):
        """ Return a copy of the effective volumne table """
        return self._effvol_table

    def doms_table(self, d_type=None):
        """ Return a copy of the doms table given type
        Inputs:
        + type: str (default=None)
            If None, return full table. Else return the doms with the input type.
            Type must be "dc" or "i3". """
        if d_type is None:
            return self._doms_table
        elif d_type == 'dc' or d_type == 'i3':
            return self._doms_table[self._doms_table['type'] == d_type]
        else:
            raise ValueError('Type must be either "dc" or "i3".')

    def detector_signals(self, E_photon, i3_dt_frac, dc_dt_frac):
        """ Compute the signals at the detector
        Inputs:
        + E_photon: float, list, tuple, ndarray
            Deposited photonic energy
        + i3_dt_frac: float
            Detector deadtime fraction for normal doms
        + dc_dt_frac:
            Detector deadtime fraction for DeepCore doms
        Outputs:
        + detector_signals: float, list, tuple, ndarray
            Signal at the detector """
        return E_photon*(self._i3_effvol*i3_dt_frac + self._dc_effvol*dc_dt_frac)

    def detector_hits(self, E_photon, i3_dt_frac, dc_dt_frac):
        """ Compute the total hits at the detector
        Inputs:
        + E_photon: float, list, tuple, ndarray
            Deposited photonic energy
        + i3_dt_frac: float
            Detector deadtime fraction for normal doms
        + dc_dt_frac:
            Detector deadtime fraction for DeepCore doms
        Outputs:
        + detector_hits: float, list, tuple, ndarray
            Total hits at the detector """
        signals = self.detector_signals(E_photon, i3_dt_frac, dc_dt_frac)
        # Possion-flutuated
        return np.random.poisson(signals)


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
