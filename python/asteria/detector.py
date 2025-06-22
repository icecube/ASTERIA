# -*- coding: utf-8 -*-
""" Module to simulate IceCube detector """

from __future__ import print_function, division

from astropy import units as u
from astropy.table import Table

import numpy as np
import numpy.lib.recfunctions as rfn
from scipy.interpolate import PchipInterpolator, InterpolatedUnivariateSpline

class Detector:

    """ Class for IceCube detector """

    def __init__(self, doms_table, effvol_table, detector_scope, max_height=1900, 
                 dc_rel_eff=1.35, ws_rel_eff=0.405):

        self._detector_scope = detector_scope

        # Read in doms table from file
        doms = Table.read(doms_table)
        doms = doms[doms['z'] <= max_height]
        
        # For Gen2
        if self._detector_scope == 'Gen2':
            # read in effective volume table
            effvol = {'IC86': np.genfromtxt(effvol_table['IC86']), 'Gen2': np.genfromtxt(effvol_table['Gen2'])}

        # For standard IceCube (IC86) and if detector_scope not defined
        else:
            # downsample geometry
            doms = doms[doms['det_type'] == 'IC86']
            # read in effective volume table
            effvol = np.genfromtxt(effvol_table['IC86'])
        self._effvol_table = self.get_effvol_table(effvol)

        # Doms effective volume DeepCore and normal doms
        doms_effvol = self.effvol(doms)
        doms_effvol[doms['om_type'] == 'dc'] = doms_effvol[doms['om_type'] == 'dc']*dc_rel_eff

        # Create DOMs table.
        doms['effvol'] = doms_effvol.flatten()
        self._doms_table = doms

#        #self._doms_table = Table(np.hstack((doms, doms_effvol)),
#        self._doms_table = Table(rfn.append_fields(doms, names='effvol', data=doms_effvol.flatten(), usemask=False),
#                                 names=['str', 'i', 'x', 'y', 'z', 'det_type', 'om_type', 'effvol'],
#                                 dtype=['i4', 'i4', 'f8', 'f8', 'f8', 'U4', 'U2', 'float64'],
#                                 meta={'Name': 'DomsTable'})
        self.n_i3_doms = np.sum(self._doms_table['om_type'] == 'i3')
        self.n_dc_doms = np.sum(self._doms_table['om_type'] == 'dc')
        self.n_md = np.sum(self._doms_table['om_type'] == 'md')
        self.n_ws = np.sum(self._doms_table['om_type'] == 'md') * 2 # There are twice as many wavelength shifters (ws) as mDOMs
        
        # Total effective volume:
        self._i3_effvol = np.sum(self._doms_table['effvol'][self._doms_table['om_type'] == 'i3'])
        self._dc_effvol = np.sum(self._doms_table['effvol'][self._doms_table['om_type'] == 'dc'])
        self._md_effvol = np.sum(self._doms_table['effvol'][self._doms_table['om_type'] == 'md'])
        self._ws_effvol = np.sum(self._doms_table['effvol'][self._doms_table['om_type'] == 'md']) * ws_rel_eff * 2 # For each mDOM there are 2 WLS tubes with a 40.5% efficiency

        # DOM Artificial deadtime
        self.deadtime = 0.25e-3  # s
        # Relative efficiency of dc DOMs compared to i3 DOMs
        self.dc_rel_eff = dc_rel_eff
        # Relative efficiency of dc WLS tube compared to  mDOMs
        self.ws_rel_eff = ws_rel_eff

        # TODO Jakob: change "_dom_" name to "_mod_" name because for other sensors are not DOMs
        # This is especially needed when talking about the single module noise rate.
        # "md_dom_bg" does not make sense.
        # Number of PMTs per i3 DOM
        self._i3_dom_num_pmt = 1
        # Number of PMTs per dc DOM
        self._dc_dom_num_pmt = 1
        # Number of PMTs per mDOM
        self._md_num_pmt = 24
      
        # Background rate and sigma for i3 DOMs (hits / s)
        # With a dead-time of 250µs
        self._i3_dom_bg_mu = 284.9 * self._i3_dom_num_pmt
        self._i3_dom_bg_sig = 26.2 * np.sqrt(self._i3_dom_num_pmt)

        # Background rate and sigma for dc DOMs
        # With a dead-time of 250µs
        self._dc_dom_bg_mu = 358.9 * self._dc_dom_num_pmt
        self._dc_dom_bg_sig = 36.0 * np.sqrt(self._dc_dom_num_pmt)

        # Scale mean (std) of single module PMT noise by the number of PMTs (sqrt(number of PMTs))
        # according to the rules of summing N independent random variables.
        # For every PMT we assume the same mean and std, therefore the addition becomes a factor N
        # for the mean and sqrt(N) for the std
    
        # Background rate and sigma for mDOMs
        # With a dead-time of 250µs
        self._md_bg_mu = 98.5 * self._md_num_pmt
        self._md_bg_sig = 13.7 * np.sqrt(self._md_num_pmt)

        # Background rate and sigma for WLS tube
        # Does not consider 250µs deadtime for now, each tube has 204.3 Hz noise from radioactive decays
        # TODO Jakob: Simulate WLS noise and run deadtime algorithm
        self._ws_bg_mu =  204.3
        self._ws_bg_sig = np.sqrt(204.3)

    @property
    def detector_scope(self):
        return self._detector_scope

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
    
    @property
    def md_bg_mu(self):
        return self._md_bg_mu

    @property
    def md_bg_sig(self):
        return self._md_bg_sig
    
    @property
    def ws_bg_mu(self):
        return self._ws_bg_mu

    @property
    def ws_bg_sig(self):
        return self._ws_bg_sig

    def set_i3_background(self, mu=284.9, sig=26.2):
        self._i3_dom_bg_mu = mu
        self._i3_dom_bg_sig = sig

    def set_dc_background(self, mu=358.9, sig=36.0):
        self._dc_dom_bg_mu = mu
        self._dc_dom_bg_sig = sig

    def set_md_background(self, mu=98.7, sig=13.7):
        self._md_bg_mu = mu
        self._md_bg_sig = sig

    def set_ws_background(self, mu=204.3, sig=np.sqrt(204.3)):
        self._ws_bg_mu = mu
        self._ws_bg_sig = sig

    def i3_dom_bg(self, dt=0.5*u.s, size=1):
        return np.random.normal(loc=self.i3_dom_bg_mu * dt.to(u.s).value,
                                scale=self.i3_dom_bg_sig * np.sqrt(dt.to(u.s).value),
                                size=size)

    def dc_dom_bg(self, dt=0.5*u.s, size=1):
        return np.random.normal(loc=self.dc_dom_bg_mu * dt.to(u.s).value,
                                scale=self.dc_dom_bg_sig * np.sqrt(dt.to(u.s).value),
                                size=size)
    
    def md_dom_bg(self, dt=0.5*u.s, size=1):
        return np.random.normal(loc=self.md_bg_mu * dt.to(u.s).value,
                                scale=self.md_bg_sig * np.sqrt(dt.to(u.s).value),
                                size=size)
    
    def ws_dom_bg(self, dt=0.5*u.s, size=1):
        return np.random.normal(loc=self.ws_bg_mu * dt.to(u.s).value,
                                scale=self.ws_bg_sig * np.sqrt(dt.to(u.s).value),
                                size=size)

    def i3_bg(self, dt=0.5*u.s, size=1):
        return np.random.normal(loc=self.i3_dom_bg_mu * dt.to(u.s).value * self.n_i3_doms,
                                scale=self.i3_dom_bg_sig * np.sqrt(dt.to(u.s).value * self.n_i3_doms),
                                size=size)

    def dc_bg(self, dt=0.5*u.s, size=1):
        return np.random.normal(loc=self.dc_dom_bg_mu * dt.to(u.s).value * self.n_dc_doms,
                                scale=self.dc_dom_bg_sig * np.sqrt(dt.to(u.s).value * self.n_dc_doms),
                                size=size)
    
    def md_bg(self, dt=0.5*u.s, size=1):
        return np.random.normal(loc=self.md_bg_mu * dt.to(u.s).value * self.n_md,
                                scale=self.md_bg_sig * np.sqrt(dt.to(u.s).value * self.n_md),
                                size=size)
    
    def ws_bg(self, dt=0.5*u.s, size=1):
        return np.random.normal(loc=self.ws_bg_mu * dt.to(u.s).value * self.n_ws,
                                scale=self.ws_bg_sig * np.sqrt(dt.to(u.s).value * self.n_ws),
                                size=size)
    @property
    def i3_total_effvol(self):
        return self._i3_effvol

    @property
    def dc_total_effvol(self):
        return self._dc_effvol
    
    @property
    def md_total_effvol(self):
        return self._md_effvol
    
    @property
    def ws_total_effvol(self):
        return self._ws_effvol

    @property
    def i3_dom_effvol(self):
        return self.i3_total_effvol / self.n_i3_doms

    @property
    def dc_dom_effvol(self):
        return self.dc_total_effvol / self.n_dc_doms
    
    @property
    def md_dom_effvol(self):
        return self.md_total_effvol / self.n_md
    
    @property
    def ws_dom_effvol(self):
        return self.ws_total_effvol / self.n_ws
    
    def get_effvol_table(self, effvol):
        """ Load effective volume table as astropy table
        Inputs:
        - effvol: ndarray, dict of ndarray
            Effective volume table from data files
        Outputs:
        - effvol_table: ndarray, dict of ndarry
            Effective volume table in astropy Table format
        """
        if self._detector_scope == 'Gen2':
            keys = effvol.keys()
            effvol_table = {}
            for key in keys:
                evt = Table(effvol[key], names=['z', 'effvol'], dtype=['f8', 'f8'], 
                            meta={'Name': 'Effective_Volume'})
                evt.sort('z')
                effvol_table[key] = evt
            return effvol_table
        else:
            evt = Table(effvol, names=['z', 'effvol'], dtype=['f8', 'f8'],
                        meta={'Name': 'Effective_Volume'})
            evt.sort('z')
            effvol_table = evt
            return effvol_table

    def effvol(self, doms):
        """ Interpolate table to to get effective volume
        Inputs:
        - doms: float, list, tuple, ndarray
            DOMs table to read of the depth for given subdetector and sensor
        Outputs:
        - vol: float, list, tuple, ndarray
            Effective volume at depth """
        # TODO Jakob: For IceCube Upgrade, where different sensors come interchangeably, this has to be adapted because
        # the current version relies on the fact that IceCube is completly made of DOMs (up to scaling) and Gen2 is completely made
        # of mDOMs. To obtain the individual effective volume we currently take a mask using the dom table and filtering for a sensor.
        # Idealy, we would replace this array based implementation with a table or dictionary. I.e. we construct a table like the doms
        # table and then instead of lopping over the detector_scope we loop over the sensor type and interpolate at the position of the
        # sensor.
        # TODO Jakob: Consider changing interpolator and set values exceeding range to edge values
        if self._detector_scope == 'Gen2':
            vol = np.array([])
            for key in self._effvol_table.keys():
                depth = doms['z'][doms['det_type']==key] #det_type is UTF-8 (b-string)
                vol_sens = InterpolatedUnivariateSpline(self._effvol_table[key]['z'], self._effvol_table[key]['effvol'],
                                                        k=3,ext=3)(depth).reshape(-1, 1)
                vol = np.append(vol, vol_sens)
            if isinstance(depth, (list, tuple, np.ndarray)):
                return vol
            # Avoid 0-dimensional array
            return float(vol)

        else:
            depth = doms['z'][doms['det_type']=='IC86'] #det_type is UTF-8 (b-string)
            vol = PchipInterpolator(self._effvol_table['z'], self._effvol_table['effvol'])(depth).reshape(-1, 1)
            if isinstance(depth, (list, tuple, np.ndarray)):
                return vol
            # Avoid 0-dimensional array
            return float(vol)
        

    @property
    def effvol_table(self):
        """ Return a copy of the effective volume table """
        return self._effvol_table

    def get_doms_table(self, det_type = None, om_type=None):
        """ Return a copy of the doms table given a det_type or om_type
        Inputs:
        - det_type: str (default=None)
            If None, return full table. Else return the doms with the input det_type.
            Type must be "IC86" or "Gen2".
        - om_type: str (default=None)
            If None, return full table. Else return the doms with the input om_type.
            Type must be "dc", "i3" or "md". """
        
        if det_type is None and om_type is None:
            return self._doms_table
        elif det_type is None and om_type is not None:
            if om_type == 'dc' or om_type == 'i3' or om_type == 'md':
                return self._doms_table[self._doms_table['om_type'] == om_type]
            else:
                raise ValueError('om_type must be either "dc", "i3" or "md".')
        elif det_type is not None:
            if det_type == 'IC86' or det_type == 'Gen2':
                return self._doms_table[self._doms_table['det_type'] == det_type]
            else:
                raise ValueError('det_type must be either "IC86" or "Gen2"')           


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
    detector_scope = config.detector.detector_scope
    
    geomfile = '/'.join([config.abs_base_path,
                         config.detector.geometry.table.path])
    effvfile = '/'.join([config.abs_base_path,
                         config.detector.effvol.table.path])

    return Detector(geomfile, effvfile, detector_scope)

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
