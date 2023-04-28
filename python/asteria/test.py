import os
from asteria.simulation import Simulation
from astropy.table import Table
import astropy.units as u
import numpy as np
from scipy.interpolate import PchipInterpolator

"""
def get_effvol_table(effvol):
    Load effective volume table as astropy table
    Inputs:
    - effvol: ndarray, dict of ndarray
        Effective volume table from data files
    Outputs:
    - effvol_table: ndarray, dict of ndarry
        Effective volume table in astropy Table format
    
    if geomscope == "Gen2":
        keys = effvol.keys()
        effvol_table = {}
        for key in keys:
            evt = Table(effvol[key], 
                                    names=['z', 'effvol'], dtype=['f8', 'f8'], 
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

def effvol(doms):
    Interpolate table to to get effective volume
    Inputs:
    - doms: float, list, tuple, ndarray
        DOMs table to read of the depth for given subdetector and sensor
    Outputs:
    - vol: float, list, tuple, ndarray
        Effective volume at depth
    if geomscope == "Gen2":
        vol = {}
        for key in _effvol_table.keys():
            depth = doms['z'][doms["det_type"]==key.encode('UTF-8')] #det_type is UTF-8 (b-string)
            vol[key] = PchipInterpolator(_effvol_table[key]['z'], _effvol_table[key]['effvol'])(depth).reshape(-1, 1)
        if isinstance(depth, (list, tuple, np.ndarray)):
            return vol
        # Avoid 0-dimensional array
        return float(vol)

    else:
        depth = doms["z"][doms["det_type"]==b"IC86"] #det_type is UTF-8 (b-string)
        vol = PchipInterpolator(_effvol_table['z'], _effvol_table['effvol'])(depth).reshape(-1, 1)
        if isinstance(depth, (list, tuple, np.ndarray)):
            return vol
        # Avoid 0-dimensional array
        return float(vol)

geomscope = "Gen2"

doms = np.genfromtxt(os.path.join(os.environ['ASTERIA'], 'data/detector/geo_IC86+Gen2.txt'), delimiter = '\t', 
                             names=['str', 'i', 'x', 'y', 'z', 'det_type', 'om_type'], 
                             dtype='i8,i8,f8,f8,f8,S4,S2')

doms = doms[doms["z"] <= 1900]

effvol_table = {"IC86": os.path.join(os.environ['ASTERIA'], 'data/detector/effectivevolume_benedikt_AHA_normalDoms.txt'),
              "Gen2": os.path.join(os.environ['ASTERIA'], 'data/detector/mDOM_eff_vol.txt')}

effvol_dict = {"IC86": np.genfromtxt(effvol_table["IC86"]), "Gen2": np.genfromtxt(effvol_table["Gen2"])}

_effvol_table = get_effvol_table(effvol_dict)

_effvol = effvol(doms)

"""
os.environ['ASTERIA'] = '/home/jakob/software/ASTERIA/ASTERIA'
string = os.path.join(os.environ['ASTERIA'], 'data/detector/Gen2_geometry_preliminary.txt')
model = {'name': 'Walk_2019',
         'param':{
             'progenitor_mass': 40*u.Msun}
         }

#sim = Simulation(model=model,
#                 distance=10 * u.kpc, 
#                 Emin=0*u.MeV, Emax=100*u.MeV, dE=1*u.MeV,
#                 tmin=-0.5*u.s, tmax=1.5*u.s, dt=1*u.ms,
#                 geomscope = None)

sim = Simulation(config="/home/jakob/software/ASTERIA/ASTERIA/data/config/example.ini")
sim.run()

