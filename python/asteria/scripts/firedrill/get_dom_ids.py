
# importing the requests library
import xml.etree.ElementTree as ET
from os import path
import numpy as np
from numpy.lib import recfunctions as rfn
from scipy.interpolate import PchipInterpolator
from asteria import config, detector
# TODO: Add DOM motherboard ID column to ASTERIA DOM Table
# TODO: Add other columns ichub? to ASTERIA DOM Table
tree = ET.parse('./data/detector.xml')
root = tree.getroot()
f_dc_str=81
dc_rel_eff = 1.35

def effvol(effvol_table, depth):
    """ Interpolate table to to get effective volumne
    Inputs:
    - depth: float, list, tuple, ndarray
        Depth to evaluate effective volumne
    Outputs:
    - vol: float, list, tuple, ndarray
        Effective volume at depth """
    vol = PchipInterpolator(effvol_table['z'], effvol_table['effvol'])(depth)
    if isinstance(depth, (list, tuple, np.ndarray)):
        return vol
    # Avoid 0-dimensional array
    return float(vol)


if not path.exists('./data/IceCube_DOM_IDs.txt'):
    with open('./data/IceCube_DOM_IDs.txt', 'w') as f:
        for child in root:
            if child.tag == 'dom':
                dom = child.attrib
                print('{0:<5d}{1:<5d}{2:s}'.format(int(dom['string']),int(dom['position']),dom['id']))
                f.write('{0:<5d}{1:<5d}{2:s}\n'.format(int(dom['string']),int(dom['position']),dom['id']))
else:
    print('Found DOM ID Table')

id_table_file = './data/IceCube_DOM_IDs.txt'
dtypes = [('str', 'f4'),
          ('i', 'f4'),
          ('mbid', 'S12')]
id_table = np.genfromtxt(id_table_file, dtype=dtypes)

dom_table_file = '../ASTERIA/data/detector/Icecube_geometry.20110102.complete.txt'
dtypes = [('str', 'f4'),
          ('i', 'f4'),
          ('x', 'f8'),
          ('y', 'f8'),
          ('z', 'f8')]
dom_table = np.genfromtxt(dom_table_file, dtype=dtypes)
dom_table = dom_table[dom_table['z'] <= 1900]

dom_type_table = np.where(dom_table['str'] < f_dc_str, 'i3', 'dc').reshape(-1, 1)
dom_type_table = np.array(dom_type_table.flatten(), dtype=[('type', 'S2')])

doms = rfn.append_fields(dom_table, names='mbid', data=id_table['mbid'])
doms = rfn.append_fields(doms, names='type', data=dom_type_table['type'])

effvol_table_file = '../ASTERIA/data/detector/effectivevolume_benedikt_AHA_normalDoms.txt'
dtypes = [('z', 'f8'),
          ('effvol', 'f8')]
effvol_table = np.genfromtxt(effvol_table_file, dtype=dtypes)
effvol_table = np.sort(effvol_table, order='z')
dom_effvol = effvol(effvol_table, doms['z']).reshape(-1, 1).flatten()
dom_effvol[doms['type'] == b'dc'] = dom_effvol[doms['type'] == b'dc']*dc_rel_eff

doms = rfn.append_fields(doms, names='effvol', data=dom_effvol)

with open('./data/full_dom_table.txt', 'w') as f:
    for row in doms:
        f.write('{0:<5d}{1:<5d}{2:<12.6f}{3:<12.6f}{4:<12.6f}{5:<15s}{6:<5s}{7:12.6f}\n'.format(
            int(row['str']), int(row['i']),
            float(row['x']), float(row['y']), float(row['z']),
            str(row['mbid'])[2:-1], str(row['type'])[2:-1],
            float(row['effvol'])
        ))

dtypes = [('str', 'f4'),
          ('i', 'f4'),
          ('x', 'f8'),
          ('y', 'f8'),
          ('z', 'f8'),
          ('mbid', 'S12'),
          ('type', 'S2'),
          ('effvol', 'f8')]

test_table = np.genfromtxt('./data/full_dom_table.txt', dtype=dtypes)
print(test_table)
# np.savetxt('./data/full_dom_table.txt', doms, fmt='%.18d %.18d %.18f %.18f %.18f %.18s %.18s %.18f')
#
# np.save('./data/full_dom_table', doms.compressed())
# test_table = np.load('./data/full_dom_table')

# # test_table = np.genfromtxt('./data/full_dom_table.txt', dtype=dtypes)
# # dom_type = [str(typestring)[4:-2] for typestring in test_table['type']]
# print(test_table)
