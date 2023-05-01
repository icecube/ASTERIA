import os
from asteria.simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import CubicSpline


os.environ['ASTERIA'] = '/home/jakob/software/ASTERIA/ASTERIA'
string = os.path.join(os.environ['ASTERIA'], 'data/detector/Gen2_geometry_preliminary.txt')

sim = Simulation(config="/home/jakob/software/ASTERIA/ASTERIA/data/config/example.ini")
sim.run()

dt = sim.detector.doms_table

dt.columns['z'][dt.columns['str']==1001]

#effvol = dt.columns['effvol'][dt.columns['str']==1001] #effective volume for str 1001
dom_pos = dt.columns['z'][dt.columns['str']==1001] #z position for str 1001
effvol_table = sim.detector._effvol_table['Gen2']['effvol']
z_table = sim.detector._effvol_table['Gen2']['z']

#effvol at DOM position
effvol_dom_pchip = PchipInterpolator(z_table,effvol_table)(dom_pos)
effvol_dom_unispline = InterpolatedUnivariateSpline(z_table,effvol_table,k=3,ext=0)(dom_pos)
effvol_dom_3spline = CubicSpline(z_table,effvol_table,bc_type='not-a-knot')(dom_pos)
effvol_dom_solution = InterpolatedUnivariateSpline(z_table,effvol_table,k=3,ext=3)(dom_pos)


#effvol continuous 
zz = np.arange(-800,600,0.1)
effvol_pchip = PchipInterpolator(z_table,effvol_table)(zz)
effvol_unispline = InterpolatedUnivariateSpline(z_table,effvol_table,k=3,ext=0)(zz)
effvol_3spline = CubicSpline(z_table,effvol_table,bc_type='not-a-knot')(zz)
effvol_solution = InterpolatedUnivariateSpline(z_table,effvol_table,k=3,ext=3)(zz)


#dom range outside of effvol range

hdom = dom_pos[dom_pos >= z_table.max()]
ldom = dom_pos[dom_pos <= z_table.min()]

fig, ax = plt.subplots(1,1)
ax.scatter(dom_pos, effvol_dom_pchip, marker = 'x', color = 'C0')
ax.scatter(dom_pos, effvol_dom_unispline, marker = 'o', color = 'C1')
ax.scatter(dom_pos, effvol_dom_3spline, marker = '^', color = 'C2')
ax.scatter(dom_pos, effvol_dom_solution, marker = 's', color = 'C3')
ax.plot(zz, effvol_pchip, color = 'C0', label = 'PchipInterpolator')
ax.plot(zz, effvol_unispline, color = 'C1', label = 'InterpolateUniversalSpline')
ax.plot(zz, effvol_3spline, color = 'C2', label = 'CubicSpline')
ax.plot(zz, effvol_solution, color = 'C3', label = 'InterpolateUniversalSpline + const')
ax.plot(z_table, effvol_table, c='k', label = 'eff. volume simulation')
ax.axvspan(hdom.min(),hdom.max(),color='red', alpha = 0.5)
ax.axvspan(ldom.min(),ldom.max(),color='red', alpha = 0.5)
ax.set_xlabel('Depth wrt AMANDA coordinate frame [m]')
ax.set_ylabel(r'Single photon effective volume [$m^3$]')
#ax.set_ylim((0,1))
ax.set_yscale('log')
ax.legend()
plt.savefig('extrapolation_issues.png')
plt.show()