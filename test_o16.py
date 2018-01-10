from interactions import Oxygen16NC, Oxygen16CC, Oxygen18
from neutrino import Flavor

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

enu = np.linspace(0.1, 200., 401) * u.MeV
nc = Oxygen16NC()
cc = Oxygen16CC()
o18 = Oxygen18()

fig, ax = plt.subplots(1,2, figsize=(10,4), sharex=True)
ax0, ax1 = ax

xs = nc.cross_section(Flavor.nu_e, enu)
ax0.plot(enu, xs, label=r'$^{16}$O NC: $\nu_\mathrm{all}+^{16}$O$\rightarrow \nu_\mathrm{all} + X$')
xs = cc.cross_section(Flavor.nu_e, enu)
ax0.plot(enu, xs, label=r'$^{16}$O CC: $\nu_e+^{16}$O$\rightarrow e^- + X$')
xs = cc.cross_section(Flavor.nu_e_bar, enu)
ax0.plot(enu, xs, label=r'$^{16}$O CC: $\overline{\nu}_e+^{16}$O$\rightarrow e^+ + X$')
xs = o18.cross_section(Flavor.nu_e, enu)
ax0.plot(enu, xs, label=r'$^{18}$O CC: $\nu_e+^{18}$O$\rightarrow e^- + X$')

ax0.set(xlabel=r'$E_\nu$ [MeV]',
        ylabel=r'$\sigma(\nu_X + ^{16}\mathrm{O})$ [$10^{-42}$ cm$^2$]',
        yscale='log',
        ylim=[1e-46,1e-37])
ax0.legend()

lep = nc.mean_lepton_energy(Flavor.nu_e, enu)
ax1.plot(enu, lep)
lep = cc.mean_lepton_energy(Flavor.nu_e, enu)
ax1.plot(enu, lep)
lep = cc.mean_lepton_energy(Flavor.nu_e_bar, enu)
ax1.plot(enu, lep)
lep = o18.mean_lepton_energy(Flavor.nu_e, enu)
ax1.plot(enu, lep)
ax1.set(xlabel=r'$E_\nu$ [MeV]',
        ylabel=r'$\langle E_e\rangle$ [MeV]')

fig.tight_layout()
plt.show()

pass
