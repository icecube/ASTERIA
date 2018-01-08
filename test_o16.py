from interactions import Oxygen16NC
from neutrino import Flavor

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

enu = np.linspace(0.1, 200., 401) * u.MeV
nc = Oxygen16NC()
xs = nc.cross_section(Flavor.nu_e, enu)
lep = nc.mean_lepton_energy(Flavor.nu_e, enu)

fig, ax = plt.subplots(1,2, figsize=(10,4), sharex=True)
ax0, ax1 = ax

ax0.plot(enu, xs, label=r'$^{16}$O NC')
ax0.set(xlabel=r'$E_\nu$ [MeV]',
        ylabel=r'$\sigma(\nu_X e^{-}\rightarrow\nu_X e^{-})$ [$10^{-42}$ cm$^2$]',
        yscale='log')
ax0.legend()

ax1.plot(enu, lep)
ax1.set(xlabel=r'$E_\nu$ [MeV]',
        ylabel=r'$\langle E_e\rangle$ [MeV]')

fig.tight_layout()
plt.show()

pass
