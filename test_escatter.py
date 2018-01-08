from interactions import ElectronScatter
from neutrino import Flavor

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

enu = np.linspace(0.1, 200., 401) * u.MeV
els = ElectronScatter()
xs_nue = els.cross_section(Flavor.nu_e, enu)

fig, ax = plt.subplots(2,1, figsize=(6,6), sharex=True,
                       gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
ax0, ax1 = ax
ax0.plot(enu, xs_nue, label=Flavor.nu_e)

for flavor in (Flavor.nu_e_bar, Flavor.nu_mu, Flavor.nu_mu_bar):
    xs = els.cross_section(flavor, enu)
    p = ax0.plot(enu, xs, label=flavor)
    ax1.plot(enu, xs/xs_nue, color=p[-1].get_color())

ax0.set(xlabel='$E$ [MeV]',
        ylabel=r'$\sigma(\nu_X e^{-}\rightarrow\nu_X e^{-})$ [$10^{-42}$ cm$^2$]',
        yscale='log')
ax1.set(xlabel='$E$ [MeV]',
        ylabel=r'$\sigma(\nu_X)/\sigma(\nu_e)$',
        ylim=(0,1.05))
ax0.legend()

fig.suptitle('Electron Scattering: J. Phys. G 29:2969, 2003')
fig.subplots_adjust(left=0.125, right=0.925, top=0.9, bottom=0.1)

plt.show()

pass
