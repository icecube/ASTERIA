import interactions
from neutrino import Flavor, neutrinos, antineutrinos

import astropy.units as u

import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle

sn_interactions = [interactions.InvBetaTab(),
                   interactions.ElectronScatter(),
                   interactions.Oxygen16CC(),
                   interactions.Oxygen16NC(),
                   interactions.Oxygen18()]

lines = ["-", "--", "-.", ":"]

flavors = [Flavor.nu_e, Flavor.nu_mu, Flavor.nu_e_bar, Flavor.nu_mu_bar]
#flavors = [Flavor.nu_e]

e_nu = np.linspace(0.1, 100., 501) * u.MeV

fig, ax = plt.subplots(1,1, figsize=(12,4))

for interaction in sn_interactions:
    color = None
    line = cycle(lines)
    for flavor in flavors:
        xs = interaction.cross_section(flavor, e_nu)
        if xs.value.any():
            label='{}: {}'.format(interaction.__class__.__name__, flavor)
            if color is None:
                p = ax.plot(e_nu, xs, next(line), label=label)
                color = p[0].get_color()
            else:
                ax.plot(e_nu, xs, next(line), label=label, color=color)

ax.grid()
ax.set(xlim=[0,100],
       xlabel=r'$E_\nu$ [MeV]',
       ylim=[1e-45, 1e-38],
       ylabel=r'$\sigma$ [cm$^2$]',
       yscale='log',
       title=r'Primary SN $\nu$ Interaction Cross Sections')

ax.legend(fontsize=8, bbox_to_anchor=(1.05,1))

fig.subplots_adjust(left=0.075, right=0.8)

plt.show()

pass
