import interactions
from neutrino import Flavor, neutrinos, antineutrinos

import astropy.units as u

import numpy as np
import matplotlib.pyplot as plt

sn_interactions = [#interactions.InvBetaTab(),
                   #interactions.ElectronScatter(),
                   interactions.Oxygen16CC(),
                   #interactions.Oxygen16NC(),
                   #interactions.Oxygen18()]
                    ]

flavors = neutrinos + antineutrinos
#flavors = [Flavor.nu_e]

e_nu = np.linspace(0.1, 100., 201) * u.MeV

fig, ax = plt.subplots(1,1, figsize=(12,4))

for interaction in sn_interactions:
    for flavor in flavors:
        xs = interaction.cross_section(flavor, e_nu)
        if xs.value.any():
            ax.plot(e_nu, xs, label=flavor)

ax.grid()
ax.set(xlim=[0,100],
       ylim=[1e-45, 1e-38],
       yscale='log')
ax.legend()

fig.tight_layout()

plt.show()

pass
