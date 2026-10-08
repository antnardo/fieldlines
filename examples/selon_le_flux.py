"""Placer les lignes selon le flux : nombre ∝ charge, et pas de ligne tracée deux fois.

À gauche, `lignes_autour` : autant de lignes autour de chaque charge, et chaque ligne
de +3 à −1 tracée deux fois. À droite, `lignes_selon_flux` : trois fois plus de lignes
autour de +3, et −1 ne reçoit que les lignes qui lui manquent.
En bas, des fils chargés (problème plan) : la densité des lignes suit exactement
l'intensité du champ.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from fieldlines import Carte, Charge, FilCharge

cadre = (-4, 4, -3, 3)
charges = [Charge(3, (-1, 0)), Charge(-1, (1, 0))]
fils = [FilCharge(1, (-1.5, -0.5)), FilCharge(-1, (1.5, -0.5)), FilCharge(1, (0, 1.2))]

autour = Carte(charges, cadre)
autour.lignes_autour(nombre=12)
flux = Carte(charges, cadre)
flux.lignes_selon_flux(nombre=24)
fils_autour = Carte(fils, cadre)
fils_autour.lignes_autour(nombre=12)
fils_flux = Carte(fils, cadre)
fils_flux.lignes_selon_flux(nombre=12)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
titres = ("lignes_autour", "lignes_selon_flux") * 2
for ax, carte, titre in zip(axes.flat, (autour, flux, fils_autour, fils_flux), titres, strict=True):
    carte.trace(ax)
    ax.set_title(f"{titre} : {len(carte)} lignes")

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
