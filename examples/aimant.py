"""Un aimant droit : le champ d'un cylindre uniformément aimanté, et ses pôles.

L'aimant équivaut à un solénoïde de même forme : les lignes le traversent du pôle sud
au pôle nord, sortent par la face nord et se referment dehors. Elles partent du plan
médian, à flux égal dans l'espace (`flux_revolution`) : les tubes de flux sont les
mêmes dedans et dehors. À droite, un aimant de même moment vu de loin est un dipôle ;
en tirets, les lignes du dipôle ponctuel, confondues avec celles de l'aimant au-delà
de quelques fois sa taille.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fieldlines import Aimant, Carte, DipoleMagnetique

aimant = Aimant(aimantation=1, centre=(0, 0), rayon=0.4, hauteur=1.2, angle=math.pi / 2)
moment = aimant.aimantation * math.pi * aimant.rayon**2 * aimant.hauteur

pres = Carte([aimant], cadre=(-2, 2, -2, 2))
pres.lignes_le_long((-0.35, 0), (0.35, 0), 10, repartition="flux_revolution")

loin = Carte([aimant], cadre=(-6, 6, -6, 6))
loin.lignes_le_long((-0.35, 0), (0.35, 0), 6, repartition="flux_revolution")
# Les lignes du dipôle partent de là où celles de l'aimant recoupent le plan équatorial,
# hors de l'aimant : près du dipôle, elles seraient de petites boucles sans rapport.
dipole = Carte([DipoleMagnetique(moment, (0, 0), angle=math.pi / 2)], cadre=(-6, 6, -6, 6))
for ligne in loin.lignes().values():
    x, y = ligne.points.T
    dehors = np.flatnonzero((y[:-1] * y[1:] < 0) & (np.abs(x[:-1]) > aimant.rayon))
    if dehors.size:
        k = dehors[0]
        croisement = x[k] - y[k] * (x[k + 1] - x[k]) / (y[k + 1] - y[k])
        dipole.ajoute_ligne((croisement, 0), fleches=[], style={"color": "tab:red", "ls": "--"})

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
pres.trace(axes[0], taille_sources=12)
axes[0].set_title("aimant droit, lignes à flux égal")
loin.trace(axes[1], taille_sources=8)
dipole.trace_lignes(axes[1], linewidth=0.8)
axes[1].set_title("vu de loin : un dipôle (tirets)")

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
