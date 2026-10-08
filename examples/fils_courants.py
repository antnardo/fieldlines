"""Champ magnétique de fils rectilignes infinis, perpendiculaires à la figure.

À gauche, deux fils parcourus en sens opposés : les lignes sont des cercles qui
entourent chaque fil. À droite, deux nappes de six fils de sens opposés : le champ est
presque uniforme entre elles, comme dans un solénoïde coupé en deux.

Les lignes du champ magnétique n'aboutissent nulle part : on les fait partir d'un
segment qui les coupe toutes, à flux égal (densité ∝ intensité du champ), et chacune
s'arrête quand elle se referme.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fieldlines import Carte, FilCourant

cadre = (-4, 4, -3, 3)
paire = Carte([FilCourant(1, (-1, 0)), FilCourant(-1, (1, 0))], cadre)
paire.lignes_le_long((-0.8, 0), (0.8, 0), 14)

nappes = Carte(
    [FilCourant(sens, (x, y)) for x in np.linspace(-2, 2, 6) for sens, y in ((1, 1), (-1, -1))],
    cadre,
)
nappes.lignes_le_long((0, -0.9), (0, 0.9), 12)
nappes.lignes_le_long((-0.2, 1.1), (-0.2, 2.9), 4)
nappes.lignes_le_long((-0.2, -1.1), (-0.2, -2.9), 4)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, carte in zip(axes, (paire, nappes), strict=True):
    carte.trace(ax)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
