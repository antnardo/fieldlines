"""Bobines de Helmholtz : deux spires coaxiales distantes de leur rayon.

Le champ est presque uniforme entre les deux spires, d'où les lignes presque
parallèles au centre. Les sources ont le même axe : la fonction de flux existe, et on
répartit les départs à flux égal dans l'espace (`flux_revolution`).
"""

from pathlib import Path

import matplotlib.pyplot as plt

from fieldlines import Carte, Spire

cadre = (-3, 3, -2.5, 2.5)
carte = Carte([Spire(1, (-0.5, 0), 1), Spire(1, (0.5, 0), 1)], cadre)
carte.lignes_le_long((0, -0.9), (0, 0.9), 16, repartition="flux_revolution")

fig, ax = plt.subplots(figsize=(6, 5))
carte.trace(ax)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
