"""Champ magnétique d'un solénoïde de 20 spires, vu en coupe.

Les lignes partent du plan médian, à l'intérieur, où le champ est presque uniforme ;
elles sortent par les extrémités et se referment à l'extérieur.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from fieldlines import Carte, Solenoide

cadre = (-5, 5, -3, 3)
carte = Carte([Solenoide(1, (0, 0), rayon=1, longueur=4, spires=20)], cadre)
carte.lignes_le_long((0, -0.95), (0, 0.95), 16)

fig, ax = plt.subplots(figsize=(8, 5))
carte.trace(ax, taille_sources=7)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
