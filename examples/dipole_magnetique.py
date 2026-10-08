"""Un dipôle magnétique et une petite spire de même moment : loin, le même champ.

Les deux cartes ont les mêmes départs, régulièrement espacés sur l'équateur (à flux
égal, ils se serreraient tous près des sources, où le champ en 1/r³ est intense) :
loin des sources, les lignes se superposent ; près d'elles, celles de la spire (en
bleu) entourent ses deux passages dans la figure, celles du dipôle (en noir)
reviennent toutes en son centre.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt

from fieldlines import Carte, DipoleMagnetique, Spire

cadre = (-3, 3, -3, 3)
rayon = 0.3
dipole = Carte([DipoleMagnetique(math.pi * rayon**2, (0, 0), angle=math.pi / 2)], cadre)
spire = Carte([Spire(1, (0, 0), rayon, angle=math.pi / 2)], cadre)

fig, ax = plt.subplots(figsize=(6, 6))
for carte, couleur in ((spire, "tab:blue"), (dipole, "k")):
    for debut, fin in (((0.4, 0), (2.9, 0)), ((-0.4, 0), (-2.9, 0))):
        carte.lignes_le_long(debut, fin, 5, repartition="reguliere", style={"color": couleur})
    carte.trace(ax, taille_sources=8, linewidth=1)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
