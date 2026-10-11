"""Une charge dans un champ uniforme : les lignes de la charge et celles du champ.

`ChampUniforme(E=(1, 0))` s'ajoute à la charge comme une source. Les lignes qui
partent de la charge s'échappent vers +x ; celles du champ uniforme, venues de la
gauche, contournent la charge. La séparatrice passe par le point de champ nul, en
(−1, 0), où q/r² = E₀. Les équipotentielles de V = −x + 1/r sont tracées en couleur.
Les sources ont un axe de révolution commun (Ox, par la charge) : les départs sur le
bord gauche sont à flux égal dans l'espace.

Pour un champ magnétique, `ChampUniforme(B=(Bx, By))` se combine de même à un fil ou
à une spire.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from fieldlines import Carte, ChampUniforme, Charge

cadre = (-3, 3, -2.5, 2.5)
carte = Carte([ChampUniforme(E=(1, 0)), Charge(1, (0, 0))], cadre)
carte.lignes_autour(nombre=12, rayon=0.1, angle=math.pi / 12)
for signe in (1, -1):
    carte.lignes_le_long((-3, signe * 0.4), (-3, signe * 2.4), 5, repartition="flux_revolution")

fig, ax = plt.subplots(figsize=(6.5, 5.5))
carte.trace(ax, taille_sources=12)
carte.trace_equipotentielles(ax, carte.niveaux(13, vmin=-3, vmax=3), norm=Normalize(-3, 3))
ax.plot(-1, 0, "o", color="tab:blue", markersize=5, zorder=10)
ax.annotate("E = 0", (-1, 0), xytext=(4, -12), textcoords="offset points", fontsize=9)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
