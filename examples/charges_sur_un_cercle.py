"""N charges égales régulièrement réparties sur un cercle, pour N = 3, 5, 6, 7, 8, 15."""

import math
from pathlib import Path

import matplotlib.pyplot as plt

from fieldlines import Carte, Charge

cadre = (-3, 3, -3, 3)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, n in zip(axes.flat, (3, 5, 6, 7, 8, 15), strict=True):
    r0 = math.sqrt(2)
    charges = [
        Charge(1, (r0 * math.cos(2 * math.pi * k / n), r0 * math.sin(2 * math.pi * k / n)))
        for k in range(n)
    ]
    carte = Carte(charges, cadre)
    carte.lignes_autour(nombre=max(108 // n, 12), rayon=0.2)
    ax.set(xlim=cadre[:2], ylim=cadre[2:], aspect="equal", title=f"N = {n}")
    carte.trace_lignes(ax, color="gray", linewidth=0.5)
    carte.trace_fleches(ax, taille=6, color="gray")
    niveaux = carte.niveaux(10, echelle="log", vmin=1, vmax=10)
    carte.trace_equipotentielles(ax, niveaux, contraste=1.5, linewidths=0.5)
    carte.trace_sources(ax, taille=3)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
