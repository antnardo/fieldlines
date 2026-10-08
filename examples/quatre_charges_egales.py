"""Quatre charges égales aux sommets d'un carré, et les points où le champ s'annule.

Niveaux logarithmiques entre deux bornes choisies, plus trois niveaux qui passent par
les points d'arrêt ; les lignes s'arrêtent d'elles-mêmes au point d'arrêt central.

D'après Exercices/Exos/EM/EM-ES15/carte-champ.py.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.optimize import brentq

from fieldlines import Carte, Charge

charges = [Charge(0.5, (x, y)) for x in (-1, 1) for y in (-1, 1)]
cadre = (-3, 3, -3, 3)
carte = Carte(charges, cadre)
carte.lignes_autour(nombre=32, rayon=0.2)
niveaux = carte.niveaux(
    10, echelle="log", vmin=0.5, vmax=5, ajoute=[1 + 1 / math.sqrt(5), 1.466454, 1.43]
)


def champ_sur_ox(x: float) -> float:
    """Ex(x, 0) au signe près, la condition d'annulation du champ sur l'axe Ox."""
    return (1 - x) ** 2 * (1 + (1 + x) ** 2) ** 3 - (1 + x) ** 2 * (1 + (1 - x) ** 2) ** 3


racine = brentq(champ_sur_ox, 0.5, 1)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set(xlim=cadre[:2], ylim=cadre[2:], aspect="equal")
carte.trace_lignes(ax, color="gray", linewidth=0.5)
carte.trace_fleches(ax, taille=7, color="gray")
carte.trace_equipotentielles(ax, niveaux, contraste=1.4, cmap="cool_r", linewidths=0.5)
carte.trace_sources(ax, taille=3)
for x, y in [(racine, 0), (-racine, 0), (0, racine), (0, -racine)]:
    ax.plot(x, y, "o", color="blue", markersize=5, zorder=10)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
