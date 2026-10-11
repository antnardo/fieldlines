"""Six fils chargés alternativement ±, perpendiculaires à la figure (problème plan).

Un fil de charge linéique λ crée E = 2λ/r : avec λ = ±1/2, V = ∓ln r. Lignes et
équipotentielles en traits fins, niveaux linéaires plus trois niveaux ajoutés.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt

from fieldlines import Carte, FilCharge

fils = [
    FilCharge((-1) ** i / 2, (math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)))
    for i in range(6)
]
cadre = (-3, 3, -3, 3)
carte = Carte(fils, cadre)
carte.lignes_autour(nombre=48, rayon=0.2)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set(xlim=cadre[:2], ylim=cadre[2:], aspect="equal")
carte.trace_lignes(ax, color="gray", linewidth=0.5)
carte.trace_fleches(ax, taille=7, color="gray")
niveaux = carte.niveaux(20, ajoute=[-0.1, 0.1, 0])
carte.trace_equipotentielles(ax, niveaux, contraste=0.2, cmap="plasma_r", linewidths=0.5)
carte.trace_sources(ax, taille=3)

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
