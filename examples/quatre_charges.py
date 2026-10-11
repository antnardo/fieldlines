"""Quatre charges alternées aux sommets d'un carré, toutes les lignes automatiques."""

from pathlib import Path

import matplotlib.pyplot as plt

from fieldlines import Carte, Charge

charges = [Charge(1, (-1, -1)), Charge(-1, (-1, 1)), Charge(1, (1, -1)), Charge(-1, (1, 1))]
cadre = (-5, 5, -5, 5)
carte = Carte(charges, cadre)
carte.lignes_autour(nombre=10, rayon=0.2)

fig, ax = plt.subplots()
ax.set(xlim=cadre[:2], ylim=cadre[2:], aspect="equal", xlabel="$x$", ylabel="$y$")
carte.trace_lignes(ax)
carte.trace_fleches(ax)
carte.trace_equipotentielles(ax, carte.niveaux(20, echelle="log", symetrique=True), contraste=3)
carte.trace_sources(ax, taille=12)

fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
