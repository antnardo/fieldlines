"""Deux charges différentes, +2 et −1 : la moitié des lignes de +2 finit sur −1.

D'après Medias/Python/EM/Figures cours/E_cartechamp_example4.py, où l'on retire à la
main les lignes de −1, déjà tracées depuis +2 ; `lignes_selon_flux` le fait seul.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from fieldlines import Carte, Charge

charges = [Charge(2, (-1, -1)), Charge(-1, (1, 0))]
cadre = (-5, 5, -5, 5)
carte = Carte(charges, cadre)
carte.lignes_selon_flux(nombre=20, rayon=0.2)

fig, ax = plt.subplots()
ax.set(xlim=cadre[:2], ylim=cadre[2:], aspect="equal", xlabel="$x$", ylabel="$y$")
carte.trace_lignes(ax)
carte.trace_fleches(ax)
carte.trace_equipotentielles(ax, carte.niveaux(20, echelle="log", symetrique=True), contraste=3)
carte.trace_sources(ax, taille=12)

fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
