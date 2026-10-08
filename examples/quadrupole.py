"""Quadrupôle : lignes automatiques, puis retouche des flèches et retrait des doublons.

`lignes_autour` trace chaque ligne d'une charge à l'autre deux fois, une fois depuis
chaque bout : on retire les doublons par leur numéro (`trace_noms` les montre). La
figure de droite obtient le même résultat sans retouche, avec `lignes_selon_flux`.

D'après Medias/Python/EM/Figures cours/E_cartechamp_example3.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from fieldlines import Carte, Charge

charges = [Charge(-1, (-1, -1)), Charge(1, (-1, 1)), Charge(1, (1, -1)), Charge(-1, (1, 1))]
cadre = (-5, 5, -5, 5)

a_la_main = Carte(charges, cadre)
a_la_main.lignes_autour(nombre=16, rayon=0.2)
a_la_main.ajuste(30, 38, fleches=[0.45])
a_la_main.ajuste(58, 2, fleches=[0.6])
a_la_main.retire(range(23, 29), range(31, 38), range(39, 46), 47, range(51, 58))

selon_flux = Carte(charges, cadre)
selon_flux.lignes_selon_flux(nombre=16, rayon=0.2)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, carte in zip(axes, (a_la_main, selon_flux), strict=True):
    ax.set(xlim=cadre[:2], ylim=cadre[2:], aspect="equal", xlabel="$x$", ylabel="$y$")
    carte.trace_lignes(ax)
    carte.trace_fleches(ax)
    niveaux = carte.niveaux(20, echelle="log", symetrique=True)
    carte.trace_equipotentielles(ax, niveaux, contraste=2.5)
    carte.trace_sources(ax, taille=12)
axes[0].set_title("lignes_autour, retouchées")
axes[1].set_title("lignes_selon_flux")

fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
