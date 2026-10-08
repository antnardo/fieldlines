"""Toutes les options sur une distribution quelconque : deux charges et un dipôle.

Lignes de départ automatiques ou à la main, retouche ou retrait d'une ligne par son
numéro, niveaux logarithmiques symétriques des équipotentielles, valeurs du potentiel.
Le numéro de chaque ligne est écrit sur la figure : c'est avec lui qu'on choisit les
lignes à retoucher ou à retirer.

D'après Medias/Python/Python Classe/EM/E_cartechamp_example1.py.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt

from fieldlines import Carte, Charge, Dipole

charges = [
    Charge(1, (-1, 0.5)),
    Charge(-1, (-1, -0.5)),
    Dipole(1, (1, 0), angle=math.pi / 2),  # moment vers le haut
]
cadre = (-20, 20, -20, 20)
carte = Carte(charges, cadre)

# 23 lignes autour de chaque source, numérotées source après source : 0-22, 23-45, 46-68.
carte.lignes_autour(nombre=23, rayon=0.2)
# Une ligne à la main, calculée loin au-delà du cadre, avec trois flèches.
carte.ajoute_ligne((1, 0.65), fleches=[0.2, 0.5, 0.8], marge=4.5)
# Retouches et retraits par numéro ; les numéros ne bougent jamais.
carte.ajuste(52, marge=2, fleches=[0.1, 0.9])
carte.retire(2, range(6, 13), 44, range(63, 68))

niveaux = carte.niveaux(20, echelle="log", symetrique=True)

fig, ax = plt.subplots()
ax.set(xlim=cadre[:2], ylim=cadre[2:], aspect="equal", xlabel="$x$", ylabel="$y$")
carte.trace_lignes(ax)
carte.trace_fleches(ax)
carte.trace_noms(ax)
equipotentielles = carte.trace_equipotentielles(ax, niveaux, contraste=5)
ax.clabel(equipotentielles, fontsize=8, fmt="%.2g")
carte.trace_sources(ax, taille=12)

fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
