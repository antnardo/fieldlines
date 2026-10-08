"""Le potentiel de deux charges ±1, et celui du dipôle de même moment, superposés.

Mêmes niveaux et mêmes couleurs pour les deux cartes : le `norm` du premier tracé sert
au second. Loin des charges, les équipotentielles se confondent.

D'après Medias/Python/EM/Figures cours/E_cartechamp_example5.py.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt

from fieldlines import Carte, Charge, Dipole

cadre = (-2.5, 2.5, -2.5, 2.5)
paire = Carte([Charge(1, (0, 0.5)), Charge(-1, (0, -0.5))], cadre)
dipole = Carte([Dipole(1, (0, 0), angle=math.pi / 2)], cadre)
niveaux = paire.niveaux(20, echelle="log", symetrique=True)

fig, ax = plt.subplots()
ax.set(xlim=cadre[:2], ylim=cadre[2:], aspect="equal", xlabel="$x$", ylabel="$y$")
equipotentielles = paire.trace_equipotentielles(ax, niveaux, contraste=3)
dipole.trace_equipotentielles(ax, niveaux, norm=equipotentielles.norm, linestyles="--")
paire.trace_sources(ax, taille=12)
dipole.trace_sources(ax, taille=12)

fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
