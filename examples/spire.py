"""Champ magnétique d'une spire circulaire, vue par la tranche (plan méridien).

À gauche, les lignes intégrées pas à pas, parties du diamètre de la spire à flux égal
du champ de la figure ; chacune s'arrête quand elle se referme. À droite, les mêmes
lignes obtenues autrement : ce sont les lignes de niveau de la fonction de flux
Ψ = ρ A_φ, exacte par les intégrales elliptiques, à niveaux équidistants (tubes de
même flux dans l'espace, plus espacés près de l'axe).
"""

from pathlib import Path

import matplotlib.pyplot as plt

from fieldlines import Carte, Spire

cadre = (-3, 3, -2.5, 2.5)
carte = Carte([Spire(1, (0, 0), 1)], cadre)  # axe selon Ox
carte.lignes_le_long((0, -0.85), (0, 0.85), 14)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
carte.trace(axes[0])
axes[0].set_title("lignes intégrées")

axes[1].set(xlim=cadre[:2], ylim=cadre[2:], aspect="equal", title="lignes de niveau de Ψ")
carte.trace_lignes_de_flux(axes[1], niveaux=[0.25 * k for k in range(1, 16)], linewidths=1)
carte.trace_sources(axes[1])

fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
