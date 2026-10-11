"""Lignes de champ électrique et magnétique, et équipotentielles, pour des figures de cours.

Des charges, des fils et des plans chargés, des dipôles, des fils et des nappes de
courant, des spires, des solénoïdes, des aimants, un champ uniforme : on les pose dans
une `Carte`, on choisit où partent les lignes, on retouche celles qui gênent par leur
nom, et on trace avec matplotlib.

>>> from fieldlines import Carte, Charge
>>> carte = Carte([Charge(2, (-1, 0)), Charge(-1, (1, 0))], cadre=(-4, 4, -3, 3))
>>> noms = carte.lignes_selon_flux(nombre=16)
>>> sum(ligne.aval.arret.name == "SOURCE" for ligne in carte.lignes().values()) > 0
True

La tête de paquet expose la carte, les sources, de quoi en écrire une (`Source`,
`Singularite`, `Noyau`, `Nature`) et les constantes qui ramènent au SI. Le reste
s'importe de son module : `fieldlines.integration` (`Options`, `Reglages`,
`LigneDeChamp`, `Arret`), `fieldlines.departs`, `fieldlines.niveaux`,
`fieldlines.geometrie` (`Cadre`, `Axe`), `fieldlines.champ` (`Distribution`, `Grille`).

La méthode (suivre pas à pas la direction du champ depuis un point de départ) vient du
billet de H. Bauke, « Visualizing streamlines », numbercrunch.de, 2013.
"""

from fieldlines.carte import Carte
from fieldlines.sources import (
    K_COULOMB,
    MU0_SUR_4PI,
    Aimant,
    ChampUniforme,
    Charge,
    Dipole,
    DipoleMagnetique,
    FilCharge,
    FilCourant,
    NappeCourant,
    Nature,
    Noyau,
    PlanCharge,
    Singularite,
    Solenoide,
    Source,
    Spire,
)

__version__ = "0.1.1"

__all__ = [
    "K_COULOMB",
    "MU0_SUR_4PI",
    "Aimant",
    "Carte",
    "ChampUniforme",
    "Charge",
    "Dipole",
    "DipoleMagnetique",
    "FilCharge",
    "FilCourant",
    "NappeCourant",
    "Nature",
    "Noyau",
    "PlanCharge",
    "Singularite",
    "Solenoide",
    "Source",
    "Spire",
    "__version__",
]
