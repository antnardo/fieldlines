"""Lignes de champ électrique et magnétique, et équipotentielles, pour des figures de cours.

Des charges, des fils chargés, des dipôles, des fils parcourus par un courant, des
spires, des solénoïdes : on les pose dans une `Carte`, on choisit où partent les
lignes, on retouche celles qui gênent par leur nom, et on trace avec matplotlib.

>>> from fieldlines import Carte, Charge
>>> carte = Carte([Charge(2, (-1, 0)), Charge(-1, (1, 0))], cadre=(-4, 4, -3, 3))
>>> noms = carte.lignes_selon_flux(nombre=16)
>>> sum(ligne.aval.arret.name == "SOURCE" for ligne in carte.lignes().values()) > 0
True

La méthode (suivre pas à pas la direction du champ depuis un point de départ) vient du
billet de H. Bauke, « Visualizing streamlines », numbercrunch.de, 2013.
"""

from fieldlines.carte import Carte, Nom
from fieldlines.champ import Distribution, Grille
from fieldlines.departs import CercleDeFlux, departs_autour, departs_le_long
from fieldlines.geometrie import Axe, Cadre
from fieldlines.integration import (
    Arret,
    Extremite,
    Integrateur,
    LigneDeChamp,
    Options,
    Reglages,
)
from fieldlines.niveaux import niveaux
from fieldlines.sources import (
    Charge,
    Dipole,
    DipoleMagnetique,
    FilCharge,
    FilCourant,
    Singularite,
    Solenoide,
    Source,
    Spire,
)

__version__ = "0.1.0"

__all__ = [
    "Arret",
    "Axe",
    "Cadre",
    "Carte",
    "CercleDeFlux",
    "Charge",
    "Dipole",
    "DipoleMagnetique",
    "Distribution",
    "Extremite",
    "FilCharge",
    "FilCourant",
    "Grille",
    "Integrateur",
    "LigneDeChamp",
    "Nom",
    "Options",
    "Reglages",
    "Singularite",
    "Solenoide",
    "Source",
    "Spire",
    "__version__",
    "departs_autour",
    "departs_le_long",
    "niveaux",
]
