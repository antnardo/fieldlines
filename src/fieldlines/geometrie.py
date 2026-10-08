"""Le cadre de la figure et les quelques calculs de plan qui servent partout.

Le cadre est la seule donnée géométrique que partagent le calcul des grilles,
l'intégration des lignes, le placement des flèches et le tracé : il vit donc dans le
module le plus bas, que tous les autres importent.
"""

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

__all__ = ["Axe", "Cadre", "Point", "en_point"]

Point: TypeAlias = tuple[float, float]


def en_point(valeur: object) -> Point:
    """Convertit `[x, y]`, `(x, y)` ou un tableau numpy en couple de flottants.

    Les scripts de cours écrivent volontiers les positions en listes ; les classes
    gelées du paquet veulent des n-uplets, qu'on peut comparer et hacher.
    """
    x, y = np.asarray(valeur, dtype=float).reshape(2)
    return (float(x), float(y))


@dataclass(frozen=True, slots=True)
class Cadre:
    """Le rectangle [xmin, xmax] × [ymin, ymax] de la figure."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self) -> None:
        if not (self.xmin < self.xmax and self.ymin < self.ymax):
            raise ValueError(f"cadre vide ou retourné : {self}")

    @classmethod
    def depuis(cls, valeur: "Cadre | tuple[float, float, float, float]") -> "Cadre":
        """Accepte un Cadre ou le quadruplet (xmin, xmax, ymin, ymax) des anciens scripts."""
        if isinstance(valeur, Cadre):
            return valeur
        xmin, xmax, ymin, ymax = (float(v) for v in valeur)
        return cls(xmin, xmax, ymin, ymax)

    @property
    def largeur(self) -> float:
        return self.xmax - self.xmin

    @property
    def hauteur(self) -> float:
        return self.ymax - self.ymin

    @property
    def diagonale(self) -> float:
        return math.hypot(self.largeur, self.hauteur)

    def elargi(self, marge: float) -> "Cadre":
        """Le cadre agrandi de `marge` fois sa largeur (et sa hauteur) de chaque côté.

        Les lignes sont calculées un peu au-delà du cadre : une ligne qui sort et
        revient (celles d'un dipôle) reste entière, et le tracé coupé au bord des axes
        ne laisse pas voir où le calcul s'est arrêté.
        """
        dx, dy = marge * self.largeur, marge * self.hauteur
        return Cadre(self.xmin - dx, self.xmax + dx, self.ymin - dy, self.ymax + dy)

    def contient(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """Vrai pour les points dans le cadre, bord compris (vectorisé)."""
        return (
            (np.asarray(x) >= self.xmin)
            & (np.asarray(x) <= self.xmax)
            & (np.asarray(y) >= self.ymin)
            & (np.asarray(y) <= self.ymax)
        )

    def distance_interieure(self, x: float, y: float) -> float:
        """Distance signée au bord : positive dedans, négative dehors.

        C'est la fonction d'événement de la sortie du cadre : continue, elle change de
        signe exactement au passage du bord.
        """
        return min(x - self.xmin, self.xmax - x, y - self.ymin, self.ymax - y)

    def axes(self, points: int) -> tuple[np.ndarray, np.ndarray]:
        """Les abscisses et ordonnées d'une grille régulière de `points` × `points`."""
        return (
            np.linspace(self.xmin, self.xmax, points),
            np.linspace(self.ymin, self.ymax, points),
        )


@dataclass(frozen=True, slots=True)
class Axe:
    """Une droite orientée du plan : un point et un angle (en radians, depuis Ox).

    C'est l'axe de révolution d'une spire, d'un solénoïde ou d'un dipôle ; la figure
    est alors un plan méridien qui le contient.
    """

    origine: Point
    angle: float

    @property
    def direction(self) -> Point:
        return (math.cos(self.angle), math.sin(self.angle))

    @property
    def normale(self) -> Point:
        """La direction de l'axe tournée de +90° : le côté « gauche » du plan méridien."""
        return (-math.sin(self.angle), math.cos(self.angle))

    def coordonnees(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Les coordonnées (z, ρ) d'un point : abscisse le long de l'axe, distance à l'axe.

        ρ est positif des deux côtés de l'axe : les grandeurs de révolution n'en
        dépendent pas, et c'est à l'appelant de reporter le signe sur la composante
        radiale d'un champ.
        """
        ux, uy = self.direction
        dx = np.asarray(x, dtype=float) - self.origine[0]
        dy = np.asarray(y, dtype=float) - self.origine[1]
        return dx * ux + dy * uy, np.abs(dy * ux - dx * uy)

    def abscisse(self, point: Point) -> float:
        """L'abscisse z de la projection d'un point sur l'axe."""
        ux, uy = self.direction
        return (point[0] - self.origine[0]) * ux + (point[1] - self.origine[1]) * uy

    def ecart(self, point: Point) -> float:
        """La distance d'un point à l'axe."""
        ux, uy = self.direction
        return abs((point[1] - self.origine[1]) * ux - (point[0] - self.origine[0]) * uy)

    def sens_relatif(self, angle: float, tolerance: float = 1e-9) -> int:
        """+1 si `angle` a la direction de l'axe, -1 s'il lui est opposé, 0 sinon."""
        produit = math.cos(angle - self.angle)
        if abs(abs(produit) - 1.0) > tolerance:
            return 0
        return 1 if produit > 0 else -1
