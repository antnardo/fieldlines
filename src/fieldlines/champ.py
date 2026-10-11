"""Une distribution de sources : champ total, potentiel, fonction de flux, grilles.

La distribution est homogène — toutes ses sources électriques, ou toutes magnétiques :
tracer les lignes d'une somme E + B n'aurait pas de sens.

Les grilles masquent les points non finis. Une grille qui passe exactement sur une
charge y trouve un potentiel infini ; l'ancien module gardait ce ±inf, qui faisait des
extrêmes infinis puis des niveaux NaN. Ici, ces points sont masqués et les extrêmes
sont ceux des valeurs finies.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from fieldlines.geometrie import Axe, Cadre
from fieldlines.sources import Charge, Lots, Nature, Singularite, Source

__all__ = ["Distribution", "Grille", "Symetrie"]

Symetrie: TypeAlias = Literal["plane", "revolution"]


@dataclass(frozen=True, slots=True)
class Grille:
    """Une grandeur scalaire sur une grille régulière : `valeurs[i, j]` en (x[j], y[i])."""

    x: np.ndarray
    y: np.ndarray
    valeurs: np.ma.MaskedArray

    @property
    def minimum(self) -> float:
        """Le minimum des valeurs finies."""
        return float(self.valeurs.min())

    @property
    def maximum(self) -> float:
        """Le maximum des valeurs finies."""
        return float(self.valeurs.max())


class Distribution:
    """Un ensemble de sources de même nature, et ce qu'on calcule à partir d'elles."""

    __slots__ = ("_axe", "_lots", "_singularites", "nature", "sources")

    def __init__(self, sources: Iterable[Source]) -> None:
        self.sources: tuple[Source, ...] = tuple(sources)
        if not self.sources:
            raise ValueError("une distribution demande au moins une source")
        natures = {source.nature for source in self.sources}
        if len(natures) > 1:
            raise ValueError("sources électriques et magnétiques mêlées : faire deux cartes")
        self.nature: Nature = natures.pop()
        self._lots = Lots(self.sources)
        self._singularites = tuple(s for source in self.sources for s in source.singularites())
        self._axe = self._cherche_axe()

    # ----------------------------------------------------------------- champ et potentiel

    def champ(self, x: np.ndarray | float, y: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """Les composantes du champ total (vectorisé, forme de x et y)."""
        return self._lots.champ(x, y)

    def champ_point(self, x: float, y: float) -> tuple[float, float]:
        """Le champ total en un point, par le chemin rapide de l'intégration."""
        return self._lots.champ_point(x, y)

    def norme(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        fx, fy = self.champ(x, y)
        return np.hypot(fx, fy)

    def potentiel(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """Le potentiel électrostatique total (champ électrique seulement)."""
        if self.nature != "E":
            raise TypeError("le champ magnétique n'a pas de potentiel scalaire ici")
        forme = np.broadcast(np.asarray(x), np.asarray(y)).shape
        total = np.zeros(forme)
        for source in self.sources:
            total = total + source.potentiel(x, y)
        return total

    # ---------------------------------------------------------------- fonction de flux

    @property
    def singularites(self) -> tuple[Singularite, ...]:
        """Les points singuliers de toutes les sources, dans l'ordre des sources."""
        return self._singularites

    def symetrie(self) -> Symetrie | None:
        """La symétrie qui donne une fonction de flux, ou None s'il n'y en a pas.

        Plane : uniquement des fils de courant. Révolution : des charges, dipôles,
        spires et solénoïdes tous sur un même axe, dipôles et spires alignés sur lui.
        """
        if all(type(s).flux_plan is not Source.flux_plan for s in self.sources):
            return "plane"
        return "revolution" if self.axe_de_revolution() is not None else None

    def axe_de_revolution(self) -> Axe | None:
        """L'axe commun de révolution des sources, s'il existe."""
        return self._axe

    def _cherche_axe(self) -> Axe | None:
        """Un axe candidat, puis sa vérification par chaque source.

        L'axe propre d'une spire, d'un solénoïde, d'un aimant ou d'un dipôle s'impose.
        Sinon, il passe par les charges : par les deux premières positions distinctes,
        ou par la seule position dans la direction qu'impose un champ uniforme ou un plan
        chargé (à défaut, Ox). Sans charge ni axe propre, pas de révolution.
        """
        axes = [axe for source in self.sources if (axe := source.axe()) is not None]
        if axes:
            candidat = axes[0]
        else:
            positions = [s.position for s in self.sources if isinstance(s, Charge)]
            if not positions:
                return None
            origine = positions[0]
            autres = [p for p in positions if p != origine]
            directions = [d for s in self.sources if (d := s.direction_axe()) is not None]
            if autres:
                angle = math.atan2(autres[0][1] - origine[1], autres[0][0] - origine[0])
            else:
                angle = directions[0] if directions else 0.0
            candidat = Axe(origine, angle)
        try:
            for source in self.sources:
                source.flux_revolution(0.0, 0.0, candidat)
        except ValueError:
            return None
        return candidat

    def flux(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """La fonction de flux Ψ, dont les lignes de niveau sont les lignes de champ.

        Elle n'existe que si la distribution a une symétrie (voir `symetrie`) ; sinon,
        ValueError. Entre deux lignes, 2π ΔΨ est le flux en révolution et ΔΨ le flux par
        unité de longueur dans le problème plan.
        """
        symetrie = self.symetrie()
        forme = np.broadcast(np.asarray(x), np.asarray(y)).shape
        total = np.zeros(forme)
        if symetrie == "plane":
            for source in self.sources:
                total = total + source.flux_plan(x, y)
            return total
        axe = self.axe_de_revolution()
        if axe is None:
            raise ValueError(
                "pas de fonction de flux : les sources ne sont ni des fils de courant seuls "
                "ni toutes sur un même axe de révolution"
            )
        for source in self.sources:
            total = total + source.flux_revolution(x, y, axe)
        return total

    # ------------------------------------------------------------------------- grilles

    def grille(self, cadre: Cadre, points: int, grandeur: Literal["potentiel", "flux"]) -> Grille:
        """Le potentiel ou la fonction de flux sur une grille de `points` × `points`."""
        x, y = cadre.axes(points)
        gx, gy = np.meshgrid(x, y)
        valeurs = self.potentiel(gx, gy) if grandeur == "potentiel" else self.flux(gx, gy)
        return Grille(x, y, np.ma.masked_invalid(valeurs))

    def champ_de_reference(self, cadre: Cadre) -> float:
        """Une valeur typique de la norme du champ dans le cadre : sa médiane.

        Les seuils de l'intégration (champ nul, ralentissement) s'expriment par rapport
        à elle : ils suivent ainsi l'échelle de la figure, quelles que soient les
        valeurs des charges.
        """
        x, y = cadre.axes(41)
        norme = self.norme(*np.meshgrid(x, y))
        finies = norme[np.isfinite(norme) & (norme > 0)]
        return float(np.median(finies)) if finies.size else 1.0

    def distance_minimale(self, singularites: Sequence[Singularite] | None = None) -> float:
        """La plus petite distance entre deux singularités (inf s'il y en a moins de deux)."""
        points = np.array([s.position for s in (singularites or self._singularites)])
        if len(points) < 2:
            return math.inf
        ecarts = np.hypot(*(points[:, None, :] - points[None, :, :]).transpose(2, 0, 1))
        ecarts[np.diag_indices(len(points))] = np.inf
        return float(ecarts.min())
