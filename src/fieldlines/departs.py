"""Où faire partir les lignes de champ.

Trois façons de choisir les points de départ :

- `departs_autour` : autour de chaque source, à intervalles réguliers d'angle (pour
  un dipôle, réguliers en sin θ, θ compté depuis son axe : c'est le flux d'un dipôle à
  travers un petit cercle, d'où des lignes plus serrées le long de l'axe). C'est la
  méthode de l'ancien `efieldlines`, qui numérote les lignes de façon prévisible.
- `departs_le_long` : le long d'un segment, à flux égal entre deux départs voisins.
- `PlacementSelonFlux` : autour des sources, à flux égal, en nombre proportionnel à
  la charge, et sans tracer deux fois la même ligne.

Le placement selon le flux procède en deux passes. D'abord, on part des sources où le
champ sort (charges positives, côté sortant des dipôles) : sur un petit cercle autour
de chacune, on calcule le flux sortant du champ total, et on place les départs à
incréments égaux de ce flux, en nombre proportionnel à la charge. On intègre ces
lignes et on note où elles arrivent. Ensuite, autour de chaque puits (charges
négatives, côté rentrant des dipôles), on repère, en coordonnée de flux rentrant, les
lignes déjà arrivées, et on n'ajoute que celles qui manquent pour atteindre son quota,
dans les plus grands intervalles laissés libres. Dans un système neutre, les lignes
d'une charge positive qui finissent sur une négative ne sont donc tracées qu'une fois.

Ce flux est celui de la figure plane : pour des fils chargés (problème plan) il est
exact, et la densité des lignes suit partout l'intensité du champ ; pour des charges
ponctuelles, c'est la convention des figures de manuel, des départs presque
équiangulaires près de chaque charge.

Cette implémentation est propre au paquet ; elle ne reprend pas le code GPL du paquet
electrostatics de T. J. Duck, qui résout le même problème autrement.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from fieldlines.champ import Distribution
from fieldlines.geometrie import Point, en_point
from fieldlines.integration import Arret, Integrateur, LigneDeChamp, Options
from fieldlines.sources import Singularite

__all__ = [
    "CercleDeFlux",
    "Depart",
    "PlacementSelonFlux",
    "Repartition",
    "departs_autour",
    "departs_le_long",
    "origine_et_phase",
]

Repartition: TypeAlias = Literal["flux", "flux_revolution", "reguliere"]


@dataclass(frozen=True, slots=True)
class Depart:
    """Un point de départ, avec la ligne déjà calculée si la méthode a dû la calculer."""

    position: Point
    ligne: LigneDeChamp | None = None


def departs_autour(
    singularite: Singularite, nombre: int, rayon: float, angle: float = 0.0
) -> list[Point]:
    """`nombre` départs sur le cercle de rayon `rayon` autour d'une singularité.

    Monopôle : angles angle + 2πk/nombre. Dipôle : réguliers en sin θ, θ compté depuis
    l'axe du dipôle, une moitié des départs du côté où le champ sort, l'autre moitié
    de l'autre côté, sans départ en double aux deux bouts de l'intervalle d'angles.
    """
    cx, cy = singularite.position
    x = np.linspace(0.0, 2.0 * math.pi, nombre, endpoint=False)
    if singularite.genre == "dipole":
        demi, rang = np.divmod(x, math.pi)
        # sin θ réparti uniformément sur ]−1, 1] dans chaque moitié.
        relatifs = demi * math.pi - np.arcsin(1.0 - 2.0 * rang / math.pi)
        angles = angle + singularite.angle + relatifs
    else:
        angles = angle + x
    return [
        (float(cx + rayon * c), float(cy + rayon * s))
        for c, s in zip(np.cos(angles), np.sin(angles), strict=True)
    ]


def origine_et_phase(singularite: Singularite, angle: float) -> tuple[float, float]:
    """L'origine des coordonnées de flux sur le cercle, et la phase des départs.

    Monopôle : un départ exactement à l'angle `angle`. Dipôle : l'origine au bord de
    la moitié sortante, et des départs décalés d'un demi-pas, ce qui les rend
    symétriques par rapport à l'axe, jamais tangents au cercle, et met une ligne sur
    l'axe quand leur nombre est impair.
    """
    if singularite.genre == "dipole":
        return singularite.angle - math.pi / 2 + angle, 0.5
    return angle, 0.0


class CercleDeFlux:
    """Le flux du champ total à travers un petit cercle autour d'un point.

    On découpe le cercle en `echantillons` arcs ; le flux sortant (ou rentrant) cumulé
    depuis l'angle `origine` sert de coordonnée le long du cercle, et des départs à
    incréments égaux de cette coordonnée se partagent le flux également.
    """

    def __init__(
        self,
        distribution: Distribution,
        centre: Point,
        rayon: float,
        origine: float = 0.0,
        echantillons: int = 2048,
    ) -> None:
        self.centre = en_point(centre)
        self.rayon = float(rayon)
        self._angles = origine + np.linspace(0.0, 2.0 * math.pi, echantillons + 1)
        milieux = 0.5 * (self._angles[:-1] + self._angles[1:])
        cx, cy = self.centre
        fx, fy = distribution.champ(cx + rayon * np.cos(milieux), cy + rayon * np.sin(milieux))
        flux = (
            (fx * np.cos(milieux) + fy * np.sin(milieux)) * rayon * (2.0 * math.pi / echantillons)
        )
        flux = np.where(np.isfinite(flux), flux, 0.0)
        self._sortant = np.concatenate(([0.0], np.cumsum(np.maximum(flux, 0.0))))
        self._entrant = np.concatenate(([0.0], np.cumsum(np.maximum(-flux, 0.0))))

    @property
    def flux_sortant(self) -> float:
        return float(self._sortant[-1])

    @property
    def flux_entrant(self) -> float:
        return float(self._entrant[-1])

    def point(self, angle: float) -> Point:
        cx, cy = self.centre
        return (cx + self.rayon * math.cos(angle), cy + self.rayon * math.sin(angle))

    def angles_sortants(self, nombre: int, phase: float = 0.0) -> np.ndarray:
        """`nombre` angles à flux sortant égal, le premier à la coordonnée phase × pas."""
        return self._angles_de(
            self._sortant, (np.arange(nombre) + phase) * self.flux_sortant / nombre
        )

    def coordonnee_entrante(self, angle: float) -> float:
        """Le flux rentrant cumulé depuis l'origine jusqu'à `angle`."""
        debut = self._angles[0]
        relatif = debut + (angle - debut) % (2.0 * math.pi)
        return float(np.interp(relatif, self._angles, self._entrant))

    def angles_entrants(self, nombre: int, deja: Sequence[float], phase: float = 0.0) -> np.ndarray:
        """Les angles à ajouter pour avoir `nombre` lignes rentrantes en tout.

        `deja` : les coordonnées (en flux rentrant) des lignes déjà arrivées. Sans
        aucune, `nombre` angles à flux égal ; sinon, on répartit les manquantes entre
        les intervalles libres, toujours dans celui où l'écart serait le plus grand,
        puis régulièrement dans chaque intervalle.
        """
        total = self.flux_entrant
        manquantes = nombre - len(deja)
        if manquantes <= 0 or total <= 0:
            return np.empty(0)
        if not deja:
            return self._angles_de(self._entrant, (np.arange(nombre) + phase) * total / nombre)
        bornes = np.sort(np.asarray(deja, dtype=float) % total)
        ecarts = np.diff(np.concatenate((bornes, [bornes[0] + total])))
        ajouts = np.zeros(len(ecarts), dtype=int)
        for _ in range(manquantes):
            ajouts[np.argmax(ecarts / (ajouts + 1))] += 1
        coordonnees = [
            debut + ecart * (k + 1) / (n + 1)
            for debut, ecart, n in zip(bornes, ecarts, ajouts, strict=True)
            for k in range(n)
        ]
        return self._angles_de(self._entrant, np.asarray(coordonnees) % total)

    def _angles_de(self, cumul: np.ndarray, coordonnees: np.ndarray) -> np.ndarray:
        """Les angles où le flux cumulé atteint chaque coordonnée (interpolation).

        Le cumul est plat là où le flux est nul (côté rentrant d'une charge positive) :
        on prend le premier arc où il atteint la valeur, jamais un palier.
        """
        indices = np.clip(np.searchsorted(cumul, coordonnees, side="left"), 1, len(cumul) - 1)
        bas, haut = cumul[indices - 1], cumul[indices]
        with np.errstate(divide="ignore", invalid="ignore"):
            poids = np.where(haut > bas, (coordonnees - bas) / (haut - bas), 0.0)
        return self._angles[indices - 1] + poids * (
            self._angles[indices] - self._angles[indices - 1]
        )


def departs_le_long(
    distribution: Distribution,
    debut: Point,
    fin: Point,
    nombre: int,
    repartition: Repartition = "flux",
    echantillons: int = 4001,
) -> list[Point]:
    """`nombre` départs sur le segment [debut, fin], au milieu de `nombre` tranches.

    - `flux` : tranches de même flux du champ de la figure, ∫|F·n| dl ; la densité
      des lignes suit alors l'intensité du champ dans le dessin, comme autour des
      charges. Pour des fils de courant, c'est exactement la variation de A_z.
    - `flux_revolution` : tranches de même flux dans l'espace, |ΔΨ| avec Ψ la fonction
      de Stokes (sources de révolution seulement) : des tubes de flux égaux, plus
      espacés près de l'axe que dans le dessin plan.
    - `reguliere` : tranches de même longueur.

    Le segment ne doit pas passer sur une source.
    """
    debut, fin = en_point(debut), en_point(fin)
    t = np.linspace(0.0, 1.0, echantillons)
    x = debut[0] + t * (fin[0] - debut[0])
    y = debut[1] + t * (fin[1] - debut[1])
    if repartition == "reguliere":
        increments = np.ones(echantillons - 1)
    elif repartition == "flux_revolution":
        if distribution.symetrie() != "revolution":
            raise ValueError("flux_revolution : les sources n'ont pas d'axe de révolution commun")
        increments = np.abs(np.diff(distribution.flux(x, y)))
    elif repartition == "flux":
        longueur = math.hypot(fin[0] - debut[0], fin[1] - debut[1])
        nx, ny = -(fin[1] - debut[1]) / longueur, (fin[0] - debut[0]) / longueur
        fx, fy = distribution.champ(0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1]))
        increments = np.abs(fx * nx + fy * ny) * longueur / (echantillons - 1)
    else:
        raise ValueError(f"répartition inconnue : {repartition!r}")
    increments = np.where(np.isfinite(increments), increments, 0.0)
    cumul = np.concatenate(([0.0], np.cumsum(increments)))
    if cumul[-1] <= 0:
        raise ValueError("aucun flux ne traverse ce segment")
    cibles = (np.arange(nombre) + 0.5) * cumul[-1] / nombre
    positions = np.interp(cibles, cumul, t)
    return [
        (float(debut[0] + c * (fin[0] - debut[0])), float(debut[1] + c * (fin[1] - debut[1])))
        for c in positions
    ]


class PlacementSelonFlux:
    """Les départs à flux égal autour des sources, sans ligne tracée deux fois.

    Voir l'en-tête du module pour la méthode. Les lignes de la première passe sont
    calculées en chemin ; elles sont rendues avec leur départ, pour ne pas les
    recalculer.
    """

    def __init__(self, integrateur: Integrateur) -> None:
        self._integrateur = integrateur

    def places(
        self,
        nombre: int,
        rayon: float,
        angle: float = 0.0,
        options: Options | None = None,
    ) -> list[Depart]:
        options = options or Options()
        distribution = self._integrateur.distribution
        singularites = distribution.singularites
        indices = [i for i, s in enumerate(singularites) if s.aboutissante and s.poids != 0]
        if not indices:
            raise ValueError("aucune source où les lignes commencent ou finissent")
        quotas = self._quotas(singularites, indices, nombre)
        reperes = {i: origine_et_phase(singularites[i], angle) for i in indices}
        cercles = {
            i: CercleDeFlux(distribution, singularites[i].position, rayon, reperes[i][0])
            for i in indices
        }

        departs: list[Depart] = []
        arrivees: dict[int, list[float]] = {i: [] for i in indices}
        for i in indices:
            s = singularites[i]
            if s.genre == "monopole" and s.poids < 0:
                continue
            for a in cercles[i].angles_sortants(quotas[i], reperes[i][1]):
                position = cercles[i].point(float(a))
                ligne = self._integrateur.integre(position, options)
                departs.append(Depart(position, ligne))
                arrivee = self._arrivee(ligne, cercles)
                if arrivee is not None:
                    j, coordonnee = arrivee
                    arrivees[j].append(coordonnee)
        for j in indices:
            s = singularites[j]
            if s.genre == "monopole" and s.poids > 0:
                continue
            for a in cercles[j].angles_entrants(quotas[j], arrivees[j], reperes[j][1]):
                departs.append(Depart(cercles[j].point(float(a))))
        return departs

    @staticmethod
    def _quotas(
        singularites: Sequence[Singularite], indices: Sequence[int], nombre: int
    ) -> dict[int, int]:
        """Le nombre de lignes de chaque source : `nombre` pour la plus forte charge,
        proportionnellement pour les autres (au moins une) ; `nombre` pour un dipôle."""
        charges = [
            abs(singularites[i].poids) for i in indices if singularites[i].genre == "monopole"
        ]
        plus_forte = max(charges, default=1.0)
        return {
            i: nombre
            if singularites[i].genre == "dipole"
            else max(1, round(nombre * abs(singularites[i].poids) / plus_forte))
            for i in indices
        }

    @staticmethod
    def _arrivee(ligne: LigneDeChamp, cercles: dict[int, CercleDeFlux]) -> tuple[int, float] | None:
        """La source où finit la ligne, et la coordonnée de flux où elle entre dans son cercle."""
        j = ligne.aval.singularite
        if ligne.aval.arret is not Arret.SOURCE or j not in cercles:
            return None
        cercle = cercles[j]
        distances = np.hypot(
            ligne.points[:, 0] - cercle.centre[0], ligne.points[:, 1] - cercle.centre[1]
        )
        dehors = np.flatnonzero(distances >= cercle.rayon)
        if not dehors.size or dehors[-1] + 1 >= len(distances):
            return None
        k = dehors[-1]
        w = (distances[k] - cercle.rayon) / (distances[k] - distances[k + 1])
        x, y = ligne.points[k] + w * (ligne.points[k + 1] - ligne.points[k])
        angle = math.atan2(y - cercle.centre[1], x - cercle.centre[0])
        return j, cercle.coordonnee_entrante(angle)
