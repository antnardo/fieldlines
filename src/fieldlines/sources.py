"""Les sources du champ électrique et du champ magnétique, et leurs formules.

Unités réduites : 1/(4π ε0) = 1 et μ0/(4π) = 1. Une charge q crée E = q/r², un fil
chargé de charge linéique λ crée E = 2λ/r, un plan chargé σ crée E = 2πσ, un fil
parcouru par I crée B = 2I/r, une nappe de courant j crée B = 2πj, une spire de rayon a
crée B = 2π I a²/(a² + z²)^(3/2) sur son axe. Pour revenir au SI, il suffit de
multiplier E et V par `K_COULOMB` = 1/(4π ε0) et B par `MU0_SUR_4PI` = μ0/(4π), les
charges, courants et longueurs étant en unités SI : les lignes de champ, elles, ne
dépendent pas des unités.

Sources électriques :

- `Charge` : charge ponctuelle (champ en 1/r², celui de l'espace) ;
- `FilCharge` : fil rectiligne infini perpendiculaire à la figure (champ en 1/r,
  problème plan) ;
- `PlanCharge` : plan infini perpendiculaire à la figure, vu par la tranche (champ
  uniforme de chaque côté) ;
- `Dipole` : dipôle ponctuel, dans le plan de la figure.

Sources magnétiques :

- `FilCourant` : fil rectiligne infini perpendiculaire à la figure, courant selon +z
  (vers le lecteur) quand I > 0 ;
- `NappeCourant` : nappe de courant plane, infinie, perpendiculaire à la figure
  (champ uniforme de chaque côté, parallèle à la nappe) ;
- `Spire` : spire circulaire vue par la tranche, la figure étant un plan méridien ;
  champ exact par les intégrales elliptiques complètes ;
- `Solenoide` : une somme de spires coaxiales ;
- `Aimant` : cylindre uniformément aimanté, le solénoïde équivalent dessiné avec ses
  pôles ;
- `DipoleMagnetique` : dipôle ponctuel, dans le plan de la figure.

Et `ChampUniforme`, électrique ou magnétique, qui s'ajoute aux autres sources.

Le champ se calcule par « lots » : chaque source donne une ou plusieurs lignes de
paramètres, et un noyau vectorisé traite d'un coup toutes les sources d'une même sorte,
pour tous les points demandés. C'est ce qui rend l'intégration des lignes rapide : un
point de la ligne, quinze charges, une seule série d'opérations numpy. Le potentiel et
la fonction de flux, qui ne servent que sur des grilles, se calculent source par source.

La fonction de flux Ψ est celle dont les lignes de niveau sont les lignes de champ :

- problème plan (fils de courant) : Ψ = A_z, et B = (∂Ψ/∂y, −∂Ψ/∂x) ;
- révolution autour d'un axe (charges sur l'axe, dipôles, spires, solénoïdes alignés
  sur lui) : la fonction de Stokes, B_z = (1/ρ) ∂Ψ/∂ρ et B_ρ = −(1/ρ) ∂Ψ/∂z ; pour une
  spire Ψ = ρ A_φ, et 2π ΔΨ est le flux à travers la surface engendrée par un arc qui
  joint deux lignes.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import ClassVar, Literal, TypeAlias

import numpy as np
from scipy.special import ellipe, ellipk

from fieldlines.geometrie import Axe, Point, en_point

__all__ = [
    "K_COULOMB",
    "MU0_SUR_4PI",
    "Aimant",
    "ChampUniforme",
    "Charge",
    "Dipole",
    "DipoleMagnetique",
    "FilCharge",
    "FilCourant",
    "Genre",
    "Lots",
    "NappeCourant",
    "Nature",
    "Noyau",
    "PlanCharge",
    "Singularite",
    "Solenoide",
    "Source",
    "Spire",
]

Nature: TypeAlias = Literal["E", "B"]
Genre: TypeAlias = Literal["monopole", "dipole", "tourbillon"]
Noyau: TypeAlias = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

# Les constantes qui ramènent au SI, pour les valeurs (pas pour les lignes) : avec q en
# coulombs, λ en C/m, σ en C/m², I en ampères, j en A/m, M en A/m et les longueurs en
# mètres, E (V/m) et V (V) valent K_COULOMB fois le champ et le potentiel calculés, B (T)
# vaut MU0_SUR_4PI fois le champ calculé.
K_COULOMB = 8.9875517923e9
"""1/(4π ε0), en N·m²/C² (CODATA 2018)."""
MU0_SUR_4PI = 1e-7
"""μ0/(4π), en T·m/A (valeur exacte jusqu'en 2019 ; CODATA 2018 : 1,000 000 000 55 × 10⁻⁷)."""

# Seuil du développement au voisinage de l'axe d'une spire. Au-dessous, la formule
# exacte de B_ρ/ρ perd ses chiffres (deux termes voisins de π/2 qui se compensent,
# erreur relative ε/m) ; le développement limité, lui, n'y fait qu'une erreur en m².
_M_AXE = 1e-6


@dataclass(frozen=True, slots=True)
class Singularite:
    """Un point où le champ diverge.

    - `monopole` : les lignes y naissent ou y meurent, en nombre proportionnel à
      `poids` (la charge, ou la charge linéique) ;
    - `dipole` : les lignes en sortent d'un côté et y rentrent de l'autre ;
    - `tourbillon` : les lignes tournent autour (fil de courant, passage d'une spire
      dans la figure) ; `poids` est l'intensité, positive vers le lecteur.

    `angle` est la direction du moment d'un dipôle (radians, depuis Ox).
    """

    position: Point
    genre: Genre
    poids: float
    angle: float = 0.0

    @property
    def aboutissante(self) -> bool:
        """Vrai si des lignes de champ peuvent finir sur ce point."""
        return self.genre != "tourbillon"


# --------------------------------------------------------------------------------------
# Noyaux vectorisés : paramètres (M, k), points x et y (N,), résultat (N,) et (N,).
# --------------------------------------------------------------------------------------


def _ecarts(p: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x[:, None] - p[:, 0], y[:, None] - p[:, 1]


def _noyau_charges(p: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx, dy = _ecarts(p, x, y)
    r2 = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        c = p[:, 2] / (r2 * np.sqrt(r2))
        return (c * dx).sum(axis=1), (c * dy).sum(axis=1)


def _noyau_fils_charges(
    p: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    dx, dy = _ecarts(p, x, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = 2.0 * p[:, 2] / (dx * dx + dy * dy)
        return (c * dx).sum(axis=1), (c * dy).sum(axis=1)


def _noyau_fils_courants(
    p: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    dx, dy = _ecarts(p, x, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = 2.0 * p[:, 2] / (dx * dx + dy * dy)
        return -(c * dy).sum(axis=1), (c * dx).sum(axis=1)


def _noyau_dipoles(p: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Dipôle ponctuel de l'espace : (3 (p·r) r − p r²) / r⁵.
    dx, dy = _ecarts(p, x, y)
    px, py = p[:, 2], p[:, 3]
    r2 = dx * dx + dy * dy
    pr = px * dx + py * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        r5 = r2 * r2 * np.sqrt(r2)
        fx = (3.0 * pr * dx - px * r2) / r5
        fy = (3.0 * pr * dy - py * r2) / r5
        return fx.sum(axis=1), fy.sum(axis=1)


def _noyau_uniformes(p: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Champs uniformes (fx, fy) : la somme des composantes, partout.
    return np.full(x.shape, p[:, 0].sum()), np.full(x.shape, p[:, 1].sum())


def _noyau_plans(p: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Plans (point px, py ; normale nx, ny ; direction du champ vx, vy ; amplitude A) :
    # F = A sign(d) v, d étant la distance signée au plan. Nul sur le plan même.
    dx, dy = _ecarts(p, x, y)
    c = p[:, 6] * np.sign(dx * p[:, 2] + dy * p[:, 3])
    return (c * p[:, 4]).sum(axis=1), (c * p[:, 5]).sum(axis=1)


def _noyau_spires(p: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spires (centre cx, cy ; axe ux, uy ; rayon a ; intensité I), formules exactes.

    Avec h l'abscisse le long de l'axe, ρ la distance à l'axe, r² = ρ² + h²,
    α² = (a − ρ)² + h², β² = (a + ρ)² + h² et m = 4aρ/β² :
    B_z = (2I/β) [K(m) + (a² − r²)/α² E(m)],
    B_ρ = (2I h/(ρβ)) [−K(m) + (a² + r²)/α² E(m)].
    On calcule B_ρ/ρ, qu'on multiplie par le vecteur radial non normé : pas de
    division par ρ sur l'axe. Près de l'axe (m petit), B_ρ/ρ = 3π I a² h/(a² + h²)^(5/2).
    """
    dx, dy = _ecarts(p, x, y)
    ux, uy, a, courant = p[:, 2], p[:, 3], p[:, 4], p[:, 5]
    h = dx * ux + dy * uy
    wx, wy = dx - h * ux, dy - h * uy
    rho2 = wx * wx + wy * wy
    rho = np.sqrt(rho2)
    a2 = a * a
    r2 = rho2 + h * h
    alpha2 = a2 + r2 - 2.0 * a * rho
    beta2 = a2 + r2 + 2.0 * a * rho
    m = 4.0 * a * rho / beta2
    k, e = ellipk(m), ellipe(m)
    beta = np.sqrt(beta2)
    with np.errstate(divide="ignore", invalid="ignore"):
        bz = 2.0 * courant / beta * (k + (a2 - r2) / alpha2 * e)
        exact = 2.0 * courant * h / (rho2 * beta) * (-k + (a2 + r2) / alpha2 * e)
        serie = 3.0 * math.pi * courant * a2 * h / (a2 + h * h) ** 2.5
        br_sur_rho = np.where(m < _M_AXE, serie, exact)
        return (bz * ux + br_sur_rho * wx).sum(axis=1), (bz * uy + br_sur_rho * wy).sum(axis=1)


class Lots:
    """Des sources regroupées par noyau, pour une évaluation vectorisée du champ."""

    __slots__ = ("_groupes",)

    # Au plus un million de couples (point, source) par appel de noyau : les grilles de
    # 512 × 512 points d'un solénoïde de 50 spires passent par tranches.
    _TAILLE_TRANCHE = 1 << 20

    def __init__(self, sources: Iterable["Source"]) -> None:
        groupes: dict[Noyau, list[tuple[float, ...]]] = {}
        for source in sources:
            for noyau, ligne in source.lots():
                groupes.setdefault(noyau, []).append(ligne)
        self._groupes = [
            (noyau, np.array(lignes, dtype=float)) for noyau, lignes in groupes.items()
        ]

    def champ(self, x: np.ndarray | float, y: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """Le champ total en des points de forme quelconque (diffusion numpy)."""
        xb, yb = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        forme = xb.shape
        xs, ys = xb.ravel(), yb.ravel()
        fx = np.zeros(xs.shape)
        fy = np.zeros(xs.shape)
        for noyau, parametres in self._groupes:
            pas = max(1, self._TAILLE_TRANCHE // len(parametres))
            for debut in range(0, xs.size, pas):
                tranche = slice(debut, debut + pas)
                gx, gy = noyau(parametres, xs[tranche], ys[tranche])
                fx[tranche] += gx
                fy[tranche] += gy
        return fx.reshape(forme), fy.reshape(forme)

    def champ_point(self, x: float, y: float) -> tuple[float, float]:
        """Le champ en un seul point : le chemin court de l'intégration."""
        xs, ys = np.array([x]), np.array([y])
        fx = fy = 0.0
        for noyau, parametres in self._groupes:
            gx, gy = noyau(parametres, xs, ys)
            fx += float(gx[0])
            fy += float(gy[0])
        return fx, fy


# --------------------------------------------------------------------------------------
# Les sources
# --------------------------------------------------------------------------------------


class Source(ABC):
    """Ce que toute source sait faire : son champ, ses singularités, ses symétries.

    Le champ passe par `lots` (voir `Lots`) ; le potentiel n'existe que pour les sources
    électriques ; la fonction de flux, seulement pour les sources compatibles avec une
    symétrie (plane ou de révolution).
    """

    __slots__ = ()

    nature: ClassVar[Nature]

    @abstractmethod
    def lots(self) -> tuple[tuple[Noyau, tuple[float, ...]], ...]:
        """Les lignes de paramètres de la source, chacune avec son noyau."""

    @abstractmethod
    def singularites(self) -> tuple[Singularite, ...]:
        """Les points où le champ de la source diverge."""

    def champ(self, x: np.ndarray | float, y: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """Les composantes du champ de la source seule (vectorisé)."""
        return Lots([self]).champ(x, y)

    def potentiel(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """Le potentiel électrostatique de la source seule (vectorisé)."""
        raise TypeError(f"{type(self).__name__} : pas de potentiel scalaire")

    def axe(self) -> Axe | None:
        """L'axe de révolution propre de la source, s'il y en a un."""
        return None

    def direction_axe(self) -> float | None:
        """La direction (radians) qu'un axe de révolution doit avoir pour cette source,
        quand elle en fixe la direction sans en fixer la position : un champ uniforme,
        un plan chargé. None si elle n'en impose aucune."""
        return None

    def flux_revolution(self, x: np.ndarray | float, y: np.ndarray | float, axe: Axe) -> np.ndarray:
        """La fonction de Stokes de la source pour une révolution autour de `axe`."""
        raise ValueError(f"{type(self).__name__} : pas de symétrie de révolution")

    def flux_plan(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """La fonction de flux d'un problème plan (lignes de champ = lignes de niveau)."""
        raise ValueError(f"{type(self).__name__} : pas de fonction de flux plane univoque")


def _distance(position: Point, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    return np.hypot(
        np.asarray(x, dtype=float) - position[0], np.asarray(y, dtype=float) - position[1]
    )


def _sur_axe(position: Point, axe: Axe, nom: str, tolerance: float = 1e-9) -> float:
    """L'abscisse de `position` sur l'axe, après avoir vérifié qu'elle y est."""
    if axe.ecart(position) > tolerance * max(1.0, abs(axe.abscisse(position))):
        raise ValueError(f"{nom} hors de l'axe de révolution")
    return axe.abscisse(position)


@dataclass(frozen=True, slots=True)
class Charge(Source):
    """Une charge ponctuelle q en `position` : E = q/r², V = q/r."""

    q: float
    position: Point
    nature: ClassVar[Nature] = "E"

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", en_point(self.position))

    def lots(self) -> tuple[tuple[Noyau, tuple[float, ...]], ...]:
        return ((_noyau_charges, (*self.position, self.q)),)

    def singularites(self) -> tuple[Singularite, ...]:
        return (Singularite(self.position, "monopole", self.q),)

    def potentiel(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return self.q / _distance(self.position, x, y)

    def flux_revolution(self, x: np.ndarray | float, y: np.ndarray | float, axe: Axe) -> np.ndarray:
        # Ψ = −q cos θ : constante sur chaque demi-axe, et Σ q cos θ = cte le long d'une
        # ligne, la propriété classique des lignes de charges alignées.
        zs = _sur_axe(self.position, axe, "charge")
        z, rho = axe.coordonnees(x, y)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -self.q * (z - zs) / np.hypot(z - zs, rho)


@dataclass(frozen=True, slots=True)
class FilCharge(Source):
    """Un fil infini perpendiculaire à la figure, de charge linéique λ.

    E = 2λ/r et V = −2λ ln r : c'est le problème plan, où le flux à travers un cercle
    ne dépend pas de son rayon (l'ancien `Ligne(q)` correspond à λ = q/2).
    """

    lineique: float
    position: Point
    nature: ClassVar[Nature] = "E"

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", en_point(self.position))

    def lots(self) -> tuple[tuple[Noyau, tuple[float, ...]], ...]:
        return ((_noyau_fils_charges, (*self.position, self.lineique)),)

    def singularites(self) -> tuple[Singularite, ...]:
        return (Singularite(self.position, "monopole", self.lineique),)

    def potentiel(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return -2.0 * self.lineique * np.log(_distance(self.position, x, y))


@dataclass(frozen=True, slots=True)
class _Plan(Source):
    """Un plan infini perpendiculaire à la figure, vu par la tranche.

    C'est la droite qui passe par `position` avec l'angle `angle` (radians, depuis Ox).
    Le champ est uniforme de chaque côté, d'amplitude `_amplitude`, dans la direction
    `_direction` d'un côté et dans la direction opposée de l'autre ; il est nul sur le
    plan même, et n'a pas de singularité.
    """

    surfacique: float
    position: Point
    angle: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", en_point(self.position))

    @property
    def tangente(self) -> Point:
        """La direction du plan dans la figure."""
        return (math.cos(self.angle), math.sin(self.angle))

    @property
    def normale(self) -> Point:
        """La normale au plan, la tangente tournée de +90° : le côté où d > 0."""
        return (-math.sin(self.angle), math.cos(self.angle))

    @abstractmethod
    def _amplitude(self) -> float:
        """L'amplitude du champ, de chaque côté."""

    @abstractmethod
    def _direction(self) -> Point:
        """La direction du champ du côté de la normale."""

    def lots(self) -> tuple[tuple[Noyau, tuple[float, ...]], ...]:
        return (
            (_noyau_plans, (*self.position, *self.normale, *self._direction(), self._amplitude())),
        )

    def singularites(self) -> tuple[Singularite, ...]:
        return ()

    def distance_signee(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """La distance au plan, positive du côté de la normale."""
        nx, ny = self.normale
        dx = np.asarray(x, dtype=float) - self.position[0]
        dy = np.asarray(y, dtype=float) - self.position[1]
        return dx * nx + dy * ny

    def flux_plan(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        # F = A sign(d) v = (∂Ψ/∂y, −∂Ψ/∂x) avec Ψ = A sign(d) (v_x (y − p_y) − v_y (x − p_x)),
        # continue à travers une nappe de courant (v = t), pas à travers un plan chargé
        # (v = n), dont les lignes partent.
        vx, vy = self._direction()
        dx = np.asarray(x, dtype=float) - self.position[0]
        dy = np.asarray(y, dtype=float) - self.position[1]
        return self._amplitude() * np.sign(self.distance_signee(x, y)) * (vx * dy - vy * dx)


@dataclass(frozen=True, slots=True)
class PlanCharge(_Plan):
    """Un plan infini de charge surfacique σ, perpendiculaire à la figure.

    E = 2πσ de chaque côté, normal au plan, qui s'en éloigne si σ > 0 ; V = −2πσ |d|.
    Deux plans de charges opposées font un condensateur plan (champ uniforme entre eux).
    """

    nature: ClassVar[Nature] = "E"

    def _amplitude(self) -> float:
        return 2.0 * math.pi * self.surfacique

    def _direction(self) -> Point:
        return self.normale

    def potentiel(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        return -2.0 * math.pi * self.surfacique * np.abs(self.distance_signee(x, y))

    def direction_axe(self) -> float | None:
        return self.angle + math.pi / 2

    def flux_revolution(self, x: np.ndarray | float, y: np.ndarray | float, axe: Axe) -> np.ndarray:
        # Autour d'un axe normal au plan, E_z = 2πσ sign(z − z_plan) : Ψ = E_z ρ²/2.
        if axe.sens_relatif(self.angle + math.pi / 2) == 0:
            raise ValueError("plan chargé non perpendiculaire à l'axe de révolution")
        z, rho = axe.coordonnees(x, y)
        return math.pi * self.surfacique * np.sign(z - axe.abscisse(self.position)) * rho * rho


@dataclass(frozen=True, slots=True)
class NappeCourant(_Plan):
    """Une nappe de courant plane, infinie, perpendiculaire à la figure.

    Le courant surfacique j est selon +z (vers le lecteur) quand j > 0 ; B = 2πj de
    chaque côté, parallèle à la nappe, dans le sens direct autour du courant comme pour
    un fil : −2πj t du côté de la normale, +2πj t de l'autre, t étant la direction de la
    nappe. Ψ = A_z = −2πj |d|. Deux nappes de courants opposés font un solénoïde infini
    (champ uniforme entre elles, nul dehors).
    """

    nature: ClassVar[Nature] = "B"

    def _amplitude(self) -> float:
        return -2.0 * math.pi * self.surfacique

    def _direction(self) -> Point:
        return self.tangente


@dataclass(frozen=True, slots=True)
class ChampUniforme(Source):
    """Un champ uniforme dans le plan de la figure, électrique ou magnétique.

    `ChampUniforme(E=(Ex, Ey))` ou `ChampUniforme(B=(Bx, By))` : les composantes sont
    celles du champ dans la figure (un champ perpendiculaire à la figure n'y a pas de
    lignes). Sans singularité, il n'a pas de lignes propres : on les place avec
    `lignes_le_long` ou `ajoute_ligne`. Ajouté aux autres sources, il fait le demi-corps
    de Rankine (fil chargé dans un champ uniforme), la charge dans un champ, le fil de
    courant dans un champ. V = −E·r ; Ψ = Fx y − Fy x (problème plan), et F_z ρ²/2 en
    révolution autour d'un axe parallèle au champ.
    """

    E: Point | None = None
    B: Point | None = None
    # La nature dépend de l'argument donné : elle est propre à chaque instance, et non
    # à la classe comme pour les autres sources.
    nature: Nature = field(init=False, repr=False, compare=False)  # type: ignore[misc]

    def __post_init__(self) -> None:
        if (self.E is None) == (self.B is None):
            raise ValueError("ChampUniforme : donner E=(Ex, Ey) ou B=(Bx, By), pas les deux")
        nom, donne = ("E", self.E) if self.E is not None else ("B", self.B)
        try:
            vecteur = en_point(donne)
        except ValueError:
            raise ValueError(
                f"ChampUniforme({nom}={donne!r}) : {nom} est un vecteur du plan de la figure, "
                f"{nom}=({nom}x, {nom}y)"
            ) from None
        if vecteur == (0.0, 0.0):
            raise ValueError("ChampUniforme : champ nul")
        object.__setattr__(self, nom, vecteur)
        object.__setattr__(self, "nature", nom)

    @property
    def vecteur(self) -> Point:
        """Les composantes du champ."""
        if self.E is not None:
            return self.E
        if self.B is not None:
            return self.B
        raise ValueError("ChampUniforme sans composantes")  # exclu par __post_init__

    @property
    def angle(self) -> float:
        """La direction du champ (radians, depuis Ox)."""
        fx, fy = self.vecteur
        return math.atan2(fy, fx)

    def lots(self) -> tuple[tuple[Noyau, tuple[float, ...]], ...]:
        return ((_noyau_uniformes, self.vecteur),)

    def singularites(self) -> tuple[Singularite, ...]:
        return ()

    def potentiel(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        if self.nature != "E":
            return super().potentiel(x, y)
        fx, fy = self.vecteur
        return -(fx * np.asarray(x, dtype=float) + fy * np.asarray(y, dtype=float))

    def direction_axe(self) -> float | None:
        return self.angle

    def flux_plan(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        fx, fy = self.vecteur
        return fx * np.asarray(y, dtype=float) - fy * np.asarray(x, dtype=float)

    def flux_revolution(self, x: np.ndarray | float, y: np.ndarray | float, axe: Axe) -> np.ndarray:
        sens = axe.sens_relatif(self.angle)
        if sens == 0:
            raise ValueError("champ uniforme non parallèle à l'axe de révolution")
        _, rho = axe.coordonnees(x, y)
        return 0.5 * sens * math.hypot(*self.vecteur) * rho * rho


@dataclass(frozen=True, slots=True)
class _DipolePonctuel(Source):
    """Un dipôle ponctuel de moment `moment`, orienté par `angle` (radians, depuis Ox).

    Le dipôle est celui de l'espace (champ en 1/r³), son moment dans le plan de la
    figure : (3 (p·r) r − p r²)/r⁵.
    """

    moment: float
    position: Point
    angle: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", en_point(self.position))

    @property
    def vecteur(self) -> Point:
        return (self.moment * math.cos(self.angle), self.moment * math.sin(self.angle))

    def lots(self) -> tuple[tuple[Noyau, tuple[float, ...]], ...]:
        return ((_noyau_dipoles, (*self.position, *self.vecteur)),)

    def singularites(self) -> tuple[Singularite, ...]:
        return (Singularite(self.position, "dipole", self.moment, self.angle),)

    def axe(self) -> Axe | None:
        return Axe(self.position, self.angle)

    def flux_revolution(self, x: np.ndarray | float, y: np.ndarray | float, axe: Axe) -> np.ndarray:
        sens = axe.sens_relatif(self.angle)
        if sens == 0:
            raise ValueError("dipôle non aligné sur l'axe de révolution")
        zs = _sur_axe(self.position, axe, "dipôle")
        z, rho = axe.coordonnees(x, y)
        with np.errstate(divide="ignore", invalid="ignore"):
            return sens * self.moment * rho * rho / np.hypot(z - zs, rho) ** 3


@dataclass(frozen=True, slots=True)
class Dipole(_DipolePonctuel):
    """Un dipôle électrique ponctuel : V = p·r/r³."""

    nature: ClassVar[Nature] = "E"

    def potentiel(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        px, py = self.vecteur
        dx = np.asarray(x, dtype=float) - self.position[0]
        dy = np.asarray(y, dtype=float) - self.position[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            return (px * dx + py * dy) / np.hypot(dx, dy) ** 3


@dataclass(frozen=True, slots=True)
class DipoleMagnetique(_DipolePonctuel):
    """Un dipôle magnétique ponctuel (petite spire, aimant vu de loin)."""

    nature: ClassVar[Nature] = "B"


@dataclass(frozen=True, slots=True)
class FilCourant(Source):
    """Un fil infini perpendiculaire à la figure, parcouru par I selon +z (vers le lecteur).

    B = 2I/r, orthoradial, dans le sens direct quand I > 0 ; Ψ = A_z = −2I ln r.
    """

    intensite: float
    position: Point
    nature: ClassVar[Nature] = "B"

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", en_point(self.position))

    def lots(self) -> tuple[tuple[Noyau, tuple[float, ...]], ...]:
        return ((_noyau_fils_courants, (*self.position, self.intensite)),)

    def singularites(self) -> tuple[Singularite, ...]:
        return (Singularite(self.position, "tourbillon", self.intensite),)

    def flux_plan(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return -2.0 * self.intensite * np.log(_distance(self.position, x, y))


@dataclass(frozen=True, slots=True)
class Spire(Source):
    """Une spire circulaire de rayon `rayon`, vue par la tranche.

    La figure est un plan méridien : l'axe de la spire passe par `centre` avec
    l'angle `angle` (radians, depuis Ox), et le courant tourne dans le sens direct
    autour de cet axe quand I > 0. La spire perce la figure en deux points :
    centre + a·n (courant vers le lecteur) et centre − a·n, n étant l'axe tourné de +90°.
    """

    intensite: float
    centre: Point
    rayon: float
    angle: float = 0.0
    nature: ClassVar[Nature] = "B"

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre", en_point(self.centre))
        if self.rayon <= 0:
            raise ValueError("le rayon d'une spire doit être positif")

    def lots(self) -> tuple[tuple[Noyau, tuple[float, ...]], ...]:
        ux, uy = math.cos(self.angle), math.sin(self.angle)
        return ((_noyau_spires, (*self.centre, ux, uy, self.rayon, self.intensite)),)

    def passages(self) -> tuple[Point, Point]:
        """Les deux points où la spire perce la figure : courant sortant, puis entrant."""
        nx, ny = -math.sin(self.angle), math.cos(self.angle)
        cx, cy = self.centre
        a = self.rayon
        return (cx + a * nx, cy + a * ny), (cx - a * nx, cy - a * ny)

    def singularites(self) -> tuple[Singularite, ...]:
        sortant, entrant = self.passages()
        return (
            Singularite(sortant, "tourbillon", self.intensite),
            Singularite(entrant, "tourbillon", -self.intensite),
        )

    def axe(self) -> Axe | None:
        return Axe(self.centre, self.angle)

    def flux_revolution(self, x: np.ndarray | float, y: np.ndarray | float, axe: Axe) -> np.ndarray:
        """Ψ = ρ A_φ = 2I β [(1 − m/2) K(m) − E(m)], sans division par ρ."""
        sens = axe.sens_relatif(self.angle)
        if sens == 0:
            raise ValueError("spire non coaxiale à l'axe de révolution")
        zs = _sur_axe(self.centre, axe, "spire")
        z, rho = axe.coordonnees(x, y)
        h = z - zs
        a = self.rayon
        beta2 = (a + rho) ** 2 + h * h
        m = 4.0 * a * rho / beta2
        return (
            sens * 2.0 * self.intensite * np.sqrt(beta2) * ((1.0 - m / 2.0) * ellipk(m) - ellipe(m))
        )


@dataclass(frozen=True, slots=True)
class Solenoide(Source):
    """Un solénoïde de `spires` spires de rayon `rayon`, sur une longueur `longueur`.

    Chaque spire porte I ; leurs centres sont au milieu de chacune des `spires`
    tranches de la longueur, sur l'axe passant par `centre` avec l'angle `angle`. Le
    champ au centre d'un solénoïde long tend vers μ0 n I = 4π (spires/longueur) I.
    """

    intensite: float
    centre: Point
    rayon: float
    longueur: float
    spires: int
    angle: float = 0.0
    nature: ClassVar[Nature] = "B"

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre", en_point(self.centre))
        if self.spires < 1 or self.longueur < 0:
            raise ValueError("un solénoïde a au moins une spire et une longueur positive")

    def composantes(self) -> tuple[Spire, ...]:
        """Les spires qui composent le solénoïde."""
        ux, uy = math.cos(self.angle), math.sin(self.angle)
        pas = self.longueur / self.spires
        cx, cy = self.centre
        abscisses = (-self.longueur / 2 + (k + 0.5) * pas for k in range(self.spires))
        return tuple(
            Spire(self.intensite, (cx + z * ux, cy + z * uy), self.rayon, self.angle)
            for z in abscisses
        )

    def lots(self) -> tuple[tuple[Noyau, tuple[float, ...]], ...]:
        return tuple(lot for spire in self.composantes() for lot in spire.lots())

    def singularites(self) -> tuple[Singularite, ...]:
        return tuple(s for spire in self.composantes() for s in spire.singularites())

    def axe(self) -> Axe | None:
        return Axe(self.centre, self.angle)

    def flux_revolution(self, x: np.ndarray | float, y: np.ndarray | float, axe: Axe) -> np.ndarray:
        return sum(
            (spire.flux_revolution(x, y, axe) for spire in self.composantes()),
            start=np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape),
        )


@dataclass(frozen=True, slots=True)
class Aimant(Source):
    """Un aimant cylindrique uniformément aimanté selon son axe : le solénoïde équivalent.

    Un cylindre de rayon `rayon` et de hauteur `hauteur`, centré en `centre`, d'axe
    orienté par `angle` (radians, depuis Ox) du pôle sud vers le pôle nord, porte
    l'aimantation M = `aimantation` selon cet axe. Son champ B, dedans comme dehors, est
    celui du courant surfacique M qui tourne sur sa surface latérale : un solénoïde de
    même forme, de courant total M × hauteur réparti sur `spires` spires. Au centre d'un
    aimant long, B = μ0 M = 4π M en unités réduites ; avec M en A/m et les longueurs en
    mètres, B (T) = `MU0_SUR_4PI` × le champ calculé.

    Le dessin montre le cylindre en coupe, ses deux pôles colorés et marqués N et S,
    sans les spires. Les lignes qui passent à moins du rayon d'arrêt de la surface
    latérale s'arrêtent sur une spire : partir de l'intérieur de l'aimant, par un
    segment de son plan médian, et ne pas frôler sa surface.
    """

    aimantation: float
    centre: Point
    rayon: float
    hauteur: float
    angle: float = 0.0
    spires: int = 64
    nature: ClassVar[Nature] = "B"

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre", en_point(self.centre))
        if self.rayon <= 0 or self.hauteur <= 0:
            raise ValueError("un aimant a un rayon et une hauteur positifs")
        if self.spires < 1:
            raise ValueError("un aimant équivaut à au moins une spire")

    def solenoide(self) -> Solenoide:
        """Le solénoïde équivalent : courant total M × hauteur, sur `spires` spires."""
        return Solenoide(
            self.aimantation * self.hauteur / self.spires,
            self.centre,
            self.rayon,
            self.hauteur,
            self.spires,
            self.angle,
        )

    def poles(self) -> tuple[Point, Point]:
        """Les centres des faces sud et nord (dans le sens de l'aimantation si M > 0)."""
        ux, uy = math.cos(self.angle), math.sin(self.angle)
        cx, cy = self.centre
        h = self.hauteur / 2
        sud, nord = (cx - h * ux, cy - h * uy), (cx + h * ux, cy + h * uy)
        return (sud, nord) if self.aimantation >= 0 else (nord, sud)

    def sommets(self) -> tuple[Point, Point, Point, Point]:
        """Les quatre coins du rectangle (le cylindre en coupe), dans l'ordre direct."""
        ux, uy = math.cos(self.angle), math.sin(self.angle)
        nx, ny = -uy, ux
        cx, cy = self.centre
        h, a = self.hauteur / 2, self.rayon
        return (
            (cx - h * ux - a * nx, cy - h * uy - a * ny),
            (cx + h * ux - a * nx, cy + h * uy - a * ny),
            (cx + h * ux + a * nx, cy + h * uy + a * ny),
            (cx - h * ux + a * nx, cy - h * uy + a * ny),
        )

    def lots(self) -> tuple[tuple[Noyau, tuple[float, ...]], ...]:
        return self.solenoide().lots()

    def singularites(self) -> tuple[Singularite, ...]:
        return self.solenoide().singularites()

    def axe(self) -> Axe | None:
        return Axe(self.centre, self.angle)

    def flux_revolution(self, x: np.ndarray | float, y: np.ndarray | float, axe: Axe) -> np.ndarray:
        return self.solenoide().flux_revolution(x, y, axe)
