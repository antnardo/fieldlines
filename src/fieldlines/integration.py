"""Le calcul d'une ligne de champ : `solve_ivp` et ses événements d'arrêt.

On suit la direction du champ, dr/dt = v F/|F|, depuis un point de départ, vers l'aval
puis vers l'amont. Quatre événements arrêtent l'intégration :

- la sortie du cadre (agrandi d'une marge) ;
- l'arrivée sur une singularité (charge, dipôle…) : le point de la source est alors
  ajouté au bout de la ligne, qui la touche ;
- la ligne refermée : la ligne repasse au point de départ dans le même sens ;
- le champ nul : la ligne tombe sur un point d'arrêt (point col entre deux charges).

Pourquoi `solve_ivp` plutôt que l'ancien `ode` à pas fixe : les événements sont
localisés exactement sur la sortie dense du schéma, au lieu d'être constatés après
coup à la précision du pas, et le pas s'adapte à la courbure. Le facteur v vaut 1 sauf
près d'une singularité ou d'un champ presque nul, où il décroît comme la distance (ou
comme le champ) : l'approche devient exponentielle, si bien que le schéma ne peut
plus enjamber la source en un pas ni osciller autour d'un point col. Le paramètre t
majore donc l'abscisse curviligne, sans lui être égal ; on rééchantillonne à la fin
en abscisse curviligne.

Le retour au départ se détecte par le signe de (r − r0)·t0 (t0 la tangente initiale),
qui change à coup sûr quand la ligne repasse, même si le pas est grand ; la distance au
départ, elle, pourrait passer sous un seuil entre deux pas sans qu'on le voie.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from typing import Any, Literal, TypeAlias

import numpy as np
from scipy.integrate import solve_ivp

from fieldlines.champ import Distribution
from fieldlines.geometrie import Cadre, Point, en_point

__all__ = ["Arret", "Extremite", "Integrateur", "LigneDeChamp", "Options", "Reglages", "Sens"]

_VIRAGE_MAXIMAL = math.radians(5.0)

Sens: TypeAlias = Literal["deux", "aval", "amont"]


class Arret(Enum):
    """Pourquoi une ligne s'arrête à l'un de ses bouts."""

    SORTIE = "sortie du cadre"
    SOURCE = "arrivée sur une source"
    BOUCLE = "ligne refermée"
    CHAMP_NUL = "point de champ nul"
    LONGUEUR = "longueur maximale atteinte"
    ECHEC = "échec du solveur"
    DEPART = "pas de calcul de ce côté"


@dataclass(frozen=True, slots=True)
class Extremite:
    """Un bout de ligne : la raison de l'arrêt, et la singularité atteinte s'il y en a une.

    `singularite` est l'indice dans `Distribution.singularites`.
    """

    arret: Arret
    singularite: int | None = None


@dataclass(frozen=True, slots=True)
class Options:
    """Les réglages propres à une ligne.

    - `fleches` : positions des flèches, en fractions de la longueur visible (dans le
      cadre) de la ligne, comptées de l'amont vers l'aval ;
    - `marge` : de combien on agrandit le cadre (en fraction de ses côtés) avant
      d'arrêter le calcul d'une ligne qui sort ;
    - `sens` : `deux` (de part et d'autre du départ), `aval` ou `amont` seulement ;
    - `longueur_max` : longueur au-delà de laquelle on abandonne (par défaut vingt fois
      la diagonale du cadre agrandi) ;
    - `style` : arguments de `Axes.plot` propres à cette ligne (couleur, tirets…).
    """

    fleches: tuple[float, ...] = (0.5,)
    marge: float = 0.25
    sens: Sens = "deux"
    longueur_max: float | None = None
    style: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fleches", tuple(float(f) for f in self.fleches))
        object.__setattr__(self, "style", dict(self.style))
        if self.sens not in ("deux", "aval", "amont"):
            raise ValueError(f"sens inconnu : {self.sens!r}")
        if any(not 0.0 <= f <= 1.0 for f in self.fleches):
            raise ValueError("les flèches se placent à des fractions entre 0 et 1")

    def remplace(self, **changements: Any) -> "Options":
        """Une copie modifiée ; un nom de réglage inconnu lève TypeError."""
        connus = {f.name for f in fields(self)}
        inconnus = set(changements) - connus
        if inconnus:
            raise TypeError(f"réglages de ligne inconnus : {', '.join(sorted(inconnus))}")
        return replace(self, **changements)

    @property
    def integration(self) -> tuple[float, Sens, float | None]:
        """Ce qui change le tracé de la ligne elle-même (et oblige à la recalculer)."""
        return (self.marge, self.sens, self.longueur_max)


@dataclass(frozen=True, slots=True)
class Reglages:
    """La précision du calcul, commune à toutes les lignes d'une carte.

    - `tolerance` : tolérance relative du schéma de Runge-Kutta ;
    - `pas` : espacement des points de la ligne, en fraction de la diagonale du cadre ;
    - `arret` : rayon d'arrêt autour d'une singularité, en fraction de la diagonale
      (réduit pour une ligne qui part plus près d'elle) ;
    - `champ_nul` : champ, en fraction du champ médian du cadre, sous lequel on
      considère la ligne arrivée sur un point d'arrêt ;
    - `ralenti` : champ, en même fraction, sous lequel on ralentit l'intégration.
    """

    tolerance: float = 1e-6
    pas: float = 1 / 1500
    arret: float = 1e-3
    champ_nul: float = 1e-7
    ralenti: float = 1e-4


@dataclass(frozen=True, slots=True)
class LigneDeChamp:
    """Une ligne calculée : ses points, rangés dans le sens du champ, et ses deux bouts."""

    points: np.ndarray
    depart: Point
    amont: Extremite
    aval: Extremite

    @property
    def fermee(self) -> bool:
        return self.aval.arret is Arret.BOUCLE

    def abscisses(self) -> np.ndarray:
        """L'abscisse curviligne de chaque point, depuis le bout amont."""
        pas = np.hypot(*np.diff(self.points, axis=0).T)
        return np.concatenate(([0.0], np.cumsum(pas)))

    @property
    def longueur(self) -> float:
        return float(self.abscisses()[-1])

    def resume(self) -> str:
        """Une ligne de texte pour le mode bavard."""
        if self.fermee:
            return f"fermée, longueur {self.longueur:.3g}"
        return (
            f"longueur {self.longueur:.3g} ; amont : {_decrit(self.amont)} ; "
            f"aval : {_decrit(self.aval)}"
        )


def _decrit(extremite: Extremite) -> str:
    if extremite.singularite is None:
        return extremite.arret.value
    return f"{extremite.arret.value} n° {extremite.singularite}"


@dataclass(frozen=True, slots=True)
class _DemiLigne:
    points: np.ndarray
    fin: Extremite


class Integrateur:
    """Calcule les lignes de champ d'une distribution dans un cadre."""

    def __init__(
        self, distribution: Distribution, cadre: Cadre, reglages: Reglages | None = None
    ) -> None:
        self._distribution = distribution
        self._cadre = cadre
        self._reglages = reglages or Reglages()
        diagonale = cadre.diagonale
        self._pas = self._reglages.pas * diagonale
        self._rayon_arret = self._reglages.arret * diagonale
        reference = distribution.champ_de_reference(cadre)
        self._champ_nul = self._reglages.champ_nul * reference
        self._ralenti = self._reglages.ralenti * reference
        positions = [s.position for s in distribution.singularites]
        self._positions = np.array(positions, dtype=float).reshape(-1, 2)
        self._aboutissantes = [s.aboutissante for s in distribution.singularites]

    @property
    def distribution(self) -> Distribution:
        return self._distribution

    @property
    def cadre(self) -> Cadre:
        return self._cadre

    def rayon_arret(self, point: Point) -> float:
        """Le rayon d'arrêt pour une ligne partant de `point` : jamais plus de la moitié
        de sa distance à la singularité la plus proche, pour ne pas partir dedans."""
        return min(self._rayon_arret, 0.5 * self._distance_singularites(point))

    def integre(self, depart: Point, options: Options | None = None) -> LigneDeChamp:
        """La ligne de champ qui passe par `depart`."""
        options = options or Options()
        depart = en_point(depart)
        domaine = self._cadre.elargi(options.marge)
        longueur_max = options.longueur_max or 20.0 * domaine.diagonale
        rayon = self.rayon_arret(depart)
        seul = np.array([depart])
        if domaine.distance_interieure(*depart) <= 0:
            fin = Extremite(Arret.SORTIE)
            return LigneDeChamp(seul, depart, fin, fin)
        if rayon <= 0:
            fin = Extremite(Arret.SOURCE, self._plus_proche(depart))
            return LigneDeChamp(seul, depart, fin, fin)

        rien = _DemiLigne(seul, Extremite(Arret.DEPART))
        aval = rien
        if options.sens != "amont":
            aval = self._demi_ligne(depart, 1.0, rayon, domaine, longueur_max)
            if aval.fin.arret is Arret.BOUCLE:
                return LigneDeChamp(aval.points, depart, aval.fin, aval.fin)
        amont = rien
        if options.sens != "aval":
            amont = self._demi_ligne(depart, -1.0, rayon, domaine, longueur_max)
            if amont.fin.arret is Arret.BOUCLE:
                points = amont.points[::-1]
                return LigneDeChamp(points, depart, amont.fin, amont.fin)
        points = np.concatenate((amont.points[::-1], aval.points[1:]))
        return LigneDeChamp(points, depart, amont.fin, aval.fin)

    # ------------------------------------------------------------------- en interne

    def _distance_singularites(self, point: Point) -> float:
        if not len(self._positions):
            return math.inf
        return float(np.min(np.hypot(*(self._positions - point).T)))

    def _plus_proche(self, point: Point) -> int:
        return int(np.argmin(np.hypot(*(self._positions - point).T)))

    def _demi_ligne(
        self, depart: Point, signe: float, rayon: float, domaine: Cadre, longueur_max: float
    ) -> _DemiLigne:
        champ = self._distribution.champ_point
        fx0, fy0 = champ(*depart)
        norme0 = math.hypot(fx0, fy0)
        if not (math.isfinite(norme0) and norme0 > self._champ_nul):
            return _DemiLigne(np.array([depart]), Extremite(Arret.CHAMP_NUL))
        tx, ty = signe * fx0 / norme0, signe * fy0 / norme0
        positions = self._positions
        approche = 4.0 * rayon
        ralenti, nul = self._ralenti, self._champ_nul
        x0, y0 = depart

        def vitesse(_t: float, r: np.ndarray) -> list[float]:
            fx, fy = champ(r[0], r[1])
            norme = math.hypot(fx, fy)
            if not (math.isfinite(norme) and norme > 0):
                return [0.0, 0.0]
            v = signe / norme
            if norme < ralenti:
                v *= norme / ralenti
            if len(positions):
                d = float(np.min(np.hypot(positions[:, 0] - r[0], positions[:, 1] - r[1])))
                if d < approche:
                    v *= d / approche
            return [fx * v, fy * v]

        def sortie(_t: float, r: np.ndarray) -> float:
            return domaine.distance_interieure(r[0], r[1])

        def source(_t: float, r: np.ndarray) -> float:
            d = np.hypot(positions[:, 0] - r[0], positions[:, 1] - r[1])
            return float(np.min(d)) - rayon

        def champ_nul(_t: float, r: np.ndarray) -> float:
            return math.hypot(*champ(r[0], r[1])) - nul

        # Retour au départ : p = (r − r0)·t0 passe de − à + quand la ligne repasse au
        # départ dans le même sens. On ne le guette que dans une bande |q| < largeur
        # autour de la tangente initiale (q, l'écart perpendiculaire), sous une forme
        # continue, max(p + δ, |q| − largeur), pour que la recherche de racine reste
        # sûre ; le léger décalage δ évite un déclenchement au départ même. La bande est
        # aussi large que le plus grand pas : un pas qui arrive de derrière le départ
        # est dans la bande avant de le franchir. Un franchissement loin du départ
        # (plus de `tolerance_retour`) n'est pas une boucle : on poursuit.
        decalage = 1e-9 * self._cadre.diagonale
        pas_maximal = 0.05 * self._cadre.diagonale
        tolerance_retour = 1e-3 * self._cadre.diagonale

        def retour(_t: float, r: np.ndarray) -> float:
            p = (r[0] - x0) * tx + (r[1] - y0) * ty
            q = (r[1] - y0) * tx - (r[0] - x0) * ty
            return max(p + decalage, abs(q) - pas_maximal)

        evenements: list[Callable[[float, np.ndarray], float]] = [sortie, champ_nul, retour]
        if len(positions):
            evenements.append(source)
        for evenement in evenements:
            evenement.terminal = True  # type: ignore[attr-defined]
            evenement.direction = 1 if evenement is retour else -1  # type: ignore[attr-defined]
        sans_retour = [e for e in evenements if e is not retour]

        morceaux: list[np.ndarray] = []
        t, r = 0.0, np.array(depart, dtype=float)
        suspendu_jusqua: float | None = None
        while True:
            actifs = evenements if suspendu_jusqua is None else sans_retour
            fin_t = longueur_max if suspendu_jusqua is None else min(suspendu_jusqua, longueur_max)
            solution = solve_ivp(
                vitesse,
                (t, fin_t),
                r,
                method="RK45",
                rtol=self._reglages.tolerance,
                atol=1e-10 * self._cadre.diagonale,
                max_step=pas_maximal,
                events=actifs,
                dense_output=True,
            )
            morceaux.append(self._echantillonne(solution))
            t, r = float(solution.t[-1]), solution.y[:, -1]
            if solution.status == -1:
                fin = Extremite(Arret.ECHEC)
                break
            if solution.status == 0:
                if t < longueur_max:
                    suspendu_jusqua = None
                    continue
                fin = Extremite(Arret.LONGUEUR)
                break
            survenu = next(e for e, ts in zip(actifs, solution.t_events, strict=True) if ts.size)
            if survenu is sortie:
                fin = Extremite(Arret.SORTIE)
                break
            if survenu is champ_nul:
                fin = Extremite(Arret.CHAMP_NUL)
                break
            if survenu is retour:
                if math.hypot(r[0] - x0, r[1] - y0) < tolerance_retour:
                    morceaux.append(np.array([depart]))
                    fin = Extremite(Arret.BOUCLE)
                    break
                # Passage loin du départ : on poursuit un moment sans guetter le retour,
                # faute de quoi l'événement, nul à la reprise, se déclencherait aussitôt.
                suspendu_jusqua = t + 2.0 * pas_maximal
                continue
            indice = self._plus_proche((float(r[0]), float(r[1])))
            if self._aboutissantes[indice]:
                morceaux.append(positions[indice][None, :])
            fin = Extremite(Arret.SOURCE, indice)
            break
        return _DemiLigne(_sans_doublons(np.concatenate(morceaux)), fin)

    def _echantillonne(self, solution: Any) -> np.ndarray:
        """Les points de la solution, assez serrés pour un tracé lisse.

        Chaque pas du schéma est redécoupé par la sortie dense : jamais deux points
        plus espacés que `pas`, ni une direction du champ qui tourne de plus de 5°
        entre deux points. Le schéma, lui, fait de grands pas là où la précision le
        permet, trop grands pour une petite boucle autour d'un fil.
        """
        ts, ys = solution.t, solution.y
        if ts.size < 2:
            return ys.T.copy()
        cordes = np.hypot(*np.diff(ys, axis=1))
        fx, fy = self._distribution.champ(ys[0], ys[1])
        directions = np.unwrap(np.arctan2(fy, fx))
        virages = np.abs(np.diff(directions))
        virages = np.where(np.isfinite(virages), virages, 0.0)
        nombres = np.maximum.reduce(
            [
                np.ones(len(cordes), dtype=int),
                np.ceil(cordes / self._pas).astype(int),
                np.ceil(virages / _VIRAGE_MAXIMAL).astype(int),
            ]
        )
        instants = [
            np.linspace(a, b, n, endpoint=False)
            for a, b, n in zip(ts[:-1], ts[1:], nombres, strict=True)
        ]
        instants.append(ts[-1:])
        return solution.sol(np.concatenate(instants)).T


def _sans_doublons(points: np.ndarray) -> np.ndarray:
    """Retire les points répétés à la suite (les jointures entre deux appels du solveur)."""
    if len(points) < 2:
        return points
    garde = np.concatenate(([True], np.any(np.diff(points, axis=0) != 0, axis=1)))
    return points[garde]
