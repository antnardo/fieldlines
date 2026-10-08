"""La carte de champ : des sources, un cadre, des lignes nommées, et leur tracé.

C'est l'objet des scripts de cours. On y place des lignes (autour des sources, selon
le flux, le long d'un segment, ou une à une), on retouche ou retire celles qui gênent
en les désignant par leur nom, puis on trace.

Les noms sont stables : les lignes placées automatiquement sont numérotées 0, 1, 2…
dans l'ordre de leur création, un numéro n'est jamais réutilisé, et retirer une ligne
ne renumérote pas les autres. On peut donc appeler `retire` autant de fois qu'on veut
(l'ancien `remove_SL` décalait les indices, d'où « ne pas appeler deux fois »). Une
ligne ajoutée à la main peut aussi recevoir un nom en texte.

Les lignes ne sont calculées qu'au besoin (au premier tracé, ou par `lignes`), et
recalculées seulement si l'on change leur départ ou un réglage qui modifie leur
tracé : changer les flèches ou le style ne coûte rien.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal, TypeAlias

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.contour import QuadContourSet

from fieldlines.champ import Distribution, Grille
from fieldlines.departs import PlacementSelonFlux, Repartition, departs_autour, departs_le_long
from fieldlines.geometrie import Cadre, Point, en_point
from fieldlines.integration import Integrateur, LigneDeChamp, Options, Reglages
from fieldlines.niveaux import Echelle, niveaux
from fieldlines.sources import Singularite, Source
from fieldlines.trace import (
    normalisation,
    position_sur_ligne,
    trace_contours,
    trace_fleche,
    trace_ligne,
    trace_nom,
    trace_sources,
)

__all__ = ["Carte", "Nom"]

Nom: TypeAlias = int | str


@dataclass(slots=True)
class _Entree:
    """Une ligne de la carte : son départ, ses réglages, et la ligne une fois calculée."""

    depart: Point
    options: Options
    ligne: LigneDeChamp | None = None


class Carte:
    """Les lignes de champ et les équipotentielles d'une distribution, dans un cadre.

    >>> from fieldlines import Carte, Charge
    >>> carte = Carte([Charge(1, (-1, 0)), Charge(-1, (1, 0))], cadre=(-3, 3, -3, 3))
    >>> carte.lignes_autour(nombre=4, rayon=0.2)
    [0, 1, 2, 3, 4, 5, 6, 7]
    >>> carte.retire(4, 5, 6, 7)
    >>> carte.ligne(0).aval.arret.value
    'arrivée sur une source'
    """

    def __init__(
        self,
        sources: Iterable[Source],
        cadre: Cadre | tuple[float, float, float, float],
        *,
        reglages: Reglages | None = None,
        verbose: bool = False,
    ) -> None:
        self.distribution = Distribution(sources)
        self.cadre = Cadre.depuis(cadre)
        self.verbose = verbose
        self._integrateur = Integrateur(self.distribution, self.cadre, reglages)
        self._entrees: dict[Nom, _Entree] = {}
        self._prochain = 0
        self._grilles: dict[tuple[str, int], Grille] = {}

    # ------------------------------------------------------------------ les lignes

    @property
    def sources(self) -> tuple[Source, ...]:
        return self.distribution.sources

    @property
    def noms(self) -> list[Nom]:
        """Les noms des lignes, dans l'ordre de leur création."""
        return list(self._entrees)

    def __len__(self) -> int:
        return len(self._entrees)

    def __contains__(self, nom: object) -> bool:
        return nom in self._entrees

    def lignes_autour(
        self, nombre: int = 16, rayon: float | None = None, angle: float = 0.0, **options: Any
    ) -> list[Nom]:
        """`nombre` lignes autour de chaque source où des lignes aboutissent.

        Charges : départs régulièrement espacés en angle sur un cercle de rayon
        `rayon`, le premier à l'angle `angle` (radians). Dipôles : départs réguliers en
        sin θ, θ compté depuis le moment. Les lignes sont numérotées source après
        source, dans l'ordre de la liste des sources. Une ligne qui va d'une charge à
        une autre est tracée depuis chacune : `retire` permet d'enlever le doublon,
        `lignes_selon_flux` de l'éviter.
        """
        rayon = self._rayon(rayon)
        reglages = Options().remplace(**options)
        noms = []
        for singularite in self._aboutissantes():
            for depart in departs_autour(singularite, nombre, rayon, angle):
                noms.append(self._ajoute(depart, reglages))
        return noms

    def lignes_selon_flux(
        self, nombre: int = 16, rayon: float | None = None, angle: float = 0.0, **options: Any
    ) -> list[Nom]:
        """Des lignes à flux égal, en nombre proportionnel aux charges, sans doublon.

        La plus forte charge reçoit `nombre` lignes, les autres en proportion de leur
        charge (au moins une) ; chaque dipôle en reçoit `nombre`. Les lignes qui vont
        d'une source à une autre ne sont tracées qu'une fois. Voir `fieldlines.departs`.
        """
        rayon = self._rayon(rayon)
        self._aboutissantes()
        reglages = Options().remplace(**options)
        placement = PlacementSelonFlux(self._integrateur)
        return [
            self._ajoute(depart.position, reglages, depart.ligne)
            for depart in placement.places(nombre, rayon, angle, reglages)
        ]

    def lignes_le_long(
        self,
        debut: Point,
        fin: Point,
        nombre: int,
        repartition: Repartition = "flux",
        **options: Any,
    ) -> list[Nom]:
        """`nombre` lignes partant du segment [debut, fin].

        `repartition="flux"` : même flux du champ de la figure entre deux lignes
        voisines (densité des lignes ∝ intensité du champ) ; `"flux_revolution"` : même
        flux dans l'espace, pour des sources de révolution ; `"reguliere"` : départs
        équidistants. C'est la façon de placer les lignes du champ magnétique, qui
        n'aboutissent nulle part : un segment qui part d'un fil, ou qui traverse une
        spire. Le segment ne doit pas passer sur une source.
        """
        reglages = Options().remplace(**options)
        departs = departs_le_long(self.distribution, debut, fin, nombre, repartition)
        return [self._ajoute(depart, reglages) for depart in departs]

    def ajoute_ligne(self, depart: Point, nom: Nom | None = None, **options: Any) -> Nom:
        """Ajoute la ligne qui passe par `depart` ; rend son nom (numéro, ou `nom`)."""
        if nom is not None and nom in self._entrees:
            raise ValueError(f"il y a déjà une ligne nommée {nom!r}")
        return self._ajoute(en_point(depart), Options().remplace(**options), nom=nom)

    def ajuste(
        self, *noms: Nom | Iterable[Nom], depart: Point | None = None, **options: Any
    ) -> None:
        """Change le départ ou les réglages (voir `Options`) des lignes nommées.

        `carte.ajuste(30, 38, fleches=[0.45])`. La ligne n'est recalculée que si son
        départ ou un réglage d'intégration change.
        """
        for nom in _aplatis(noms):
            entree = self._entree(nom)
            nouvelles = entree.options.remplace(**options)
            if depart is not None:
                entree.depart = en_point(depart)
                entree.ligne = None
            if nouvelles.integration != entree.options.integration:
                entree.ligne = None
            entree.options = nouvelles

    def retire(self, *noms: Nom | Iterable[Nom]) -> None:
        """Retire les lignes nommées : `carte.retire(2, 6, range(7, 13))`."""
        a_retirer = _aplatis(noms)
        for nom in a_retirer:
            self._entree(nom)
        for nom in a_retirer:
            del self._entrees[nom]

    def ligne(self, nom: Nom) -> LigneDeChamp:
        """La ligne nommée, calculée au besoin."""
        entree = self._entree(nom)
        if entree.ligne is None:
            entree.ligne = self._integrateur.integre(entree.depart, entree.options)
            if self.verbose:
                print(f"ligne {nom} : {entree.ligne.resume()}")
        return entree.ligne

    def options(self, nom: Nom) -> Options:
        """Les réglages de la ligne nommée."""
        return self._entree(nom).options

    def lignes(self) -> dict[Nom, LigneDeChamp]:
        """Toutes les lignes, calculées au besoin."""
        return {nom: self.ligne(nom) for nom in self._entrees}

    # ------------------------------------------------- potentiel, flux et niveaux

    def grille(
        self, grandeur: Literal["potentiel", "flux"] = "potentiel", points: int = 512
    ) -> Grille:
        """Le potentiel (champ électrique) ou la fonction de flux sur une grille du cadre."""
        cle = (grandeur, points)
        if cle not in self._grilles:
            self._grilles[cle] = self.distribution.grille(self.cadre, points, grandeur)
        return self._grilles[cle]

    def niveaux(
        self,
        nombre: int = 20,
        echelle: Echelle = "lineaire",
        symetrique: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        seuil: float = 1e-3,
        ajoute: Iterable[float] = (),
        points: int = 512,
    ) -> np.ndarray:
        """Des niveaux d'équipotentielles (voir `fieldlines.niveaux`).

        Sans `vmin` ni `vmax`, les bornes sont les extrêmes finis du potentiel sur la
        grille du cadre.
        """
        if vmin is None or vmax is None:
            grille = self.grille("potentiel", points)
            vmin = grille.minimum if vmin is None else vmin
            vmax = grille.maximum if vmax is None else vmax
        return niveaux(vmin, vmax, nombre, echelle, symetrique, seuil, ajoute)

    # ------------------------------------------------------------------- le tracé

    def trace_lignes(self, ax: Axes, **style: Any) -> None:
        """Trace toutes les lignes (`style` : arguments de `Axes.plot`).

        Le style propre d'une ligne (`ajuste(nom, style={...})`) l'emporte.
        """
        for nom, ligne in self.lignes().items():
            trace_ligne(ax, ligne, **{**style, **self._entrees[nom].options.style})

    def trace_fleches(
        self, ax: Axes, taille: float = 8.0, style: str = "fancy", **reglages: Any
    ) -> None:
        """Les flèches de chaque ligne, aux fractions de sa longueur visible.

        `taille` en points, `style` un arrowstyle de matplotlib (`fancy`, `-|>`…),
        `reglages` des arguments de `FancyArrowPatch` (couleur…). Une ligne de couleur
        propre a des flèches de sa couleur.
        """
        for nom, ligne in self.lignes().items():
            options = self._entrees[nom].options
            couleur = {"color": options.style["color"]} if "color" in options.style else {}
            for fraction in options.fleches:
                position = position_sur_ligne(ligne, fraction, self.cadre)
                if position is not None:
                    trace_fleche(
                        ax, *position, taille=taille, style=style, **{**reglages, **couleur}
                    )

    def trace_noms(self, ax: Axes, fraction: float = 0.5, **reglages: Any) -> None:
        """Écrit le nom de chaque ligne en son milieu visible : pour choisir quoi retoucher."""
        for nom, ligne in self.lignes().items():
            position = position_sur_ligne(ligne, fraction, self.cadre)
            if position is not None:
                trace_nom(ax, position[0], nom, **reglages)

    def trace_equipotentielles(
        self,
        ax: Axes,
        niveaux: Sequence[float] | np.ndarray | None = None,
        contraste: float = 3.0,
        cmap: str | Colormap = "RdYlBu_r",
        norm: Normalize | None = None,
        points: int = 512,
        **reglages: Any,
    ) -> QuadContourSet:
        """Les équipotentielles, colorées par `cmap`.

        Sans `norm`, les couleurs couvrent [Vmin, Vmax]/10^contraste (Vmin, Vmax : les
        extrêmes finis de la grille) : augmenter le contraste distingue les niveaux
        faibles. Passer le `norm` d'un premier tracé (`cs.norm`) pour comparer deux
        cartes avec les mêmes couleurs. Rend le `QuadContourSet` (pour `ax.clabel`).
        """
        grille = self.grille("potentiel", points)
        if niveaux is None:
            niveaux = self.niveaux(points=points)
        echelle = norm if norm is not None else normalisation(grille, contraste)
        return trace_contours(ax, grille, niveaux, cmap=cmap, norm=echelle, **reglages)

    def trace_lignes_de_flux(
        self,
        ax: Axes,
        niveaux: int | Sequence[float] | np.ndarray = 20,
        points: int = 400,
        **reglages: Any,
    ) -> QuadContourSet:
        """Les lignes de champ comme lignes de niveau de la fonction de flux Ψ.

        Exact pour des fils de courant (Ψ = A_z) et pour des sources de révolution
        (Ψ = ρ A_φ pour des spires) : des niveaux également espacés donnent des tubes de
        flux égal. `niveaux` : un nombre de niveaux régulièrement espacés entre les
        extrêmes de Ψ sur la grille, ou la liste des niveaux. Noir par défaut.
        """
        grille = self.grille("flux", points)
        if isinstance(niveaux, int):
            niveaux = np.linspace(grille.minimum, grille.maximum, niveaux + 2)[1:-1]
        return trace_contours(ax, grille, niveaux, **{"colors": "k", **reglages})

    def trace_sources(self, ax: Axes, taille: float = 10.0) -> None:
        """Les sources : charges (aire ∝ |q|), dipôles, courants ⊙ et ⊗ (taille en points)."""
        trace_sources(ax, self.sources, taille)

    def trace(
        self, ax: Axes, taille_fleches: float = 8.0, taille_sources: float = 10.0, **style: Any
    ) -> None:
        """Lignes, flèches et sources d'un coup, et un repère orthonormé limité au cadre."""
        ax.set_xlim(self.cadre.xmin, self.cadre.xmax)
        ax.set_ylim(self.cadre.ymin, self.cadre.ymax)
        ax.set_aspect("equal")
        self.trace_lignes(ax, **style)
        self.trace_fleches(ax, taille=taille_fleches)
        self.trace_sources(ax, taille=taille_sources)

    # ------------------------------------------------------------------- en interne

    def _ajoute(
        self,
        depart: Point,
        options: Options,
        ligne: LigneDeChamp | None = None,
        nom: Nom | None = None,
    ) -> Nom:
        if nom is None:
            nom = self._prochain
            self._prochain += 1
        self._entrees[nom] = _Entree(en_point(depart), options, ligne)
        return nom

    def _entree(self, nom: Nom) -> _Entree:
        try:
            return self._entrees[nom]
        except KeyError:
            raise KeyError(f"pas de ligne nommée {nom!r} (lignes : {self.noms})") from None

    def _aboutissantes(self) -> list[Singularite]:
        singularites = [s for s in self.distribution.singularites if s.aboutissante]
        if not singularites:
            raise ValueError(
                "aucune source où les lignes commencent ou finissent : placer les lignes "
                "avec lignes_le_long ou ajoute_ligne"
            )
        return singularites

    def _rayon(self, rayon: float | None) -> float:
        """Le rayon des cercles de départ : par défaut le dixième de la plus petite
        distance entre deux sources, sans dépasser 2 % de la diagonale du cadre."""
        if rayon is not None:
            if rayon <= 0:
                raise ValueError("le rayon des départs doit être positif")
            return float(rayon)
        distance = self.distribution.distance_minimale(self._aboutissantes())
        return min(0.1 * distance, 0.02 * self.cadre.diagonale)


def _aplatis(noms: Iterable[Any]) -> list[Nom]:
    """Les noms donnés un à un ou en listes (`range`, listes d'indices), sans doublon."""
    resultat: list[Nom] = []
    for nom in noms:
        if isinstance(nom, str):
            resultat.append(nom)
        elif isinstance(nom, Integral):
            resultat.append(int(nom))
        else:
            resultat.extend(_aplatis(nom))
    return list(dict.fromkeys(resultat))
