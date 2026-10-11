"""Le tracé : lignes, flèches, noms des lignes, équipotentielles, sources.

Les tailles (flèches, symboles des sources) sont en points typographiques, et non plus
en unités de la figure comme dans l'ancien module : une même taille donne le même
rendu quel que soit le cadre.

Les flèches se placent à une fraction de la longueur *visible* de la ligne (la partie
dans le cadre) et s'orientent selon le segment de la ligne qui les porte, parcouru
dans le sens du champ. L'ancien module prenait la moyenne des points suivants, qui
n'existent plus en bout de ligne : une flèche placée à 0,999 pointait à rebours.
Chaque flèche est un patch ajouté aux axes, et non plus un élément de
`PatchCollection` : la collection figeait la forme de la flèche dans les unités de la
figure.
"""

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import transforms
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon

from fieldlines.champ import Grille
from fieldlines.geometrie import Cadre, Point
from fieldlines.integration import LigneDeChamp
from fieldlines.sources import (
    Aimant,
    Charge,
    Dipole,
    DipoleMagnetique,
    FilCharge,
    FilCourant,
    NappeCourant,
    PlanCharge,
    Solenoide,
    Source,
    Spire,
)

__all__ = [
    "normalisation",
    "position_sur_ligne",
    "trace_contours",
    "trace_fleche",
    "trace_ligne",
    "trace_nom",
    "trace_sources",
]

ROUGE = "r"
BLEU = "b"


def trace_ligne(ax: Axes, ligne: LigneDeChamp, **style: Any) -> Line2D:
    """Trace une ligne ; noir et trait plein par défaut."""
    reglages = {"color": "k", **style}
    (trait,) = ax.plot(ligne.points[:, 0], ligne.points[:, 1], **reglages)
    return trait


def position_sur_ligne(
    ligne: LigneDeChamp, fraction: float, cadre: Cadre
) -> tuple[Point, Point] | None:
    """Le point à `fraction` de la longueur visible, et la tangente unitaire (sens du champ).

    None si aucun segment de la ligne n'est entièrement dans le cadre.
    """
    points = ligne.points
    if len(points) < 2:
        return None
    pas = np.hypot(*np.diff(points, axis=0).T)
    dedans = cadre.contient(points[:, 0], points[:, 1])
    visibles = np.where(dedans[:-1] & dedans[1:], pas, 0.0)
    cumul = np.concatenate(([0.0], np.cumsum(visibles)))
    total = cumul[-1]
    if total <= 0:
        return None
    # Une cible strictement positive tombe dans un segment visible : searchsorted à
    # gauche saute les paliers des parties hors cadre.
    cible = min(max(fraction * total, 1e-12 * total), total)
    k = int(np.clip(np.searchsorted(cumul, cible, side="left"), 1, len(points) - 1))
    w = (cible - cumul[k - 1]) / pas[k - 1]
    a, b = points[k - 1], points[k]
    x, y = a + w * (b - a)
    tx, ty = (b - a) / pas[k - 1]
    return (float(x), float(y)), (float(tx), float(ty))


def trace_fleche(
    ax: Axes,
    point: Point,
    direction: Point,
    taille: float = 8.0,
    style: str = "fancy",
    **reglages: Any,
) -> FancyArrowPatch:
    """Une flèche de `taille` points, pointe en `point`, dans la direction `direction`.

    Le segment porteur est minuscule : seule sa direction compte, la forme de la
    flèche vient de `style` (un arrowstyle de matplotlib) et de `taille`.
    """
    x, y = point
    longueur = 1e-6 * (abs(x) + abs(y) + 1.0)
    tx, ty = direction
    arguments = {"color": "k", "zorder": 9, "shrinkA": 0, "shrinkB": 0, **reglages}
    fleche = FancyArrowPatch(
        (x - longueur * tx, y - longueur * ty),
        (x, y),
        arrowstyle=style,
        mutation_scale=taille,
        **arguments,
    )
    ax.add_patch(fleche)
    return fleche


def trace_nom(ax: Axes, position: Point, nom: object, **reglages: Any) -> None:
    """Écrit le nom d'une ligne à côté de `position` (décalé de quelques points)."""
    arguments = {"fontsize": 8, "color": "0.25", "zorder": 11, **reglages}
    ax.annotate(str(nom), position, xytext=(3, 3), textcoords="offset points", **arguments)


def normalisation(grille: Grille, contraste: float) -> mcolors.Normalize:
    """L'échelle de couleurs des équipotentielles.

    Les couleurs couvrent [Vmin, Vmax]/10^contraste, Vmin et Vmax étant les extrêmes
    finis de la grille : plus le contraste est grand, plus les niveaux faibles se
    distinguent, les forts saturant la palette. C'est le réglage de l'ancien module.
    """
    echelle = 10.0**contraste
    return mcolors.Normalize(vmin=grille.minimum / echelle, vmax=grille.maximum / echelle)


def trace_contours(
    ax: Axes,
    grille: Grille,
    niveaux: Iterable[float] | None = None,
    **reglages: Any,
) -> QuadContourSet:
    """Les lignes de niveau d'une grille (équipotentielles, lignes de flux)."""
    gx, gy = np.meshgrid(grille.x, grille.y)
    if niveaux is not None:
        reglages["levels"] = np.asarray(list(niveaux), dtype=float)
    return ax.contour(gx, gy, grille.valeurs, **reglages)


# ----------------------------------------------------------------------------- sources


def trace_sources(ax: Axes, sources: Iterable[Source], taille: float = 10.0) -> None:
    """Dessine chaque source : disque rouge ou bleu pour une charge (aire ∝ |q|),
    flèche pour un dipôle, ⊙ ou ⊗ pour un courant qui traverse la figure, droite pour
    un plan, rectangle aux pôles N et S pour un aimant ; rien pour un champ uniforme."""
    for source in sources:
        if isinstance(source, (Charge, FilCharge)):
            valeur = source.q if isinstance(source, Charge) else source.lineique
            _disque(ax, source.position, valeur, taille)
        elif isinstance(source, (Dipole, DipoleMagnetique)):
            fond = ROUGE if isinstance(source, Dipole) else "white"
            _dipole(ax, source.position, source.angle, taille, fond)
        elif isinstance(source, FilCourant):
            _courant(ax, source.position, source.intensite, taille)
        elif isinstance(source, Spire):
            _spire(ax, source, taille, profil=True)
        elif isinstance(source, Solenoide):
            for spire in source.composantes():
                _spire(ax, spire, 0.7 * taille, profil=False)
        elif isinstance(source, Aimant):
            _aimant(ax, source, taille)
        elif isinstance(source, PlanCharge):
            _droite(ax, source.position, source.angle, ROUGE if source.surfacique > 0 else BLEU)
        elif isinstance(source, NappeCourant):
            _droite(ax, source.position, source.angle, "k")
            _courant(ax, source.position, source.surfacique, taille)


def _disque(ax: Axes, position: Point, valeur: float, taille: float) -> None:
    couleur = ROUGE if valeur > 0 else BLEU
    ax.plot(
        *position,
        "o",
        markersize=taille * math.sqrt(abs(valeur)),
        markerfacecolor=couleur,
        markeredgecolor="0.4",
        zorder=10,
    )


def _dipole(ax: Axes, position: Point, angle: float, taille: float, fond: str) -> None:
    """Un disque de `taille` points, et une flèche noire de 1,4 × taille qui le traverse.

    Le disque coloré garde le dipôle visible au milieu des lignes qui y convergent.
    """
    ax.plot(
        *position, "o", markersize=taille, markerfacecolor=fond, markeredgecolor="0.4", zorder=10
    )
    echelle = ax.figure.dpi_scale_trans
    dx, dy = 0.7 * taille * math.cos(angle) / 72, 0.7 * taille * math.sin(angle) / 72
    pointe = ax.transData + transforms.ScaledTranslation(dx, dy, echelle)
    queue = ax.transData + transforms.ScaledTranslation(-dx, -dy, echelle)
    ax.annotate(
        "",
        xy=position,
        xycoords=pointe,
        xytext=position,
        textcoords=queue,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "k",
            "lw": 1.2,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 0.6 * taille,
        },
        zorder=11,
    )


def _courant(ax: Axes, position: Point, intensite: float, taille: float) -> None:
    """⊙ si le courant vient vers le lecteur (I > 0), ⊗ sinon."""
    ax.plot(
        *position, "o", markersize=taille, markerfacecolor="white", markeredgecolor="k", zorder=10
    )
    if intensite > 0:
        ax.plot(*position, "o", markersize=0.25 * taille, color="k", zorder=11)
    else:
        ax.plot(*position, "x", markersize=0.6 * taille, color="k", markeredgewidth=1.2, zorder=11)


def _spire(ax: Axes, spire: Spire, taille: float, profil: bool) -> None:
    sortant, entrant = spire.passages()
    if profil:
        ax.plot(*zip(sortant, entrant, strict=True), color="0.5", lw=1, ls="--", zorder=9)
    _courant(ax, sortant, spire.intensite, taille)
    _courant(ax, entrant, -spire.intensite, taille)


def _droite(ax: Axes, position: Point, angle: float, couleur: str) -> None:
    """Un plan vu par la tranche : une droite qui traverse toute la figure."""
    x, y = position
    ax.axline(position, (x + math.cos(angle), y + math.sin(angle)), color=couleur, lw=2, zorder=10)


def _aimant(ax: Axes, aimant: Aimant, taille: float) -> None:
    """Le cylindre en coupe : deux moitiés colorées, marquées S et N, sous les lignes.

    Le fond pâle (zorder 1) laisse voir les lignes qui traversent l'aimant, le contour
    et les lettres passent au-dessus.
    """
    s1, s2, s3, s4 = aimant.sommets()  # s1 → s2 : le long de l'axe, côté −n ; s4 → s3 : côté +n
    milieu_bas = (0.5 * (s1[0] + s2[0]), 0.5 * (s1[1] + s2[1]))
    milieu_haut = (0.5 * (s4[0] + s3[0]), 0.5 * (s4[1] + s3[1]))
    sud, nord = aimant.poles()
    arriere, avant = (s1, s4), (s2, s3)  # les deux faces : vers −u, vers +u
    sud_vers_moins_u = aimant.aimantation >= 0
    moitie_arriere = [arriere[0], milieu_bas, milieu_haut, arriere[1]]
    moitie_avant = [milieu_bas, avant[0], avant[1], milieu_haut]
    moitie_sud, moitie_nord = (
        (moitie_arriere, moitie_avant) if sud_vers_moins_u else (moitie_avant, moitie_arriere)
    )
    ax.add_patch(Polygon(moitie_sud, closed=True, fc="#c6dbef", ec="none", zorder=1))
    ax.add_patch(Polygon(moitie_nord, closed=True, fc="#fcbba1", ec="none", zorder=1))
    ax.add_patch(Polygon([s1, s2, s3, s4], closed=True, fc="none", ec="k", lw=1.2, zorder=10))
    for lettre, pole in (("S", sud), ("N", nord)):
        x = 0.5 * (pole[0] + aimant.centre[0])
        y = 0.5 * (pole[1] + aimant.centre[1])
        ax.text(x, y, lettre, ha="center", va="center", fontsize=taille, weight="bold", zorder=11)
