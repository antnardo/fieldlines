"""Les niveaux des équipotentielles (ou de toute autre grandeur scalaire).

Quatre répartitions, celles des figures de cours :

- linéaire : `nombre` niveaux de vmin à vmax ;
- linéaire symétrique : ±(`nombre` niveaux de `seuil` à max(|vmin|, |vmax|)) ;
- logarithmique : `nombre` niveaux en progression géométrique de vmin à vmax, tous de
  même signe ;
- logarithmique symétrique : ± une progression géométrique de `seuil` à
  max(|vmin|, |vmax|), pour les distributions où le potentiel change de signe et varie
  de plusieurs ordres de grandeur (dipôle, quadrupôle).

Les versions symétriques n'ont pas de niveau zéro (`ajoute=[0]` l'ajoute).
"""

from collections.abc import Iterable
from typing import Literal, TypeAlias

import numpy as np

__all__ = ["Echelle", "niveaux"]

Echelle: TypeAlias = Literal["lineaire", "log"]


def niveaux(
    vmin: float,
    vmax: float,
    nombre: int = 20,
    echelle: Echelle = "lineaire",
    symetrique: bool = False,
    seuil: float = 1e-3,
    ajoute: Iterable[float] = (),
) -> np.ndarray:
    """Les niveaux, triés et sans doublon.

    ValueError si les bornes ne sont pas finies, si `nombre` < 1, ou pour une échelle
    logarithmique non symétrique sur un intervalle qui contient zéro.
    """
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        raise ValueError(f"bornes non finies : {vmin}, {vmax}")
    if nombre < 1:
        raise ValueError("il faut au moins un niveau")
    if echelle not in ("lineaire", "log"):
        raise ValueError(f"échelle inconnue : {echelle!r}")
    vmin, vmax = min(vmin, vmax), max(vmin, vmax)
    if symetrique:
        extreme = max(abs(vmin), abs(vmax))
        if seuil <= 0 or extreme <= seuil:
            raise ValueError(f"il faut 0 < seuil < max(|vmin|, |vmax|) = {extreme:g}")
        if echelle == "log":
            positifs = np.geomspace(seuil, extreme, nombre)
        else:
            positifs = np.linspace(seuil, extreme, nombre)
        resultat = np.concatenate((-positifs[::-1], positifs))
    elif echelle == "lineaire":
        resultat = np.linspace(vmin, vmax, nombre)
    elif vmin > 0:
        resultat = np.geomspace(vmin, vmax, nombre)
    elif vmax < 0:
        resultat = -np.geomspace(-vmax, -vmin, nombre)[::-1]
    else:
        raise ValueError(
            f"échelle logarithmique sur [{vmin:g}, {vmax:g}], qui contient zéro : "
            "prendre symetrique=True, ou des bornes de même signe"
        )
    return np.union1d(resultat, np.asarray(list(ajoute), dtype=float))
