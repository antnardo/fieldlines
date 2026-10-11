import math

import numpy as np
import pytest
from matplotlib.patches import FancyArrowPatch

from fieldlines.champ import Distribution
from fieldlines.geometrie import Cadre
from fieldlines.integration import Arret, Extremite, Integrateur, LigneDeChamp
from fieldlines.sources import (
    Aimant,
    ChampUniforme,
    Charge,
    Dipole,
    FilCharge,
    FilCourant,
    NappeCourant,
    PlanCharge,
    Solenoide,
    Spire,
)
from fieldlines.trace import (
    normalisation,
    position_sur_ligne,
    trace_contours,
    trace_fleche,
    trace_ligne,
    trace_sources,
)

CADRE = Cadre(-1, 1, -1, 1)
BOUT = Extremite(Arret.SORTIE)


def droite(*points):
    return LigneDeChamp(np.array(points, dtype=float), points[0], BOUT, BOUT)


class TestPositionSurLigne:
    def test_milieu_de_la_partie_visible(self):
        # De x = −3 à x = 1 : seule la partie [−1, 1] est visible, son milieu est 0.
        ligne = droite((-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0))
        point, direction = position_sur_ligne(ligne, 0.5, CADRE)
        assert point == pytest.approx((0, 0)) and direction == pytest.approx((1, 0))

    @pytest.mark.parametrize(("fraction", "attendu"), [(0.0, (-1, 0)), (1.0, (1, 0))])
    def test_bouts_visibles(self, fraction, attendu):
        ligne = droite((-3, 0), (-1, 0), (0, 0), (1, 0), (2, 0))
        point, direction = position_sur_ligne(ligne, fraction, CADRE)
        assert point == pytest.approx(attendu) and direction == pytest.approx((1, 0))

    def test_ligne_hors_cadre(self):
        assert position_sur_ligne(droite((2, 2), (3, 3)), 0.5, CADRE) is None

    @pytest.mark.parametrize("fraction", [0.999, 1.0])
    def test_fleche_en_bout_dans_le_sens_du_champ(self, fraction):
        # L'ancien module inversait les flèches placées en bout de ligne.
        distribution = Distribution([Charge(1, (-1, 0)), Charge(-1, (1, 0))])
        cadre = Cadre(-3, 3, -3, 3)
        ligne = Integrateur(distribution, cadre).integre(
            (-1 + 0.2 * math.cos(1), 0.2 * math.sin(1))
        )
        point, direction = position_sur_ligne(ligne, fraction, cadre)
        ecart = np.array(point) - (1, 0)
        assert np.dot(direction, -ecart) > 0  # la flèche pointe vers la charge négative


def test_trace_fleche_orientee(ax):
    fleche = trace_fleche(ax, (0.5, 0.5), (0.0, -1.0), taille=10)
    assert isinstance(fleche, FancyArrowPatch) and fleche in ax.patches
    (xa, ya), (xb, yb) = fleche._posA_posB
    assert xb == pytest.approx(xa) and yb < ya


def test_trace_ligne_noire_par_defaut(ax):
    trait = trace_ligne(ax, droite((0, 0), (1, 1)))
    assert trait.get_color() == "k"


def test_normalisation_contraste():
    distribution = Distribution([Charge(1, (0.05, 0.05))])
    grille = distribution.grille(CADRE, 11, "potentiel")
    norme = normalisation(grille, 2.0)
    assert norme.vmax == pytest.approx(grille.maximum / 100)


def test_trace_contours_niveaux(ax):
    grille = Distribution([Charge(1, (0.05, 0.05))]).grille(CADRE, 21, "potentiel")
    contours = trace_contours(ax, grille, [1, 2, 3])
    np.testing.assert_allclose(contours.levels, [1, 2, 3])


@pytest.mark.parametrize(
    ("sources", "lignes", "annotations"),
    [
        ([Charge(1, (0, 0)), Charge(-2, (1, 0)), FilCharge(1, (0, 1))], 3, 0),
        ([Dipole(1, (0, 0))], 1, 1),
        ([FilCourant(1, (0, 0)), FilCourant(-1, (1, 0))], 4, 0),
        ([Spire(1, (0, 0), 1)], 5, 0),
        ([Solenoide(1, (0, 0), 1, 2, 3)], 12, 0),
        ([Aimant(1, (0, 0), 0.5, 1)], 0, 2),  # les lettres S et N
        ([PlanCharge(1, (0, 0), 0.3)], 1, 0),  # une droite
        ([NappeCourant(1, (0, 0))], 3, 0),  # une droite et le symbole ⊙
        ([ChampUniforme(B=(1, 0))], 0, 0),  # rien à dessiner
    ],
)
def test_trace_sources(ax, sources, lignes, annotations):
    trace_sources(ax, sources)
    assert len(ax.lines) == lignes and len(ax.texts) == annotations


def test_trace_aimant_deux_poles_et_un_contour(ax):
    trace_sources(ax, [Aimant(1, (0, 0), 0.5, 1, angle=math.pi / 2)])
    assert len(ax.patches) == 3
    lettres = {texte.get_text(): texte.get_position() for texte in ax.texts}
    assert lettres["S"][1] < 0 < lettres["N"][1]
