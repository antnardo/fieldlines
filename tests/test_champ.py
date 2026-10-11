import math

import numpy as np
import pytest

from fieldlines.champ import Distribution
from fieldlines.geometrie import Cadre
from fieldlines.sources import (
    ChampUniforme,
    Charge,
    Dipole,
    DipoleMagnetique,
    FilCharge,
    FilCourant,
    NappeCourant,
    PlanCharge,
    Solenoide,
    Spire,
)


def test_distribution_vide_refusee():
    with pytest.raises(ValueError):
        Distribution([])


def test_natures_melees_refusees():
    with pytest.raises(ValueError, match="mêlées"):
        Distribution([Charge(1, (0, 0)), FilCourant(1, (1, 0))])


def test_champ_somme_des_sources():
    sources = [Charge(1, (0, 0)), FilCharge(2, (1, 1)), Dipole(1, (-1, 0), 1.0)]
    distribution = Distribution(sources)
    attendu = np.sum([s.champ(0.5, -0.5) for s in sources], axis=0)
    assert distribution.champ(0.5, -0.5) == pytest.approx(tuple(attendu))
    assert distribution.champ_point(0.5, -0.5) == pytest.approx(tuple(attendu))


def test_champ_garde_la_forme_des_entrees():
    fx, fy = Distribution([Charge(1, (0, 0))]).champ(np.ones((3, 4)), np.full((3, 4), 2.0))
    assert fx.shape == fy.shape == (3, 4)


def test_potentiel_magnetique_refuse():
    with pytest.raises(TypeError):
        Distribution([FilCourant(1, (0, 0))]).potentiel(1.0, 1.0)


def test_grille_sur_une_charge_masquee_et_finie():
    # La grille 5 × 5 de [−1, 1]² passe exactement sur la charge : l'ancien module
    # y trouvait un potentiel infini, puis des niveaux NaN.
    grille = Distribution([Charge(1, (0, 0)), Charge(-1, (0.5, 0.5))]).grille(
        Cadre(-1, 1, -1, 1), 5, "potentiel"
    )
    assert grille.valeurs.mask[2, 2] and grille.valeurs.mask[3, 3]
    assert math.isfinite(grille.minimum) and math.isfinite(grille.maximum)


@pytest.mark.parametrize(
    ("sources", "attendu"),
    [
        ([FilCourant(1, (0, 0)), FilCourant(-1, (1, 2))], "plane"),
        (
            [
                Spire(1, (0, 0), 1),
                Solenoide(1, (3, 0), 1, 2, 4),
                DipoleMagnetique(1, (-2, 0), math.pi),
            ],
            "revolution",
        ),
        ([Spire(1, (0, 0), 1), Spire(1, (0, 3), 1)], None),
        ([Spire(1, (0, 0), 1), Spire(1, (2, 0), 1, angle=0.5)], None),
        ([Charge(1, (0, 0)), Charge(-1, (1, 1)), Charge(2, (3, 3))], "revolution"),
        ([Charge(1, (0, 0)), Charge(-1, (1, 1)), Charge(2, (3, 0))], None),
        ([Charge(1, (0, 0)), Dipole(1, (2, 0))], "revolution"),
        ([FilCharge(1, (0, 0))], None),
        ([ChampUniforme(B=(1, 0))], "plane"),
        ([ChampUniforme(B=(1, 0)), FilCourant(1, (0, 1))], "plane"),
        ([NappeCourant(1, (0, 0)), NappeCourant(-1, (0, 1))], "plane"),
        ([PlanCharge(1, (0, 0)), ChampUniforme(E=(0, 1))], "plane"),
        ([ChampUniforme(E=(0, 1)), Charge(1, (2, 0))], "revolution"),  # axe vertical
        ([ChampUniforme(E=(0, 1)), Charge(1, (2, 0)), PlanCharge(1, (0, -1))], "revolution"),
        ([ChampUniforme(B=(1, 0)), Spire(1, (0, 0), 1)], "revolution"),
        ([ChampUniforme(B=(0, 1)), Spire(1, (0, 0), 1)], None),
        ([ChampUniforme(E=(1, 0)), Charge(1, (0, 0)), Charge(1, (0, 2))], None),
    ],
)
def test_symetrie(sources, attendu):
    assert Distribution(sources).symetrie() == attendu


def test_axe_de_revolution_oriente_par_le_champ_uniforme():
    distribution = Distribution([ChampUniforme(E=(0, 1)), Charge(1, (2, 0))])
    axe = distribution.axe_de_revolution()
    assert axe is not None and axe.origine == (2.0, 0.0)
    assert axe.sens_relatif(math.pi / 2) != 0


def test_flux_sans_symetrie_refuse():
    with pytest.raises(ValueError, match="pas de fonction de flux"):
        Distribution([Charge(1, (0, 0)), Charge(1, (1, 0)), Charge(1, (0, 1))]).flux(2.0, 2.0)


def test_flux_constant_sur_l_axe_pour_des_charges():
    distribution = Distribution([Charge(1, (-1, 0)), Charge(-1, (1, 0))])
    psi = distribution.flux(np.array([-3.0, 3.0]), np.array([0.0, 0.0]))
    assert psi == pytest.approx([0.0, 0.0])


def test_champ_de_reference_positif():
    reference = Distribution([Charge(1, (0, 0))]).champ_de_reference(Cadre(-1, 1, -1, 1))
    assert 0 < reference < math.inf


def test_distance_minimale():
    distribution = Distribution([Charge(1, (0, 0)), Charge(1, (3, 4)), Charge(1, (0, 1))])
    assert distribution.distance_minimale() == pytest.approx(1.0)
