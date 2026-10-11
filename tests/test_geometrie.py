import math

import numpy as np
import pytest

from fieldlines.geometrie import Axe, Cadre, distance_au_segment, en_point


def test_en_point_accepte_une_liste():
    assert en_point([1, 2.5]) == (1.0, 2.5)


def test_cadre_vide_refuse():
    with pytest.raises(ValueError):
        Cadre(1, 0, 0, 1)


def test_cadre_depuis_quadruplet():
    assert Cadre.depuis((-1, 2, -3, 4)) == Cadre(-1.0, 2.0, -3.0, 4.0)


def test_cadre_elargi_ajoute_la_marge_de_chaque_cote():
    assert Cadre(-20, 20, -10, 10).elargi(0.25) == Cadre(-30, 30, -15, 15)


@pytest.mark.parametrize(("x", "y", "attendu"), [(0, 0, 1.0), (0.5, 0, 0.5), (2, 0, -1.0)])
def test_distance_interieure_signee(x, y, attendu):
    assert Cadre(-1, 1, -1, 1).distance_interieure(x, y) == pytest.approx(attendu)


def test_contient_vectorise():
    dedans = Cadre(0, 1, 0, 1).contient(np.array([0.5, 2.0]), np.array([0.5, 0.5]))
    assert dedans.tolist() == [True, False]


def test_axe_coordonnees():
    axe = Axe((1, 1), math.pi / 2)
    z, rho = axe.coordonnees(np.array([1.0, -2.0]), np.array([3.0, 0.0]))
    np.testing.assert_allclose(z, [2.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(rho, [0.0, 3.0], atol=1e-12)


@pytest.mark.parametrize(("angle", "sens"), [(0.3, 1), (0.3 + math.pi, -1), (1.0, 0)])
def test_axe_sens_relatif(angle, sens):
    assert Axe((0, 0), 0.3).sens_relatif(angle) == sens


@pytest.mark.parametrize(
    ("x", "y", "attendu"), [(1, 1, 1.0), (3, 0, 1.0), (-1, 0, 1.0), (0.5, -2, 2.0), (1.5, 0, 0.0)]
)
def test_distance_au_segment(x, y, attendu):
    assert distance_au_segment((0, 0), (2, 0), x, y) == pytest.approx(attendu)


def test_distance_au_segment_vectorisee_et_degeneree():
    distances = distance_au_segment((1, 1), (1, 1), np.array([1.0, 4.0]), np.array([1.0, 5.0]))
    np.testing.assert_allclose(distances, [0.0, 5.0])
