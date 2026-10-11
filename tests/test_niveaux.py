import math

import numpy as np
import pytest

from fieldlines.niveaux import niveaux


def test_lineaire():
    np.testing.assert_allclose(niveaux(-1, 3, 5), [-1, 0, 1, 2, 3])


def test_lineaire_symetrique_sans_zero():
    # seuil relatif : 0,25 × max(|vmin|, |vmax|) = 1.
    resultat = niveaux(-2, 4, 3, symetrique=True, seuil=0.25)
    np.testing.assert_allclose(resultat, [-4, -2.5, -1, 1, 2.5, 4])


def test_log_symetrique():
    resultat = niveaux(-50, 100, 3, echelle="log", symetrique=True, seuil=0.01)
    np.testing.assert_allclose(resultat, [-100, -10, -1, 1, 10, 100])


def test_seuil_relatif_par_defaut_trois_decades():
    # Le seuil suit le potentiel extrême : une grille qui passe près d'une charge ne
    # fait plus descendre les niveaux jusqu'à 10⁻³ (empilés sur la surface V = 0).
    resultat = niveaux(-5000, 5000, 4, echelle="log", symetrique=True)
    np.testing.assert_allclose(resultat[4:], [5, 50, 500, 5000])


def test_log_positif():
    np.testing.assert_allclose(niveaux(1, 100, 3, echelle="log"), [1, 10, 100])


def test_log_negatif():
    np.testing.assert_allclose(niveaux(-100, -1, 3, echelle="log"), [-100, -10, -1])


def test_log_qui_change_de_signe_refuse():
    # L'ancien module rendait des NaN.
    with pytest.raises(ValueError, match="symetrique"):
        niveaux(-1, 10, 5, echelle="log")


@pytest.mark.parametrize("borne", [math.inf, -math.inf, math.nan])
def test_bornes_non_finies_refusees(borne):
    with pytest.raises(ValueError):
        niveaux(0, borne)


def test_ajoute_trie_sans_doublon():
    np.testing.assert_allclose(niveaux(0, 1, 3, ajoute=[0.5, 0.25, 2]), [0, 0.25, 0.5, 1, 2])


def test_bornes_inversees_acceptees():
    np.testing.assert_allclose(niveaux(3, -1, 5), [-1, 0, 1, 2, 3])


@pytest.mark.parametrize("seuil", [0, 1, 2, -0.1])
def test_seuil_hors_de_0_1_refuse(seuil):
    with pytest.raises(ValueError, match="fraction"):
        niveaux(-1, 1, 5, symetrique=True, seuil=seuil)
