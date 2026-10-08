import math

import numpy as np
import pytest

from fieldlines.carte import Carte
from fieldlines.integration import Arret
from fieldlines.sources import Charge, Dipole, FilCourant, Spire

PAIRE = [Charge(1, (-1, 0)), Charge(-1, (1, 0))]
CADRE = (-3, 3, -3, 3)


@pytest.fixture
def carte():
    carte = Carte(PAIRE, CADRE)
    carte.lignes_autour(nombre=8, rayon=0.2)
    return carte


class TestNoms:
    def test_numerotation_source_apres_source(self, carte):
        assert carte.noms == list(range(16))

    def test_retirer_ne_renumerote_pas(self, carte):
        carte.retire(2, range(5, 8))
        carte.retire(9)
        assert carte.noms == [0, 1, 3, 4, 8, *range(10, 16)]

    def test_retirer_un_nom_inconnu(self, carte):
        carte.retire(3)
        with pytest.raises(KeyError, match="3"):
            carte.retire(3)

    def test_retirer_est_tout_ou_rien(self, carte):
        with pytest.raises(KeyError):
            carte.retire(1, 99)
        assert 1 in carte

    def test_numeros_jamais_reutilises(self, carte):
        carte.retire(15)
        assert carte.ajoute_ligne((0, 1)) == 16

    def test_nom_en_texte(self, carte):
        assert carte.ajoute_ligne((0, 1), nom="axe") == "axe"
        with pytest.raises(ValueError):
            carte.ajoute_ligne((0, 2), nom="axe")

    def test_entiers_numpy_acceptes(self, carte):
        carte.retire(np.arange(3))
        assert 0 not in carte and 3 in carte


class TestReglages:
    def test_sans_initialisation_prealable(self):
        # L'ancien module exigeait set_default_SL_integration_parameters avant tout.
        carte = Carte(PAIRE, CADRE)
        nom = carte.ajoute_ligne((0, 0.5))
        assert carte.options(nom).fleches == (0.5,)
        assert carte.ligne(nom).aval.arret is Arret.SOURCE

    def test_ajuste_le_depart_a_zero(self, carte):
        # L'ancien change_parameter ignorait x0 = 0.
        avant = carte.ligne(0)
        carte.ajuste(0, depart=(0, 0))
        assert carte.ligne(0).depart == (0.0, 0.0) and carte.ligne(0) is not avant

    def test_ajuster_les_fleches_ne_recalcule_pas(self, carte):
        avant = carte.ligne(1)
        carte.ajuste(1, 2, fleches=[0.1, 0.9])
        assert carte.ligne(1) is avant and carte.options(2).fleches == (0.1, 0.9)

    def test_ajuster_la_marge_recalcule(self, carte):
        avant = carte.ligne(4)
        carte.ajuste(4, marge=2)
        assert carte.ligne(4) is not avant

    def test_reglage_inconnu(self, carte):
        with pytest.raises(TypeError):
            carte.ajuste(0, enlarge=3)

    def test_rayon_par_defaut(self):
        carte = Carte(PAIRE, CADRE)
        carte.lignes_autour(nombre=4)
        # Le dixième de la distance entre les charges, sans dépasser 2 % de la diagonale.
        attendu = min(0.1 * 2, 0.02 * math.hypot(6, 6))
        assert math.dist(carte.ligne(0).depart, (-1, 0)) == pytest.approx(attendu)


class TestPlacements:
    def test_lignes_autour_sans_source_aboutissante(self):
        with pytest.raises(ValueError, match="lignes_le_long"):
            Carte([FilCourant(1, (0, 0))], CADRE).lignes_autour()

    def test_selon_flux_moins_de_lignes_que_autour(self):
        carte = Carte(PAIRE, CADRE)
        carte.lignes_selon_flux(nombre=8, rayon=0.2)
        assert len(carte) < 16

    def test_le_long_pour_un_champ_magnetique(self):
        carte = Carte([FilCourant(1, (0, 0))], CADRE)
        noms = carte.lignes_le_long((0.2, 0), (2.5, 0), 4)
        assert all(carte.ligne(nom).fermee for nom in noms)

    def test_dipole_lignes_autour(self):
        carte = Carte([Dipole(1, (0, 0), math.pi / 2)], CADRE)
        assert len(carte.lignes_autour(nombre=12, rayon=0.3)) == 12


class TestEquipotentielles:
    def test_niveaux_finis_meme_sur_une_charge(self):
        carte = Carte([Charge(1, (0, 0)), Charge(-1, (1, 1))], (-2, 2, -2, 2))
        niveaux = carte.niveaux(10, echelle="log", symetrique=True, points=5)
        assert np.all(np.isfinite(niveaux)) and len(niveaux) == 20

    def test_bornes_imposees(self, carte):
        np.testing.assert_allclose(carte.niveaux(3, vmin=0, vmax=1), [0, 0.5, 1])

    def test_grille_mise_en_cache(self, carte):
        assert carte.grille("potentiel", 64) is carte.grille("potentiel", 64)


class TestTrace:
    def test_trace_complet(self, carte, ax):
        carte.trace(ax)
        carte.trace_noms(ax)
        contours = carte.trace_equipotentielles(ax, points=64)
        assert len(ax.lines) == 16 + 2
        assert len(ax.patches) == 16
        assert len(ax.texts) == 16
        assert contours.norm.vmax == pytest.approx(carte.grille("potentiel", 64).maximum / 1000)

    def test_style_propre_a_une_ligne(self, carte, ax):
        carte.ajuste(3, style={"color": "r", "linestyle": "--"})
        carte.trace_lignes(ax, color="gray")
        carte.trace_fleches(ax)
        assert ax.lines[3].get_color() == "r" and ax.lines[2].get_color() == "gray"
        assert ax.patches[3].get_facecolor() != ax.patches[2].get_facecolor()

    def test_lignes_de_flux_d_une_spire(self, ax):
        carte = Carte([Spire(1, (0, 0), 1)], CADRE)
        contours = carte.trace_lignes_de_flux(ax, 5, points=64)
        assert len(contours.levels) == 5

    def test_verbose(self, carte, capsys):
        carte.verbose = True
        carte.ligne(0)
        assert "ligne 0 :" in capsys.readouterr().out
