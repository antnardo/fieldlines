import math

import numpy as np
import pytest

from fieldlines.champ import Distribution
from fieldlines.geometrie import Cadre
from fieldlines.integration import Arret, Integrateur, Options
from fieldlines.sources import Charge, DipoleMagnetique, FilCourant, Spire

CADRE = Cadre(-4, 4, -4, 4)


@pytest.fixture
def paire():
    return Integrateur(Distribution([Charge(1, (-1, 0)), Charge(-1, (1, 0))]), CADRE)


class TestLignesElectriques:
    def test_va_de_la_charge_positive_a_la_negative(self, paire):
        ligne = paire.integre((-1 + 0.2 * math.cos(1.0), 0.2 * math.sin(1.0)))
        assert ligne.amont.arret is Arret.SOURCE and ligne.amont.singularite == 0
        assert ligne.aval.arret is Arret.SOURCE and ligne.aval.singularite == 1
        assert tuple(ligne.points[0]) == (-1.0, 0.0)
        assert tuple(ligne.points[-1]) == (1.0, 0.0)

    def test_tangente_au_champ(self, paire):
        ligne = paire.integre((-1 + 0.2 * math.cos(2.0), 0.2 * math.sin(2.0)))
        milieux = 0.5 * (ligne.points[1:] + ligne.points[:-1])
        cordes = np.diff(ligne.points, axis=0)
        loin = np.min(np.hypot(milieux[:, 0, None] - [-1, 1], milieux[:, 1, None]), axis=1) > 0.1
        ex, ey = paire.distribution.champ(milieux[loin, 0], milieux[loin, 1])
        tx, ty = cordes[loin].T
        produit_vectoriel = (tx * ey - ty * ex) / (np.hypot(tx, ty) * np.hypot(ex, ey))
        produit_scalaire = tx * ex + ty * ey
        assert np.max(np.abs(produit_vectoriel)) < 1e-3
        assert np.all(produit_scalaire > 0)

    def test_potentiel_decroit_le_long_de_la_ligne(self, paire):
        ligne = paire.integre((-1 + 0.2 * math.cos(0.5), 0.2 * math.sin(0.5)))
        potentiel = paire.distribution.potentiel(*ligne.points[1:-1].T)
        assert np.all(np.diff(potentiel) < 0)

    def test_charge_seule_sort_du_cadre_agrandi(self):
        integrateur = Integrateur(Distribution([Charge(1, (0, 0))]), CADRE)
        ligne = integrateur.integre((0.2, 0.0), Options(marge=0.25))
        assert ligne.aval.arret is Arret.SORTIE
        assert ligne.points[-1, 0] == pytest.approx(6.0)

    def test_s_arrete_au_point_de_champ_nul(self):
        integrateur = Integrateur(Distribution([Charge(1, (-1, 0)), Charge(1, (1, 0))]), CADRE)
        ligne = integrateur.integre((-0.8, 0.0))
        assert ligne.aval.arret is Arret.CHAMP_NUL
        assert np.hypot(*ligne.points[-1]) < 1e-3

    def test_sens_aval_seulement(self, paire):
        ligne = paire.integre((-0.5, 0.5), Options(sens="aval"))
        assert ligne.amont.arret is Arret.DEPART
        assert tuple(ligne.points[0]) == (-0.5, 0.5)

    def test_longueur_maximale(self, paire):
        ligne = paire.integre((0.0, 0.5), Options(longueur_max=0.3))
        assert ligne.aval.arret is Arret.LONGUEUR
        assert ligne.longueur <= 0.6 + 1e-9

    def test_depart_hors_du_cadre(self, paire):
        ligne = paire.integre((10.0, 0.0))
        assert ligne.aval.arret is Arret.SORTIE and len(ligne.points) == 1

    def test_depart_sur_une_charge(self, paire):
        ligne = paire.integre((1.0, 0.0))
        assert ligne.aval.arret is Arret.SOURCE and ligne.aval.singularite == 1


class TestLignesMagnetiques:
    @pytest.mark.parametrize("rayon", [0.01, 0.3, 1.5, 3.0])
    def test_fil_ligne_circulaire_fermee(self, rayon):
        integrateur = Integrateur(Distribution([FilCourant(1, (0.5, 0.5))]), CADRE)
        ligne = integrateur.integre((0.5 + rayon, 0.5))
        assert ligne.fermee
        rayons = np.hypot(ligne.points[:, 0] - 0.5, ligne.points[:, 1] - 0.5)
        assert np.ptp(rayons) < 1e-5 * max(rayon, 1.0)
        assert tuple(ligne.points[0]) == tuple(ligne.points[-1])
        assert ligne.longueur == pytest.approx(2 * math.pi * rayon, rel=1e-3)

    def test_fil_tourne_dans_le_sens_direct(self):
        integrateur = Integrateur(Distribution([FilCourant(1, (0, 0))]), CADRE)
        x, y = integrateur.integre((1.0, 0.0)).points[:3].T
        assert np.all(np.diff(np.unwrap(np.arctan2(y, x))) > 0)

    @pytest.mark.parametrize("hauteur", [0.5, 0.8, 0.99, 1.3])
    def test_spire_ligne_fermee_a_flux_constant(self, hauteur):
        distribution = Distribution([Spire(1, (0, 0), 1)])
        ligne = Integrateur(distribution, Cadre(-6, 6, -6, 6)).integre((0.0, hauteur))
        assert ligne.fermee
        psi = distribution.flux(*ligne.points.T)
        assert np.ptp(psi) < 1e-4 * abs(psi[0])

    def test_dipole_magnetique_lignes_du_dipole_au_dipole(self):
        distribution = Distribution([DipoleMagnetique(1, (0, 0), angle=math.pi / 2)])
        ligne = Integrateur(distribution, CADRE).integre((0.3, 0.4))
        assert ligne.amont.arret is Arret.SOURCE and ligne.aval.arret is Arret.SOURCE
        psi = distribution.flux(*ligne.points[1:-1].T)
        assert np.ptp(psi) < 1e-4 * abs(psi[0])


class TestOptions:
    def test_fraction_hors_de_0_1_refusee(self):
        with pytest.raises(ValueError):
            Options(fleches=(1.5,))

    def test_sens_inconnu_refuse(self):
        with pytest.raises(ValueError):
            Options(sens="haut")  # type: ignore[arg-type]

    def test_remplace_nom_inconnu(self):
        with pytest.raises(TypeError, match="Emin"):
            Options().remplace(Emin=1e-6)

    def test_remplace_garde_le_reste(self):
        options = Options(marge=2.0).remplace(fleches=[0.1, 0.9])
        assert options.marge == 2.0 and options.fleches == (0.1, 0.9)

    def test_style_copie(self):
        style = {"color": "r"}
        options = Options(style=style)
        style["color"] = "b"
        assert options.style == {"color": "r"}

    def test_fleches_un_seul_nombre(self):
        assert Options(fleches=0.3).fleches == (0.3,)
        assert Options().remplace(fleches=np.float64(0.2)).fleches == (0.2,)

    @pytest.mark.parametrize("valeur", ["0.3", object()])
    def test_fleches_ni_nombre_ni_liste_refusees(self, valeur):
        with pytest.raises(TypeError, match="fleches"):
            Options(fleches=valeur)

    def test_remplace_style_donne_directement(self):
        # `ajuste(nom, color="r")` : le message dit de passer par style={...}.
        with pytest.raises(TypeError, match=r'style=\{"color": …\}'):
            Options().remplace(color="r")
        with pytest.raises(TypeError, match="color, linewidth"):
            Options().remplace(linewidth=2, color="r")
