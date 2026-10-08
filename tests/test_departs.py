import math

import numpy as np
import pytest

from fieldlines.champ import Distribution
from fieldlines.departs import (
    CercleDeFlux,
    PlacementSelonFlux,
    departs_autour,
    departs_le_long,
    origine_et_phase,
)
from fieldlines.geometrie import Cadre
from fieldlines.integration import Integrateur
from fieldlines.sources import Charge, Dipole, FilCharge, FilCourant, Singularite, Spire

CADRE = Cadre(-4, 4, -3, 3)


def angles(points, centre):
    return np.arctan2(np.array(points)[:, 1] - centre[1], np.array(points)[:, 0] - centre[0])


class TestDepartsAutour:
    def test_charge_angles_reguliers(self):
        points = departs_autour(Singularite((1, 2), "monopole", 1.0), 8, 0.5, angle=0.1)
        distances = np.hypot(np.array(points)[:, 0] - 1, np.array(points)[:, 1] - 2)
        np.testing.assert_allclose(distances, 0.5)
        np.testing.assert_allclose(np.diff(np.unwrap(angles(points, (1, 2)))), 2 * math.pi / 8)
        assert angles(points, (1, 2))[0] == pytest.approx(0.1)

    def test_dipole_sans_depart_en_double(self):
        # L'ancien module prenait les deux bouts de linspace(0, 2π) : un départ en double.
        points = departs_autour(Singularite((0, 0), "dipole", 1.0, math.pi / 2), 23, 0.2)
        assert len({(round(x, 12), round(y, 12)) for x, y in points}) == 23

    def test_dipole_regulier_en_sinus(self):
        # Moment vers +y : la moitié sortante est y > 0, θ compté depuis +y.
        points = departs_autour(Singularite((0, 0), "dipole", 1.0, math.pi / 2), 10, 1.0)
        sortants = sorted(x for x, y in points[:5])
        np.testing.assert_allclose(np.diff(sortants), 0.4, atol=1e-12)


class TestCercleDeFlux:
    def test_charge_isolee_departs_equiangulaires(self):
        cercle = CercleDeFlux(Distribution([Charge(2, (0, 0))]), (0, 0), 0.1)
        np.testing.assert_allclose(cercle.angles_sortants(8), np.arange(8) * math.pi / 4, atol=1e-9)

    def test_flux_d_une_charge_isolee(self):
        # Flux du champ de la figure à travers un cercle : 2π q / r ; rien n'entre.
        cercle = CercleDeFlux(Distribution([Charge(2, (0, 0))]), (0, 0), 0.1)
        assert cercle.flux_sortant == pytest.approx(2 * math.pi * 2 / 0.1)
        assert cercle.flux_entrant == pytest.approx(0.0)

    def test_complete_les_plus_grands_intervalles(self):
        cercle = CercleDeFlux(Distribution([Charge(-1, (0, 0))]), (0, 0), 0.1)
        total = cercle.flux_entrant
        deja = [0.0, total / 8, 2 * total / 8]
        ajouts = cercle.angles_entrants(8, deja)
        assert len(ajouts) == 5
        attendus = np.arange(3, 8) * math.pi / 4
        np.testing.assert_allclose(np.sort(ajouts % (2 * math.pi)), attendus, atol=1e-6)

    def test_rien_a_ajouter_si_le_quota_est_atteint(self):
        cercle = CercleDeFlux(Distribution([Charge(-1, (0, 0))]), (0, 0), 0.1)
        assert cercle.angles_entrants(2, [0.0, 1.0, 2.0]).size == 0


class TestDepartsLeLong:
    def test_repartition_reguliere(self):
        points = departs_le_long(Distribution([Charge(1, (5, 5))]), (0, 0), (4, 0), 4, "reguliere")
        assert points == pytest.approx([(0.5, 0), (1.5, 0), (2.5, 0), (3.5, 0)])

    def test_fil_de_courant_rayons_en_progression_geometrique(self):
        # Ψ = −2I ln r : à ΔΨ égal, les rayons forment une suite géométrique.
        points = departs_le_long(Distribution([FilCourant(1, (0, 0))]), (0.1, 0), (10, 0), 5)
        rapports = np.array(points)[1:, 0] / np.array(points)[:-1, 0]
        np.testing.assert_allclose(rapports, rapports[0], rtol=1e-3)

    def test_revolution_flux_egal_dans_l_espace(self):
        distribution = Distribution([Spire(1, (0, 0), 1)])
        points = np.array(departs_le_long(distribution, (0, 0), (0, 0.9), 6, "flux_revolution"))
        psi = distribution.flux(points[:, 0], points[:, 1])
        np.testing.assert_allclose(np.diff(psi), psi[0] * 2, rtol=1e-3)

    def test_revolution_sans_axe_refusee(self):
        with pytest.raises(ValueError, match="révolution"):
            departs_le_long(
                Distribution([FilCourant(1, (0, 0))]), (1, 0), (2, 0), 3, "flux_revolution"
            )

    def test_repartition_inconnue(self):
        with pytest.raises(ValueError):
            departs_le_long(Distribution([FilCourant(1, (0, 0))]), (1, 0), (2, 0), 3, "uniforme")

    def test_sans_symetrie_flux_du_plan_egal(self):
        distribution = Distribution([Charge(1, (0, 1)), Charge(1, (2, 1)), Charge(-1, (1, -1))])
        points = np.array(departs_le_long(distribution, (-1, 0), (3, 0), 5))
        x = np.linspace(-1, 3, 20001)
        _, ey = distribution.champ(x, np.zeros_like(x))
        cumul = np.concatenate(([0], np.cumsum(np.abs(0.5 * (ey[1:] + ey[:-1])) * np.diff(x))))
        tranches = np.diff(np.interp(np.concatenate(([-1], points[:, 0], [3])), x, cumul))
        np.testing.assert_allclose(tranches[1:-1], tranches[1], rtol=1e-3)
        assert tranches[0] == pytest.approx(tranches[1] / 2, rel=1e-3)

    def test_segment_sans_flux_refuse(self):
        with pytest.raises(ValueError):
            departs_le_long(Distribution([FilCharge(1, (0, 0))]), (1, 0), (2, 0), 3)


class TestPlacementSelonFlux:
    def test_paire_neutre_sans_doublon(self):
        integrateur = Integrateur(Distribution([Charge(1, (-1, 0)), Charge(-1, (1, 0))]), CADRE)
        departs = PlacementSelonFlux(integrateur).places(16, 0.2)
        depuis_plus = [d for d in departs if d.ligne is not None]
        arrivees = sum(d.ligne.aval.singularite == 1 for d in depuis_plus)
        depuis_moins = [d for d in departs if math.dist(d.position, (1, 0)) == pytest.approx(0.2)]
        assert len(depuis_plus) == 16
        assert len(depuis_moins) == 16 - arrivees

    def test_quotas_proportionnels_aux_charges(self):
        integrateur = Integrateur(Distribution([Charge(2, (-1, 0)), Charge(-1, (1, 0))]), CADRE)
        departs = PlacementSelonFlux(integrateur).places(16, 0.2)
        assert sum(d.ligne is not None for d in departs) == 16
        # Toute la charge −1 reçoit des lignes de +2 : il n'y a rien à ajouter autour d'elle.
        assert all(d.ligne is not None for d in departs)

    def test_dipole_symetrique_par_rapport_a_son_axe(self):
        integrateur = Integrateur(Distribution([Dipole(1, (0, 0), math.pi / 2)]), CADRE)
        departs = PlacementSelonFlux(integrateur).places(5, 0.3)
        x = sorted(d.position[0] for d in departs if d.ligne is not None)
        np.testing.assert_allclose(x, -np.array(x[::-1]), atol=1e-9)
        assert x[2] == pytest.approx(0.0, abs=1e-9)

    def test_sans_source_aboutissante_refuse(self):
        integrateur = Integrateur(Distribution([FilCourant(1, (0, 0))]), CADRE)
        with pytest.raises(ValueError):
            PlacementSelonFlux(integrateur).places(8, 0.2)


@pytest.mark.parametrize(
    ("singularite", "attendu"),
    [
        (Singularite((0, 0), "monopole", 1.0), (0.3, 0.0)),
        (Singularite((0, 0), "dipole", 1.0, 1.0), (1.0 - math.pi / 2 + 0.3, 0.5)),
    ],
)
def test_origine_et_phase(singularite, attendu):
    assert origine_et_phase(singularite, 0.3) == pytest.approx(attendu)
