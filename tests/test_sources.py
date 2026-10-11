import math

import numpy as np
import pytest

from fieldlines.geometrie import Axe
from fieldlines.sources import (
    K_COULOMB,
    MU0_SUR_4PI,
    Aimant,
    ChampUniforme,
    Charge,
    Dipole,
    DipoleMagnetique,
    FilCharge,
    FilCourant,
    Lots,
    NappeCourant,
    PlanCharge,
    Solenoide,
    Source,
    Spire,
)

MU0 = 4 * math.pi  # unités réduites : μ0/(4π) = 1


def biot_savart_spire(spire, x, y, n=4000):
    """Le champ d'une spire par la loi de Biot et Savart, intégrée numériquement."""
    ux, uy = math.cos(spire.angle), math.sin(spire.angle)
    nx, ny = -uy, ux
    phi = np.linspace(0, 2 * math.pi, n, endpoint=False)
    # Points de la spire dans l'espace (x, y, z), z hors de la figure.
    px = spire.centre[0] + spire.rayon * np.cos(phi) * nx
    py = spire.centre[1] + spire.rayon * np.cos(phi) * ny
    pz = spire.rayon * np.sin(phi)
    dlx = -np.sin(phi) * nx * spire.rayon
    dly = -np.sin(phi) * ny * spire.rayon
    dlz = np.cos(phi) * spire.rayon
    rx, ry, rz = x - px, y - py, -pz
    r3 = (rx**2 + ry**2 + rz**2) ** 1.5
    bx = (dly * rz - dlz * ry) / r3
    by = (dlz * rx - dlx * rz) / r3
    pas = 2 * math.pi / n
    return spire.intensite * bx.sum() * pas, spire.intensite * by.sum() * pas


class TestChampsElectriques:
    def test_coulomb(self):
        ex, ey = Charge(2, (1, -1)).champ(4.0, 3.0)
        assert (ex, ey) == pytest.approx((2 * 3 / 125, 2 * 4 / 125))

    def test_potentiel_coulomb(self):
        assert Charge(2, (1, -1)).potentiel(4.0, 3.0) == pytest.approx(2 / 5)

    def test_fil_charge_en_un_sur_r(self):
        ex, ey = FilCharge(1.5, (0, 0)).champ(3.0, 4.0)
        assert (ex, ey) == pytest.approx((2 * 1.5 * 3 / 25, 2 * 1.5 * 4 / 25))

    def test_dipole_loin_de_deux_charges(self):
        d = 1e-3
        paire = Lots([Charge(1, (0, d / 2)), Charge(-1, (0, -d / 2))])
        dipole = Dipole(d, (0, 0), angle=math.pi / 2)
        x, y = np.array([0.3, -1.0, 2.0]), np.array([0.7, 0.2, -1.5])
        np.testing.assert_allclose(paire.champ(x, y), dipole.champ(x, y), rtol=1e-5)

    @pytest.mark.parametrize(
        "source",
        [
            Charge(1.3, (0.2, -0.4)),
            FilCharge(-0.7, (1, 1)),
            Dipole(2, (0, 1), 0.7),
            PlanCharge(0.4, (0.3, 0.2), angle=0.6),
            ChampUniforme(E=(0.8, -0.3)),
        ],
    )
    def test_champ_est_moins_le_gradient_du_potentiel(self, source, gradient):
        x, y = 1.7, -0.9
        dvx, dvy = gradient(source.potentiel, x, y)
        ex, ey = source.champ(x, y)
        assert (ex, ey) == pytest.approx((-dvx, -dvy), rel=1e-6)

    def test_flux_de_revolution_d_une_charge(self, gradient):
        # E_ρ = −(1/ρ) ∂Ψ/∂z et E_z = (1/ρ) ∂Ψ/∂ρ, l'axe étant Ox.
        charge = Charge(1.5, (0.5, 0))
        axe = Axe((0, 0), 0.0)
        x, y = 1.3, 0.8
        dpz, dprho = gradient(lambda a, b: charge.flux_revolution(a, b, axe), x, y)
        ex, ey = charge.champ(x, y)
        assert (ex, ey) == pytest.approx((dprho / y, -dpz / y), rel=1e-6)

    def test_charge_hors_axe_refusee(self):
        with pytest.raises(ValueError):
            Charge(1, (0, 1)).flux_revolution(0.0, 0.0, Axe((0, 0), 0.0))


class TestChampsMagnetiques:
    def test_fil_courant_regle_de_la_main_droite(self):
        source = FilCourant(2, (0, 0))
        assert source.champ(2.0, 0.0) == pytest.approx((0.0, 2.0))
        assert source.champ(0.0, 2.0) == pytest.approx((-2.0, 0.0))

    def test_fil_courant_derive_de_a_z(self, gradient):
        source = FilCourant(-1.2, (0.3, 0.1))
        dax, day = gradient(source.flux_plan, 1.1, -0.6)
        assert source.champ(1.1, -0.6) == pytest.approx((day, -dax), rel=1e-6)

    @pytest.mark.parametrize("z", [-3.0, -0.5, 0.0, 0.2, 2.0])
    def test_spire_sur_son_axe(self, z):
        spire = Spire(1.3, (0.5, -0.2), 0.7, angle=0.4)
        ux, uy = math.cos(0.4), math.sin(0.4)
        bx, by = spire.champ(0.5 + z * ux, -0.2 + z * uy)
        attendu = MU0 * 1.3 * 0.7**2 / (2 * (0.7**2 + z**2) ** 1.5)
        assert (bx, by) == pytest.approx((attendu * ux, attendu * uy), rel=1e-12, abs=1e-14)

    @pytest.mark.parametrize(("x", "y"), [(0.3, 0.4), (-1.2, 0.95), (2.0, -1.5), (0.01, 1.3)])
    def test_spire_comme_biot_et_savart(self, x, y):
        spire = Spire(0.8, (0.1, 0.2), 1.0, angle=0.3)
        assert spire.champ(x, y) == pytest.approx(biot_savart_spire(spire, x, y), rel=1e-9)

    @pytest.mark.parametrize("rho", [1e-9, 1e-4])
    def test_spire_pres_de_l_axe_raccord_du_developpement(self, rho):
        # De part et d'autre du seuil du développement limité, B_ρ suit
        # 3π I a² h ρ/(a² + h²)^(5/2).
        h = 0.6
        _, by = Spire(1.0, (0, 0), 1.0).champ(h, rho)
        assert by == pytest.approx(3 * math.pi * h * rho / (1 + h * h) ** 2.5, rel=1e-6)

    def test_spire_loin_ressemble_a_un_dipole(self):
        spire = Spire(1.5, (0, 0), 0.2, angle=0.9)
        dipole = DipoleMagnetique(1.5 * math.pi * 0.2**2, (0, 0), angle=0.9)
        x, y = np.array([10.0, -7.0]), np.array([3.0, 8.0])
        ecart = np.hypot(*np.subtract(spire.champ(x, y), dipole.champ(x, y)))
        np.testing.assert_array_less(ecart, 1e-3 * np.hypot(*dipole.champ(x, y)))

    def test_spire_derive_de_sa_fonction_de_flux(self, gradient):
        spire = Spire(1.1, (0, 0), 0.8)
        axe = Axe((0, 0), 0.0)
        x, y = 0.4, 0.5
        dpz, dprho = gradient(lambda a, b: spire.flux_revolution(a, b, axe), x, y)
        assert spire.champ(x, y) == pytest.approx((dprho / y, -dpz / y), rel=1e-6)

    def test_flux_de_spire_loin_comme_dipole(self):
        axe = Axe((0, 0), 0.0)
        spire = Spire(1.0, (0, 0), 0.1)
        dipole = DipoleMagnetique(math.pi * 0.01, (0, 0))
        assert spire.flux_revolution(3.0, 4.0, axe) == pytest.approx(
            dipole.flux_revolution(3.0, 4.0, axe), rel=1e-3
        )

    def test_solenoide_au_centre(self):
        # Solénoïde fini : B = μ0 n I (L/2) / sqrt((L/2)² + R²) au centre.
        solenoide = Solenoide(1.0, (0, 0), 1.0, 10.0, 400)
        bx, by = solenoide.champ(0.0, 0.0)
        attendu = MU0 * 40 * 1.0 * 5 / math.sqrt(25 + 1)
        assert (bx, by) == pytest.approx((attendu, 0.0), rel=1e-4, abs=1e-9)

    def test_solenoide_est_la_somme_de_ses_spires(self):
        solenoide = Solenoide(0.5, (1, 2), 0.6, 3.0, 7, angle=1.0)
        somme = Lots(solenoide.composantes()).champ(0.4, 0.1)
        assert solenoide.champ(0.4, 0.1) == pytest.approx(somme)

    def test_dipole_magnetique_sans_potentiel(self):
        with pytest.raises(TypeError):
            DipoleMagnetique(1, (0, 0)).potentiel(1.0, 1.0)


class TestSingularites:
    def test_spire_perce_la_figure_en_deux_points(self):
        sortant, entrant = Spire(1, (0, 0), 2.0).singularites()
        assert sortant.position == pytest.approx((0, 2)) and sortant.poids == 1
        assert entrant.position == pytest.approx((0, -2)) and entrant.poids == -1

    def test_solenoide_deux_points_par_spire(self):
        assert len(Solenoide(1, (0, 0), 1, 2, 5).singularites()) == 10

    def test_dipole_aboutissant_et_oriente(self):
        (s,) = Dipole(1, (0, 0), 0.5).singularites()
        assert s.aboutissante and s.genre == "dipole" and s.angle == 0.5

    def test_fil_courant_non_aboutissant(self):
        assert not FilCourant(1, (0, 0)).singularites()[0].aboutissante


def test_source_abstraite():
    with pytest.raises(TypeError):
        Source()  # type: ignore[abstract]


def test_position_en_liste_convertie():
    assert Charge(1, [1, 2]).position == (1.0, 2.0)


def test_lots_somme_des_sources():
    sources = [Charge(1, (0, 0)), Charge(-2, (1, 0)), Spire(1, (0, 2), 0.5)]
    total = Lots(sources).champ(0.3, 0.7)
    attendu = np.sum([s.champ(0.3, 0.7) for s in sources], axis=0)
    assert total == pytest.approx(tuple(attendu))


def test_lots_champ_point_egal_au_champ():
    lots = Lots([Charge(1, (0, 0)), Dipole(1, (1, 1), 0.3)])
    assert lots.champ_point(0.5, -0.2) == pytest.approx(lots.champ(0.5, -0.2))


class TestConstantesSI:
    def test_k_coulomb_est_un_sur_quatre_pi_epsilon_zero(self):
        assert pytest.approx(1 / (4 * math.pi * 8.8541878128e-12), rel=1e-9) == K_COULOMB

    def test_mu_zero_sur_quatre_pi(self):
        assert pytest.approx(1.25663706212e-6 / (4 * math.pi), rel=1e-8) == MU0_SUR_4PI

    def test_coulomb_en_si(self):
        # Deux charges de 1 µC à 1 m : E = 8 988 V/m.
        ex, _ = Charge(1e-6, (0, 0)).champ(1.0, 0.0)
        assert K_COULOMB * ex == pytest.approx(8988, rel=1e-3)

    def test_fil_en_si(self):
        # 10 A à 1 cm : B = μ0 I/(2π r) = 0,2 mT.
        _, by = FilCourant(10, (0, 0)).champ(0.01, 0.0)
        assert MU0_SUR_4PI * by == pytest.approx(2e-4, rel=1e-9)


class TestChampUniforme:
    def test_constant_partout(self):
        x, y = np.array([0.0, 3.0, -7.0]), np.array([1.0, -2.0, 5.0])
        ex, ey = ChampUniforme(E=(1.5, -0.5)).champ(x, y)
        np.testing.assert_allclose(ex, 1.5)
        np.testing.assert_allclose(ey, -0.5)

    def test_nature_selon_l_argument(self):
        assert ChampUniforme(E=(1, 0)).nature == "E"
        assert ChampUniforme(B=[0, 2]).nature == "B"
        assert ChampUniforme(B=[0, 2]).vecteur == (0.0, 2.0)

    @pytest.mark.parametrize("arguments", [{}, {"E": (1, 0), "B": (0, 1)}, {"E": (0, 0)}])
    def test_arguments_refuses(self, arguments):
        with pytest.raises(ValueError, match="ChampUniforme"):
            ChampUniforme(**arguments)

    def test_scalaire_refuse(self):
        with pytest.raises(ValueError, match=r"B=\(Bx, By\)"):
            ChampUniforme(B=1.0)  # type: ignore[arg-type]

    def test_sans_singularite(self):
        assert ChampUniforme(E=(1, 0)).singularites() == ()

    def test_magnetique_sans_potentiel(self):
        with pytest.raises(TypeError):
            ChampUniforme(B=(1, 0)).potentiel(0.0, 0.0)

    def test_derive_de_sa_fonction_de_flux_plane(self, gradient):
        source = ChampUniforme(B=(0.7, -1.1))
        dax, day = gradient(source.flux_plan, 0.4, 0.9)
        assert source.champ(0.4, 0.9) == pytest.approx((day, -dax), rel=1e-6)

    def test_flux_de_revolution_autour_d_un_axe_parallele(self, gradient):
        source = ChampUniforme(B=(-2.0, 0.0))
        axe = Axe((0, 0), 0.0)
        x, y = 0.3, 0.8
        dpz, dprho = gradient(lambda a, b: source.flux_revolution(a, b, axe), x, y)
        assert source.champ(x, y) == pytest.approx((dprho / y, -dpz / y), rel=1e-6)

    def test_flux_de_revolution_axe_oblique_refuse(self):
        with pytest.raises(ValueError, match="parallèle"):
            ChampUniforme(B=(1, 0)).flux_revolution(0.0, 0.0, Axe((0, 0), 0.3))


def champ_de_fils(fabrique, surfacique, x, y, demi_longueur=1000.0, pas=0.05):
    """Le champ d'une nappe plane le long de Ox, comme somme de fils serrés."""
    abscisses = np.arange(-demi_longueur, demi_longueur, pas) + pas / 2
    return Lots([fabrique(surfacique * pas, (a, 0.0)) for a in abscisses]).champ(x, y)


class TestPlans:
    def test_plan_charge_comme_une_nappe_de_fils(self):
        plan = PlanCharge(0.3, (0, 0))
        np.testing.assert_allclose(
            plan.champ(0.0, 0.5), champ_de_fils(FilCharge, 0.3, 0.0, 0.5), rtol=1e-3, atol=1e-6
        )
        assert plan.champ(0.0, 0.5) == pytest.approx((0.0, 2 * math.pi * 0.3))
        assert plan.champ(0.0, -0.5) == pytest.approx((0.0, -2 * math.pi * 0.3))

    def test_nappe_de_courant_comme_une_nappe_de_fils(self):
        nappe = NappeCourant(0.3, (0, 0))
        np.testing.assert_allclose(
            nappe.champ(0.0, 0.5), champ_de_fils(FilCourant, 0.3, 0.0, 0.5), rtol=1e-3, atol=1e-6
        )
        assert nappe.champ(0.0, 0.5) == pytest.approx((-2 * math.pi * 0.3, 0.0))
        assert nappe.champ(0.0, -0.5) == pytest.approx((2 * math.pi * 0.3, 0.0))

    def test_plan_oblique(self):
        plan = PlanCharge(1.0, (1, 1), angle=math.pi / 4)
        # Du côté de la normale (−1, 1)/√2, le champ est 2π n.
        attendu = 2 * math.pi / math.sqrt(2)
        assert plan.champ(0.0, 2.0) == pytest.approx((-attendu, attendu))

    def test_nappe_derive_de_a_z(self, gradient):
        nappe = NappeCourant(-0.8, (0.2, -0.1), angle=0.5)
        dax, day = gradient(nappe.flux_plan, 1.0, 0.7)
        assert nappe.champ(1.0, 0.7) == pytest.approx((day, -dax), rel=1e-6)

    def test_plan_charge_flux_de_revolution(self, gradient):
        plan = PlanCharge(0.5, (0.5, 0), angle=math.pi / 2)  # le plan x = 0,5
        axe = Axe((0, 0), 0.0)
        x, y = 1.3, 0.8
        dpz, dprho = gradient(lambda a, b: plan.flux_revolution(a, b, axe), x, y)
        assert plan.champ(x, y) == pytest.approx((dprho / y, -dpz / y), rel=1e-6)
        with pytest.raises(ValueError, match="perpendiculaire"):
            plan.flux_revolution(0.0, 0.0, Axe((0, 0), 0.3))

    def test_sans_singularite(self):
        assert PlanCharge(1, (0, 0)).singularites() == NappeCourant(1, (0, 0)).singularites() == ()

    def test_condensateur_plan(self):
        # Deux plans de charges opposées : E = 4πσ entre eux, nul dehors.
        armatures = Lots([PlanCharge(1, (0, 1)), PlanCharge(-1, (0, -1))])
        assert armatures.champ(0.0, 0.0) == pytest.approx((0.0, -4 * math.pi))
        assert armatures.champ(0.0, 2.0) == pytest.approx((0.0, 0.0))


class TestAimant:
    def test_est_le_solenoide_equivalent(self):
        aimant = Aimant(2.0, (1, 2), 0.5, 3.0, angle=0.7, spires=9)
        solenoide = Solenoide(2.0 * 3.0 / 9, (1, 2), 0.5, 3.0, 9, angle=0.7)
        x, y = np.array([0.4, 2.0]), np.array([0.1, 3.5])
        np.testing.assert_allclose(aimant.champ(x, y), solenoide.champ(x, y))
        assert aimant.singularites() == solenoide.singularites()

    def test_champ_au_centre_d_un_aimant_long(self):
        # B = μ0 M (L/2)/sqrt((L/2)² + R²), soit 4π M en unités réduites.
        aimant = Aimant(1.0, (0, 0), 0.1, 20.0, spires=2000)
        bx, by = aimant.champ(0.0, 0.0)
        assert bx == pytest.approx(MU0 * 10 / math.sqrt(100 + 0.01), rel=1e-4)
        assert by == pytest.approx(0.0, abs=1e-9)

    def test_neodyme_en_si(self):
        # μ0 M = 1,3 T (néodyme) : au centre d'un aimant long, B ≈ 1,3 T.
        aimantation = 1.3 / (4 * math.pi * MU0_SUR_4PI)
        aimant = Aimant(aimantation, (0, 0), 0.01, 1.0, spires=2000)
        bx, _ = aimant.champ(0.0, 0.0)
        assert MU0_SUR_4PI * bx == pytest.approx(1.3 * 0.5 / math.hypot(0.5, 0.01), rel=1e-3)

    def test_loin_ressemble_a_un_dipole(self):
        aimant = Aimant(1.5, (0, 0), 0.2, 1.0, angle=0.9)
        dipole = DipoleMagnetique(1.5 * math.pi * 0.2**2 * 1.0, (0, 0), angle=0.9)
        x, y = np.array([10.0, -7.0]), np.array([3.0, 8.0])
        ecart = np.hypot(*np.subtract(aimant.champ(x, y), dipole.champ(x, y)))
        np.testing.assert_array_less(ecart, 1e-2 * np.hypot(*dipole.champ(x, y)))

    def test_poles_dans_le_sens_de_l_aimantation(self):
        sud, nord = Aimant(1, (0, 0), 0.5, 2.0, angle=math.pi / 2).poles()
        assert sud == pytest.approx((0, -1)) and nord == pytest.approx((0, 1))
        sud, nord = Aimant(-1, (0, 0), 0.5, 2.0, angle=math.pi / 2).poles()
        assert sud == pytest.approx((0, 1)) and nord == pytest.approx((0, -1))

    def test_sommets_du_rectangle(self):
        sommets = Aimant(1, (1, 0), 0.5, 2.0).sommets()
        np.testing.assert_allclose(sommets, [(0, -0.5), (2, -0.5), (2, 0.5), (0, 0.5)])

    def test_flux_de_revolution(self):
        aimant = Aimant(1.0, (0, 0), 0.3, 1.0, spires=8)
        axe = Axe((0, 0), 0.0)
        assert aimant.flux_revolution(0.2, 0.7, axe) == pytest.approx(
            aimant.solenoide().flux_revolution(0.2, 0.7, axe)
        )

    @pytest.mark.parametrize(
        "arguments", [(1, (0, 0), 0, 1), (1, (0, 0), 1, -1), (1, (0, 0), 1, 1, 0, 0)]
    )
    def test_dimensions_refusees(self, arguments):
        with pytest.raises(ValueError):
            Aimant(*arguments)
