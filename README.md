# fieldlines

Lignes de champ électrique et magnétique, et équipotentielles, pour des figures de
cours : des charges, des fils chargés, des dipôles, des fils parcourus par un courant,
des spires, des solénoïdes. On les pose dans une carte, on choisit où partent les
lignes, on retouche celles qui gênent en les désignant par leur nom, et on trace avec
matplotlib.

```python
import matplotlib.pyplot as plt
from fieldlines import Carte, Charge

carte = Carte([Charge(2, (-1, 0)), Charge(-1, (1, 0))], cadre=(-4, 4, -3, 3))
carte.lignes_selon_flux(nombre=16)

fig, ax = plt.subplots()
carte.trace(ax)  # lignes, flèches, charges
carte.trace_equipotentielles(ax, carte.niveaux(20, echelle="log", symetrique=True))
plt.show()
```

## Origine

La méthode — suivre pas à pas la direction du champ depuis un point de départ — vient
du billet de H. Bauke, [Visualizing streamlines](http://numbercrunch.de/blog/2013/05/visualizing-streamlines/)
(numbercrunch.de, 2013). Le paquet reprend et développe `efieldlines`, module écrit
en 2017 à partir de ce billet pour les figures du cours d'électromagnétisme, et y
ajoute le champ magnétique.

Le placement des lignes selon le flux est propre au paquet : il ne reprend pas le code
du paquet [electrostatics](https://github.com/tomduck/electrostatics) de T. J. Duck
(GPL), qui résout le même problème autrement.

## Installer

Python 3.11 ou plus récent, avec numpy, scipy et matplotlib.

```bash
pip install -e .            # ou mettre src/ dans le PYTHONPATH
uv sync --group dev         # pour développer : pytest, ruff, mypy
```

## Les sources

Unités réduites : 1/(4π ε0) = 1 et μ0/(4π) = 1. Pour revenir au SI, multiplier E par
1/(4π ε0) et B par μ0/(4π) ; les lignes de champ ne dépendent pas des unités. Les
positions sont des couples (x, y), les angles sont en radians, comptés depuis Ox.

| Source | Champ | Remarque |
| --- | --- | --- |
| `Charge(q, position)` | E = q/r², V = q/r | charge ponctuelle |
| `FilCharge(lineique, position)` | E = 2λ/r, V = −2λ ln r | fil infini perpendiculaire à la figure |
| `Dipole(moment, position, angle)` | (3 (p·r) r − p r²)/r⁵ | dipôle ponctuel dans le plan |
| `FilCourant(intensite, position)` | B = 2I/r | courant vers le lecteur si I > 0 (⊙) |
| `Spire(intensite, centre, rayon, angle)` | exact, intégrales elliptiques | vue par la tranche ; `angle` : son axe |
| `Solenoide(intensite, centre, rayon, longueur, spires, angle)` | somme de spires | vu en coupe |
| `DipoleMagnetique(moment, position, angle)` | comme `Dipole` | petite spire, aimant vu de loin |

Une carte ne mêle pas sources électriques et magnétiques.

## La carte

`Carte(sources, cadre)` prend la liste des sources et le cadre `(xmin, xmax, ymin, ymax)`.
On y place des lignes de quatre façons, qui se combinent :

- `lignes_autour(nombre, rayon, angle)` : `nombre` lignes autour de chaque source où
  des lignes aboutissent (charges, fils chargés, dipôles), régulièrement espacées en
  angle (en sin θ pour un dipôle). Une ligne qui va d'une charge à une autre est tracée
  depuis chacune ;
- `lignes_selon_flux(nombre, rayon, angle)` : à flux égal, en nombre proportionnel à
  la charge (`nombre` pour la plus forte), sans tracer deux fois la même ligne : autour
  d'une charge négative, on n'ajoute que les lignes qui ne lui sont pas déjà arrivées ;
- `lignes_le_long(debut, fin, nombre, repartition)` : depuis un segment, à flux égal
  du champ de la figure (`"flux"`), à flux égal dans l'espace pour des sources de
  révolution (`"flux_revolution"`), ou à intervalles réguliers (`"reguliere"`). C'est
  ainsi qu'on place les lignes du champ magnétique, qui n'aboutissent nulle part ;
- `ajoute_ligne(depart, nom=None)` : une ligne à la main.

Chaque ligne a un nom : un numéro, attribué dans l'ordre de création et jamais
réutilisé, ou un texte choisi. `trace_noms(ax)` les écrit sur la figure ; on retouche
ou retire ensuite par nom, autant de fois qu'on veut :

```python
carte.lignes_autour(nombre=16, rayon=0.2)
carte.ajuste(30, 38, fleches=[0.45])          # flèche à 45 % de la longueur visible
carte.ajuste(12, marge=2, style={"color": "r"})
carte.retire(range(23, 29), 47)
```

Les réglages d'une ligne (`Options`) :

| Réglage | Défaut | Rôle |
| --- | --- | --- |
| `fleches` | `(0.5,)` | positions des flèches, en fractions de la longueur visible |
| `marge` | `0.25` | le calcul continue jusqu'à cette fraction du cadre au-delà de ses bords |
| `sens` | `"deux"` | `"aval"` ou `"amont"` pour ne suivre la ligne que d'un côté du départ |
| `longueur_max` | 20 diagonales | longueur au-delà de laquelle on abandonne |
| `style` | `{}` | arguments de `Axes.plot` propres à la ligne (ses flèches prennent sa couleur) |

## Le tracé

| Méthode | Ce qu'elle trace |
| --- | --- |
| `trace_lignes(ax, **style)` | les lignes (noires par défaut) |
| `trace_fleches(ax, taille=8, style="fancy")` | les flèches, taille en points |
| `trace_noms(ax)` | le nom de chaque ligne, pour choisir quoi retoucher |
| `trace_equipotentielles(ax, niveaux, contraste=3, cmap="RdYlBu_r", norm=None)` | les équipotentielles |
| `trace_lignes_de_flux(ax, niveaux)` | les lignes de champ comme lignes de niveau de la fonction de flux |
| `trace_sources(ax, taille=10)` | charges (aire ∝ \|q\|), dipôles, courants ⊙ et ⊗ |
| `trace(ax)` | lignes, flèches et sources, cadre et repère orthonormé |

Les niveaux des équipotentielles viennent de `carte.niveaux(nombre, echelle, symetrique,
vmin, vmax, seuil, ajoute)` : linéaires ou logarithmiques (`echelle="log"`), symétriques
(± une progression de `seuil` au maximum de |V|) ou non, entre les extrêmes du
potentiel sur la grille ou entre des bornes données, plus des niveaux ajoutés. Les
couleurs couvrent [Vmin, Vmax]/10^`contraste` ; `trace_equipotentielles` rend le
`QuadContourSet`, à passer à `ax.clabel` pour écrire les valeurs, ou dont on réutilise
le `norm` pour tracer une autre carte avec les mêmes couleurs.

## La fonction de flux

Pour des fils de courant (problème plan), Ψ = A_z ; pour des sources qui ont toutes le
même axe de révolution (charges sur l'axe, dipôles et spires alignés sur lui), Ψ est la
fonction de Stokes, ρ A_φ pour une spire. Les lignes de champ sont les lignes de niveau
de Ψ : `trace_lignes_de_flux` les trace directement, et `lignes_le_long(...,
repartition="flux_revolution")` place les départs à ΔΨ égal. `carte.distribution.flux(x, y)`
en donne la valeur.

## Le calcul des lignes

Chaque ligne est intégrée par `scipy.integrate.solve_ivp`, vers l'aval puis vers
l'amont de son départ, et s'arrête sur un événement : sortie du cadre agrandi de sa
marge, arrivée sur une source (la ligne finit alors au centre de la source), retour au
point de départ (ligne fermée du champ magnétique), ou champ nul (point d'arrêt). Près
d'une source ou d'un point d'arrêt, l'intégration ralentit pour ne pas les enjamber.
La précision se règle par `Carte(..., reglages=Reglages(...))` ; `verbose=True` écrit
comment chaque ligne s'est arrêtée.

## Passer d'efieldlines à fieldlines

| efieldlines | fieldlines |
| --- | --- |
| `Figure(charges, xmin, xmax, ymin, ymax)` | `Carte(charges, cadre=(xmin, xmax, ymin, ymax))` |
| `Ligne(q, pos)` | `FilCharge(q / 2, pos)` |
| `Dipole(p, pos, alpha=a)` | `Dipole(p, pos, angle=math.pi / 2 - a)` |
| `set_default_SL_integration_parameters(...)` | inutile : réglages par défaut, ou par ligne |
| `create_SL_with_automatic_starting_points(R, N)` | `lignes_autour(nombre=N, rayon=R)` |
| `append_new_SL(x0, y0, arrow_positions=…, enlarge=c)` | `ajoute_ligne((x0, y0), fleches=…, marge=(c - 1) / 2)` (cadre centré en 0) |
| `tweak_SL(i, ...)` | `ajuste(i, ...)` |
| `remove_SL([i, j])` | `retire(i, j)`, sans décalage des numéros |
| `compute_streamlines()`, `compute_arrows(...)` | inutiles : le calcul se fait au tracé |
| `compute_potential(gridsize)` | inutile, ou `trace_equipotentielles(..., points=gridsize)` |
| `set_automatic_levels(Nlevels, logscale, symmetric, Vmin, Vmax, deltaV)` | `niveaux(nombre, echelle, symetrique, vmin, vmax, seuil)` |
| `set_levels(l)`, `add_levels(l)` | une liste de niveaux, ou `niveaux(..., ajoute=l)` |
| `plot_streamlines(ax)`, `plot_arrows(ax)` | `trace_lignes(ax)`, `trace_fleches(ax)` |
| `display_streamline_identifier(ax)` | `trace_noms(ax)` |
| `plot_contours(ax, contraste, cmap, scale)` | `trace_equipotentielles(ax, niveaux, contraste, cmap, norm)` |
| `plot_contours_label(ax)` | `ax.clabel(trace_equipotentielles(...))` |
| `plot_charges(ax, chargesize)` | `trace_sources(ax, taille)` |

Les flèches se placent désormais sur la partie visible de la ligne, et leurs tailles
sont en points : `arrowsize`, qui dépendait du cadre, disparaît.

## Exemples

Le dossier `examples/` reprend les figures du cours (cartes de champ de charges, de fils
chargés, quadrupôle, approximation dipolaire, exercice EM-ES15) et ajoute le champ
magnétique : fils, spire, bobines de Helmholtz, solénoïde, dipôle magnétique. Chaque
script enregistre sa figure en PDF à côté de lui.

## Développer

```bash
uv run ruff format src tests examples
uv run ruff check src tests examples
uv run mypy
uv run --python 3.11 pytest
uv run --python 3.14 pytest
```

Les tests confrontent les champs aux formules (Coulomb, fil infini, axe d'une spire,
loi de Biot et Savart, dipôle lointain, solénoïde) et vérifient les propriétés des
lignes : tangentes au champ, d'une charge positive à une négative, potentiel
décroissant, lignes du champ magnétique fermées et à flux constant.

## Licence

Le code — le paquet (`src/`), les tests et les scripts — est sous licence MIT
(`LICENSE`). Les exemples (`examples/`), qui sont du contenu scientifique et
pédagogique plutôt que des algorithmes, sont sous licence Creative Commons
Attribution - Pas d'utilisation commerciale - Partage dans les mêmes conditions 4.0
International, CC BY-NC-SA 4.0 (`LICENSE-CC-BY-NC-SA-4.0`).
