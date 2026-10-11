# fieldlines

[![PyPI](https://img.shields.io/pypi/v/fieldlines.svg)](https://pypi.org/project/fieldlines/)
[![CI](https://github.com/antnardo/fieldlines/actions/workflows/ci.yml/badge.svg)](https://github.com/antnardo/fieldlines/actions/workflows/ci.yml)
[![Licence MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](https://github.com/antnardo/fieldlines/blob/main/LICENSE)

Lignes de champ électrique et magnétique, et équipotentielles, pour des figures de
cours : des charges, des fils et des plans chargés, des dipôles, des fils et des nappes
de courant, des spires, des solénoïdes, des aimants, un champ uniforme. On les pose
dans une carte, on choisit où partent les lignes, on retouche celles qui gênent en les
désignant par leur nom, et on trace avec matplotlib.

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

Le paquet s'installe depuis PyPI (`pip install fieldlines`) ou depuis GitHub (ci-dessous).

## Sommaire

- [Ce qu'il fait](#ce-quil-fait)
- [Installation](#installation)
- [Prérequis](#prérequis)
- [Prise en main](#prise-en-main)
- [Les sources](#les-sources)
- [Unités réduites et valeurs SI](#unités-réduites-et-valeurs-si)
- [Recettes pour le champ magnétique](#recettes-pour-le-champ-magnétique)
- [Pièges](#pièges)
- [Les modules](#les-modules)
- [Le calcul des lignes](#le-calcul-des-lignes)
- [Exemples](#exemples)
- [Tester](#tester)
- [Antériorité](#antériorité)
- [Licence](#licence)

## Ce qu'il fait

- **Des lignes intégrées sur le champ exact**, pas interpolé sur une grille : chaque
  ligne suit `scipy.integrate.solve_ivp` et s'arrête sur un événement localisé
  exactement — arrivée sur une charge (la ligne la touche), sortie du cadre, ligne
  refermée (champ magnétique), point de champ nul.
- **Des départs qui ont un sens physique.** Autour des charges à angles réguliers ;
  **selon le flux**, en nombre proportionnel à la charge et sans ligne tracée deux
  fois ; le long d'un segment, à flux égal dans la figure ou dans l'espace.
- **Des lignes nommées et retouchables** : `ajuste(12, fleches=[0.3, 0.8])`,
  `retire(range(23, 29))`, des numéros qui ne bougent jamais.
- **Le champ magnétique à la hauteur du champ électrique** : spire exacte (intégrales
  elliptiques), solénoïde, bobines de Helmholtz, aimant, nappe de courant, champ
  uniforme ; lignes fermées ; fonction de flux Ψ exacte, dont les lignes de niveau sont
  les lignes de champ, pour des tubes de flux égal.
- **Des sources personnalisées** en une vingtaine de lignes (`Source`), pour un
  potentiel imposé ou un champ qui n'est celui d'aucune distribution connue.
- **Des unités réduites** pour les lignes, et deux constantes pour revenir aux volts et
  aux teslas.

## Installation

**Avec pip depuis PyPI**, quand le réseau et pip sont disponibles :

```bash
pip install fieldlines
```

(ou `uv add fieldlines`). **Depuis GitHub**, pour la version de développement :

```bash
pip install git+https://github.com/antnardo/fieldlines
```

**Avec la roue d'une release**, pour un poste qui a pip mais pas git : chaque
[release](https://github.com/antnardo/fieldlines/releases) porte
`fieldlines-<version>-py3-none-any.whl` (et le sdist `.tar.gz`, avec `examples/`), qui
s'installe par `pip install fieldlines-<version>-py3-none-any.whl`.

**Copier le dossier**, sans pip ni réseau : le dossier `src/fieldlines/` se pose à côté
des scripts, et `from fieldlines import Carte, Charge` fonctionne, à condition
d'exécuter le script depuis son dossier.

**Pour développer**, un clone et [uv](https://docs.astral.sh/uv/) :

```bash
git clone https://github.com/antnardo/fieldlines
cd fieldlines && uv sync && uv run pytest
```

## Prérequis

Python 3.11 ou plus récent, avec numpy (1.26 ou plus), scipy (1.11 ou plus) et
matplotlib (3.8 ou plus). Le paquet est testé sous Python 3.11 et 3.14.

## Prise en main

`Carte(sources, cadre)` prend la liste des sources et le cadre `(xmin, xmax, ymin, ymax)`
de la figure ; une carte ne mêle pas sources électriques et magnétiques. Les positions
sont des couples `(x, y)`, les angles sont en radians, comptés depuis Ox.

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
ou retire ensuite par nom, autant de fois qu'on veut, et on relit ce qu'on a réglé :

```python
carte.lignes_autour(nombre=16, rayon=0.2)
carte.ajuste(30, 38, fleches=[0.45])          # flèche à 45 % de la longueur visible
carte.ajuste(12, marge=2, style={"color": "r"})
carte.retire(range(23, 29), 47)
carte.options(12).fleches, carte.depart(12)   # les réglages et le départ de la ligne 12
```

Les réglages d'une ligne (`Options`, dans `fieldlines.integration`) :

| Réglage | Défaut | Rôle |
| --- | --- | --- |
| `fleches` | `(0.5,)` | positions des flèches, en fractions de la longueur visible ; une liste ou un seul nombre |
| `marge` | `0.25` | le calcul continue jusqu'à cette fraction du cadre au-delà de ses bords |
| `sens` | `"deux"` | `"aval"` ou `"amont"` pour ne suivre la ligne que d'un côté du départ |
| `longueur_max` | 20 diagonales | longueur au-delà de laquelle on abandonne |
| `style` | `{}` | arguments de `Axes.plot` propres à la ligne (ses flèches prennent sa couleur) |

Le tracé :

| Méthode | Ce qu'elle trace |
| --- | --- |
| `trace_lignes(ax, **style)` | les lignes (noires par défaut) |
| `trace_fleches(ax, taille=8, style="fancy")` | les flèches, taille en points |
| `trace_noms(ax)` | le nom de chaque ligne, pour choisir quoi retoucher |
| `trace_equipotentielles(ax, niveaux, contraste=3, cmap="RdYlBu_r", norm=None)` | les équipotentielles (champ électrique) |
| `trace_lignes_de_flux(ax, niveaux)` | les lignes de champ comme lignes de niveau de la fonction de flux |
| `trace_sources(ax, taille=10)` | charges (aire ∝ \|q\|), dipôles, courants ⊙ et ⊗, plans, aimants |
| `trace(ax)` | lignes, flèches et sources, cadre et repère orthonormé |
| `enregistre(chemin, figsize=None)` | `trace` dans une figure neuve, enregistrée puis fermée |

Les niveaux des équipotentielles viennent de `carte.niveaux(nombre, echelle, symetrique,
vmin, vmax, seuil, ajoute)` : linéaires ou logarithmiques (`echelle="log"`), symétriques
(± une progression de `seuil` × max(|V|) au maximum de |V|, `seuil=1e-3` par défaut)
ou non, entre les extrêmes du potentiel sur la grille ou entre des bornes données, plus
des niveaux ajoutés. Sur une carte magnétique, `niveaux` porte sur la fonction de flux
(`grandeur="flux"`), pour `trace_lignes_de_flux`. Les couleurs couvrent
[Vmin, Vmax]/10^`contraste` ; `trace_equipotentielles` rend le `QuadContourSet`, à
passer à `ax.clabel` pour écrire les valeurs, ou dont on réutilise le `norm` pour
tracer une autre carte avec les mêmes couleurs.

Les valeurs elles-mêmes : `carte.champ(x, y)`, `carte.potentiel(x, y)` et
`carte.flux(x, y)` (vectorisés) ; `carte.grille("potentiel")` ou `carte.grille("flux")`
sur une grille du cadre.

## Les sources

Chaque source électrique a son symétrique magnétique quand il existe. Les formules sont
en unités réduites (ci-dessous).

| Électrique | Magnétique | Champ |
| --- | --- | --- |
| `Charge(q, position)` | — | E = q/r², V = q/r |
| `FilCharge(lineique, position)` | `FilCourant(intensite, position)` | E = 2λ/r, V = −2λ ln r ; B = 2I/r, courant vers le lecteur si I > 0 (⊙) |
| `PlanCharge(surfacique, position, angle)` | `NappeCourant(surfacique, position, angle)` | E = 2πσ normal au plan ; B = 2πj parallèle à la nappe, opposé de part et d'autre |
| `Dipole(moment, position, angle)` | `DipoleMagnetique(moment, position, angle)` | (3 (p·r) r − p r²)/r⁵, dipôle ponctuel de l'espace |
| — | `Spire(intensite, centre, rayon, angle)` | exact, intégrales elliptiques ; vue par la tranche, `angle` : son axe |
| — | `Solenoide(intensite, centre, rayon, longueur, spires, angle)` | somme de spires, vu en coupe |
| — | `Aimant(aimantation, centre, rayon, hauteur, angle, spires=64)` | le solénoïde équivalent, dessiné avec ses pôles N et S |
| `ChampUniforme(E=(Ex, Ey))` | `ChampUniforme(B=(Bx, By))` | uniforme, s'ajoute aux autres sources |

Fils, plans et nappes sont perpendiculaires à la figure (problème plan) ; spires,
solénoïdes, aimants et dipôles sont dans un plan méridien qui contient leur axe. Un
plan ou une nappe est la droite qui passe par `position` avec l'angle `angle`.

Une source personnalisée hérite de `Source` et donne son noyau vectorisé (`lots`) et
ses singularités (`singularites`), plus `potentiel`, `flux_plan` ou `flux_revolution`
si elle les a : `fieldlines.sources` en est la documentation, `ChampUniforme` le plus
court modèle ; un potentiel imposé sur une frontière en est un autre.

## Unités réduites et valeurs SI

Le paquet calcule avec 1/(4π ε0) = 1 et μ0/(4π) = 1 : une charge q crée E = q/r², un
fil parcouru par I crée B = 2I/r. Les lignes de champ et les équipotentielles ne
dépendent pas de ce choix ; les valeurs, si. Ce que `ax.clabel` écrit sur les
équipotentielles est q/r, pas des volts.

Pour revenir au SI, les charges en coulombs (λ en C/m, σ en C/m²), les courants en
ampères (j et M en A/m) et les longueurs en mètres :

```python
from fieldlines import K_COULOMB, MU0_SUR_4PI  # 1/(4π ε0) = 8,99·10⁹ ; μ0/(4π) = 10⁻⁷

ex, ey = carte.champ(x, y)
E_si = K_COULOMB * ex, K_COULOMB * ey          # V/m, et V = K_COULOMB * carte.potentiel(x, y)
bx, by = carte_magnetique.champ(x, y)
B_si = MU0_SUR_4PI * bx, MU0_SUR_4PI * by      # T
ax.clabel(contours, fmt=lambda v: f"{K_COULOMB * v:.2g} V")
```

E et B n'ont pas le même facteur : les cartes de ce paquet ne se comparent pas d'un
champ à l'autre en valeur.

## Recettes pour le champ magnétique

**La convention d'axe.** La figure est un plan méridien. L'axe d'une spire, d'un
solénoïde ou d'un aimant passe par `centre` dans la direction `angle` (radians depuis
Ox ; Ox par défaut, donc un axe horizontal) ; le courant tourne dans le sens direct
autour de cet axe quand I > 0, et la spire perce la figure en ⊙ du côté de l'axe tourné
de +90° (en haut pour un axe horizontal) et en ⊗ de l'autre. Pour un axe vertical,
`angle=math.pi / 2`. Les fils et les nappes sont perpendiculaires à la figure, I > 0
vers le lecteur (⊙).

**Où faire partir les lignes.** Elles n'aboutissent nulle part : `lignes_autour`
refuse, et `lignes_le_long` prend un segment que chaque ligne de la famille voulue
coupe une fois, sans passer à moins du rayon d'arrêt d'une source.

| Source | Segment de départ | Exemple |
| --- | --- | --- |
| un fil | du fil vers le bord, `(r0, 0)` → `(R, 0)` ; à flux égal, les rayons des cercles sont en progression géométrique | `fils_courants.py` |
| plusieurs fils | un segment par famille de lignes (autour d'un fil, autour de deux…), depuis le fil ou depuis un point de champ nul | `fils_courants.py` |
| une spire | son diamètre dans la figure, `(0, −0.9 R)` → `(0, 0.9 R)` ; `"flux"` suit l'intensité du champ dans le dessin, `"flux_revolution"` fait des tubes de flux égaux dans l'espace | `spire.py` |
| un solénoïde | son plan médian, à l'intérieur, `(0, −0.95 R)` → `(0, 0.95 R)` | `solenoide.py` |
| des bobines de Helmholtz | le plan médian entre les deux spires, `"flux_revolution"` | `bobines_de_helmholtz.py` |
| un aimant | son plan médian, à l'intérieur ; les lignes le traversent du pôle S au pôle N et se referment dehors | `aimant.py` |
| un dipôle magnétique | `lignes_autour` convient (les lignes sortent du dipôle et y rentrent), ou un segment sur l'équateur | `dipole_magnetique.py` |
| un champ uniforme et un fil | un segment que toutes les lignes coupent, l'axe vertical par exemple | `champ_uniforme.py` (pour une charge) |

Un nombre pair de départs sur le diamètre d'une spire évite la ligne de l'axe, qui ne
se referme qu'à l'infini. Un aimant est le solénoïde équivalent : `Aimant(M, centre,
rayon, hauteur, angle)` vaut `Solenoide(M * hauteur / spires, centre, rayon, hauteur,
spires, angle)`, et se dessine en coupe avec ses pôles ; pour un aimant réel, M en A/m
et B (T) = `MU0_SUR_4PI` × le champ calculé.

**La fonction de flux.** Pour des fils de courant (problème plan), Ψ = A_z ; pour des
sources qui ont toutes le même axe de révolution (charges sur l'axe, dipôles, spires,
solénoïdes et aimants alignés sur lui, champ uniforme parallèle), Ψ est la fonction de
Stokes, ρ A_φ pour une spire. Les lignes de champ sont les lignes de niveau de Ψ :
`trace_lignes_de_flux(ax, niveaux)` les trace directement et exactement,
`carte.niveaux(nombre, grandeur="flux")` donne des niveaux, et
`lignes_le_long(..., repartition="flux_revolution")` place les départs à ΔΨ égal
(`spire.py`, panneau de droite).

## Pièges

- `lignes_autour` et `lignes_selon_flux` sur une carte magnétique lèvent `ValueError` :
  les lignes n'y aboutissent nulle part, prendre `lignes_le_long` ou `ajoute_ligne`.
- `lignes_le_long` refuse un segment qui passe à moins du rayon d'arrêt d'une source
  (`Reglages.arret` × diagonale du cadre, 10⁻³ par défaut) : les lignes s'y
  arrêteraient aussitôt. Écarter le segment, ou réduire `arret`.
- `ajuste(nom, color="r")` lève `TypeError` : un style de tracé se donne dans
  `style={"color": "r"}`. `fleches=0.3` (un seul nombre) est accepté.
- `niveaux(..., symetrique=True, seuil=s)` : `seuil` est une fraction de max(|V|), pas
  une valeur absolue. Un seuil absolu empilait des niveaux sur la surface V = 0.
- Une carte ne mêle pas E et B : deux cartes, tracées sur les mêmes axes.
- `flux`, `trace_lignes_de_flux` et `repartition="flux_revolution"` demandent une
  symétrie : fils de courant seuls, ou sources de même axe (dipôles et spires alignés
  sur lui, charges dessus). Sinon, `ValueError`.
- Les valeurs sont en unités réduites : voir ci-dessus avant de lire un `clabel`.
- `lignes_autour` trace deux fois chaque ligne qui va d'une charge à une autre ; les
  retirer par nom, ou prendre `lignes_selon_flux`.

## Les modules

| Module | Contenu |
| --- | --- |
| `fieldlines` | la tête de paquet : `Carte`, les sources, `Source`, `Singularite`, `Noyau`, `Nature`, `K_COULOMB`, `MU0_SUR_4PI` |
| `fieldlines.carte` | `Carte` |
| `fieldlines.sources` | les sources et leurs noyaux vectorisés, `Lots`, les constantes |
| `fieldlines.champ` | `Distribution` (champ, potentiel, flux, symétrie, grilles), `Grille` |
| `fieldlines.departs` | `departs_autour`, `departs_le_long`, `CercleDeFlux`, `PlacementSelonFlux` |
| `fieldlines.integration` | `Integrateur`, `Options`, `Reglages`, `LigneDeChamp`, `Arret` |
| `fieldlines.niveaux` | `niveaux` |
| `fieldlines.trace` | les fonctions de tracé (lignes, flèches, noms, contours, sources) |
| `fieldlines.geometrie` | `Cadre`, `Axe`, `distance_au_segment` |

## Le calcul des lignes

Chaque ligne est intégrée par `scipy.integrate.solve_ivp`, vers l'aval puis vers
l'amont de son départ, et s'arrête sur un événement : sortie du cadre agrandi de sa
marge, arrivée sur une source (la ligne finit alors au centre de la source), retour au
point de départ (ligne fermée du champ magnétique), ou champ nul (point d'arrêt). Près
d'une source ou d'un point d'arrêt, l'intégration ralentit pour ne pas les enjamber.
La précision se règle par `Carte(..., reglages=Reglages(...))` ; `verbose=True` écrit
comment chaque ligne s'est arrêtée. Le champ se calcule par lots : un noyau numpy par
sorte de source, pour tous les points et toutes les sources d'un coup.

## Exemples

![Quadrupôle : lignes de champ et équipotentielles, à gauche retouchées à la main, à droite placées selon le flux](https://raw.githubusercontent.com/antnardo/fieldlines/main/examples/images/quadrupole.png)

La galerie [`examples/README.md`](examples/README.md) décrit chaque exemple avec sa
figure. Chaque script enregistre sa figure en PDF à côté de lui.

## Tester

```bash
uv run --python 3.11 pytest
uv run --python 3.14 pytest
uv run ruff check src tests examples && uv run ruff format --check src tests examples
uv run mypy
uv build
```

Les tests confrontent les champs aux formules (Coulomb, fil infini, nappe de fils, axe
d'une spire, loi de Biot et Savart, dipôle lointain, solénoïde, aimant) et vérifient
les propriétés des lignes : tangentes au champ, d'une charge positive à une négative,
potentiel décroissant, lignes du champ magnétique fermées et à flux constant.
L'intégration continue (`.github/workflows/ci.yml`) rejoue tout cela à chaque push
sous Python 3.11 et 3.14 ; à chaque release publiée, `publish.yml` vérifie que
l'étiquette est la version du paquet et joint la roue et le sdist à la release.

## Antériorité

La méthode — suivre pas à pas la direction du champ depuis un point de départ — vient
du billet de H. Bauke, [Visualizing streamlines](http://numbercrunch.de/blog/2013/05/visualizing-streamlines/)
(numbercrunch.de, 2013). Le paquet la développe depuis 2017 pour les figures d'un
cours d'électromagnétisme, et y ajoute le champ magnétique.

`matplotlib.streamplot` trace des lignes interpolées sur une grille, brisées, avec les
flèches près des départs ; rien n'y atteint exactement une charge et une boucle y est
reparcourue. [magpylib](https://magpylib.readthedocs.io/) calcule le champ d'aimants et
de bobines en trois dimensions (sa spire et son solénoïde coïncident avec ceux de ce
paquet à 10⁻¹⁰ près) mais trace par `streamplot` ; il est le bon outil pour un aimant
réel ou du 3D. Le placement des lignes selon le flux est propre au paquet : il ne
reprend pas le code du paquet [electrostatics](https://github.com/tomduck/electrostatics)
de T. J. Duck (GPL), qui résout le même problème autrement. Sans équivalent ailleurs :
les départs selon le flux sans doublon, l'arrêt exact, les flèches à une fraction de
la longueur visible, la fonction de Stokes, la retouche par nom.

## Licence

Le code — le paquet (`src/`), les tests et les scripts — est sous licence MIT
(`LICENSE`). Les exemples (`examples/`), qui sont du contenu scientifique et
pédagogique plutôt que des algorithmes, sont sous licence Creative Commons
Attribution - Pas d'utilisation commerciale - Partage dans les mêmes conditions 4.0
International, CC BY-NC-SA 4.0 (`LICENSE-CC-BY-NC-SA-4.0`).
