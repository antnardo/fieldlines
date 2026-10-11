# Changelog

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) et le
[versionnage sémantique](https://semver.org/lang/fr/).

## [Non publié]

### Ajouté

- La galerie des exemples `examples/README.md`, l'image de chaque exemple dans
  `examples/images/`.

## [0.1.1] - 2026-10-11

### Modifié

- Publié sur PyPI : `pip install fieldlines`. Le workflow de release y envoie la roue et le sdist par « trusted publishing », sans jeton ; GitHub garde les artefacts joints à la release.

## [0.1.0] - 2026-10-10

Première version : le champ électrique et le champ magnétique.

### Ajouté

- Sources magnétiques : fil rectiligne infini, nappe de courant plane, spire circulaire
  exacte (intégrales elliptiques complètes), solénoïde (somme de spires), aimant
  cylindrique uniformément aimanté (`Aimant`, le solénoïde équivalent, dessiné en coupe
  avec ses pôles N et S), dipôle magnétique.
- Sources électriques : plan chargé infini (`PlanCharge`), le symétrique du fil chargé ;
  et `ChampUniforme`, électrique (`E=(Ex, Ey)`) ou magnétique (`B=(Bx, By)`), qui
  s'ajoute aux autres sources (charge ou fil dans un champ uniforme).
- Constantes `K_COULOMB` = 1/(4π ε0) et `MU0_SUR_4PI` = μ0/(4π), pour passer des
  unités réduites du paquet aux valeurs SI.
- Fonction de flux (A_z pour des fils, fonction de Stokes pour des sources de
  révolution), et lignes de champ comme ses lignes de niveau ; l'axe de révolution
  peut être orienté par un champ uniforme ou un plan chargé.
- Lignes du champ magnétique fermées : l'intégration s'arrête quand la ligne repasse
  à son point de départ.
- Placement des lignes selon le flux : nombre proportionnel à la charge, et aucune
  ligne tracée deux fois ; départs à flux égal le long d'un segment.
- Lignes nommées de façon stable, réglages par ligne (flèches, marge, sens, longueur
  maximale, style) ; `options(nom)` et `depart(nom)` lisent ce que `ajuste` change.
- Sur la carte : `champ`, `potentiel` et `flux` en un point, `niveaux` de la fonction
  de flux pour une carte magnétique, `enregistre` qui trace et écrit la figure.
- Garde-fous : un segment de départ qui passe à moins du rayon d'arrêt d'une source est
  refusé ; `fleches` accepte un seul nombre ; un style de tracé donné directement à
  `ajuste` (`color=`) renvoie vers `style={...}`.
- Tests numériques : champs confrontés aux formules, à la loi de Biot et Savart, à des
  nappes de fils ; propriétés des lignes. Exemples tirés des figures du cours.
- Licences : MIT pour le code, CC BY-NC-SA 4.0 pour les exemples. Intégration
  continue (pytest sous Python 3.11 et 3.14, ruff, mypy, `uv build`) et publication
  de la roue et du sdist à chaque release GitHub.

### Modifié

- Intégration par `solve_ivp`, avec des événements pour la sortie du cadre, l'arrivée
  sur une source, la ligne refermée et le champ nul, au lieu d'un pas fixe et de
  seuils sur la norme du champ (`Emin`, `Emax`).
- Les flèches se placent à une fraction de la longueur visible de la ligne ; les
  tailles des flèches et des sources sont en points.
- Le dipôle est orienté par l'angle de son moment depuis Ox (`angle`), et non plus
  depuis Oy dans le sens horaire (`alpha`) ; le fil chargé (`FilCharge`) prend sa
  charge linéique λ.
- Le seuil des niveaux symétriques (`niveaux(..., symetrique=True, seuil=...)`) est
  relatif : une fraction de max(|vmin|, |vmax|), 10⁻³ par défaut. Un seuil absolu
  empilait des niveaux sur la surface V = 0 dès que la grille passait près d'une
  charge.
- La tête de paquet n'expose que la carte, les sources, de quoi en écrire une
  (`Source`, `Singularite`, `Noyau`, `Nature`) et les constantes ; le reste s'importe
  de son module.

### Corrigé

- Flèches placées en bout de ligne inversées.
- Réglages par défaut indisponibles avant `set_default_SL_integration_parameters`.
- Potentiel infini puis niveaux NaN quand la grille passe sur une charge ; échelle
  logarithmique non symétrique sur un potentiel qui change de signe.
- Indices décalés après `remove_SL`, qu'on ne pouvait pas appeler deux fois.
- `change_parameter` qui ignorait un départ en 0 et ne recalculait pas la ligne.
- Départ en double autour d'un dipôle.
- `NotImplemented` au lieu d'une méthode abstraite ; étiquettes placées avec
  `0.5*Aur + 0.5*Aur` ; `tl` reconstruit à partir de `y`.
