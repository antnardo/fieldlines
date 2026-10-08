# Changelog

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) et le
[versionnage sémantique](https://semver.org/lang/fr/).

## [Unreleased]

## [0.1.0] — non publiée

Première version. Le calcul reprend le module privé `efieldlines` (2017), réécrit et
étendu au champ magnétique ; l'API change entièrement (voir le README pour la
correspondance).

### Ajouté

- Sources magnétiques : fil rectiligne infini, spire circulaire exacte (intégrales
  elliptiques complètes), solénoïde (somme de spires), dipôle magnétique.
- Fonction de flux (A_z pour des fils, fonction de Stokes pour des sources de
  révolution), et lignes de champ comme ses lignes de niveau.
- Lignes du champ magnétique fermées : l'intégration s'arrête quand la ligne repasse
  à son point de départ.
- Placement des lignes selon le flux : nombre proportionnel à la charge, et aucune
  ligne tracée deux fois ; départs à flux égal le long d'un segment.
- Lignes nommées de façon stable, réglages par ligne (flèches, marge, sens, longueur
  maximale, style).
- Tests numériques : champs confrontés aux formules et à la loi de Biot et Savart,
  propriétés des lignes. Exemples tirés des figures du cours.

### Modifié

- Intégration par `solve_ivp`, avec des événements pour la sortie du cadre, l'arrivée
  sur une source, la ligne refermée et le champ nul, au lieu d'un pas fixe et de
  seuils sur la norme du champ (`Emin`, `Emax`).
- Les flèches se placent à une fraction de la longueur visible de la ligne ; les
  tailles des flèches et des sources sont en points.
- Le dipôle est orienté par l'angle de son moment depuis Ox (`angle`), et non plus
  depuis Oy dans le sens horaire (`alpha`) ; le fil chargé (`FilCharge`) prend sa
  charge linéique λ (l'ancien `Ligne(q)` valait λ = q/2).

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
