# Spécification du monde (étape 1)

Cette spec décrit **exactement** ce que fait le simulateur ; les valeurs numériques citées sont les
défauts de `configs/antworld.yaml` (tous modifiables). Si le code et ce document divergent, c'est un bug.

## 1. Vue d'ensemble

Une colonie de fourmis vit sur une grille 32×32 générée à partir d'une seed. Elle doit récolter de la
**nourriture** (qui repousse lentement) et du **minerai** (fini) pour se nourrir, grandir et construire.
L'« agent » est la colonie : à chaque **pas macro** il choisit une action de haut niveau ; les fourmis
individuelles exécutent ensuite des règles simples pendant `ticks_per_step` (=5) **ticks**.

- **Tick** : unité de temps de la simulation. Chaque fourmi agit une fois par tick.
- **Pas macro** (= pas d'environnement pour les agents) : 1 décision, 5 ticks. Un épisode dure au plus
  `max_steps` (=400) pas macro, soit 2000 ticks.

## 2. Le monde

- **Grille** `size × size` (32×32), connexité à 4 voisins (N, E, S, O). Les bords sont des murs.
- **Terrain** : case *libre* ou *roche* (infranchissable). Une case est indexée par `y*size + x`.
- **Nid** : une case libre dans la zone centrale (`nest_margin` ≤ x,y < size − `nest_margin`), avec
  un carré libre autour. Le nid est l'endroit où les fourmis déposent, où naissent les nouvelles
  fourmis et où se font les constructions.
- **Nourriture** : certaines cases sont des *zones de nourriture*, avec une capacité `cap` propre à la
  case (8 à 16). Une case démarre pleine.
- **Minerai** : cases disjointes de la nourriture, avec une quantité initiale (8 à 18). **Ne repousse jamais.**
  Environ 16 cases et **~196 unités par monde** (166 à 223 sur 200 seeds) : il en faut 250 pour tout construire,
  donc la nurserie (150) est finançable mais pas tout : arbitrage nurserie / grenier.

### Génération procédurale (`generate_world(cfg, seed)`)

Tout vient de `SeedManager(seed).rng("worldgen/<tentative>")`.

1. Bruit de Perlin fractal (2 octaves) → roche là où le bruit dépasse le quantile `1 − rock_fraction` (16 %).
2. Nid tiré au hasard dans la zone centrale, case libre + voisinage libre.
3. Toute case libre **inaccessible depuis le nid** devient roche (pas de poches isolées).
4. Nourriture : bruit de Perlin → les `food_fraction` (14 %) cases accessibles au bruit le plus élevé,
   hors du voisinage immédiat du nid (distance de chemin ≥ `min_resource_dist` = 3). Capacité proportionnelle
   au bruit. Minerai : idem sur les cases restantes, `ore_fraction` (2 %), avec un autre bruit.
5. **Test de jouabilité** : assez de cases accessibles (≥ 60 % de la grille), nourriture et minerai
   accessibles, nourriture la plus proche à ≤ 14 pas du nid et minerai à ≤ 22, au moins 12 cases de
   nourriture et 4 de minerai. Sinon on retire une nouvelle tentative (≤ 50), puis erreur.

## 3. Dynamique des ressources

### Nourriture : repousse logistique lente
À chaque tick, pour chaque case de nourriture d'état `a` et de capacité `cap` :

    a ← a + g · (a + s·cap) · (1 − a/cap)        g = 0.01,  s = 0.05

Le terme logistique fait que la repousse est lente quand la case est presque vide (**seul le petit terme
`s` « graine » permet de repartir de 0**) et ralentit près de la capacité : une case fortement
surexploitée se rétablit lentement. Mesuré : une case vide met 296 ticks pour atteindre la mi-capacité et 501 pour 90 %.

### Minerai : stock fini
Aucune repousse. Une fois épuisé, c'est fini. Il ne sert qu'à construire (§6).

### Récolte
Une fourmi sur une case contenant au moins 1 unité en prend `min(carry_capacity=2, ⌊quantité⌋)`
(nombre entier d'unités ; les fractions résiduelles restent sur la case).

## 4. Brouillard de guerre

- La colonie a une carte **partagée** `known` (cases déjà vues) et, pour chaque ressource, la
  **dernière quantité vue** par case (`seen`). Elle peut être périmée : la nourriture a pu repousser
  ou être mangée depuis.
- À chaque tick, chaque fourmi révèle le carré de rayon `forager_vision` (1) autour d'elle, `scout_vision`
  (2) pour les éclaireuses (la vision ne dépend que du rôle, même quand la fourmi porte une charge) ; le nid révèle un rayon `nest_vision` (5) en permanence. La vision traverse
  la roche (simplification).
- Les fourmis connaissent le terrain réel (elles retrouvent leur chemin), mais **ne vont que vers des
  ressources qu'elles croient présentes** (`seen ≥ 1`).

## 5. Les fourmis (règles individuelles)

Chaque fourmi a : une case, un rôle, une charge (type + nombre d'unités), un cap (pour les éclaireuses).
Elle se déplace d'**une case par tick**. Règle de priorité, évaluée chaque tick :

1. **Si elle porte une charge** (quel que soit son rôle) : elle rentre au nid par le plus court chemin ;
   arrivée au nid, elle dépose. Nourriture : ajoutée au stock **dans la limite de la capacité de
   stockage** ; l'excédent est perdu (gaspillage). Minerai : toujours accepté.
2. Sinon, selon le rôle :
   - **Cueilleuse de nourriture / de minerai** : si sa case contient la ressource, elle récolte ;
     sinon elle suit le plus court chemin vers la case de ressource **crue présente** la plus proche
     (champ de flux calculé par BFS, recalculé quand la carte `seen` change d'état « ≥ 1 »). S'il n'y en a
     aucune, elle rentre au nid et attend.
   - **Éclaireuse** : marche aléatoire persistante. Chaque tick, avec la probabilité `scout_turn_prob`
     (0.15) elle choisit un nouveau cap : avec la probabilité `scout_bias` (0.6) c'est la direction cible
     de l'action (si elle est choisie), sinon une direction uniforme. Un obstacle force un nouveau cap.
   - **Bâtisseuse** : va au nid ; si une construction est en cours, ajoute 1 unité de travail par tick.

Les collisions n'existent pas (plusieurs fourmis par case).

## 6. La colonie

### Stocks et bâtiments
- `food_stock`, `ore_stock`. Départ : 12 fourmis, 30 de nourriture, 0 de minerai.
- **Capacité de stockage de nourriture** = 60 + 60 × niveau du **grenier**.
- **Population maximale** = 40 + 20 × niveau de la **nurserie**.
- Construction (nurserie ou grenier, niveau 0 → 4) : coût en minerai payé **au début** (nurserie
  `15·(niveau+1)`, grenier `10·(niveau+1)`), puis `200·(niveau+1)` unités de travail apportées par les
  bâtisseuses au nid. Une seule construction à la fois ; une demande pendant une construction est ignorée.

### Faim et mort
- Chaque tick, la colonie consomme `0.02 × population` de nourriture.
- Si le stock ne suffit pas, il est vidé et une famine frappe : chaque tick, `0.01 × population ×
  (fraction du besoin non couvert)` fourmis meurent (accumulateur fractionnaire, pas d'aléa). Les
  **plus récemment nées meurent d'abord** (les indices les plus élevés) ; leur charge est perdue.
- **La colonie meurt quand il ne reste plus aucune fourmi.** L'épisode se termine alors (terminaison).

### Naissances
Chaque tick, un compteur de ponte avance de `lay_rate` (0.25). Quand il atteint 1 : si
`food_stock ≥ birth_cost (4) + food_reserve (10)` et que la population est sous la capacité, une fourmi
naît au nid (rôle cueilleuse de nourriture), `birth_cost` de nourriture est consommé. Sinon le compteur
reste plafonné à 1. Au plus une naissance par tick.

## 7. L'action macro

Une action est un vecteur de 5 entiers (`MultiDiscrete([4, 4, 4, 3, 5])`, 960 combinaisons) :

| # | Dimension | Valeurs |
|---|---|---|
| 0 | Part d'éclaireuses | 0 %, 10 %, 25 %, 50 % de la population |
| 1 | Part de bâtisseuses | 0 %, 10 %, 25 %, 50 % de la population |
| 2 | Part de minerai parmi les cueilleuses restantes | 0 %, 25 %, 50 %, 100 % |
| 3 | Construction | rien, nurserie, grenier |
| 4 | Cible d'exploration | libre, Nord, Est, Sud, Ouest |

Les fourmis restantes (après éclaireuses et bâtisseuses) cueillent : une part minerai, le reste nourriture.
**Réaffectation** : les rôles existants sont conservés autant que possible ; le surplus d'un rôle
(les fourmis aux indices les plus élevés) est réaffecté aux rôles déficitaires. Des bâtisseuses sans
construction restent inactives au nid (c'est à l'agent de ne pas les gaspiller).

## 8. Ordre des opérations d'un tick

1. Repousse de la nourriture.
2. Chaque fourmi agit (règles §5) puis révèle sa vision. Le nid révèle sa vision.
3. Fin de construction si le travail requis est atteint.
4. Consommation de nourriture, famine, morts.
5. Naissance éventuelle.
6. Recalcul des champs de flux si la carte `seen` a changé d'état.

## 9. Conservation (vérifiée par les tests)

Compteurs cumulés dans l'état. Pour la nourriture :
- monde : `nourriture_monde = initiale + repousse − récoltée`
- fourmis : `récoltée = livrée + gaspillée + perdue(mortes) + portée_actuellement`
- stock : `stock = initial + livrée − mangée − consommée_pour_naissances`

Pour le minerai : `monde = initial − récolté` ; `récolté = livré + perdu + porté` ; `stock = livré − dépensé`.
Population : `n = initial + naissances − morts`.

## 10. Interface agent (étape 2 : `antworld.env.AntWorldEnv`)

- **Action** : `MultiDiscrete([4,4,4,3,5])` (§7). **Épisode** : `terminated` si la colonie meurt, `truncated` à 400 pas.
- **Observation** (partielle, float32 dans [0,1]) : `map` (8, 32, 32) et `state` (24,). La colonie observe sa
  *propre connaissance* : cases déjà vues, dernières quantités vues (périmées possible), zone visible à
  l'instant, position de ses fourmis ; rien de ce qui est dans le brouillard. `state` = effectifs, stocks,
  niveaux, construction en cours, répartition des rôles, temps écoulé, coût du prochain niveau, cible
  d'exploration. Détail des canaux : en-tête de `src/antworld/observation.py`.
  `obs.fog: false` (config) donne toute la carte (ablation de l'étape 8).
- **Récompense** : `r = Φ(s') − Φ(s) + w_survive·[vivante]`, `Φ = w_pop·fourmis + w_food·nourriture + w_ore·valeur(minerai)` ;
  poids par défaut 1,0 / 0,05 / 0,05 ; survie 0,1 par pas. Valeur du minerai : le stock **utile** (coût du prochain
  niveau de nurserie + de grenier) à pleine valeur, le **surplus** à −0,5 × (une pénalité légère), le
  minerai investi dans un bâtiment à pleine valeur (construire est neutre, le gain vient de la capacité). Le retour
  d'un épisode vaut donc `Φ(fin) − Φ(début) + survie` : aucune boucle ne peut créer de récompense gratuite.
- **Sauvegarde/restauration** : `world.snapshot()/restore()/clone()`, `env.get_state()/set_state()` (état
  dynamique + état du générateur aléatoire ; le layout est régénéré depuis la seed).

## 11. Hors périmètre de l'étape 1 (ajouts par paliers plus tard)

Champignon à cultiver (investissement puis gain différé), chasse (risquée mais rentable), prédateurs,
plusieurs colonies, changement de règles en cours de partie (la nourriture cesse de repousser — étape 8).
