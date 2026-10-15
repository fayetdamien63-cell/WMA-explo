# Décisions de conception

Format : décision — raison. Les plus récentes en bas.

## Étape 0

- **Package `common/` ajouté à la structure suggérée** — config, seeds et device sont utilisés
  aussi bien par le simulateur que par l'apprentissage ; les mettre dans `antworld/` ferait
  dépendre l'apprentissage du simulateur pour des utilitaires sans rapport.
- **Layout `src/` + `pyproject.toml` (setuptools), install en mode éditable** — imports propres
  (`from common.seeding import ...`) sans bricoler `sys.path`. Pas de poetry/uv : pip suffit.
- **Seeds : un `SeedManager` qui dérive des flux nommés d'une seed maître (`np.random.SeedSequence`)**
  — un générateur par composant (monde, fourmis, agent...) : ajouter un tirage dans un composant ne
  décale pas les autres. Le nom est haché avec SHA-256, pas `hash()` (randomisé par processus).
- **`seed_everything` fixe aussi les générateurs globaux** (random, np.random, torch) — uniquement
  pour les bibliothèques tierces (init des poids, SB3). Notre code n'y touche pas.
- **Config : dataclasses + YAML, clés inconnues = erreur, surcharges `a.b=valeur`** — une faute de
  frappe dans un hyperparamètre ne doit pas être ignorée en silence. Pas de Hydra/OmegaConf :
  ~100 lignes lisibles suffisent. `ProjectConfig` ne contient que seed/device/runs_dir ;
  chaque étape ajoute ses sections.
- **`get_device()` : cuda > mps > cpu ; un device demandé explicitement mais indisponible lève
  une erreur** — éviter qu'un entraînement de plusieurs heures tourne sur CPU par accident.
  Surcharge possible via `ANTWORLD_DEVICE`.
- **Dépendances par extras** (`env`, `rl`, `viz`, `dev`) — on n'installe que ce que l'étape en cours
  utilise. Numba : seulement si le test de performance de l'étape 1 l'exige.
- **PyTorch hors des index par défaut documenté dans le README** — la roue dépend de la plateforme
  (CUDA sur Windows, MPS sur Mac).

## Étape 1

- **Actions : `MultiDiscrete([4,4,4,3,5])` (960 combinaisons), validé par l'utilisateur** — parts d'éclaireuses,
  de bâtisseuses et de minerai par paliers, choix de construction, direction d'exploration. Les
  allocations continues (curiosité de l'utilisateur) sont reportées après l'étape 3 ou à l'étape 9.
- **Récompense, grille 32×32, 2 ressources, brouillard dès le début : confirmés par l'utilisateur.**
- **Numba pour le noyau de simulation** — la règle était « Numba seulement si le test de perf l'exige ».
  Mesure préalable en NumPy pur (scratchpad, non conservée) : un seul champ de flux BFS vectorisé coûte
  279 µs (le budget total d'un pas macro pour 5 000 pas/s est de 200 µs) et un tick vectorisé *très
  réduit* (sans naissances, rôles, construction, famine) coûte déjà 16 µs, soit 80 µs par pas de 5 ticks.
  Avec toutes les règles NumPy pur ne pouvait pas tenir la cible. Résultat avec Numba : ~15 000 pas/s.
  Débogage possible sans compilation : `NUMBA_DISABLE_JIT=1` (trajectoire vérifiée identique bit à bit).
- **Fourmis individuelles explicites** (case, rôle, charge, cap) plutôt qu'un modèle agrégé de « voyages » :
  plus lisible, rendu fidèle, et prépare l'extension « fourmis individuelles à politique partagée » (étape 9).
- **État = 4 tableaux NumPy (`grid`, `flow`, `ants`, `col`) + constantes d'indices nommées** (`indices.py`) —
  compatible Numba, sauvegarde/restauration d'état = copie de tableaux (étape 2), peu d'arguments au noyau.
- **L'aléa du noyau est pré-tiré côté Python** (`rng.random((ticks, n+ticks, 3))` par pas macro) — pas de
  générateur caché dans Numba, tout vient du `SeedManager` (flux « worldgen/k » et « sim »).
- **Pas macro = 5 ticks, épisode = 400 pas macro** — assez long pour des dynamiques lentes (repousse :
  ~300 ticks = 60 pas macro) et des horizons du world model de 10 à 50 pas ; modifiable dans la config.
- **Les fourmis connaissent le terrain réel mais pas les ressources** : le brouillard cache *où sont les
  ressources* (carte `seen`, éventuellement périmée), pas la topologie. Évite des fourmis bloquées sur
  « ressource vue mais chemin inconnu » ; la vision traverse la roche (simplification).
- **Famine déterministe** (accumulateur, pas de tirage aléatoire) ; les dernières nées meurent d'abord —
  mort = simple décrément du compteur de population, pas de compactage de tableau.
- **Repousse logistique avec terme « graine »** — une case surexploitée repart lentement (dynamique lente
  voulue) mais repart.
- **Constructions** (nurserie → capacité de population, grenier → stockage de nourriture), niveaux 0 à 4, à
  coût croissant en minerai : donne un usage au minerai et un compromis réel (investir du travail et
  des cueilleuses dans la construction plutôt que dans la nourriture).
- **Validation de config** : stock initial ≤ capacité de stockage, population initiale ≤ capacité, etc. —
  trouvée utile pendant le développement (un test avait une config incohérente).

### Problèmes de conception repérés à l'étape 1 (traités à l'étape 2 : voir plus bas)
1. **Le minerai n'a plus d'usage une fois les 8 niveaux construits** : une politique aléatoire finit avec
   180 de minerai en stock en moyenne (326 pour une politique qui construit), alors que 250 suffisent pour tout
   construire. Si la récompense compte « ressources accumulées » en incluant le minerai, **accumuler du
   minerai inutile serait récompensé** (récompense exploitable). Piste : ne pas inclure le minerai dans la récompense
   (sa valeur passe par la population via la nurserie), ou le plafonner. À trancher à l'étape 2 avec le test anti-triche.
2. **Plafond de population de base (40) trop vite atteint** : la règle fixe « 10 % d'éclaireuses, tout
   le reste en nourriture » atteint 39,5 fourmis en moyenne sur 20 seeds (0 % de colonies mortes),
   contre 28,4 pour l'aléatoire. La marge pour un agent plus fin dépend donc surtout de la nurserie
   (minerai). La marge réelle sera mesurée à l'étape 3 (porte heuristique/oracle).
3. La quantité totale de nourriture est quasi constante d'un monde à l'autre (112 à 122 cases) : les mondes
   varient par la forme et les distances, pas par l'abondance. Acceptable pour l'instant.

## Étape 2

- **Minerai (décision de l'utilisateur) : il peut compter dans la récompense, mais le surplus est légèrement
  pénalisant ; la quantité par monde est réduite.** `ore_fraction` 0,04 → 0,02 et quantités 10–25 → 8–18 :
  ~196 de minerai par monde (avant : 507) pour 250 nécessaires pour tout construire.
- **Récompense = différence de potentiel + bonus de survie** (`reward.py`). Φ = 1,0·fourmis + 0,05·nourriture +
  0,05·valeur du minerai. Retour d'épisode = Φ(fin) − Φ(début) + survie : pas de boucle exploitable *par
  construction*. La « croissance de population » est le terme Δ(fourmis) ; « ressources accumulées » = Δ(stocks).
- **Valeur du minerai** : stock utile (prochain niveau de nurserie + de grenier) à pleine valeur ; surplus à
  −0,5 × valeur ; minerai investi dans un bâtiment à pleine valeur (construire est neutre pour la récompense, le
  gain vient de la capacité). Effet mesuré : thésauriser ~189 minerai fait perdre 3 points sur ~69 (légère).
- **Observation = la connaissance de la colonie, pas la vérité terrain** : carte 8 canaux (visible à l'instant,
  connue, roche connue, nourriture/minerai *dernières valeurs vues*, nid, éclaireuses, autres fourmis) + 24 scalaires
  d'état interne. On donne les dernières quantités vues (carte de la colonie) mais pas leur âge : l'agent n'est pas
  privé de mémoire à ce point, mais le monde reste partiellement observable (cases cachées, valeurs périmées).
- **`Dict` d'observations (carte + état)** plutôt qu'un vecteur aplati — la carte pourra passer dans un CNN
  (étapes 5–7). Pour PPO/SB3 (étape 4), un extracteur de caractéristiques dédié sera nécessaire (le CNN par défaut
  de SB3 exige ≥ 36 pixels de côté).
- **Sans brouillard = nid avec vision sur toute la carte** (`obs.fog: false`) : aucun code supplémentaire dans le noyau.
- **`reset(seed)` : la seed fixe exactement le monde** (et l'aléa de la partie). Sans seed, un monde est tiré
  du générateur de l'env. La liste de seeds d'entraînement/test sera gérée par le banc d'évaluation (étape 3).
- **État sauvegardé = tableaux dynamiques + état du générateur aléatoire ; le layout est régénéré depuis la seed**
  (`EnvState`), donc restaurable dans un autre processus ou un autre env. `clone()` partage le layout (pas de régénération).
- **Test anti-triche** (`eval/anticheat.py`, `scripts/anticheat_report.py`) : 2 références + 9 stratégies absurdes
  sur 20 seeds, plus une borne supérieure théorique du retour.

### Bug trouvé et corrigé pendant l'étape 2
Une éclaireuse *chargée* ne révélait qu'un rayon de 1 dans le simulateur mais l'observation lui en attribuait 2 :
l'observation déclarait « visible » des cases jamais vues. Attrapé par le test « le brouillard ne fuit pas », qui
ne le voyait que sur certaines trajectoires (d'où des seeds de test plus nombreuses). Règle unifiée : la vision ne
dépend que du rôle.

## Étape 3

- **Interface `Agent`** : `reset(agent_seed)`, `act(obs, env=None)`, compteurs `sim_steps` (pas de simulateur consommés pour
  planifier) et `train_env_steps` (pas d'environnement d'entraînement). Seul un agent `privileged` reçoit l'env (l'oracle) ; le harnais
  passe `None` aux autres, vérifié par un test espion : l'heuristique et les agents appris ne peuvent pas tricher.
- **Heuristique = observation partielle uniquement** (carte + état interne), pour que ce soit une comparaison équitable avec les agents
  appris. Constantes dans `HeuristicConfig` / `configs/baselines.yaml`, **réglées sur les seeds d'entraînement (0..39)**, jamais sur celles de test.
  Enseignement du réglage : *arrêter les éclaireuses une fois la carte explorée est une grosse erreur* (66 → 95 de retour) car la carte de la colonie
  se périme (la nourriture repousse, mais personne ne repasse voir les cases vues vides) : les éclaireuses servent aussi à rafraîchir la carte.
  Les autres variantes testées (±1 palier, seuils) étaient dans le bruit (±2) : non retenues pour ne pas surajuster.
- **Seeds : trois plages disjointes** (`eval/seeds.py`) : entraînement [0, 100 000), validation [500 000, ...), test [1 000 000, ...).
  Le test vérifie la disjonction. Réglages et budgets sur train/val ; résultats rapportés sur test uniquement.
- **L'aléa d'un épisode dérive de (seed d'agent, seed du monde)** : le résultat ne dépend ni de l'ordre d'évaluation ni du parallélisme (testé : séquentiel = parallèle).
- **Parallélisme par `ProcessPoolExecutor` en mode `spawn`** — portable Mac / Windows / Linux ; l'agent est reconstruit dans chaque worker à partir de son nom et de la config (picklable).
- **Statistiques sans scipy** : IC à 95 % par bootstrap percentile (10 000 rééchantillonnages, seed fixe → déterministe, couverture vérifiée ≈ 95 % par simulation),
  comparaisons **appariées** (mêmes mondes : la variance entre mondes est grande, l'appariement la neutralise), nombre de mondes nécessaire = (1,96·σ/h)².
- **Oracle = CEM sur le vrai simulateur** (`agents/planning.py` générique + `agents/oracle.py`) : il voit tout le monde (même le brouillard) et simule la
  dynamique exacte, mais **ne connaît pas les futurs tirages aléatoires** de la partie réelle (`perfect_foresight: false`, chaque simulation de planification
  utilise un aléa différent) : sinon il exploiterait le hasard futur, ce qui n'a aucun sens pour un agent. Même planificateur qu'à l'étape 6 (seule la fonction de simulation change).
- **Plan par blocs de 5 pas** (une action tenue 5 pas macro) : les rôles changent lentement ; divise par 5 la dimension de recherche. MPC : on exécute 5 pas puis on replanifie, en réutilisant la distribution décalée.
- **Tests « lents »** (`pytest -m slow`) exclus du `pytest` par défaut : évaluations multi-seeds avec l'oracle (~minutes).
- **Critère de la porte fixé avant de voir les chiffres** : sur les seeds de test, ordre aléatoire < heuristique < oracle avec IC 95 % des différences appariées > 0, et heuristique ≤ 85 % de l'écart aléatoire→oracle.
