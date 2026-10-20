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

### Budget de l'oracle (mesuré sur 8 mondes d'ENTRAÎNEMENT 0..7 ; heuristique sur les mêmes : 99,5)
Échantillons × itérations, horizon en pas macro, replanification tous les `r` pas. Retour moyen ± erreur standard, temps total pour 8 épisodes sur 4 cœurs :

| Configuration | Retour | Temps |
|---|---|---|
| 64×3, H15 | 101,5 ± 6,1 | 62 s |
| 64×3, H30 | 110,0 ± 3,8 | 121 s |
| 64×3, H30, **hasard futur connu** | 110,9 ± 2,9 | 126 s |
| 32×2, H30 | 108,5 ± 4,8 | 40 s |
| 128×4, H30 | 113,9 ± 4,4 | 365 s |
| 64×3, H45 | 122,6 ± 4,9 | 176 s |
| 32×2, H60 / H80 / H120 | 126,2 / 129,2 / 136,2 | 72 / 94 / 135 s |
| 64×3, H60 | 130,3 ± 3,7 | 231 s |
| 32×2, H120, replan 10 | 122,4 ± 5,0 | 70 s |
| 32×2, H200 / H400, replan 10 | 127,8 / 138,2 ± 7,1 | 76 / 72 s |
| 64×3, H400, replan 10 | 145,5 ± 5,9 | 214 s |
| 128×3, H400, replan 10 | 148,6 ± 5,8 | 434 s |
| **64×3, H400, replan 5 (retenu)** | **153,2 ± 5,9** | 428 s |

Lectures : (1) l'**horizon** compte beaucoup plus que le nombre d'échantillons (+50 points de H15 à H400) ; (2) **replanifier plus souvent** aide
(+8 de replan 10 à 5) ; (3) le nombre d'échantillons est quasi saturé (128×3 ≈ 64×3, dans le bruit) ; (4) connaître le hasard futur ne change presque rien
(+0,9, dans le bruit), donc le choix de ne pas le donner à l'oracle ne le pénalise pas. Retenu : 64×3, horizon = tout l'épisode, replanification tous les 5 pas (≈ 200 s/épisode).
**L'oracle reste une borne haute APPROCHÉE : son score augmentait encore avec le calcul quand on s'est arrêté.**

### ⚠ Problème de conception détecté à l'étape 3 : la récompense v1 encourage « hiberner puis exploser »
La récompense par potentiel (étape 2) fait que le retour d'un épisode = Φ(état final) − Φ(état initial) + survie : **seul l'état à la fin compte**, pas la trajectoire.
L'oracle (qui voit tout et planifie jusqu'à la fin de l'épisode) en tire une stratégie que ni l'heuristique ni nos stratégies « absurdes » codées à la main n'avaient testée :
il garde la colonie minuscule pendant une grande partie de la partie (les cases de nourriture se reconstituent à pleine capacité), puis la fait exploser dans les ~50 derniers pas.
Mesures (20 mondes de test) :

| Agent | Retour | Population MOYENNE sur l'épisode | Population minimale (moy., plus bas) | Population finale | Part du retour dans les 50 derniers pas |
|---|---|---|---|---|---|
| aléatoire | 58,3 | 33,3 | 12,4 (6) | 22,9 | 8 % |
| heuristique | 93,3 | 63,8 | 13,0 (13) | 58,7 | 2 % |
| oracle | 152,1 | **30,9** | **4,7 (2)** | 111,2 | **36 %** |

L'oracle a un retour 63 % supérieur à l'heuristique avec une population moyenne deux fois plus basse, des colonies tombées jusqu'à 2 fourmis (risque d'extinction) et plus du tiers de son score gagné à la toute fin.
C'est un effet de **fin d'horizon** : légitime pour l'objectif tel que défini, mais très probablement pas ce que « prospérer » veut dire. Conséquences : la borne haute est gonflée par une
stratégie « sacrifier le présent pour la fin », les écarts entre agents ne mesurent pas seulement la qualité de gestion, un agent appris doit découvrir une stratégie à très long délai (hibernation de ~200 pas) et
le test anti-triche de l'étape 2 (stratégies codées à la main) ne pouvait pas le voir. **Décision de l'utilisateur : récompense « prospérité » (v2), appliquée ci-dessous.**

### Récompense « prospérité » (v2) — décision de l'utilisateur après l'alerte ci-dessus
`r_t = w_pop_level · fourmis_t + Δ(w_food·nourriture + w_ore·valeur du minerai) + w_survive·[vivante]`, avec `w_pop_level = 0,0025` par fourmi et par pas
(≈ 64 points pour 64 fourmis sur 400 pas). La population est récompensée **à chaque pas** (aire sous la courbe) ; les stocks restent des potentiels (mêmes valeurs que
la v1, même pénalité légère du surplus de minerai). Retour = `w_pop_level·Σ fourmis + Φ(fin) − Φ(début) + survie`. Effet mesuré (oracle, 8 mondes d'entraînement) : plus
de creux (population minimale 93 % de la moyenne), plus d'explosion finale (13-15 % du retour dans les 50 derniers pas, contre 36 %).
- L'oracle évalue ses séquences avec `reward.sequence_return` (identité télescopique), vérifiée par test égale à la somme des récompenses de l'env.
- **Garde-fous anti-exploitation ajoutés au banc d'évaluation** : pour chaque épisode `min_pop` (creux après la phase de croissance de 100 pas), `steady_mean_pop`, `late_return_share`
  (part du retour dans les 50 derniers pas) ; `exploitation_warnings` alerte au-delà de 25 % de gain final ou d'un creux < 35 % de la moyenne, et `run_baselines.py` **invalide la porte** en cas d'alerte.
  Vérifié rétroactivement : il aurait signalé l'oracle v1 (14 % et 36 %) et aucun des deux autres agents. La phase de croissance est exclue car toutes les colonies démarrent à 12 fourmis.
- **Test lent d'un optimiseur** (`tests/test_baselines_gate.py`) : un oracle de budget moyen ne doit déclencher aucune alerte et garder une population moyenne ≥ 90 % de celle de l'heuristique.
  Leçon : un test anti-triche qui n'utilise que des stratégies codées à la main ne voit pas ce qu'un optimiseur trouve.
- **Heuristique non modifiée** : réévaluée sous la v2 sur les seeds d'entraînement, toutes les variantes testées restent dans le bruit (±2) ; elle obtient 107,7 ± 1,5 sur les 40 mondes d'entraînement.

## Étape 4

- **Monde simplifié = test de plomberie, pas un banc de comparaison** (`configs/antworld_simple.yaml` : nourriture seule, sans brouillard, donc sans minerai ni bâtiments). Il est *trop facile* :
  heuristique 80,0 ≈ oracle 80,6 (plafond : 40 fourmis + stock plein), aléatoire 63,9 ; même une politique NON entraînée y atteint 77 dans 3 cas sur 5 (l'argmax d'un réseau aléatoire
  tombe sur l'action « tout le monde cueille »). L'indicateur « pas pour atteindre 90 % de l'heuristique » y vaut 0 : **sans information**. La référence d'efficacité vient du monde complet.
  Le critère « écart heuristique/oracle » de l'étape 3 n'a aucun sens dans ce monde (il y est « non atteint » : normal, tout est résolu).
- **PPO = stable-baselines3, `MultiInputPolicy` + extracteur maison** (`models/features.py` : 3 convolutions de pas 2 + MLP d'état, ~90 k paramètres) : le CNN par défaut de SB3 exige ≥ 36 pixels de côté.
  Hyperparamètres dans `configs/ppo.yaml` (8 envs × 128 pas, lot 256, 4 époques, γ = 0,995, λ = 0,95, entropie 0,01, récompense normalisée par `VecNormalize` pour l'apprentissage seulement).
  Ils n'ont pas été optimisés : c'est une baseline standard, pas une baseline réglée (limite à garder en tête pour les comparaisons de l'étape 8).
- **Mondes d'entraînement tirés dans les seeds [0, 1000)** par un flux `rng_seed` dérivé de `SeedManager(seed)` (paramètre `world_seeds`/`rng_seed` de `AntWorldEnv`) ; la graine d'environnement de SB3 n'est PAS utilisée
  (elle ferait des seeds consécutives des runs corrélés : seed s et s+1 partageraient des environnements). Courbes sur 10 mondes de VALIDATION fixes ; résultat final sur 20 mondes de TEST. Vérifié par tests (+ mutations).
- **Politique évaluée : stochastique = RÉFÉRENCE, déterministe (argmax) rapportée aussi.** Décision prise APRÈS avoir vu les résultats du premier run, et assumée comme telle : le diagnostic (`scripts/diagnose_ppo.py`) montre
  que les politiques entraînées ne sont pas effondrées (entropie 3,8 à 4,6 sur un maximum de 6,87) et que la politique stochastique — celle que PPO optimise réellement — bat son argmax de 2 à 15 points
  (seed 0 : 115,6 contre 104,3 ; seed 4 : 111,3 contre 96,6 en validation). Sur une action multi-discrète à 5 dimensions, l'argmax par dimension n'est pas l'action la plus probable de la politique jointe et n'est pas ce qui est optimisé.
  Les deux chiffres sont toujours donnés ; les courbes enregistrent `mean_return` (déterministe) et `mean_return_stochastic`.
- **Le « plateau à ~60 » du premier run était un artefact de l'évaluation déterministe, pas un plateau d'apprentissage** (première lecture erronée, corrigée ici). Le premier run (1 M pas, évaluation déterministe seule) montrait 5 seeds
  bloquées ~200-300 k pas à ~60, et j'avais d'abord soupçonné un problème d'exploration. Le diagnostic (politiques non effondrées, entropie élevée) puis le second run, qui enregistre aussi la politique stochastique, montrent que
  **la politique stochastique est déjà à ~105 à 200 k pas** (seed 0 : 105,4 contre 58,7 en déterministe au même point). L'argmax par dimension d'une politique multi-discrète à entropie élevée peut tomber sur une combinaison d'actions sans rapport avec la politique apprise.
- **Référence d'efficacité en échantillons (monde complet, 5 seeds, politique stochastique, mondes de validation)** : 90 % du niveau de l'heuristique (98,5) en **78 k pas** (médiane ; 68-81 k) ; 100 % (109,5) en **282 k pas** (192-374 k ; ≈ 865 s d'entraînement CPU) ;
  plateau lent ensuite (~115 en validation, 110,9 en test à 2 M pas). Politique déterministe : 691 k / 1,18 M pas. C'est la référence à battre pour le world model (étapes 7-8).

## Étape 5

- **Données** (`models/data.py`, `agents/data_policies.py`) : 6 sources de trajectoires, mélange 25 % aléatoire / 15 % aléatoire par blocs (une action tenue 5-20 pas, comme la structure des plans CEM) /
  15 % heuristique / 15 % heuristique bruitée (ε = 0,25 par dimension) / 20 % PPO stochastique (une des 5 seeds par épisode) / 10 % politiques dégénérées (actions constantes qui font stagner ou mourir la colonie).
  Raison : un world model ne sait que ce qu'il a vu ; les planificateurs (étape 6) iront explorer des actions atypiques, et la mort d'une colonie doit apparaître dans les données (11 % des épisodes d'entraînement).
  Cartes stockées en uint8 (erreur ≤ 0,002, identique pour le modèle et la baseline) : 640 épisodes = 56 Mo compressés.
  Quatre jeux, plages de seeds disjointes : `train` (mondes 0..999), `val`, `eval_seen` (MÊMES mondes que train, nouvelles trajectoires), `eval_unseen` (mondes de test, jamais vus).
- **Architecture** (`models/worldmodel.py`, ~0,8 M de paramètres) : encodeur CNN+MLP → GRU latent (256) → décodeurs de carte, d'état, de récompense et de fin d'épisode. Deux modes de pas : *observer*
  (la vraie observation est injectée après la prédiction : entraînement à 1 pas) et *boucle ouverte* (aucune observation : seul ce que verra un planificateur). Une prédiction est toujours faite AVANT d'observer sa cible
  (testé : pas de fuite de l'avenir, cohérence observer/imaginer).
- **Deux variantes comparées honnêtement** : `latent seul` (spécification minimale de l'énoncé : carte décodée du seul vecteur de 256 valeurs) et `résiduel` (le décodeur prédit un résidu par rapport à la carte de base,
  = dernière observation ou, en imagination, dernière prédiction du modèle). Hypothèse testée : un vecteur de 256 valeurs ne peut pas mémoriser une carte 8×32×32 exacte, donc ne peut pas battre « rien ne change » sur les parties statiques.
  La variante résiduelle démarre EXACTEMENT à « rien ne change » (testé), donc le cadre de comparaison est honnête.
- **Pertes normalisées** : w = 1/(variance de changement à 1 pas + ε) par canal/dimension : une prédiction « rien ne change » coûte ≈ 1 par canal dynamique, tous les canaux comptent pareil, et le terme de récompense
  (normalisé par son écart-type) est du même ordre. Entraînement en 3 phases : 1 pas observé (16 pas), puis imagination de 8 pas, puis de 24 pas ; 1 500 itérations chacune (lot 16).
- **Métriques** (`eval/wm_metrics.py`) : en BOUCLE OUVERTE depuis 5 instants de départ par épisode (10 pas de contexte observés), horizons 1, 5, 10, 25, 50 ; compétence = 1 − erreur modèle / erreur « rien ne change » (macro par canal dynamique
  et globale) ; récompense CUMULÉE sur k pas (R², contre persistance et moyenne) ; IC 95 % par bootstrap d'ÉPISODES. Vérifié sur de faux modèles : persistance → compétence exactement 0, modèle parfait → 1.
- **Critères de la porte fixés AVANT les résultats** (mondes jamais vus, 10 pas) : compétence carte ≥ 0,25, compétence état ≥ 0,25, R² de la récompense cumulée ≥ 0,5, compétence carte à 1 pas ≥ 0,5, écart « mondes vus − jamais vus » ≤ 0,10.

### Trois défauts trouvés et corrigés pendant l'étape (aucun n'a atteint un résultat rapporté)
1. **Pondération des pertes** : les poids normalisés à moyenne 1 rendaient les termes de carte (0,001) et d'état (0,002) ~700 fois plus petits que celui de la récompense (0,7) ; le modèle n'apprenait presque que la récompense.
   Vu dans le journal d'un entraînement que j'ai interrompu à 1 000 itérations. Poids corrigés (sans renormalisation).
2. **Résidu en espace logit** `σ(logit(base)+δ)` : pour une case valant exactement 0 ou 1, le gradient est atténué d'un facteur ~10³ ; les canaux binaires (visible, connu, roche) restaient *exactement* à « rien ne change », même en sur-apprentissage
   d'un petit lot. Remplacé par un résidu additif borné « straight-through ». Test de régression ajouté. (Découvert grâce au test de sur-apprentissage, qui échouait.)
3. **Rétropropagation à travers les cartes prédites réinjectées** : instable (perte 302 et divergence avec un pas de 1e-3 sur un petit lot). `detach_base=True` par défaut : seul l'état latent est rétropropagé dans le temps.
