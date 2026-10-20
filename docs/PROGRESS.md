# Avancement

| Étape | État |
|---|---|
| 0 — Fondations | ✅ validée (commit `15e2a5b`) |
| 1 — Simulateur | ✅ validée (commit `6064638`) ; ajustement : minerai réduit (étape 2) |
| 2 — Env Gymnasium | ✅ validée (commit `b967d93`) |
| 3 — Agents de référence et banc d'évaluation | ✅ validée avec la récompense v2 (commit `b8541d6`) |
| 4 — Baseline PPO | ✅ validée (commit `a424de4`) |
| 5 — World model supervisé | ✅ acceptée par l'utilisateur (option B puis C) : porte révisée atteinte à 5, 10, 25 et 50 pas ; **1 pas manqué (34 % du plafond pour 40 %)**, non bloquant |
| 6 — Planification dans le modèle | ✅ validée (feu vert de l'utilisateur) |
| 7 — Dreamer-lite | ⚠️ résultats prêts : **porte NON atteinte** (voir ci-dessous), en attente de décision de l'utilisateur |
| 8 à 9 | — |

## Étape 0 — résultats de la porte

Environnement de test : Linux, Python 3.13, torch 2.14.1+cpu (pas de GPU).

- `pytest` : 25 passés / 25 (config 7, device 8, seeding 10).
- Déterminisme : `scripts/check_determinism.py --seed 42` dans deux processus avec `PYTHONHASHSEED`
  différents → sorties identiques (empreinte `f64ecb04c9a3edbe`) ; seed 43 → autre empreinte.
- Tenseur PyTorch passé sur le device détecté (`cpu` ici).
- Non vérifié ici : les branches CUDA et MPS de `get_device()` (simulées seulement). À confirmer sur le
  MacBook (`mps`) et le PC RTX (`cuda`) avec `pytest tests/test_device.py`.

## Étape 1 — résultats

`pytest` : **78 passés / 78** (~10 s). Détail des critères :

| Critère | Résultat |
|---|---|
| Déterminisme (mêmes seed + actions → même trajectoire) | hash de l'état identique à chaque pas sur 120 pas ; identique entre deux processus avec `PYTHONHASHSEED` différents ; seeds ou actions différentes → trajectoires différentes |
| Invariants (24 épisodes complets : 6 seeds × 4 politiques, vérifiés à chaque pas) | aucune quantité négative, nourriture ≤ capacité, fourmis sur cases libres, stock ≤ capacité de stockage, compteurs monotones |
| Conservation | monde + fourmis + stock + consommé = initial + repousse, nourriture et minerai, + bilan de population ; le test détecte bien 5 triches simulées (test du test) |
| Mondes jouables | 200 seeds : 200 mondes distincts, tous jouables, vérifiés par un BFS indépendant du générateur ; nourriture la plus proche à 3–9 pas, minerai à 3–20 ; 112–122 cases de nourriture ; générés dès la 1re tentative (le filtre de jouabilité ne rejette rien sur ces seeds) |
| Performance (cible > 5 000 pas/s) | **~15 000 pas macro/s** (~75 000 ticks/s), population moyenne ≈ 25 ; **6 465 pas/s avec une colonie de ~200 fourmis** ; génération d'un monde : 2,2 ms |
| Règles | 14 scénarios faits main : aller-retour de cueillette, brouillard (ressource jamais vue ignorée), débordement de stockage, famine et mort de la colonie, naissances (stock/capacité), parts de rôles, coût et durée de construction, éclaireuses et cible d'exploration, minerai fini, repousse (~mi-capacité à ~300 ticks) |

### Parties de contrôle (20 seeds, 400 pas) — voir `docs/DECISIONS.md` pour les problèmes détectés

| Politique fixe | Population finale moyenne | Colonies mortes |
|---|---|---|
| aléatoire | 28,4 (10–49) | 0 % |
| tout en nourriture, 0 éclaireuse | 14,8 (0–40) | 5 % |
| 10 % d'éclaireuses | 39,5 (31–40) | 0 % |
| 10 % éclaireuses + 10 % bâtisseuses + nurserie + minerai 25 % | 46,5 (28–67) | 0 % |

### Sorties visuelles (non commitées, dans `runs/step1/`)
`python scripts/play_random.py --seed 3` produit `random_seed3.gif` (partie complète), `_sheet.png`
(planche de 6 instants), `.png` (courbes population/stocks/exploration).

## Étape 2 — résultats

`pytest` : **107 passés / 107** (~25 s). Critères :

| Critère | Résultat |
|---|---|
| `check_env` Gymnasium | passe (avec et sans `render_mode="rgb_array"`) |
| Formes et bornes | `observation_space.contains(obs)` à chaque pas de 4 épisodes complets ; map (8,32,32), state (24,), float32 dans [0,1], tout fini |
| Pas de fuite du brouillard | sur 8 seeds, tout ce qui est non nul (nourriture, minerai, roche, visible) est sur une case connue ; au départ ~<30 % de la carte connue |
| Épisodes finis | 3 politiques × 5 seeds : toujours `terminated` XOR `truncated` ; colonie condamnée → `terminated` avant 400 pas |
| Déterminisme env | mêmes seed + actions → mêmes observations et récompenses ; seed différente → autre monde |
| Restauration | restaurer puis rejouer 60 pas redonne exactement le même état (hash) ; snapshot non altéré par la suite ; clone indépendant ; `get_state/set_state` entre deux envs différents donne les mêmes obs et récompenses |
| Récompense | retour = Φ(fin) − Φ(début) + survie à 1e-9 près sur 6 épisodes complets ; surplus de minerai pénalisé ; investir est neutre ; mourir est négatif |
| Vitesse env (obs + récompense) | ~10 000 pas/s ; simulateur seul ~18 000 pas/s |

Les tests ont été éprouvés par mutation : 4 bugs volontaires (restauration sans RNG, bonus de thésaurisation
non potentiel, fuite du brouillard, pénalité de surplus à 0) sont tous détectés.

### Test anti-triche (20 seeds, 400 pas, `python scripts/anticheat_report.py`)

| Stratégie | Retour moyen | Pop. finale | Minerai final | Colonies mortes |
|---|---|---|---|---|
| **référence sensée** | **88,9** | 52,8 | 24 | 0 % |
| 10 % d'éclaireuses, rien d'autre | 68,7 | 39,5 | 0 | 0 % |
| thésauriser (25 % minerai, jamais construire) | 65,7 | 39,6 | 189 | 0 % |
| oscillation extrême des rôles | 57,0 | 29,9 | 0 | 0 % |
| aléatoire | 55,7 | 20,1 | 20 | 0 % |
| spam construction | 50,4 | 15,1 | 25 | 0 % |
| ne rien faire (tous cueillent la nourriture) | 40,2 | 14,8 | 0 | 5 % |
| 50 % de bâtisseuses oisives | 37,3 | 12,3 | 0 | 5 % |
| tous au minerai / minerai + 10 % éclaireuses / personne ne cueille | −5,3 à −5,7 | 0 | – | 100 % |

Aucune stratégie absurde ne dépasse la référence (en moyenne ni au maximum par seed) ; tous les retours sont sous la borne théorique.

## Étape 3 — résultats (seeds de TEST 1 000 000..1 000 019, récompense « prospérité » v2)

`pytest` : 151 tests rapides passés (+ 3 tests lents `-m slow`, dont un test d'OPTIMISEUR anti-exploitation). Évaluation : `python scripts/run_baselines.py --split test --n-worlds 20 --workers 4` (sorties `runs/step3/test/`).

| Agent | Retour moyen [IC 95 %] | Écart-type entre mondes | Pop. moy. / creux (après pas 100) / finale | Colonies mortes | Pas simulés/épisode | Temps/épisode | ms/décision |
|---|---|---|---|---|---|---|---|
| aléatoire (5 seeds d'agent × 20 mondes) | 80,7 [76,8 ; 84,7] | 9,1 | 33,3 / 18,3 / 22,9 | 0 % | 0 | 0,1 s | 0,0 |
| heuristique | 110,4 [106,4 ; 114,4] | 9,2 | 63,8 / 50,0 / 58,6 | 0 % | 0 | 0,1 s | 0,0 |
| oracle (CEM 64×3, H400, replan 5) | 133,9 [131,8 ; 135,9] | 4,8 | 73,3 / 68,2 / 87,0 | 0 % | 3 110 400 | 381,6 s | 953,7 |

- Différences appariées (IC 95 %) : heuristique − aléatoire **+29,7 [+25,9 ; +33,5]** ; oracle − heuristique **+23,5 [+20,4 ; +26,7]**. L'ordre est net.
- Score normalisé de l'heuristique (aléatoire = 0, oracle = 1) : **0,56** (seuil fixé avant les mesures : 0,85) → **PORTE ATTEINTE**.
- **Diagnostic anti-exploitation : RAS** (population minimale après le pas 100 = 93 % de la moyenne pour l'oracle ; plus de creux ni d'explosion finale). Les courbes montrent un oracle qui fait croître sa population et son stock de nourriture régulièrement, et une heuristique qui culmine (~82 fourmis au pas 100) puis redescend (~58) faute de nourriture : la marge d'amélioration est réelle.
- **Variance** : 83 % de la variance du retour de l'agent aléatoire vient du MONDE (entre mondes 80,2 ; aléa de l'agent 16,2) → comparaisons appariées indispensables. Mondes nécessaires pour un IC 95 % à ±5 % : aléatoire 20, heuristique 11, oracle 2.
  **Recommandation pour la suite : ≥ 20 mondes de test appariés, ≥ 5 seeds d'entraînement par agent appris.**
- Coût de l'oracle : 3,1 M de pas simulés et ~380 s par épisode (4 processus en parallèle) contre 400 pas d'environnement pour les autres : `sim_steps` et le temps réel sont journalisés à part. Budget choisi sur 8 mondes d'entraînement : 64×3, H400, replan 5 → 131,4 ± 3,5 (32×2, H60 : 123,4 ± 4,2 ; 32×2, H400 : 123,4 ± 4,4). Borne haute APPROCHÉE.

### Historique : la récompense v1 était exploitable
Avec l'ancienne récompense (potentiel de l'état final), l'oracle « hibernait puis explosait » (population moyenne 30,9, creux à 2 fourmis, 36 % du retour dans les 50 derniers pas) pour un retour de 152,1. Résultats v1 conservés dans `runs/step3_v1/` (non versionné). Détail et correction : `docs/DECISIONS.md`.

## Étape 4 — résultats

`pytest` : 152 tests rapides passés (+ 3 lents, dont `test_ppo_learns_on_the_simple_world`). Toutes les mesures : 5 seeds d'entraînement, mondes d'entraînement tirés dans les seeds [0, 1000), courbes sur 10 mondes de validation, résultats finaux sur les 20 mondes de test (mêmes que l'étape 3).

### Monde simplifié (nourriture seule, sans brouillard) — 100 k pas, 5 seeds
| Agent | Retour (test) |
|---|---|
| aléatoire | 63,9 [60,8 ; 67,3] |
| heuristique | 80,0 [79,3 ; 80,4] |
| oracle (CEM 32×2, H60) | 80,6 [80,5 ; 80,6] |
| **PPO (5 seeds)** | **80,0** sur chacune des 5 seeds (+16,1 contre l'aléatoire) |

Critère « PPO bat nettement l'aléatoire » : **atteint**. Mais ce monde est **trop facile** (plafond 80,6 atteint par l'heuristique ; une politique non entraînée y obtient déjà 77 dans 3 cas sur 5) : il valide la plomberie, pas l'efficacité. Voir DECISIONS.md.

### Monde complet — 2 M pas par seed, 5 seeds (`runs/step4/ppo_full_2M/`)
| Agent | Retour (test, 20 mondes) | Écart apparié |
|---|---|---|
| aléatoire | 80,7 [76,8 ; 84,7] | |
| heuristique | 110,4 [106,4 ; 114,4] | |
| **PPO, politique stochastique (référence)** | **110,9 [109,5 ; 112,2]** (par seed : 108,4 ; 112,8 ; 112,5 ; 110,6 ; 110,1) | − heuristique **+0,5 [−2,0 ; +3,0]** ; − aléatoire **+30,2 [+26,3 ; +33,7]** ; oracle − PPO **+23,0 [+20,3 ; +25,9]** |
| PPO, politique déterministe (argmax) | 110,0 [108,6 ; 111,4] | − heuristique −0,4 [−4,1 ; +2,9] |
| oracle | 133,9 [131,8 ; 135,9] | |

PPO **égale l'heuristique, sans la dépasser**, et reste **23 points sous l'oracle** : c'est la marge qu'un modèle du monde pourrait chercher à combler. Pas d'exploitation de la récompense (population moyenne 62,7, creux 52,2, 11 % du retour dans les 50 derniers pas).

**Référence d'efficacité en échantillons (politique stochastique, courbe de validation, médiane [min-max] sur 5 seeds)** :
| Cible | Pas d'environnement | Temps d'entraînement (CPU) |
|---|---|---|
| 90 % de l'heuristique (98,5) | **78 k** [68-81 k] | 240 s |
| 100 % de l'heuristique (109,5) | **282 k** [192-374 k] | 865 s |
| plateau lent ensuite | ~115 (validation) à 2 M pas | 3 650 s pour 2 M pas |
Politique déterministe (argmax) : 691 k et 1,18 M pas : l'évaluation par argmax **surestime fortement le coût en échantillons** (voir DECISIONS.md).

Courbes : `runs/step4/ppo_full_2M/learning_curves_stochastic.png` (et `learning_curves.png` pour l'argmax).
Premier run (1 M pas, déterministe seul) : 102,4 [100,0 ; 104,8] en test, conservé dans `runs/step4/ppo_full/`.

### Limites
- Hyperparamètres PPO non optimisés (baseline standard) ; 5 seeds ; débit ~600-800 pas/s par processus sur CPU (1 thread). Les 2 M de pas par seed ont pris ~1 h 02 (5 processus en parallèle sur 4 cœurs).
- L'écart PPO − heuristique de la courbe de validation (+5) est plus grand qu'en test (+0,5) : 10 mondes de validation, bruit de ± 2,5 ; la mesure de référence est celle de test (20 mondes).
- Un PPO plus long ou mieux réglé pourrait faire mieux : **commande pour la machine avec GPU** : `python scripts/train_ppo.py --world full --steps 5000000 --seeds 0 1 2 3 4 --workers 1 --refs-val runs/step4/full_baselines/val/summary.json --refs-test runs/step3/test/summary.json --out runs/step4/ppo_full_5M` ; à lancer aussi avec `device: cuda` dans `configs/ppo.yaml` pour comparer le temps réel (le petit réseau peut ne pas gagner à passer sur GPU).

## Étape 5 — World model supervisé (résultats)

`pytest` : **192 passés** (+ 3 lents). Tests éprouvés par mutation (5 bugs volontaires, dont la réintroduction de 3 des défauts trouvés pendant l'étape : tous détectés).
Données : 640 trajectoires (`runs/step5/data`, 56 Mo) — train 400 (326 mondes, 11 % de colonies mortes), val 40, **eval_seen** 100 (mondes d'entraînement, nouvelles trajectoires), **eval_unseen** 100 (mondes de test). Politiques : aléatoire, aléatoire par blocs, heuristique, heuristique bruitée, PPO stochastique, dégénérées.
Modèle : `résiduel`, 0,82 M de paramètres, entraîné **30 min sur CPU** (4 500 itérations en 3 phases : 1 pas observé, imagination de 8 puis 24 pas). Évaluation : boucle ouverte (10 pas de contexte puis AUCUNE observation), 497 fenêtres, IC 95 % par bootstrap d'épisodes.

### Résultats, mondes JAMAIS VUS (compétence = 1 − erreur du modèle / erreur de « rien ne change » ; 0 = pas mieux)
| Horizon (pas macro) | 1 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|
| Carte (macro par canal dynamique) | +0,27 [+0,26 ; +0,28] | +0,26 | **+0,29** [+0,28 ; +0,29] | +0,34 | +0,38 |
| État interne (macro) | +0,78 | +0,81 | **+0,82** | +0,78 | +0,74 |
| Récompense cumulée, R² (modèle) | 0,43 | 0,57 | **0,58** | 0,54 | 0,35 (IC large) |
| Récompense cumulée, R² (« dernière récompense ») | −0,97 | −4,6 | −6,4 | −11,0 | −14,7 |
| Plafond carte (état complet connu, voir plus bas) | 0,78 | 0,64 | 0,61 | 0,64 | 0,69 |

Mondes vus et jamais vus : écart ≤ 0,01 en compétence de carte à tous les horizons (pas de sur-apprentissage aux mondes d'entraînement). Le modèle bat « rien ne change » à tous les horizons (IC strictement > 0).
Par canal à 10 pas : visible +0,41, connu +0,34, roche +0,04, nourriture +0,11, minerai +0,15, éclaireuses +0,50, autres fourmis +0,46, nid exclu (immobile).

### Porte (critères fixés AVANT les résultats, mondes jamais vus)
| Critère | Valeur | |
|---|---|---|
| Compétence carte à 10 pas ≥ 0,25 | +0,287 | ✅ |
| Compétence état à 10 pas ≥ 0,25 | +0,821 | ✅ |
| R² de la récompense cumulée à 10 pas ≥ 0,5 | +0,578 | ✅ |
| Écart vus − jamais vus (carte, 10 pas) ≤ 0,10 | +0,005 | ✅ |
| **Compétence carte à 1 pas ≥ 0,5** | **+0,268** | ❌ |

**Les critères portant sur l'erreur à 10 pas (celui de l'énoncé) sont atteints ; le critère supplémentaire à 1 pas, que j'avais ajouté, ne l'est pas.** Non relâché : diagnostic ci-dessous et décision à l'utilisateur.

### Diagnostic du critère à 1 pas : plafond de prédictibilité (rejeu du simulateur, 32 futurs aléatoires par fenêtre)
Variance irréductible mesurée EXACTEMENT : on rejoue l'épisode (déterministe, vérifié contre les cartes enregistrées) jusqu'à l'instant de départ, puis on lance 32 futurs avec les mêmes actions et des aléas différents. Plafond d'un prédicteur qui **connaîtrait tout l'état caché** :
carte 0,78 / 0,64 / 0,61 / 0,64 / 0,69 (aux 5 horizons) ; état 1,00 → 0,93 ; récompense R² 1,00 → 0,97. **Le modèle atteint 34 % / 41 % / 47 % / 53 % / 55 % du plafond de carte.**
Plafond / modèle par canal à 1 pas : visible 0,79/0,42, connu 0,67/0,26, roche 0,67/0,02, nourriture 0,80/0,10, minerai 0,81/0,10, éclaireuses 0,73/0,52, autres fourmis 1,00/0,45.
**Ce plafond est trop généreux pour un modèle qui ne voit que les observations** : il connaît le contenu du brouillard (d'où 0,67 pour la roche) et la charge et le cap de chaque fourmi. Borne plus réaliste, même en supposant connues les cases qui seront révélées : roche ≤ 0,16, minerai ≤ 0,31, nourriture ≤ 0,50
(le contenu d'une case jamais vue est inconnaissable). Moyenne sur les 7 canaux dynamiques ≲ 0,5-0,6 à 1 pas pour un modèle idéal sur le reste : **le seuil de 0,5 est essentiellement hors d'atteinte pour un modèle à observations seules**, et de toute façon très au-dessus de ce que nous obtenons (0,27).
Leviers pour monter (non testés ici, voir la commande plus bas) : plus de calcul (la validation plafonne dans chaque phase après 4 500 itérations de lot 16), modèle plus grand, plus de données (400 épisodes seulement).

### Comparaison des deux architectures (mondes jamais vus)
| | résiduel | latent vectoriel seul |
|---|---|---|
| Compétence carte à 1 / 10 / 50 pas | +0,27 / +0,29 / +0,38 | **−33,6 / −2,9 / −0,46** |
| Compétence état à 1 / 10 / 50 pas | +0,78 / +0,82 / +0,74 | −3,7 / +0,64 / +0,74 |
| R² récompense cumulée à 1 / 10 / 50 pas | 0,43 / 0,58 / 0,35 | 0,38 / 0,42 / 0,58 |
Hypothèse confirmée : un vecteur de 256 valeurs ne peut pas reproduire exactement une carte 8×32×32 ; sur les canaux STATIQUES (roche −12,7, nourriture −5,7, minerai −2,3 à 10 pas) il est bien pire que « rien ne change ». Il ne se rapproche de la persistance qu'aux grands horizons, où celle-ci se dégrade aussi.
**Réserve** : comparaison confondue — la variante « latent seul » a dû être entraînée avec un plancher de poids différent (7,7·10⁻⁴ au lieu de 5·10⁻⁶), car avec le plancher bas le nid (poids 2·10⁵) rendait son optimisation dégénérée (perte 129).

### Ce que montre la planche « rêve contre réalité » (`runs/step5/wm_residual/eval/dream_vs_reality_*.png`)
Le fond statique (roche, terrain connu, nourriture vue) est reproduit fidèlement jusqu'à 50 pas ; les fourmis individuelles ne le sont pas (le modèle, déterministe, prédit une moyenne diffuse sur une dynamique stochastique : zone claire floue, nid net). Pas de modèle génératif stochastique à ce stade (extension possible).

### Coût de calcul (axe « temps réel »)
Entraînement : 30 min CPU (4 cœurs) pour le résiduel, 19 min pour le latent seul. **Imagination sur CPU : ~720 pas/s (lot 1), ~3 700 pas/s (lot 64), ~2 700 pas/s (lot 256)** — soit MOINS que le vrai simulateur (7 000-15 000 pas/s par cœur) :
sur CPU, planifier dans le modèle n'est pas plus rapide que planifier dans le simulateur ; l'avantage éventuel se jouera sur GPU et en pas d'environnement consommés (à mesurer aux étapes 6-8).

### Défauts trouvés et corrigés (détail dans DECISIONS.md)
Poids de perte renormalisés (×700 de déséquilibre), résidu en logit (canaux binaires bloqués), rétropropagation instable à travers les cartes réinjectées, plancher relatif des poids (canaux faibles sous-pondérés). Aucun n'a affecté un résultat rapporté ci-dessus : tous les chiffres viennent du modèle final.

### Commandes (machine avec GPU)
```
python scripts/collect_data.py --out runs/step5/data_big --n-train 2000          # plus de données (~30 min, 6 Go de RAM environ)
python scripts/train_world_model.py --data runs/step5/data_big --out runs/step5/wm_residual_big --iters 6000 6000 6000 --batch-size 64 --device cuda
python scripts/eval_world_model.py --model runs/step5/wm_residual_big --data runs/step5/data_big --compare runs/step5/wm_vector
python scripts/bayes_ceiling.py --model runs/step5/wm_residual_big --data runs/step5/data_big --split eval_unseen
```
(Les modèles PPO de l'étape 4, `runs/step4/ppo_full_2M/ppo_seed*.zip`, sont nécessaires à la collecte ; sinon `--no-ppo`.)

### Décision de l'utilisateur sur la porte de l'étape 5 : option B puis C
**B (critère révisé)** : « compétence de carte ≥ 40 % du plafond mesuré (prédicteur à état complet), à chaque horizon », en remplacement de « ≥ 0,5 à 1 pas ». Seuil proposé avant application, NON ajusté ensuite.
Résultat (`runs/step5/wm_residual/gate_revised.json`) : part du plafond **34 % (1 pas) ❌, 41 % (5), 47 % (10), 53 % (25), 55 % (50) ✅** ; les quatre critères à 10 pas (compétence carte, compétence état, R² de la récompense, écart vus/jamais vus) restent ✅.
La porte reste donc formellement **non atteinte à 1 pas** ; l'utilisateur a décidé d'avancer malgré cela. **C** : étape 6 avec un horizon de planification ≤ 25 pas (R² de la récompense cumulée 0,54 à 25 pas, 0,35 à 50 pas).
Le plafond est mis en cache avec les données (`runs/step5/data/ceiling_eval_unseen.json`, indépendant du modèle) et sert à tout modèle futur.


## Étape 6 — Planification CEM/MPC dans le modèle appris (résultats)

Mêmes mondes de TEST pour tout (8 mondes, seeds 1 000 000…, planificateur sans accès privilégié ; retour moyen d'épisode, IC 95 % bootstrap sur les mondes).
Références sur ces 8 mondes : aléatoire 75,9 · heuristique 106,3 · oracle (simulateur vrai, horizon complet) 133,3. Score normalisé = (retour − aléatoire)/(oracle − aléatoire).

### Test 1 — le planificateur est-il bon ? (même code, vrai simulateur à la place du modèle)
| Planificateur (simulateur vrai) | Retour | Normalisé | Temps/épisode |
|---|---|---|---|
| H25, 32 séq./plan | 109,6 [102,3 ; 115,6] | 0,59 | 8 s |
| H25, 192 séq./plan | 113,3 [108,6 ; 117,4] | 0,65 | 44 s |
| H50, 96 séq./plan | 120,3 [115,6 ; 125,0] | 0,77 | 40 s |
| H100, 96 séq./plan | 127,8 [124,2 ; 131,4] | 0,90 | 78 s |
Le score monte régulièrement avec l'horizon et tend vers l'oracle (horizon complet) : **l'écart de l'horizon 25 à l'oracle est dû à l'horizon, pas au planificateur** (testé, pas supposé). Le planificateur est donc validé ; il bat l'heuristique dès H25 (+7,0 [+2,4 ; +11,7] à 192 séq.).

### Test 2 — score contre budget dans le modèle appris (horizon 25 sauf mention)
| Séquences/plan | Retour | Normalisé | Temps/épisode (CPU) | Pas imaginés/épisode |
|---|---|---|---|---|
| 16 | 83,5 [77,7 ; 88,0] | 0,13 | 29 s | 31 k |
| 32 | 97,4 [90,1 ; 103,9] | 0,37 | 51 s | 62 k |
| 96 | 99,7 [95,7 ; 104,1] | 0,41 | 113 s | 187 k |
| 192 | 96,2 [90,2 ; 102,3] | 0,35 | 243 s | 374 k |
| 384 | 98,6 [94,3 ; 103,1] | 0,40 | 490 s | 749 k |
| 96, H10 | 98,4 [93,4 ; 103,8] | 0,39 | 48 s | 76 k |
| 96, H15 | 98,8 [94,1 ; 103,6] | 0,40 | 69 s | 114 k |
Le score bat l'aléatoire (+20 à +24) mais **reste sous l'heuristique** (−6,7 [−12,2 ; −1,3] à 96 séq. ; −7,8 à 384) et **plafonne vers 96-100 dès ~32 séquences** : multiplier le budget par 12 n'apporte rien (les IC se recouvrent entièrement). Raccourcir l'horizon (10, 15) n'aide pas non plus. Le même planificateur avec le vrai simulateur à H25 atteint 109,6-113,3 : **l'écart de ~12 à 15 points vient de l'erreur du modèle**.
Figure : `docs/img/step6_budget_curve.png` (score vs budget, vs temps réel, exploitation).

### Test 3 — le planificateur exploite-t-il les erreurs du modèle ? **Oui.**
Récompense cumulée sur 25 pas, prédit − réel (le plan choisi est rejoué dans le vrai simulateur depuis l'état réel ; contrôle : 8 séquences aléatoires) :
| Séq./plan | 16 | 32 | 96 | 192 | 384 |
|---|---|---|---|---|---|
| plan CHOISI | +0,65 | +1,33 | +2,63 | +3,39 | +3,38 |
| séquences ALÉATOIRES | +0,18 | +0,91 | +1,32 | +1,09 | +0,89 |
| corrélation prédit/réel (plans choisis) | 0,80 | 0,66 | 0,34 | 0,58 | 0,38 |
Le modèle est optimiste sur les plans choisis, et d'autant plus que la recherche est poussée (malédiction de l'optimiseur) ; les séquences aléatoires, non optimisées contre lui, le sont bien moins. Le biais reste de l'ordre de 3 unités de retour sur 25 pas, contre ~1 pour le hasard. Le classement des plans devient moins fiable (corrélation en baisse), ce qui explique le plafond.

### Limites
8 mondes seulement (IC larges : ±5) ; planificateur myope (H ≤ 25, pas de valeur terminale) ; une seule graine de planification par monde ; le modèle est celui de l'étape 5 (porte formellement non atteinte à 1 pas, voir ci-dessus) ; imagination sur CPU ~2 700 pas/s, plus lente que le vrai simulateur.

### Pistes (pas encore faites)
Pénaliser l'incertitude (ensemble de modèles), ré-entraîner le modèle sur les données collectées par le planificateur (c'est la boucle de l'étape 7), valeur terminale apprise (étape 7). Les trois attaquent la cause mesurée (modèle exploité et horizon court).

### Commandes
```
python scripts/run_planner.py --mode sim   --horizon 25 --samples 64 --iters 3 --n-worlds 20 --workers 4 --threads 1 --tag sim_h25_64x3
python scripts/run_planner.py --mode model --horizon 25 --samples 32 --iters 3 --n-worlds 20 --workers 4 --threads 1 --tag model_32x3
python scripts/plot_planner.py runs/step6
```
(`--device cuda` pour l'imagination sur GPU.)


## Étape 7 — Dreamer-lite : acteur-critique entraîné dans l'imagination (résultats)

**Verdict : porte NON atteinte.** Critère : « bat l'heuristique avec moins de pas d'environnement que PPO » (PPO : ~78 k pas pour 90 % de l'heuristique, ~282 k pour 100 %). Dreamer-lite ne l'atteint pas de façon durable : **0 graine sur 4** ne tient 98,5 (90 % de l'heuristique) sur la courbe de validation, et le retour final de test est **au niveau de l'aléatoire**.

### Ce qui a été construit (voir DECISIONS.md)
Boucle complète sans aucune donnée préchargée (ni PPO ni le modèle de l'étape 5 : chaque pas d'environnement est compté) : échauffement aléatoire de 4 000 pas → jouer un épisode avec la politique courante (stochastique) → mettre à jour le modèle (de zéro, en ligne) → rêver depuis des croyances réelles → mettre à jour acteur et critique (rendements λ, REINFORCE). `src/models/dreamer.py` (noyau), `src/agents/dreamer_agent.py` (agent, boucle, mesures rêve/réalité), `scripts/train_dreamer.py`, `configs/dreamer.yaml`, `tests/test_dreamer.py` (14 tests : rendements λ vérifiés à la main, monde jouet à crédit retardé, contrôle négatif sur du bruit, arrêt du rêve à la fin de l'épisode, tampon, reproductibilité de bout en bout ; 3 mutations sur 4 détectées, voir DECISIONS.md).

### Résultats finals (config retenue : horizon 8, ancrage du critique, 100 000 pas, 4 graines)
Courbe de validation (10 mondes, politique stochastique, moyenne des 4 graines aux pas 4 k, 14 k, …, 94 k, 100 k) : **82,3 – 91,8 – 88,8 – 91,7 – 87,6 – 94,5 – 82,3 – 92,2 – 87,9 – 81,0 – 82,1**. Aléatoire = 82,6 ; heuristique = 109,5 ; cible « 90 % » = 98,5.
| Graine | Pic de validation | Moyenne des 5 derniers points | Test (20 mondes, politique finale) |
|---|---|---|---|
| 0 | 101,2 | 92,1 | **102,2** |
| 1 | 94,8 | 87,0 | 77,1 |
| 2 | 96,1 | 79,0 | 71,3 |
| 3 | 98,2 | 82,3 | 79,5 |
Test, moyenne des 4 graines : **82,5** (aléatoire 80,7 ; heuristique 110,4 ; PPO à 2 M pas 110,9 ; oracle 133,9). Seule la graine 0 finit haut, sur un dernier point de validation haut : 3 graines sur 4 finissent sous l'aléatoire. **Le gain précoce (+9 vers 14 k pas) n'est pas conservé ni prolongé.**
Temps d'entraînement : 3 700 à 4 900 s CPU pour 100 k pas (1 à 2 threads), contre ~240 s pour que PPO atteigne 90 % de l'heuristique : **~20× plus lent en temps réel, et moins bon par pas d'environnement**. Figure : `docs/img/step7_dreamer_vs_ppo.png`.
Les évaluations de validation (10 mondes, une graine d'agent) ont un bruit de l'ordre de ± 5 ; les oscillations de ± 10 d'un point à l'autre dépassent ce bruit.

### Ce que les diagnostics ont établi (mesuré)
1. **Premier essai (réglages par défaut, 60 k pas, 1 graine) : aucun apprentissage** (79,8 à la fin ; 77,6 à 24 k). Journal : l'entropie de la politique s'effondre de 6,9 à 1,4 en ~3 000 pas, la récompense rêvée reste à ~0,2 par pas.
2. **Diagnostic à modèle FIXE** (`scripts/diag_dreamer_fixed_model.py` : acteur-critique entraîné dans le modèle de l'étape 5, qui est bien meilleur que le modèle en ligne ; croyances de départ tirées des données de l'étape 5 ; **n'est pas un résultat d'efficacité** car ces données incluent des trajectoires de PPO) :
   - réglages par défaut : récompense rêvée +40 % (0,21 → 0,30) mais **retour réel inchangé (78,4)** : l'acteur optimise des erreurs du modèle ;
   - plancher d'avantage 1,0 et horizon 5 : **81 → 96,9** (95,3 à 1 000 mises à jour) ; horizon 15 : plus lent et écart rêve−réel plus grand (+0,87 contre +0,35 pour des séquences aléatoires à 500 mises à jour) ;
   - ancrage du critique sur les retours réels : biais du critique de −15 à −0,6 (mesuré), mais **pas de gain de retour** (88,5 à 1 000 mises à jour avec H5).
   **Plafond ≈ 95–97 avec un bon modèle fixe**, soit le même que le planificateur de l'étape 6 dans ce modèle (98–100) et loin des 110–113 du même planificateur dans le vrai simulateur à horizon 25.
3. **Boucle en ligne (modèle appris de zéro)**, deux variantes à 100 k pas : H8 + ancrage (graine 0 : moyenne des 5 derniers points 92,1) et H5 sans ancrage (73,1 ; deux effondrements à 52,7 et 31,4, population moyenne 7 = colonie quasi éteinte, alors que l'écart rêve−réel à 5 pas restait faible). Puis 3 graines de plus de la première : bilan ci-dessus.
4. **Le modèle appris en ligne n'est jamais nettement meilleur que « rien ne change »** : la perte de carte normalisée (1 = persistance) reste entre 1 et 4 sur toute la durée (journaux `dreamer_seed*_train.jsonl`), malgré 4 440 mises à jour (40 pour 1 000 pas).
5. L'écart rêve−réel de la politique finale est petit (|écart| ≲ 0,3 sur 8 pas) : à cet horizon, le modèle n'est pas visiblement exploité par la politique.

### Ce qui n'est PAS établi
La **cause de la dégradation après ~50 k pas** (courbes qui retombent vers l'aléatoire). Candidats, aucun testé : modèle qui ne s'améliore pas assez (point 4), non-stationnarité modèle/critique/politique, information de long terme (bâtiments, stocks) mal portée par le modèle à 8 pas. Une seule configuration finale ; budget de réglage faible (4 diagnostics à modèle fixe de 1 000 mises à jour, 2 variantes en ligne) ; le réglage n'a utilisé que les mondes de validation.

### Écarts à Dreamer pur (à ablater à l'étape 8)
Critique ancré sur des retours réels du tampon (`real_value_weight`, 0 = Dreamer pur) ; REINFORCE (actions discrètes) et non rétropropagation à travers le modèle ; modèle déterministe sans variable latente stochastique ; troncature de l'épisode suivie explicitement dans les rêves.

### Options (décision de l'utilisateur)
1. Passer à l'étape 8 avec Dreamer-lite tel quel : comparaison honnête à nombre de pas égal (il perdra contre PPO sur l'efficacité) et tests de généralisation / changement de règle, où un modèle peut aider.
2. D'abord améliorer le modèle (peut se faire sur GPU) : beaucoup plus de mises à jour et de données ; ensemble de modèles avec pénalité de désaccord ; variable latente stochastique (type RSSM).
3. Tester d'abord les causes candidates (courbe de perte du modèle contre mises à jour, ablation du critique, modèle figé après N pas).

### Commandes
```
python scripts/train_dreamer.py --seeds 0 1 2 3 4 --workers 1 --threads 4 --steps 200000 --out runs/step7/dreamer \
    --refs-val runs/step4/full_baselines/val/summary.json --refs-test runs/step3/test/summary.json     # config retenue = configs/dreamer.yaml
python scripts/train_dreamer.py ... --set ac.horizon=15 ac.real_value_weight=0 wm_updates_per_1k=100      # surcharges "a.b=valeur"
python scripts/diag_dreamer_fixed_model.py --updates 1000 --set ac.horizon=5 --out runs/step7/diag
python scripts/plot_dreamer.py --dreamer runs/step7/dreamer/dreamer_seed*_curve.jsonl --ppo runs/step4/ppo_full_2M/ppo_seed*_curve.jsonl --refs runs/step4/full_baselines/val/summary.json
```
(`device: cuda` dans `configs/dreamer.yaml` pour le GPU.)
