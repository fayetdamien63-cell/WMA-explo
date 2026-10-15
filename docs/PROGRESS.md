# Avancement

| Étape | État |
|---|---|
| 0 — Fondations | ✅ validée (commit `15e2a5b`) |
| 1 — Simulateur | ✅ validée (commit `6064638`) ; ajustement : minerai réduit (étape 2) |
| 2 — Env Gymnasium | ✅ tests passés, en attente de feu vert |
| 3 à 9 | — |

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
