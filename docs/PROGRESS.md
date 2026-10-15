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
