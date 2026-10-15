# Avancement

| Étape | État |
|---|---|
| 0 — Fondations | ✅ validée (commit `15e2a5b`) |
| 1 — Simulateur | ✅ tests passés, **en attente de la porte (regard de l'utilisateur sur une partie aléatoire)** |
| 2 à 9 | — |

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
