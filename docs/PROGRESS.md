# Avancement

| Étape | État |
|---|---|
| 0 — Fondations | ✅ terminée (en attente de feu vert) |
| 1 — Simulateur | ⏳ non commencée |
| 2 à 9 | — |

## Étape 0 — résultats de la porte

Environnement de test : Linux, Python 3.13, torch 2.14.1+cpu (pas de GPU).

- `pytest` : **25 passés / 25** (config 7, device 8, seeding 10).
- Déterminisme : `scripts/check_determinism.py --seed 42` lancé dans deux processus avec des
  `PYTHONHASHSEED` différents → sorties et empreinte identiques (`f64ecb04c9a3edbe`) ; seed 43 → autre
  empreinte (`c7ed8ffe58690721`).
- Tenseur PyTorch : produit matriciel sur le device détecté (`cpu` ici) correct.

### Non vérifié ici
- Les branches **CUDA** et **MPS** de `get_device()` ne sont testées que par simulation de la
  disponibilité (monkeypatch) : cette sandbox n'a ni GPU ni Mac. À confirmer chez toi :
  `pytest tests/test_device.py` sur le MacBook (attendu : device `mps`) et sur le PC RTX (`cuda`).
- La reproductibilité exacte des nombres **entre machines/OS** n'est pas garantie pour torch
  (seulement dans une même configuration) ; NumPy avec `SeedSequence` est stable entre OS.
