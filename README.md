# antworld-wm

Banc d'essai pédagogique : comparer des agents (aléatoire, heuristique, oracle, PPO, planification
dans un world model, Dreamer-lite) dans un monde de fourmilières généré procéduralement.
Avancement : `docs/PROGRESS.md`. Décisions : `docs/DECISIONS.md`.

## Installation (Python 3.11+)

```bash
python -m venv .venv
# macOS/Linux : source .venv/bin/activate      Windows : .venv\Scripts\activate
```

PyTorch dépend de la machine — l'installer d'abord, puis le projet :

- **MacBook (MPS)** : `pip install torch`
- **PC Windows + RTX (CUDA)** : `pip install torch --index-url https://download.pytorch.org/whl/cu124`
  (adapter `cu124` à la version CUDA indiquée sur https://pytorch.org)
- **CPU seul** : `pip install torch --index-url https://download.pytorch.org/whl/cpu`

```bash
pip install -e ".[dev,viz]"  # extras : dev (pytest), viz (rendu/GIF), env, rl (étapes suivantes)
pytest                       # lance tous les tests
python scripts/check_determinism.py --seed 42
python scripts/bench_sim.py                       # vitesse du simulateur sur cette machine
python scripts/play_random.py --seed 3            # partie aléatoire -> runs/step1/*.gif et *.png
python scripts/run_baselines.py --split test --n-worlds 20 --workers 4   # aléatoire, heuristique, oracle (étape 3)
python scripts/train_ppo.py --world simple --steps 100000 --workers 4    # PPO, monde simplifié (étape 4)
python scripts/train_ppo.py --world full --steps 2000000 --workers 5     # PPO, monde complet (~1 h sur 4 cœurs)
pytest -m slow                                                           # tests longs
```

## Conventions

- Toute l'aléa passe par `common.seeding.SeedManager` (jamais de générateur créé à la main).
- Les paramètres vivent dans `configs/*.yaml`, chargés par `common.config.load_config`.
- Le device d'apprentissage vient de `common.device.get_device()` ; `ANTWORLD_DEVICE=cpu` le force.
