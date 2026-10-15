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
