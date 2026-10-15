"""Étape 3 (test long) : l'ordre aléatoire < heuristique < oracle est net, sur les seeds de VALIDATION.

    pytest -m slow tests/test_baselines_gate.py

Le verdict final de la porte se lit sur les seeds de test via scripts/run_baselines.py.
"""

import pytest

from agents.factory import BaselinesConfig
from eval.harness import evaluate, returns_by_world
from eval.seeds import val_seeds
from eval.stats import paired_difference
from helpers import CONFIG_PATH, default_cfg
from common.config import load_config

pytestmark = pytest.mark.slow


def test_random_below_heuristic_below_oracle_on_validation_worlds():
    cfg = default_cfg()
    base = load_config(BaselinesConfig, CONFIG_PATH.parent / "baselines.yaml")
    worlds = val_seeds(8)
    per = {}
    for name, seeds in (("random", (0, 1, 2)), ("heuristic", (0,)), ("oracle", (0,))):
        res = evaluate(name, cfg, base, worlds, agent_seeds=seeds, workers=4)
        per[name] = returns_by_world(res, name, worlds)
    h_r = paired_difference(per["heuristic"], per["random"])
    o_h = paired_difference(per["oracle"], per["heuristic"])
    print(f"\nheuristique − aléatoire {h_r.mean:+.1f} [{h_r.ci_low:+.1f}, {h_r.ci_high:+.1f}] ; "
          f"oracle − heuristique {o_h.mean:+.1f} [{o_h.ci_low:+.1f}, {o_h.ci_high:+.1f}]")
    assert h_r.ci_low > 0 and o_h.ci_low > 0
