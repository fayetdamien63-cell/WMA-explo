"""Étape 3 (test long) : l'ordre aléatoire < heuristique < oracle est net, sur les seeds de VALIDATION.

    pytest -m slow tests/test_baselines_gate.py

Le verdict final de la porte se lit sur les seeds de test via scripts/run_baselines.py.
"""

import pytest

from agents.factory import BaselinesConfig
from eval.harness import evaluate, exploitation_warnings, returns_by_world
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


def test_a_planning_oracle_does_not_exploit_the_reward():
    """Test d'un OPTIMISEUR (pas d'une stratégie codée à la main) : leçon de l'étape 3.

    Un oracle qui planifie jusqu'à la fin de l'épisode ne doit pas sacrifier la colonie : pas d'alerte
    d'exploitation, et une population moyenne comparable à celle de l'heuristique.
    """
    import copy
    cfg = default_cfg()
    base = load_config(BaselinesConfig, CONFIG_PATH.parent / "baselines.yaml")
    base = copy.deepcopy(base)
    base.oracle.cem.n_samples, base.oracle.cem.n_iters, base.oracle.cem.replan_every = 32, 2, 10   # budget moyen
    worlds = val_seeds(4)
    oracle = evaluate("oracle", cfg, base, worlds, workers=4)
    heur = evaluate("heuristic", cfg, base, worlds)
    print(f"\noracle : pop moy {sum(r.mean_pop for r in oracle) / 4:.1f}, creux {sum(r.min_pop for r in oracle) / 4:.1f}, "
          f"part finale {sum(r.late_return_share for r in oracle) / 4:.2f} | heuristique : pop moy {sum(r.mean_pop for r in heur) / 4:.1f}")
    assert not exploitation_warnings(oracle)
    assert sum(r.mean_pop for r in oracle) >= 0.9 * sum(r.mean_pop for r in heur)
