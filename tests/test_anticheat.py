"""Étape 2 : test anti-triche — aucune stratégie absurde n'obtient un score aberrant."""

import pytest

from eval.anticheat import evaluate
from helpers import default_cfg


@pytest.fixture(scope="module")
def table():
    t = evaluate(default_cfg(), seeds=range(20))
    print("\n" + "\n".join(f"  {n:52s} retour moyen {r['return']:7.1f}  pop {r['pop']:5.1f}  minerai {r['ore']:6.1f}"
                           for n, r in sorted(t.items(), key=lambda kv: -kv[1]["return"])))
    return t


def test_no_absurd_strategy_beats_the_sensible_reference(table):
    reference = table["référence sensée"]["return"]
    for name, r in table.items():
        if r["kind"] == "absurde":
            assert r["return"] < reference, f"« {name} » ({r['return']:.1f}) bat la référence sensée ({reference:.1f})"
            assert r["max_return"] < table["référence sensée"]["max_return"], name


def test_doing_nothing_or_exploring_nothing_is_worse_than_exploring(table):
    assert table["ne rien faire (tous cueillent la nourriture)"]["return"] < table["10 % d'éclaireuses, rien d'autre"]["return"]


def test_hoarding_ore_does_not_pay(table):
    """Thésauriser du minerai en surplus fait légèrement baisser le score, sans l'effondrer."""
    hoard, plain = table["thésauriser : 25 % minerai, ne jamais construire"], table["10 % d'éclaireuses, rien d'autre"]
    assert hoard["ore"] > 100                       # la stratégie thésaurise bien
    assert hoard["return"] < plain["return"]        # pénalité effective...
    assert hoard["return"] > 0.8 * plain["return"]  # ... mais légère


def test_returns_stay_below_the_theoretical_bound(table):
    for name, r in table.items():
        assert r["min_margin_to_bound"] > 0, name


def test_reward_is_exactly_potential_based(table):
    for name, r in table.items():
        assert r["max_telescope_error"] < 1e-9, name
