"""Étape 1 : invariants et conservation, vérifiés à CHAQUE pas de parties entières."""

import numpy as np
import pytest

from antworld import AntWorld
from helpers import check_invariants, default_cfg, random_play

CFG = default_cfg()

POLICIES = {
    "aléatoire": lambda w, rng: rng.integers(0, w.action_nvec),
    "tout nourriture": lambda w, rng: np.array([0, 0, 0, 0, 0]),
    "extrêmes": lambda w, rng: np.array(rng.choice([0, 3], 3).tolist() + [rng.integers(0, 3), rng.integers(0, 5)]),
    "bâtisseuses + minerai": lambda w, rng: np.array([1, 2, 3, rng.integers(1, 3), rng.integers(0, 5)]),
}


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_invariants_hold_throughout_an_episode(seed, policy):
    w = AntWorld.from_seed(CFG, seed)
    rng = np.random.default_rng(seed)
    check_invariants(w)
    previous = w.col.copy()
    while not w.done:
        w.step(POLICIES[policy](w, rng))
        check_invariants(w)
        # Les compteurs cumulés ne diminuent jamais.
        from antworld import indices as ix
        assert (w.col[ix.C_HARVESTED:ix.N_COL] >= previous[ix.C_HARVESTED:ix.N_COL] - 1e-12).all()
        previous = w.col.copy()


def test_episodes_always_end():
    for seed in range(10):
        w = AntWorld.from_seed(CFG, seed)
        for _ in random_play(w, np.random.default_rng(seed), CFG.sim.max_steps + 50):
            pass
        assert w.done and w.macro_step <= CFG.sim.max_steps


def test_invariant_checker_detects_cheating():
    """Test du test : créer de la ressource « de nulle part » doit faire échouer les invariants."""
    from antworld import indices as ix
    for tamper in (
        lambda w: w.col.__setitem__(ix.C_FOOD_STOCK, w.col[ix.C_FOOD_STOCK] + 5.0),   # stock gonflé
        lambda w: w.col.__setitem__(ix.C_ORE_STOCK, w.col[ix.C_ORE_STOCK] + 1.0),
        lambda w: w.grid.__setitem__((ix.G_AMOUNT + ix.R_ORE, np.flatnonzero(w.layout.ore_amount)[0]), 999.0),
        lambda w: w.grid.__setitem__((ix.G_AMOUNT + ix.R_FOOD, 0), -1.0),
        lambda w: w.col.__setitem__(ix.C_N_ANTS, w.col[ix.C_N_ANTS] + 1),             # fourmi apparue sans naissance
    ):
        w = AntWorld.from_seed(CFG, 1)
        for _ in random_play(w, np.random.default_rng(1), 30):
            pass
        check_invariants(w)
        tamper(w)
        with pytest.raises(AssertionError):
            check_invariants(w)
