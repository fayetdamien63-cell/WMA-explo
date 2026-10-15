"""Étape 0 : l'aléa est entièrement piloté par la seed maître."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from common.seeding import SeedManager, seed_everything

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_determinism.py"


def test_same_seed_same_numbers():
    a = SeedManager(42).rng("world").random(100)
    b = SeedManager(42).rng("world").random(100)
    assert np.array_equal(a, b)


def test_different_seed_different_numbers():
    a = SeedManager(42).rng("world").random(100)
    b = SeedManager(43).rng("world").random(100)
    assert not np.array_equal(a, b)


def test_streams_are_independent_of_each_other():
    """Tirer dans le flux "ants" ne doit pas perturber le flux "world"."""
    seeds = SeedManager(1)
    world_alone = seeds.rng("world").random(10)
    seeds.rng("ants").random(1000)  # activité sur un autre flux
    world_after = seeds.rng("world").random(10)
    assert np.array_equal(world_alone, world_after)
    assert not np.array_equal(world_alone, seeds.rng("ants").random(10))


def test_int_seed_and_child_are_deterministic_and_in_range():
    s = SeedManager(5)
    assert s.int_seed("x") == SeedManager(5).int_seed("x")
    assert s.int_seed("x") != s.int_seed("y")
    assert 0 <= s.int_seed("x") < 2**31
    assert s.child("w0").rng("g").random() == SeedManager(5).child("w0").rng("g").random()


@pytest.mark.parametrize("bad", [-1, 1.5, True, "3"])
def test_invalid_seed_rejected(bad):
    with pytest.raises(ValueError):
        SeedManager(bad)


def test_seed_everything_fixes_global_generators():
    import torch

    seed_everything(7)
    first = (np.random.rand(3), torch.randn(3))
    seed_everything(7)
    second = (np.random.rand(3), torch.randn(3))
    assert np.array_equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def _run_script(seed: int, hashseed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--seed", str(seed)],
        capture_output=True, text=True, env=env, check=True,
    )
    return out.stdout


def test_script_is_reproducible_across_processes():
    """Deux processus distincts (et PYTHONHASHSEED différents) : sortie identique."""
    out1 = _run_script(42, hashseed="1")
    out2 = _run_script(42, hashseed="999")
    assert out1 == out2
    assert out1 != _run_script(43, hashseed="1")
