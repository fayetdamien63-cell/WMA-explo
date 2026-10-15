"""Étape 1 : mêmes seed et actions => même trajectoire, y compris entre processus."""

import os
import subprocess
import sys

import numpy as np

from antworld import AntWorld
from helpers import default_cfg, random_play, state_hash

CFG = default_cfg()


def trajectory(world_seed: int, action_seed: int, steps: int = 120) -> list[str]:
    w = AntWorld.from_seed(CFG, world_seed)
    return [state_hash(w)] + [state_hash(x) for x in random_play(w, np.random.default_rng(action_seed), steps)]


def test_same_seed_and_actions_give_identical_trajectory():
    assert trajectory(5, 9) == trajectory(5, 9)


def test_different_world_or_actions_diverge():
    assert trajectory(5, 9)[-1] != trajectory(6, 9)[-1]
    assert trajectory(5, 9)[-1] != trajectory(5, 10)[-1]


def test_trajectory_is_identical_across_processes():
    code = (
        "import sys; sys.path.insert(0, 'tests');"
        "from test_world_determinism import trajectory; print(trajectory(5, 9)[-1])"
    )
    outs = []
    for hashseed in ("1", "777"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                   env=env, check=True).stdout.strip())
    assert outs[0] == outs[1] == trajectory(5, 9)[-1]
