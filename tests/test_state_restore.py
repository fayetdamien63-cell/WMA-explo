"""Étape 2 : sauvegarde / restauration d'état complète (nécessaire à l'oracle)."""

import numpy as np

from antworld import AntWorld
from antworld.env import AntWorldEnv
from helpers import default_cfg, state_hash

CFG = default_cfg()


def actions(seed, n):
    return np.random.default_rng(seed).integers(0, [4, 4, 4, 3, 5], size=(n, 5))


def play(world, acts):
    return [state_hash(world) if not world.step(a) else None for a in acts]  # step renvoie None


def test_restore_replays_exactly_the_same_future():
    w = AntWorld.from_seed(CFG, 4)
    for a in actions(0, 40):
        w.step(a)
    snap = w.snapshot()
    future = actions(1, 60)
    first = play(w, future)
    final_hash = state_hash(w)
    w.restore(snap)                       # retour au passé
    assert state_hash(w) != final_hash
    assert play(w, future) == first and state_hash(w) == final_hash


def test_restore_then_a_different_future_differs():
    w = AntWorld.from_seed(CFG, 4)
    snap = w.snapshot()
    play(w, actions(1, 40))
    h1 = state_hash(w)
    w.restore(snap)
    play(w, actions(2, 40))
    assert state_hash(w) != h1


def test_snapshot_is_not_modified_by_later_steps():
    w = AntWorld.from_seed(CFG, 4)
    snap = w.snapshot()
    before = [a.copy() for a in (snap.grid, snap.flow, snap.ants, snap.col)]
    play(w, actions(0, 50))
    for a, b in zip(before, (snap.grid, snap.flow, snap.ants, snap.col)):
        assert np.array_equal(a, b)


def test_snapshot_can_be_restored_many_times():
    w = AntWorld.from_seed(CFG, 6)
    snap, acts = w.snapshot(), actions(3, 30)
    hashes = set()
    for _ in range(3):
        w.restore(snap)
        play(w, acts)
        hashes.add(state_hash(w))
    assert len(hashes) == 1


def test_clone_is_independent_and_identical():
    w = AntWorld.from_seed(CFG, 7)
    for a in actions(0, 25):
        w.step(a)
    c = w.clone()
    assert state_hash(c) == state_hash(w) and c.layout is w.layout
    acts = actions(1, 40)
    play(c, acts)
    h_clone = state_hash(c)
    assert state_hash(w) != h_clone, "avancer le clone ne doit pas toucher l'original"
    play(w, acts)
    assert state_hash(w) == h_clone, "même état + mêmes actions => même futur (le générateur aléatoire est copié)"


def test_env_state_roundtrip_across_env_instances():
    env_a, env_b = AntWorldEnv(CFG), AntWorldEnv(CFG)
    env_a.reset(seed=9)
    for a in actions(0, 30):
        env_a.step(a)
    state = env_a.get_state()
    future = actions(1, 50)
    out_a = [env_a.step(a) for a in future]

    env_b.set_state(state)                                  # un autre env, jamais reset
    out_b = [env_b.step(a) for a in future]
    for (oa, ra, ta, tra, _), (ob, rb, tb, trb, _) in zip(out_a, out_b):
        assert ra == rb and ta == tb and tra == trb
        assert np.array_equal(oa["map"], ob["map"]) and np.array_equal(oa["state"], ob["state"])

    env_a.set_state(state)                                  # et le même env, remonté dans le temps
    out_a2 = [env_a.step(a) for a in future]
    assert [r for _, r, *_ in out_a] == [r for _, r, *_ in out_a2]


def test_restore_works_after_the_colony_died():
    w = AntWorld.from_seed(CFG, 2)
    alive_snap = w.snapshot()
    while w.alive:
        w.step([3, 3, 0, 0, 0])
    assert not w.alive
    w.restore(alive_snap)
    assert w.alive and w.macro_step == 0
