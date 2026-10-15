"""Étape 1 : bruit de Perlin et générateur de mondes."""

import numpy as np
import pytest

from antworld.config import WorldGenConfig
from antworld.kernels import bfs_from
from antworld.layout import generate_world, playability_problem
from antworld.noise import fractal_noise, perlin


def test_perlin_deterministic_and_bounded():
    a = perlin(32, 8.0, np.random.default_rng(1))
    b = perlin(32, 8.0, np.random.default_rng(1))
    assert np.array_equal(a, b) and a.shape == (32, 32)
    assert np.abs(a).max() < 1.0
    assert not np.array_equal(a, perlin(32, 8.0, np.random.default_rng(2)))


def test_perlin_is_smooth():
    """Deux cases voisines diffèrent bien moins que deux cases tirées au hasard (bruit blanc)."""
    field = fractal_noise(64, 8.0, 1, np.random.default_rng(0))
    neighbor_diff = np.abs(np.diff(field, axis=1)).mean()
    shuffled = np.random.default_rng(0).permutation(field.ravel()).reshape(field.shape)
    assert neighbor_diff < 0.4 * np.abs(np.diff(shuffled, axis=1)).mean()
    assert 0.0 <= field.min() and field.max() == pytest.approx(1.0)


def test_generate_world_is_deterministic():
    cfg = WorldGenConfig()
    a, b = generate_world(cfg, 7), generate_world(cfg, 7)
    for name in ("terrain", "food_cap", "ore_amount", "nest_next", "nest_dist"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.nest == b.nest
    assert not np.array_equal(a.terrain, generate_world(cfg, 8).terrain)


def test_many_worlds_are_playable_and_varied():
    cfg = WorldGenConfig()
    seen, food_dists, ore_dists, attempts, food_cells = set(), [], [], [], []
    for seed in range(200):
        lay = generate_world(cfg, seed)
        s = lay.size
        # Test indépendant de celui du générateur : on recalcule l'accessibilité depuis le nid.
        src = np.zeros(s * s, np.uint8)
        src[lay.nest] = 1
        _, dist = bfs_from(lay.terrain, s, src)
        food, ore = lay.food_cap > 0, lay.ore_amount > 0
        assert lay.terrain[lay.nest] == 0
        assert (dist[food] >= cfg.min_resource_dist).all() and (dist[ore] >= cfg.min_resource_dist).all()
        assert food.sum() >= cfg.min_food_cells and ore.sum() >= cfg.min_ore_cells
        assert (dist[food] >= 0).all() and (dist[ore] >= 0).all(), "ressource inaccessible depuis le nid"
        assert not (food & ore).any(), "nourriture et minerai sur la même case"
        assert (lay.terrain[food] == 0).all() and (lay.terrain[ore] == 0).all()
        assert (lay.food_cap[food] >= cfg.food_cap_min - 1e-9).all() and (lay.food_cap[food] <= cfg.food_cap_max + 1e-9).all()
        assert playability_problem(lay, cfg) is None
        seen.add(lay.terrain.tobytes() + lay.food_cap.tobytes())
        food_dists.append(dist[food].min()); ore_dists.append(dist[ore].min())
        attempts.append(lay.attempts); food_cells.append(int(food.sum()))
    assert len(seen) == 200, "des seeds différentes ont donné des mondes identiques"
    assert len(set(food_dists)) > 3 and len(set(ore_dists)) > 3, "pas assez de diversité de distances"
    print(f"\n[worldgen 200 seeds] tentatives moy {np.mean(attempts):.2f} (max {max(attempts)}) ; "
          f"distance nourriture la plus proche {min(food_dists)}..{max(food_dists)} ; "
          f"minerai {min(ore_dists)}..{max(ore_dists)} ; cases de nourriture {min(food_cells)}..{max(food_cells)}")
