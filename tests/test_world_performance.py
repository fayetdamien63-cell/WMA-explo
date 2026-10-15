"""Étape 1 : performance (cible > 5 000 pas macro/s sur 32x32, à ajuster)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from bench_sim import measure  # noqa: E402

from helpers import default_cfg  # noqa: E402

TARGET_STEPS_PER_S = 5000


def test_simulator_speed():
    r = measure(default_cfg())
    print(f"\n[perf] {r['steps_per_s']:.0f} pas/s, {r['ticks_per_s']:.0f} ticks/s, "
          f"génération de monde {r['world_gen_ms']:.2f} ms")
    assert r["steps_per_s"] > TARGET_STEPS_PER_S
