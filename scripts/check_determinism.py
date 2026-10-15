"""Affiche des nombres issus de la seed donnée ; deux exécutions doivent être identiques.

    python scripts/check_determinism.py --seed 42
"""

import argparse
import hashlib

import numpy as np

from common.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seeds = seed_everything(args.seed)
    numbers = {
        "world_gen (numpy)": seeds.rng("world_gen").random(5),
        "ants (numpy)": seeds.rng("ants").integers(0, 100, 5),
        "global numpy": np.random.rand(3),
    }
    try:
        import torch
        numbers["torch randn"] = torch.randn(3).numpy()
    except ImportError:
        pass

    for name, values in numbers.items():
        print(f"{name:20s} {values}")
    # Empreinte unique de tout ce qui précède : pratique pour comparer deux exécutions.
    blob = b"".join(np.asarray(v).tobytes() for v in numbers.values())
    print("empreinte", hashlib.sha256(blob).hexdigest()[:16])


if __name__ == "__main__":
    main()
