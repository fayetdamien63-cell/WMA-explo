"""Entraîne le world model sur les trajectoires collectées (scripts/collect_data.py).

    python scripts/train_world_model.py --data runs/step5/data --out runs/step5/wm_residual
    python scripts/train_world_model.py --data runs/step5/data --out runs/step5/wm_vector --vector-latent   # variante « latent seul »
    python scripts/train_world_model.py ... --iters 3000 3000 3000 --batch-size 64 --device cuda            # gros entraînement (GPU)

Écrit model.pt (modèle + configuration), train_log.jsonl (perte par itération, validation tous les `val_every`).
"""

import argparse
import dataclasses
from pathlib import Path

import yaml

from common.config import load_config
from models.data import TrajectoryDataset, load_split
from models.train import WMTrainConfig, train_world_model
from models.worldmodel import WMConfig

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "runs" / "step5" / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "step5" / "wm_residual")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "worldmodel.yaml")
    ap.add_argument("--vector-latent", action="store_true", help="variante sans résidu : carte décodée depuis le seul vecteur latent")
    ap.add_argument("--iters", type=int, nargs="+", default=None, help="itérations par phase (3 valeurs)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--weight-floor", type=float, default=None, help="plancher des poids de perte (défaut 5e-6 ; 7.7e-4 = premier essai)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    wm_cfg = load_config_dict(WMConfig, raw.get("model", {}))
    tr_cfg = load_config_dict(WMTrainConfig, raw.get("train", {}))
    if args.vector_latent:
        wm_cfg.residual = False
    for k, v in (("phase_iters", tuple(args.iters) if args.iters else None), ("batch_size", args.batch_size),
                 ("device", args.device), ("seed", args.seed), ("weight_floor", args.weight_floor)):
        if v is not None:
            setattr(tr_cfg, k, v)
    train = TrajectoryDataset(load_split(args.data / "train.npz"))
    val = TrajectoryDataset(load_split(args.data / "val.npz"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.yaml").write_text(yaml.safe_dump({"model": dataclasses.asdict(wm_cfg), "train": dataclasses.asdict(tr_cfg)}), encoding="utf-8")
    train_world_model(train, val, wm_cfg, tr_cfg, args.out)


def load_config_dict(cls, data: dict):
    """Remplit une dataclass depuis un dictionnaire (les clés inconnues sont une erreur)."""
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"clés inconnues pour {cls.__name__} : {sorted(unknown)}")
    return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in data.items()})


if __name__ == "__main__":
    main()
