"""Entraînement du world model : 3 phases (1 pas observé, puis boucle ouverte de plus en plus longue).

Pourquoi des phases ? Prédire à 1 pas (avec la vraie observation à chaque pas) est facile et donne une dynamique de base.
Mais un planificateur utilisera le modèle en BOUCLE OUVERTE (sans observation) : les erreurs s'accumulent, et le modèle doit
être entraîné sur ses propres prédictions. On allonge donc progressivement l'horizon de boucle ouverte (8 puis 24 pas).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from common.device import get_device
from common.seeding import SeedManager, seed_everything
from models.data import TrajectoryDataset
from models.worldmodel import WMConfig, WorldModel, count_parameters


@dataclass
class WMTrainConfig:
    batch_size: int = 16
    phase_iters: tuple[int, ...] = (1500, 1500, 1500)      # itérations par phase
    phase_context: tuple[int, ...] = (16, 8, 8)             # pas OBSERVÉS (prédictions à 1 pas)
    phase_horizon: tuple[int, ...] = (0, 8, 24)             # pas en BOUCLE OUVERTE
    phase_lr: tuple[float, ...] = (1e-3, 5e-4, 3e-4)
    grad_clip: float = 1.0
    loss_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 0.1)  # carte, état, récompense, fin d'épisode
    detach_base: bool = False
    val_every: int = 250
    val_batches: int = 4
    log_every: int = 50
    seed: int = 0
    device: str = "auto"
    torch_threads: int = 4


def to_tensors(batch: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


def fit_norm(model: WorldModel, train: TrajectoryDataset) -> None:
    """Fixe les normalisations de perte à partir des données d'entraînement (variance de changement à 1 pas)."""
    st = train.persistence_stats()
    r = np.concatenate([train.a["rewards"][e, : train.lengths[e]] for e in range(len(train))])
    model.set_norm(st["map_var"], st["state_var"], float(r.mean()), float(r.std()))


@torch.no_grad()
def validate(model: WorldModel, val: TrajectoryDataset, ctx: int, horizon: int, cfg: WMTrainConfig, device) -> dict[str, float]:
    """Perte sur des fenêtres de validation FIXES (même seed à chaque appel : comparable d'un point à l'autre)."""
    model.eval()
    rng = np.random.default_rng(12345)
    infos = []
    for _ in range(cfg.val_batches):
        b = to_tensors(val.sample_batch(cfg.batch_size, ctx + horizon, rng), device)
        _, info = model.loss(b, ctx, cfg.loss_weights, cfg.detach_base)
        infos.append(info)
    model.train()
    return {k: float(np.mean([i[k] for i in infos])) for k in infos[0]}


def save_checkpoint(path: Path, model: WorldModel, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": asdict(model.cfg), "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_checkpoint(path: str | Path, device="cpu") -> WorldModel:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = WMConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items()})
    model = WorldModel(cfg)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def train_world_model(train: TrajectoryDataset, val: TrajectoryDataset, wm_cfg: WMConfig, cfg: WMTrainConfig,
                      out_dir: str | Path, log=print) -> WorldModel:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.torch_threads)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    model = WorldModel(wm_cfg).to(device)
    fit_norm(model, train)
    rng = SeedManager(cfg.seed).rng("wm/batches")
    log(f"World model {'résiduel' if wm_cfg.residual else 'latent seul'} : {count_parameters(model):,} paramètres, device {device}")
    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")
    t0, it_global = time.perf_counter(), 0
    for phase, (iters, ctx, hor, lr) in enumerate(zip(cfg.phase_iters, cfg.phase_context, cfg.phase_horizon, cfg.phase_lr), 1):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        log(f"--- phase {phase} : {iters} itérations, contexte observé {ctx}, boucle ouverte {hor}, lr {lr} ---")
        for it in range(1, iters + 1):
            it_global += 1
            batch = to_tensors(train.sample_batch(cfg.batch_size, ctx + hor, rng), device)
            loss, info = model.loss(batch, ctx, cfg.loss_weights, cfg.detach_base)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            rec = {"phase": phase, "iter": it_global, "wall_s": time.perf_counter() - t0, "grad_norm": float(gn), **info}
            if it % cfg.val_every == 0 or it == iters:
                rec.update({f"val_{k}": v for k, v in validate(model, val, ctx, hor, cfg, device).items()})
                save_checkpoint(out_dir / "model.pt", model, {"iter": it_global, "phase": phase})
                log(f"[phase {phase} it {it:5d}] perte {info['loss']:.4f} (val {rec['val_loss']:.4f}) carte {info['map']:.3f} "
                    f"état {info['state']:.3f} récompense {info['reward']:.3f}  ({rec['wall_s']:.0f} s)")
            elif it % cfg.log_every == 0:
                log(f"[phase {phase} it {it:5d}] perte {info['loss']:.4f}")
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
    save_checkpoint(out_dir / "model.pt", model, {"iter": it_global, "phase": len(cfg.phase_iters), "train_wall_s": time.perf_counter() - t0})
    return model
