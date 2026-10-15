"""Joue une partie avec des actions aléatoires et produit un GIF + une courbe de synthèse.

    python scripts/play_random.py --seed 3
    python scripts/play_random.py --seed 3 --policy food_only      # autres politiques de contrôle

Sorties dans runs/step1/ (ignoré par git).
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from antworld import AntWorld, AntWorldConfig  # noqa: E402
from antworld.render import render_rgb  # noqa: E402
from common.config import load_config  # noqa: E402
from common.seeding import SeedManager  # noqa: E402

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "antworld.yaml"


def policy_action(name: str, world: AntWorld, rng: np.random.Generator) -> np.ndarray:
    if name == "random":
        return rng.integers(0, world.action_nvec)
    if name == "noop":          # tout le monde cueille de la nourriture, rien d'autre
        return np.array([0, 0, 0, 0, 0])
    if name == "food_only":     # 10 % d'éclaireuses toujours en libre exploration
        return np.array([1, 0, 0, 0, 0])
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--policy", default="random")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/step1"))
    ap.add_argument("--every", type=int, default=2, help="une image de GIF tous les n pas macro")
    args = ap.parse_args()

    cfg = load_config(AntWorldConfig, CONFIG)
    world = AntWorld.from_seed(cfg, args.seed)
    rng = SeedManager(args.seed).rng(f"policy/{args.policy}")
    steps = args.steps or cfg.sim.max_steps
    args.out.mkdir(parents=True, exist_ok=True)

    frames, history = [Image.fromarray(render_rgb(world))], [world.stats()]
    sheet_steps = {0, 10, 30, 80, 160, steps - 1}   # images de la planche-contact
    sheet = {0: frames[0]}
    for t in range(steps):
        world.step(policy_action(args.policy, world, rng))
        history.append(world.stats())
        if (t + 1) % args.every == 0 or (t + 1) in sheet_steps or world.done:
            img = Image.fromarray(render_rgb(world))
            if (t + 1) % args.every == 0:
                frames.append(img)
            if (t + 1) in sheet_steps:
                sheet[t + 1] = img
        if world.done:
            break
    sheet[world.macro_step] = Image.fromarray(render_rgb(world))
    tag = f"{args.policy}_seed{args.seed}"
    frames[0].save(args.out / f"{tag}.gif", save_all=True, append_images=frames[1:], duration=80, loop=0)

    keys = ["ants", "food", "ore", "known"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    t = [h["step"] for h in history]
    axes[0].plot(t, [h["ants"] for h in history], label="fourmis")
    axes[0].plot(t, [h["pop_cap"] for h in history], "--", label="capacité")
    axes[0].set_title("Population"); axes[0].legend()
    axes[1].plot(t, [h["food"] for h in history], label="nourriture (stock)")
    axes[1].plot(t, [h["ore"] for h in history], label="minerai (stock)")
    axes[1].set_title("Stocks"); axes[1].legend()
    axes[2].plot(t, [100 * h["known"] for h in history])
    axes[2].set_title("% de la carte explorée")
    for ax in axes:
        ax.set_xlabel("pas macro"); ax.grid(alpha=0.3)
    fig.suptitle(f"politique={args.policy}  seed={args.seed}  fin au pas {history[-1]['step']}"
                 f" ({'colonie morte' if not world.alive else 'survie'})")
    fig.tight_layout(); fig.savefig(args.out / f"{tag}.png", dpi=110)
    Image.fromarray(render_rgb(world)).save(args.out / f"{tag}_final.png")
    # Planche-contact : quelques instants de la partie côte à côte (3 colonnes).
    shots = [sheet[k] for k in sorted(sheet)][:6]
    w, h = shots[0].size
    cols = 3
    board = Image.new("RGB", (cols * w, ((len(shots) + cols - 1) // cols) * h))
    for i, im in enumerate(shots):
        board.paste(im, ((i % cols) * w, (i // cols) * h))
    board.save(args.out / f"{tag}_sheet.png")
    print(f"{tag}: pas={history[-1]['step']} fourmis={history[-1]['ants']} nourriture={history[-1]['food']:.1f} "
          f"minerai={history[-1]['ore']:.1f} nurserie={history[-1]['nursery']} grenier={history[-1]['granary']} "
          f"exploré={100 * history[-1]['known']:.0f}%")
    print("sorties :", args.out / f"{tag}.gif", args.out / f"{tag}.png")


if __name__ == "__main__":
    main()
