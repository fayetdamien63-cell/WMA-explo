"""Évalue un world model : erreur selon l'horizon (1, 5, 10, 25, 50 pas) contre « rien ne change », mondes vus et jamais vus,
verdict de la porte, courbes, et comparaison côte à côte du rêve et de la réalité.

    python scripts/eval_world_model.py --model runs/step5/wm_residual --data runs/step5/data
    python scripts/eval_world_model.py --model runs/step5/wm_residual --compare runs/step5/wm_vector   # tableau des deux variantes
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from antworld.render import render_obs_map  # noqa: E402
from eval.wm_metrics import CHANNELS, CONTEXT, HORIZONS, evaluate_model, valid_windows  # noqa: E402
from models.data import TrajectoryDataset, load_split  # noqa: E402
from models.train import load_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Critères de la porte, fixés AVANT de voir les résultats (mondes JAMAIS VUS, horizon 10 pas) :
GATE = {"map_skill_macro_h10": 0.25, "state_skill_macro_h10": 0.25, "reward_r2_h10": 0.5, "map_skill_macro_h1": 0.5,
        "max_seen_unseen_gap_h10": 0.10}


def gate_verdict(seen: dict, unseen: dict) -> dict:
    g = lambda r, h, k: r["horizons"][str(h)][k]["value"]
    checks = {
        "compétence carte à 10 pas (jamais vus) ≥ %.2f" % GATE["map_skill_macro_h10"]: (g(unseen, 10, "map_skill_macro"), g(unseen, 10, "map_skill_macro") >= GATE["map_skill_macro_h10"]),
        "compétence état à 10 pas (jamais vus) ≥ %.2f" % GATE["state_skill_macro_h10"]: (g(unseen, 10, "state_skill_macro"), g(unseen, 10, "state_skill_macro") >= GATE["state_skill_macro_h10"]),
        "R² de la récompense cumulée à 10 pas (jamais vus) ≥ %.2f" % GATE["reward_r2_h10"]: (g(unseen, 10, "reward_r2"), g(unseen, 10, "reward_r2") >= GATE["reward_r2_h10"]),
        "compétence carte à 1 pas (jamais vus) ≥ %.2f" % GATE["map_skill_macro_h1"]: (g(unseen, 1, "map_skill_macro"), g(unseen, 1, "map_skill_macro") >= GATE["map_skill_macro_h1"]),
        "écart vus − jamais vus (carte, 10 pas) ≤ %.2f" % GATE["max_seen_unseen_gap_h10"]: (g(seen, 10, "map_skill_macro") - g(unseen, 10, "map_skill_macro"),
            g(seen, 10, "map_skill_macro") - g(unseen, 10, "map_skill_macro") <= GATE["max_seen_unseen_gap_h10"]),
    }
    return {"checks": {k: {"value": v, "ok": bool(ok)} for k, (v, ok) in checks.items()}, "passed": all(ok for _, ok in checks.values())}


def plot_curves(results: dict[str, dict[str, dict]], out: Path) -> None:
    """results[variante][jeu] -> résumé. Courbes de compétence (carte, état) et R² de la récompense selon l'horizon."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for (name, sets), color in zip(results.items(), colors):
        for split, ls in (("eval_seen", "--"), ("eval_unseen", "-")):
            if split not in sets:
                continue
            for ax, key, title in zip(axes, ("map_skill_macro", "state_skill_macro", "reward_r2"),
                                      ("Compétence carte (macro par canal)", "Compétence état (macro)", "R² récompense cumulée")):
                v = [sets[split]["horizons"][str(h)][key] for h in HORIZONS]
                y = [x["value"] for x in v]
                ax.plot(HORIZONS, y, ls, color=color, marker="o", label=f"{name} — {'jamais vus' if split == 'eval_unseen' else 'mondes vus'}")
                if split == "eval_unseen":
                    ax.fill_between(HORIZONS, [x["ci_low"] for x in v], [x["ci_high"] for x in v], color=color, alpha=0.15)
                ax.set_title(title)
    for ax in axes:
        ax.axhline(0, color="gray", lw=1); ax.set_xscale("log"); ax.set_xticks(HORIZONS, [str(h) for h in HORIZONS])
        ax.set_xlabel("horizon (pas macro, boucle ouverte)"); ax.grid(alpha=0.3)
    axes[0].axhline(GATE["map_skill_macro_h10"], color="black", ls=":", lw=1)
    axes[0].set_ylabel("1 − erreur modèle / erreur « rien ne change »"); axes[0].legend(fontsize=8)
    fig.suptitle("0 = pas mieux que « rien ne change » ; tirets : mondes vus, trait plein : mondes jamais vus (bande = IC 95 % par épisode)")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


@torch.no_grad()
def dream_vs_reality(model, ds: TrajectoryDataset, episode: int, start: int, out: Path, horizons=(1, 5, 10, 25, 50)) -> None:
    """Planche : réalité / rêve / erreur, à plusieurs horizons de boucle ouverte, depuis la même observation de départ."""
    H = max(horizons)
    w = ds.window(episode, start - CONTEXT, CONTEXT + H)
    b = {k: torch.as_tensor(v)[None] for k, v in w.items()}
    out_ = model.rollout(b["maps"], b["states"], b["actions"], n_forced=CONTEXT)
    cols = [("depart", b["maps"][0, CONTEXT].numpy(), b["maps"][0, CONTEXT].numpy())]
    for h in horizons:
        cols.append((f"+{h} pas", b["maps"][0, CONTEXT + h].numpy(), out_["map"][0, CONTEXT + h - 1].numpy()))
    scale, label_h = 6, 16
    rows = ("realite", "reve (boucle ouverte)", "erreur (cyan = nulle, rouge = forte)")        # ASCII : la police par défaut de PIL n'a pas les accents
    img = Image.new("RGB", (len(cols) * 32 * scale, label_h + 3 * 32 * scale + 3 * label_h), (20, 20, 24))
    d = ImageDraw.Draw(img)
    for ci, (name, real, dream) in enumerate(cols):
        x0 = ci * 32 * scale
        d.text((x0 + 4, 2), name, fill=(240, 240, 240))
        err = np.abs(dream - real).sum(axis=0)
        heat = np.stack([np.clip(err * 255, 0, 255)] * 1 + [np.clip(255 - err * 255, 0, 255)] * 2, axis=-1).astype(np.uint8)
        for ri, panel in enumerate((render_obs_map(real, scale), render_obs_map(dream, scale), np.kron(heat, np.ones((scale, scale, 1), np.uint8)))):
            y0 = label_h + ri * (32 * scale + label_h)
            img.paste(Image.fromarray(panel), (x0, y0 + label_h))
            if ci == 0:
                d.text((x0 + 4, y0 + 2), rows[ri], fill=(255, 215, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="dossier contenant model.pt")
    ap.add_argument("--data", type=Path, default=ROOT / "runs" / "step5" / "data")
    ap.add_argument("--compare", type=Path, nargs="*", default=[], help="autres dossiers de modèles à tracer sur les mêmes courbes")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = args.out or args.model / "eval"
    out.mkdir(parents=True, exist_ok=True)
    sets = {s: TrajectoryDataset(load_split(args.data / f"{s}.npz")) for s in ("eval_seen", "eval_unseen")}

    results = {}
    for folder in [args.model, *args.compare]:
        model = load_checkpoint(folder / "model.pt")
        label = "résiduel" if model.cfg.residual else "latent seul"
        results[label] = {s: evaluate_model(model, ds) for s, ds in sets.items()}
        (folder / "eval_results.json").write_text(json.dumps(results[label], indent=2), encoding="utf-8")
        print(f"\n=== {label} ({folder}) ===")
        for s in sets:
            r = results[label][s]
            print(f"[{s}] {r['n_windows']} fenêtres, {r['n_episodes']} épisodes")
            print("  horizon | carte : compétence [IC 95 %] (MSE modèle / « rien ne change ») | état : compétence | récompense cumulée R² (modèle / persistance / moyenne)")
            for h in HORIZONS:
                x = r["horizons"][str(h)]
                m = x["map_skill_macro"]
                print(f"  {h:7d} | {m['value']:+.2f} [{m['ci_low']:+.2f}, {m['ci_high']:+.2f}] ({x['map_mse_model']['value']:.5f} / {x['map_mse_persist']['value']:.5f}) | "
                      f"{x['state_skill_macro']['value']:+.2f} | {x['reward_r2']['value']:+.2f} / {x['reward_r2_persist']['value']:+.2f} / {x['reward_r2_mean']['value']:+.2f}")
            x10 = r["horizons"]["10"]["map_skill_by_channel"]
            print("  compétence par canal à 10 pas : " + ", ".join(f"{c} {x10[c]:+.2f}" for c in CHANNELS))
        verdict = gate_verdict(results[label]["eval_seen"], results[label]["eval_unseen"])
        results[label]["gate"] = verdict
        print("  PORTE :")
        for k, c in verdict["checks"].items():
            print(f"    {'OK ' if c['ok'] else 'NON'} {k} : {c['value']:+.3f}")
        print(f"  -> {'ATTEINTE' if verdict['passed'] else 'NON ATTEINTE'}")
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_curves({k: {s: v for s, v in d.items() if s != "gate"} for k, d in results.items()}, out / "horizon_curves.png")

    # Planche rêve / réalité : un épisode de chaque politique, sur les mondes jamais vus
    model = load_checkpoint(args.model / "model.pt")
    ds = sets["eval_unseen"]
    wins = valid_windows(ds, starts=(120,))
    for e, s in wins[:: max(len(wins) // 4, 1)][:4]:
        dream_vs_reality(model, ds, e, s, out / f"dream_vs_reality_ep{e}_t{s}.png")
    print("écrit", out)


if __name__ == "__main__":
    main()
