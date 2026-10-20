"""Évaluation du world model : erreur de prédiction en BOUCLE OUVERTE selon l'horizon, contre la baseline « rien ne change ».

Protocole (identique pour tous les jeux et toutes les variantes) :
  - fenêtres : pour chaque épisode et chaque instant de départ s ∈ STARTS (assez loin pour que s + 50 reste dans l'épisode),
    le modèle observe les `context` (=10) pas qui précèdent s puis o_s, PUIS n'utilise plus aucune observation : à partir des
    seules actions a_s, a_{s+1}, ... il prédit o_{s+k}, k = 1..50.
  - baseline « rien ne change » : prédire o_{s+k} = o_s (carte et état) ; pour la récompense : r̂ = dernière récompense observée.
  - compétence (« skill ») = 1 − erreur du modèle / erreur de la baseline : 1 = parfait, 0 = pas mieux que « rien ne change »,
    < 0 = pire. Calculée par canal de la carte (macro-moyenne sur les canaux dynamiques) et pour l'ensemble.
  - récompense : on compare la récompense CUMULÉE sur k pas (ce que verra un planificateur) ; R² sur les fenêtres.
  - intervalles de confiance à 95 % : bootstrap sur les ÉPISODES (les fenêtres d'un même épisode ne sont pas indépendantes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from models.data import TrajectoryDataset
from models.worldmodel import WorldModel

HORIZONS = (1, 5, 10, 25, 50)
STARTS = (10, 60, 120, 200, 300)
CONTEXT = 10
CHANNELS = ("visible", "known", "rock", "food", "ore", "nest", "scouts", "others")


@dataclass
class WindowStats:
    """Sommes d'erreurs quadratiques par fenêtre w, horizon h, canal/dimension (tout est additif : facile à ré-agréger)."""

    episode: np.ndarray              # (W,) indice d'épisode de chaque fenêtre
    horizons: tuple[int, ...]
    sse_map_model: np.ndarray        # (W, H, 8)  somme sur les cases
    sse_map_persist: np.ndarray
    sse_state_model: np.ndarray      # (W, H, 24)
    sse_state_persist: np.ndarray
    rew_true: np.ndarray             # (W, H) récompense cumulée sur k pas
    rew_model: np.ndarray
    rew_persist: np.ndarray          # k × dernière récompense observée
    rew_mean: np.ndarray             # k × moyenne des récompenses d'entraînement
    done_true: np.ndarray            # (W, H) la colonie est morte dans les k pas
    done_prob: np.ndarray            # (W, H) probabilité maximale prédite de mort dans les k pas
    n_cells: int


def valid_windows(ds: TrajectoryDataset, starts=STARTS, context=CONTEXT, horizon=max(HORIZONS)) -> list[tuple[int, int]]:
    return [(e, s) for e in range(len(ds)) for s in starts if s - context >= 0 and s + horizon <= ds.lengths[e]]


@torch.no_grad()
def window_stats(model: WorldModel, ds: TrajectoryDataset, horizons=HORIZONS, starts=STARTS, context=CONTEXT,
                 batch: int = 25, device="cpu") -> WindowStats:
    model.eval()
    H = max(horizons)
    wins = valid_windows(ds, starts, context, H)
    if not wins:
        raise ValueError("aucune fenêtre valide (épisodes trop courts)")
    keep = [h - 1 for h in horizons]
    acc = {k: [] for k in ("sm", "sp", "ssm", "ssp", "rt", "rm", "rp", "rmean", "dt", "dp")}
    cells = None
    for i in range(0, len(wins), batch):
        chunk = wins[i: i + batch]
        data = [ds.window(e, s - context, context + H) for e, s in chunk]
        b = {k: torch.as_tensor(np.stack([d[k] for d in data]), device=device) for k in data[0]}
        out = model.rollout(b["maps"], b["states"], b["actions"], n_forced=context)
        cells = b["maps"].shape[-1] * b["maps"].shape[-2]
        pm, ps = out["map"][:, context:], out["state"][:, context:]               # prédictions de o_{s+1..s+H}
        tm, ts = b["maps"][:, context + 1:], b["states"][:, context + 1:]
        base_m, base_s = b["maps"][:, context: context + 1], b["states"][:, context: context + 1]      # o_s
        sq = lambda a: a.pow(2)
        acc["sm"].append(sq(pm - tm).sum(dim=(3, 4))[:, keep].cpu().numpy())
        acc["sp"].append(sq(base_m - tm).sum(dim=(3, 4))[:, keep].cpu().numpy())
        acc["ssm"].append(sq(ps - ts)[:, keep].cpu().numpy())
        acc["ssp"].append(sq(base_s - ts)[:, keep].cpu().numpy())
        r_true, r_pred = b["rewards"][:, context:], out["reward"][:, context:]
        last_r = b["rewards"][:, context - 1: context]                              # dernière récompense observée (pas s-1)
        cum = lambda r: r.cumsum(dim=1)[:, keep].cpu().numpy()
        acc["rt"].append(cum(r_true)); acc["rm"].append(cum(r_pred))
        ks = torch.as_tensor(horizons, dtype=torch.float32)
        acc["rp"].append((last_r * ks[None]).cpu().numpy())
        acc["rmean"].append((model.reward_mean * ks[None].expand(len(chunk), -1)).cpu().numpy())
        d_true = b["dones"][:, context:].float().cummax(dim=1).values[:, keep]
        d_prob = torch.sigmoid(out["done_logit"][:, context:]).cummax(dim=1).values[:, keep]
        acc["dt"].append(d_true.cpu().numpy()); acc["dp"].append(d_prob.cpu().numpy())
    cat = {k: np.concatenate(v) for k, v in acc.items()}
    return WindowStats(np.array([e for e, _ in wins]), tuple(horizons), cat["sm"], cat["sp"], cat["ssm"], cat["ssp"],
                       cat["rt"], cat["rm"], cat["rp"], cat["rmean"], cat["dt"], cat["dp"], cells)


def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    var = ((true - true.mean()) ** 2).sum()
    return float(1.0 - ((true - pred) ** 2).sum() / var) if var > 1e-12 else float("nan")


def summarize(stats: WindowStats, n_boot: int = 300, seed: int = 0) -> dict:
    """Agrège par horizon ; IC à 95 % par bootstrap d'épisodes. Renvoie un dictionnaire JSON-compatible."""
    eps = np.unique(stats.episode)
    rng = np.random.default_rng(seed)
    idx_by_ep = {e: np.flatnonzero(stats.episode == e) for e in eps}

    def metrics(idx: np.ndarray, h: int) -> dict[str, float]:
        sm, sp = stats.sse_map_model[idx, h], stats.sse_map_persist[idx, h]
        ssm, ssp = stats.sse_state_model[idx, h], stats.sse_state_persist[idx, h]
        map_macro, map_total = _skill_idx(sm, sp, 1e-6 * stats.n_cells)
        st_macro, st_total = _skill_idx(ssm, ssp, 1e-9)
        return {"map_skill_macro": map_macro, "map_skill_total": map_total, "state_skill_macro": st_macro, "state_skill_total": st_total,
                "reward_r2": _r2(stats.rew_true[idx, h], stats.rew_model[idx, h]),
                "reward_r2_persist": _r2(stats.rew_true[idx, h], stats.rew_persist[idx, h]),
                "reward_r2_mean": _r2(stats.rew_true[idx, h], stats.rew_mean[idx, h]),
                "reward_mae": float(np.abs(stats.rew_true[idx, h] - stats.rew_model[idx, h]).mean()),
                "reward_mae_persist": float(np.abs(stats.rew_true[idx, h] - stats.rew_persist[idx, h]).mean()),
                "map_mse_model": float(sm.sum() / (len(idx) * stats.n_cells * sm.shape[1])),
                "map_mse_persist": float(sp.sum() / (len(idx) * stats.n_cells * sp.shape[1])),
                "state_mse_model": float(ssm.mean()), "state_mse_persist": float(ssp.mean())}

    out = {"n_windows": int(len(stats.episode)), "n_episodes": int(len(eps)), "horizons": {}}
    all_idx = np.arange(len(stats.episode))
    for j, h in enumerate(stats.horizons):
        point = metrics(all_idx, j)
        boots = {k: [] for k in point}
        for _ in range(n_boot):
            pick = rng.choice(eps, size=len(eps))
            idx = np.concatenate([idx_by_ep[e] for e in pick])
            for k, v in metrics(idx, j).items():
                boots[k].append(v)
        res = {}
        for k, v in point.items():
            lo, hi = np.nanquantile(boots[k], [0.025, 0.975]) if not np.isnan(boots[k]).all() else (float("nan"),) * 2
            res[k] = {"value": v, "ci_low": float(lo), "ci_high": float(hi)}
        # canaux : compétence par canal de la carte
        per_ch = 1.0 - stats.sse_map_model[:, j].sum(0) / np.maximum(stats.sse_map_persist[:, j].sum(0), 1e-12)
        res["map_skill_by_channel"] = {c: float(v) for c, v in zip(CHANNELS, per_ch)}
        res["done"] = {"windows_with_death": int(stats.done_true[:, j].sum()), "auc": _auc(stats.done_true[:, j], stats.done_prob[:, j])}
        out["horizons"][str(h)] = res
    return out


def _skill_idx(sse_model: np.ndarray, sse_base: np.ndarray, min_base: float) -> tuple[float, float]:
    """Compétence macro (par colonne dynamique) et globale à partir de (W, C)."""
    m, b = sse_model.sum(axis=0), sse_base.sum(axis=0)
    dyn = b > min_base * len(sse_model)
    macro = float(np.mean(1.0 - m[dyn] / b[dyn])) if dyn.any() else float("nan")
    return macro, float(1.0 - m.sum() / max(b.sum(), 1e-12))


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC (probabilité qu'un positif ait un score supérieur à un négatif) ; NaN s'il manque une classe."""
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def evaluate_model(model: WorldModel, ds: TrajectoryDataset, device="cpu", n_boot: int = 300, **kw) -> dict:
    """Statistiques de fenêtres (`kw` : horizons, starts, context, batch) puis agrégation avec `n_boot` rééchantillonnages."""
    return summarize(window_stats(model, ds, device=device, **kw), n_boot=n_boot)
