"""Choix du matériel de calcul pour l'apprentissage (le simulateur reste sur CPU)."""

from __future__ import annotations

import os

import torch

ENV_VAR = "ANTWORLD_DEVICE"  # permet de forcer un device sans toucher au code


def get_device(preference: str = "auto") -> torch.device:
    """Renvoie le meilleur device disponible : CUDA, puis MPS, puis CPU.

    `preference` : "auto" (défaut), "cuda", "mps" ou "cpu". La variable d'environnement
    ANTWORLD_DEVICE l'emporte si elle est définie. Demander explicitement un device
    indisponible lève une erreur plutôt que de retomber silencieusement sur le CPU
    (un entraînement lancé sur le CPU par erreur coûterait des heures).
    """
    preference = os.environ.get(ENV_VAR, preference).lower()
    if preference not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Device inconnu : {preference!r} (attendu : auto, cuda, mps, cpu)")

    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available()

    if preference == "auto":
        return torch.device("cuda" if cuda_ok else "mps" if mps_ok else "cpu")
    if preference == "cuda" and not cuda_ok:
        raise RuntimeError("Device 'cuda' demandé mais CUDA n'est pas disponible.")
    if preference == "mps" and not mps_ok:
        raise RuntimeError("Device 'mps' demandé mais MPS n'est pas disponible.")
    return torch.device(preference)
