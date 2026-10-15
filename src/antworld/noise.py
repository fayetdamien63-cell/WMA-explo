"""Bruit de Perlin (bruit à gradients) en NumPy pur, pour fabriquer des zones lisses.

Principe : on tire un vecteur gradient aléatoire à chaque noeud d'un réseau grossier (un noeud
tous les `scale` cases). Pour une case, on prend les 4 noeuds qui l'entourent, on mesure à quel
point la case est « dans le sens » de chaque gradient (produit scalaire), puis on interpole avec une
courbe douce. Résultat : des collines continues, de taille caractéristique ~`scale` cases.
"""

from __future__ import annotations

import numpy as np


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)  # 6t^5 - 15t^4 + 10t^3 : pente nulle aux bords


def perlin(size: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Bruit de Perlin (size, size), valeurs ≈ dans [-0.7, 0.7]."""
    n_nodes = int(np.ceil(size / scale)) + 2
    angles = rng.random((n_nodes, n_nodes)) * 2.0 * np.pi
    gx, gy = np.cos(angles), np.sin(angles)

    coords = np.arange(size) / scale
    x, y = np.meshgrid(coords, coords)              # x : colonne, y : ligne
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)

    def corner(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        return gx[iy, ix] * (x - ix) + gy[iy, ix] * (y - iy)

    u, v = _fade(x - x0), _fade(y - y0)
    top = corner(x0, y0) * (1 - u) + corner(x0 + 1, y0) * u
    bottom = corner(x0, y0 + 1) * (1 - u) + corner(x0 + 1, y0 + 1) * u
    return top * (1 - v) + bottom * v


def fractal_noise(size: int, scale: float, octaves: int, rng: np.random.Generator,
                  persistence: float = 0.5) -> np.ndarray:
    """Somme d'octaves de Perlin (détails de plus en plus fins), normalisée dans [0, 1]."""
    total = np.zeros((size, size))
    amplitude = 1.0
    for octave in range(octaves):
        total += amplitude * perlin(size, max(scale / 2**octave, 1.5), rng)
        amplitude *= persistence
    lo, hi = total.min(), total.max()
    return (total - lo) / (hi - lo)
