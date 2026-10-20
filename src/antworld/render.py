"""Rendu visuel simple : image RGB construite directement avec NumPy (pas de matplotlib ici),
plus un petit bandeau de texte avec Pillow. Utilisable plus tard par `render_mode="rgb_array"`.

Lecture de l'image :
  - sable = case libre, brun foncé = roche, étoile dorée = nid
  - vert = nourriture (plus clair = plus pleine), orange = minerai
  - cases assombries = brouillard (jamais vues) : on les montre quand même, pour que l'humain voie
    ce que la colonie ignore
  - points : cueilleuses de nourriture (jaune), de minerai (cyan), éclaireuses (rouge),
    bâtisseuses (violet) ; un point blanc au centre = la fourmi porte une charge
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from antworld import indices as ix
from antworld.world import AntWorld

SAND = np.array([222, 205, 160], np.float64)
ROCK = np.array([70, 60, 55], np.float64)
FOOD = np.array([60, 170, 60], np.float64)
ORE = np.array([235, 130, 30], np.float64)
ROLE_COLORS = {ix.ROLE_FOOD: (250, 230, 40), ix.ROLE_ORE: (30, 220, 230),
               ix.ROLE_SCOUT: (230, 30, 30), ix.ROLE_BUILD: (170, 60, 220)}


def render_rgb(world: AntWorld, scale: int = 14, hud: bool = True) -> np.ndarray:
    """Image (H, W, 3) uint8 de l'état courant."""
    s = world.size
    lay = world.layout
    img = np.where(lay.terrain.reshape(s, s, 1) == 1, ROCK, SAND).astype(np.float64)

    # Ressources (vérité terrain), intensité selon le remplissage.
    food = world.grid[ix.G_AMOUNT + ix.R_FOOD].reshape(s, s)
    cap = np.maximum(lay.food_cap.reshape(s, s), 1e-9)
    fill = np.clip(food / cap, 0, 1)[..., None]
    has_food = (lay.food_cap.reshape(s, s) > 0)[..., None]
    img = np.where(has_food, SAND * (1 - 0.25 - 0.65 * fill) + FOOD * (0.25 + 0.65 * fill), img)
    ore = world.grid[ix.G_AMOUNT + ix.R_ORE].reshape(s, s)
    ore_fill = np.clip(ore / max(world.cfg.world.ore_max, 1e-9), 0, 1)[..., None]
    img = np.where((ore > 0)[..., None], SAND * (1 - 0.4 - 0.6 * ore_fill) + ORE * (0.4 + 0.6 * ore_fill), img)

    # Brouillard : on assombrit les cases jamais vues.
    unknown = (world.grid[ix.G_KNOWN].reshape(s, s) < 0.5)[..., None]
    img = np.where(unknown, img * 0.35, img)

    img = np.kron(img, np.ones((scale, scale, 1)))
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Nid (carré doré) et fourmis (petits carrés).
    def paint(cell: int, color, frac: float, offset: float = 0.0) -> None:
        y, x = divmod(int(cell), s)
        half = max(int(scale * frac / 2), 1)
        cy = int(y * scale + scale / 2 + offset)
        cx = int(x * scale + scale / 2 + offset)
        img[cy - half: cy + half, cx - half: cx + half] = color

    paint(lay.nest, (255, 215, 0), 0.9)
    ants = world.ants[: world.n_ants]
    for i in range(len(ants)):
        paint(ants[i, ix.A_CELL], ROLE_COLORS[int(ants[i, ix.A_ROLE])], 0.45)
        if ants[i, ix.A_LOAD] > 0:
            paint(ants[i, ix.A_CELL], (255, 255, 255), 0.15)

    if not hud:
        return img
    bar = Image.new("RGB", (img.shape[1], 34), (25, 25, 30))
    st = world.stats()
    d = ImageDraw.Draw(bar)
    d.text((4, 3), f"pas {st['step']:3d}  fourmis {st['ants']:3d}/{st['pop_cap']}  "
                   f"nourriture {st['food']:5.1f}/{world.storage_cap:.0f}  minerai {st['ore']:4.1f}", fill=(235, 235, 235))
    counts = world.role_counts()
    d.text((4, 18), f"cueill.N {counts[0]} M {counts[1]}  eclair. {counts[2]}  bat. {counts[3]}  "
                    f"nurserie {st['nursery']} grenier {st['granary']}  vu {100 * st['known']:.0f}%", fill=(200, 200, 120))
    return np.vstack([img, np.asarray(bar)])


def render_obs_map(m: np.ndarray, scale: int = 8) -> np.ndarray:
    """Image RGB (32·scale)² d'une CARTE D'OBSERVATION (8 canaux dans [0, 1], voir observation.py).

    Sert à comparer rêve et réalité : accepte aussi des valeurs prédites (probabilités), qui apparaissent « floues ».
    Lecture : brouillard = gris foncé ; sable = connu ; brun = roche ; vert = nourriture vue ; orange = minerai vu ;
    or = nid ; rouge = éclaireuses ; jaune = autres fourmis ; une case connue mais non visible à cet instant est assombrie.
    """
    visible, known, rock, food, ore, nest, scouts, others = (np.clip(m[i], 0, 1)[..., None] for i in range(8))
    fog = np.array([38.0, 38.0, 44.0])
    ground = SAND * (1 - rock) + ROCK * rock
    ground = ground * (1 - 0.7 * food) + FOOD * 0.7 * food
    ground = ground * (1 - 0.8 * ore) + ORE * 0.8 * ore
    ground = ground * (0.72 + 0.28 * visible)                         # les cases hors de vue sont un peu plus sombres
    img = fog * (1 - known) + ground * known
    img = img * (1 - nest) + np.array([255.0, 215.0, 0.0]) * nest
    img = img * (1 - others) + np.array([250.0, 230.0, 40.0]) * others
    img = img * (1 - scouts) + np.array([230.0, 30.0, 30.0]) * scouts
    return np.clip(np.kron(img, np.ones((scale, scale, 1))), 0, 255).astype(np.uint8)
