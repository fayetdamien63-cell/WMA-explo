"""Noyau de simulation compilé par Numba : toutes les règles du monde (voir docs/WORLD_SPEC.md).

Le code est écrit comme de simples boucles Python sur des tableaux : lisible, et rapide une fois
compilé. Pour déboguer avec un vrai débogueur Python, lancer avec NUMBA_DISABLE_JIT=1 (très lent).

Aucune fonction ici ne tire d'aléa elle-même : les nombres aléatoires d'un pas macro sont tirés
en une fois côté Python (`rand`, générateur du SeedManager) et passés en argument. C'est ce qui
rend l'état sauvegardable/restaurable et le tout reproductible.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from antworld.indices import (
    A_CELL, A_HEADING, A_LOAD, A_LOAD_TYPE, A_ROLE,
    ACT_BUILD_CHOICE, ACT_BUILD_SHARE, ACT_ORE_SHARE, ACT_SCOUT, ACT_TARGET,
    C_BIRTHS, C_BUILD_PROGRESS, C_BUILD_REQUIRED, C_BUILD_TYPE, C_DEATHS, C_DELIVERED, C_DIRTY,
    C_EXPLORE_TARGET, C_FOOD_BROOD, C_FOOD_EATEN, C_FOOD_OVERFLOW, C_FOOD_REGROWN, C_FOOD_STOCK,
    C_GRANARY_LVL, C_HARVESTED, C_LAY_ACC, C_LOST, C_N_ANTS, C_NURSERY_LVL, C_ORE_SPENT,
    C_ORE_STOCK, C_STARVE_ACC, C_TICK,
    G_AMOUNT, G_KNOWN, G_SEEN,
    P_BASE_CAP, P_BASE_STORAGE, P_BIAS, P_BIRTH_COST, P_BUILD_SHARES, P_BUILD_WORK, P_CARRY,
    P_EAT, P_FORAGER_VIS, P_GRANARY_COST, P_GRANARY_GAIN, P_LAY_RATE, P_MAX_ANTS, P_MAX_LEVEL,
    P_NEST_VIS, P_NURSERY_COST, P_NURSERY_GAIN, P_ORE_SHARES, P_REGROW, P_REGROW_SEED,
    P_RESERVE, P_SCOUT_SHARES, P_SCOUT_VIS, P_SIZE, P_STARVE, P_TURN_PROB,
    R_FOOD, R_ORE, ROLE_BUILD, ROLE_FOOD, ROLE_ORE, ROLE_SCOUT,
)


# ------------------------------------------------------------------------------------------------
# Géométrie et plus courts chemins
# ------------------------------------------------------------------------------------------------

@njit(cache=True)
def neighbor(cell, d, size):
    """Case voisine dans la direction d (0 N, 1 E, 2 S, 3 O), ou −1 si hors de la grille."""
    x = cell % size
    y = cell // size
    if d == 0:
        y -= 1
    elif d == 1:
        x += 1
    elif d == 2:
        y += 1
    else:
        x -= 1
    if x < 0 or x >= size or y < 0 or y >= size:
        return -1
    return y * size + x


@njit(cache=True)
def _bfs(terrain, size, is_source, nxt, dist):
    """Parcours en largeur multi-sources sur les cases libres.

    Remplit pour chaque case atteignable : `dist` (nombre de pas jusqu'à la source la plus proche)
    et `nxt` (la case voisine à prendre pour s'en rapprocher ; −1 sur une source ou si inaccessible).
    L'ordre des voisins est fixe (N, E, S, O) : le résultat est déterministe.
    """
    n = size * size
    queue = np.empty(n, np.int32)
    for c in range(n):
        dist[c] = -1
        nxt[c] = -1
    head = 0
    tail = 0
    for c in range(n):
        if is_source[c] != 0 and terrain[c] == 0:
            dist[c] = 0
            queue[tail] = c
            tail += 1
    while head < tail:
        c = queue[head]
        head += 1
        for d in range(4):
            nb = neighbor(c, d, size)
            if nb >= 0 and dist[nb] < 0 and terrain[nb] == 0:
                dist[nb] = dist[c] + 1
                nxt[nb] = c
                queue[tail] = nb
                tail += 1


@njit(cache=True)
def _bfs_wrapper(terrain, size, is_source):
    n = size * size
    nxt = np.empty(n, np.int32)
    dist = np.empty(n, np.int32)
    _bfs(terrain, size, is_source, nxt, dist)
    return nxt, dist


def bfs_from(terrain: np.ndarray, size: int, is_source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(nxt, dist) du plus court chemin vers les sources (version appelable depuis Python)."""
    return _bfs_wrapper(terrain, size, is_source)


@njit(cache=True)
def recompute_flow(grid, flow, terrain, size, r):
    """Recalcule le champ de flux de la ressource r : vers les cases où l'on CROIT qu'il y en a."""
    n = size * size
    src = np.zeros(n, np.uint8)
    for c in range(n):
        if grid[G_SEEN + r, c] >= 1.0:
            src[c] = 1
    dist = np.empty(n, np.int32)
    _bfs(terrain, size, src, flow[r], dist)


# ------------------------------------------------------------------------------------------------
# Vision / brouillard
# ------------------------------------------------------------------------------------------------

@njit(cache=True)
def reveal(grid, col, cell, radius, size):
    """Révèle le carré de rayon `radius` autour de `cell` et met à jour les quantités « vues ».

    Si une case change d'état « contient au moins 1 unité » dans la carte de la colonie, on marque
    le champ de flux de cette ressource à recalculer.
    """
    x = cell % size
    y = cell // size
    for yy in range(max(0, y - radius), min(size, y + radius + 1)):
        for xx in range(max(0, x - radius), min(size, x + radius + 1)):
            c = yy * size + xx
            grid[G_KNOWN, c] = 1.0
            for r in range(2):
                real = grid[G_AMOUNT + r, c]
                old = grid[G_SEEN + r, c]
                if real != old:
                    if (real >= 1.0) != (old >= 1.0):
                        col[C_DIRTY + r] = 1.0
                    grid[G_SEEN + r, c] = real


# ------------------------------------------------------------------------------------------------
# Action macro : répartition des rôles et lancement d'une construction
# ------------------------------------------------------------------------------------------------

@njit(cache=True)
def _round_half_up(x):
    return int(math.floor(x + 0.5))


@njit(cache=True)
def apply_action(ants, col, P, action):
    """Applique l'action macro : cible d'exploration, rôles des fourmis, début de construction."""
    n = int(col[C_N_ANTS])
    col[C_EXPLORE_TARGET] = action[ACT_TARGET]

    # --- effectifs visés par rôle --------------------------------------------------------------
    n_scout = min(_round_half_up(P[P_SCOUT_SHARES + action[ACT_SCOUT]] * n), n)
    n_build = min(_round_half_up(P[P_BUILD_SHARES + action[ACT_BUILD_SHARE]] * n), n - n_scout)
    rest = n - n_scout - n_build
    n_ore = _round_half_up(P[P_ORE_SHARES + action[ACT_ORE_SHARE]] * rest)
    target = np.zeros(4, np.int64)
    target[ROLE_FOOD] = rest - n_ore
    target[ROLE_ORE] = n_ore
    target[ROLE_SCOUT] = n_scout
    target[ROLE_BUILD] = n_build

    # --- réaffectation minimale : on libère le surplus (indices les plus élevés d'abord) ------
    count = np.zeros(4, np.int64)
    for i in range(n):
        count[ants[i, A_ROLE]] += 1
    pool = np.empty(n, np.int64)
    n_pool = 0
    for i in range(n - 1, -1, -1):
        role = ants[i, A_ROLE]
        if count[role] > target[role]:
            count[role] -= 1
            pool[n_pool] = i
            n_pool += 1
    # --- ... et on le redistribue aux rôles déficitaires -------------------------------------
    k = 0
    for role in range(4):
        while count[role] < target[role] and k < n_pool:
            ants[pool[k], A_ROLE] = role
            count[role] += 1
            k += 1

    # --- construction : une seule à la fois, coût en minerai payé au départ --------------------
    choice = action[ACT_BUILD_CHOICE]
    if choice > 0 and col[C_BUILD_TYPE] == 0:
        level = col[C_NURSERY_LVL] if choice == 1 else col[C_GRANARY_LVL]
        if level < P[P_MAX_LEVEL]:
            base_cost = P[P_NURSERY_COST] if choice == 1 else P[P_GRANARY_COST]
            cost = base_cost * (level + 1.0)
            if col[C_ORE_STOCK] >= cost:
                col[C_ORE_STOCK] -= cost
                col[C_ORE_SPENT] += cost
                col[C_BUILD_TYPE] = choice
                col[C_BUILD_PROGRESS] = 0.0
                col[C_BUILD_REQUIRED] = P[P_BUILD_WORK] * (level + 1.0)


# ------------------------------------------------------------------------------------------------
# Un tick de simulation
# ------------------------------------------------------------------------------------------------

@njit(cache=True)
def _kill_youngest(ants, col, k):
    """Retire les k fourmis aux indices les plus élevés (les dernières nées) ; leur charge est perdue."""
    n = int(col[C_N_ANTS])
    k = min(k, n)
    for _ in range(k):
        n -= 1
        if ants[n, A_LOAD] > 0:
            col[C_LOST + ants[n, A_LOAD_TYPE]] += ants[n, A_LOAD]
        ants[n, A_LOAD] = 0
        col[C_DEATHS] += 1.0
    col[C_N_ANTS] = n


@njit(cache=True)
def tick(grid, flow, ants, col, terrain, food_cap, food_cells, nest_next, nest, P, rand_t):
    """Avance le monde d'un tick (ordre : voir WORLD_SPEC §8). `rand_t` : (>= n_fourmis, 3) tirages."""
    size = int(P[P_SIZE])

    # 1. Repousse logistique de la nourriture --------------------------------------------------
    g = P[P_REGROW]
    s = P[P_REGROW_SEED]
    regrown = 0.0
    for k in range(food_cells.shape[0]):
        c = food_cells[k]
        cap = food_cap[c]
        a = grid[G_AMOUNT + R_FOOD, c]
        a_new = a + g * (a + s * cap) * (1.0 - a / cap)
        if a_new > cap:
            a_new = cap
        grid[G_AMOUNT + R_FOOD, c] = a_new
        regrown += a_new - a
    col[C_FOOD_REGROWN] += regrown

    # 2. Les fourmis agissent, puis voient ----------------------------------------------------
    n = int(col[C_N_ANTS])
    storage_cap = P[P_BASE_STORAGE] + col[C_GRANARY_LVL] * P[P_GRANARY_GAIN]
    carry = int(P[P_CARRY])
    building = col[C_BUILD_TYPE] > 0
    target_dir = int(col[C_EXPLORE_TARGET])

    for i in range(n):
        cell = ants[i, A_CELL]
        role = ants[i, A_ROLE]
        # La vision ne dépend que du rôle (même chargée, une éclaireuse voit loin) : l'observation
        # de la colonie (observation.py) utilise exactement la même règle.
        vision = int(P[P_SCOUT_VIS]) if role == ROLE_SCOUT else int(P[P_FORAGER_VIS])

        if ants[i, A_LOAD] > 0:
            # Règle 1 : une fourmi chargée rentre au nid et dépose, quel que soit son rôle.
            if cell == nest:
                amount = float(ants[i, A_LOAD])
                r = ants[i, A_LOAD_TYPE]
                if r == R_FOOD:
                    space = max(storage_cap - col[C_FOOD_STOCK], 0.0)
                    accepted = min(amount, space)
                    col[C_FOOD_STOCK] += accepted
                    col[C_DELIVERED + R_FOOD] += accepted
                    col[C_FOOD_OVERFLOW] += amount - accepted
                else:
                    col[C_ORE_STOCK] += amount
                    col[C_DELIVERED + R_ORE] += amount
                ants[i, A_LOAD] = 0
            else:
                cell = nest_next[cell]

        elif role == ROLE_SCOUT:
            # Éclaireuse : marche aléatoire persistante, biaisée vers la direction cible.
            heading = ants[i, A_HEADING]
            if rand_t[i, 0] < P[P_TURN_PROB]:
                if target_dir > 0 and rand_t[i, 1] < P[P_BIAS]:
                    heading = target_dir - 1
                else:
                    heading = int(rand_t[i, 2] * 4.0) % 4
            nb = neighbor(cell, heading, size)
            if nb >= 0 and terrain[nb] == 0:
                cell = nb
            else:
                heading = int(rand_t[i, 2] * 4.0) % 4  # bloquée : nouveau cap au prochain tick
            ants[i, A_HEADING] = heading

        elif role == ROLE_BUILD:
            # Bâtisseuse : va au nid, puis apporte une unité de travail par tick.
            if cell != nest:
                cell = nest_next[cell]
            elif building:
                col[C_BUILD_PROGRESS] += 1.0

        else:
            # Cueilleuse (role == ressource visée) : récolte ici, ou va vers la ressource crue la plus proche.
            r = role
            avail = int(math.floor(grid[G_AMOUNT + r, cell]))
            if avail >= 1:
                take = min(carry, avail)
                grid[G_AMOUNT + r, cell] -= take
                ants[i, A_LOAD] = take
                ants[i, A_LOAD_TYPE] = r
                col[C_HARVESTED + r] += take
            else:
                nxt = flow[r, cell]
                if nxt >= 0:
                    cell = nxt
                elif cell != nest:
                    cell = nest_next[cell]  # rien à faire : on rentre attendre au nid

        ants[i, A_CELL] = cell
        reveal(grid, col, cell, vision, size)

    reveal(grid, col, nest, int(P[P_NEST_VIS]), size)

    # 3. Fin de construction -------------------------------------------------------------------
    if col[C_BUILD_TYPE] > 0 and col[C_BUILD_PROGRESS] >= col[C_BUILD_REQUIRED]:
        if col[C_BUILD_TYPE] == 1:
            col[C_NURSERY_LVL] += 1.0
        else:
            col[C_GRANARY_LVL] += 1.0
        col[C_BUILD_TYPE] = 0.0
        col[C_BUILD_PROGRESS] = 0.0
        col[C_BUILD_REQUIRED] = 0.0

    # 4. Consommation de nourriture et famine --------------------------------------------------
    n = int(col[C_N_ANTS])
    need = n * P[P_EAT]
    if need > 0.0:
        if col[C_FOOD_STOCK] >= need:
            col[C_FOOD_STOCK] -= need
            col[C_FOOD_EATEN] += need
            col[C_STARVE_ACC] = 0.0
        else:
            eaten = col[C_FOOD_STOCK]
            col[C_FOOD_STOCK] = 0.0
            col[C_FOOD_EATEN] += eaten
            unmet_fraction = (need - eaten) / need
            col[C_STARVE_ACC] += n * P[P_STARVE] * unmet_fraction
            deaths = int(col[C_STARVE_ACC])
            col[C_STARVE_ACC] -= deaths
            _kill_youngest(ants, col, deaths)

    # 5. Naissance (au plus une par tick) ------------------------------------------------------
    n = int(col[C_N_ANTS])
    col[C_LAY_ACC] += P[P_LAY_RATE]
    if col[C_LAY_ACC] >= 1.0:
        pop_cap = P[P_BASE_CAP] + col[C_NURSERY_LVL] * P[P_NURSERY_GAIN]
        if (n > 0 and n < pop_cap and n < int(P[P_MAX_ANTS])
                and col[C_FOOD_STOCK] >= P[P_BIRTH_COST] + P[P_RESERVE]):
            col[C_FOOD_STOCK] -= P[P_BIRTH_COST]
            col[C_FOOD_BROOD] += P[P_BIRTH_COST]
            ants[n, A_CELL] = nest
            ants[n, A_ROLE] = ROLE_FOOD
            ants[n, A_LOAD] = 0
            ants[n, A_LOAD_TYPE] = 0
            ants[n, A_HEADING] = 0
            col[C_N_ANTS] = n + 1
            col[C_BIRTHS] += 1.0
            col[C_LAY_ACC] -= 1.0
        else:
            col[C_LAY_ACC] = 1.0  # pas de stock de ponte : on ne « capitalise » pas

    # 6. Champs de flux périmés -----------------------------------------------------------------
    for r in range(2):
        if col[C_DIRTY + r] > 0.0:
            recompute_flow(grid, flow, terrain, size, r)
            col[C_DIRTY + r] = 0.0

    col[C_TICK] += 1.0


@njit(cache=True)
def run_ticks(grid, flow, ants, col, terrain, food_cap, food_cells, nest_next, nest, P, rand):
    """Joue `rand.shape[0]` ticks d'affilée (un pas macro). S'arrête si la colonie est morte."""
    for t in range(rand.shape[0]):
        if col[C_N_ANTS] < 1.0:
            break
        tick(grid, flow, ants, col, terrain, food_cap, food_cells, nest_next, nest, P, rand[t])
