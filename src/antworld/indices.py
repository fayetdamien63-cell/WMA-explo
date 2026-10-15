"""Disposition de l'état du simulateur dans quelques tableaux NumPy.

Pourquoi des tableaux plutôt que des objets Python ? (1) Numba les compile en boucles rapides ;
(2) sauvegarder/restaurer l'état complet (étape 2, nécessaire à l'oracle) = copier 4 tableaux ;
(3) appeler le noyau avec peu d'arguments coûte moins cher.

Ces constantes nomment les colonnes/cases : on écrit `ants[i, A_CELL]` plutôt que `ants[i, 0]`.
Les cases de la grille sont indexées à plat : `cell = y * size + x`.
"""

# --- Ressources (sert aussi d'indice pour grid, flow, compteurs) --------------------------------
R_FOOD = 0
R_ORE = 1

# --- grid : float64, forme (5, N) ---------------------------------------------------------------
G_AMOUNT = 0      # grid[R_FOOD], grid[R_ORE] : quantités réelles dans le monde
G_SEEN = 2        # grid[G_SEEN + r] : dernière quantité vue par la colonie (peut être périmée)
G_KNOWN = 4       # 1.0 si la case a déjà été vue
N_GRID = 5

# --- flow : int32, forme (2, N) : prochaine case vers la ressource crue présente la plus proche -
# (−1 = aucune / on est dessus)

# --- ants : int32, forme (max_ants, N_ANT_FIELDS) ----------------------------------------------
A_CELL = 0
A_ROLE = 1
A_LOAD_TYPE = 2   # R_FOOD ou R_ORE (significatif si A_LOAD > 0)
A_LOAD = 3        # unités portées
A_HEADING = 4     # cap des éclaireuses : 0 N, 1 E, 2 S, 3 O
N_ANT_FIELDS = 5

ROLE_FOOD = 0     # cueilleuse de nourriture  (== R_FOOD)
ROLE_ORE = 1      # cueilleuse de minerai     (== R_ORE)
ROLE_SCOUT = 2
ROLE_BUILD = 3
N_ROLES = 4

# --- col : float64, scalaires de la colonie et compteurs cumulés --------------------------------
C_FOOD_STOCK = 0
C_ORE_STOCK = 1
C_N_ANTS = 2
C_NURSERY_LVL = 3
C_GRANARY_LVL = 4
C_BUILD_TYPE = 5          # 0 rien, 1 nurserie, 2 grenier
C_BUILD_PROGRESS = 6
C_BUILD_REQUIRED = 7
C_LAY_ACC = 8
C_STARVE_ACC = 9
C_EXPLORE_TARGET = 10     # 0 libre, 1..4 = N, E, S, O
C_DIRTY = 11              # C_DIRTY + r : le champ de flux de la ressource r est à recalculer
C_TICK = 13
# Compteurs cumulés (pour la conservation et les statistiques). `+ r` = nourriture / minerai.
C_HARVESTED = 14          # sorti du monde par les fourmis
C_DELIVERED = 16          # entré dans le stock
C_LOST = 18               # perdu avec une fourmi morte
C_FOOD_REGROWN = 20
C_FOOD_OVERFLOW = 21      # livré mais jeté (stock plein)
C_FOOD_EATEN = 22
C_FOOD_BROOD = 23         # nourriture transformée en fourmis nouvelles
C_ORE_SPENT = 24
C_BIRTHS = 25
C_DEATHS = 26
N_COL = 27

# --- P : paramètres (float64), figés pour toute la partie ---------------------------------------
P_SIZE = 0
P_CARRY = 1
P_EAT = 2
P_STARVE = 3
P_REGROW = 4
P_REGROW_SEED = 5
P_FORAGER_VIS = 6
P_SCOUT_VIS = 7
P_NEST_VIS = 8
P_TURN_PROB = 9
P_BIAS = 10
P_LAY_RATE = 11
P_BIRTH_COST = 12
P_RESERVE = 13
P_BASE_CAP = 14
P_NURSERY_GAIN = 15
P_BASE_STORAGE = 16
P_GRANARY_GAIN = 17
P_MAX_LEVEL = 18
P_NURSERY_COST = 19
P_GRANARY_COST = 20
P_BUILD_WORK = 21
P_MAX_ANTS = 22
P_SCOUT_SHARES = 23       # 4 valeurs
P_BUILD_SHARES = 27       # 4 valeurs
P_ORE_SHARES = 31         # 4 valeurs
N_PARAMS = 35

# --- Action macro : 5 entiers ------------------------------------------------------------------
ACT_SCOUT = 0
ACT_BUILD_SHARE = 1
ACT_ORE_SHARE = 2
ACT_BUILD_CHOICE = 3      # 0 rien, 1 nurserie, 2 grenier
ACT_TARGET = 4            # 0 libre, 1..4 = N, E, S, O
