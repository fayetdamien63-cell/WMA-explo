"""Affiche le tableau du test anti-triche de la récompense (20 seeds, épisodes complets).

    python scripts/anticheat_report.py
"""

from pathlib import Path

from antworld import AntWorldConfig
from common.config import load_config
from eval.anticheat import evaluate

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "antworld.yaml"

if __name__ == "__main__":
    cfg = load_config(AntWorldConfig, CONFIG)
    table = evaluate(cfg)
    print(f"{'stratégie':50s} {'type':10s} {'retour moy':>10s} {'max':>7s} {'pop fin':>7s} {'morte':>6s} {'minerai':>8s} {'niv.':>5s}")
    for name, r in sorted(table.items(), key=lambda kv: -kv[1]["return"]):
        print(f"{name:50s} {r['kind']:10s} {r['return']:10.1f} {r['max_return']:7.1f} {r['pop']:7.1f} "
              f"{100 * r['dead']:5.0f}% {r['ore']:8.1f} {r['levels']:5.1f}")
