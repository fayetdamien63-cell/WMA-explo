"""Construction des agents à partir de leur nom et de la configuration (picklable : sert aux workers)."""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.heuristic import HeuristicAgent, HeuristicConfig
from agents.oracle import OracleAgent, OracleConfig
from agents.random_agent import RandomAgent
from antworld.config import AntWorldConfig


@dataclass
class BaselinesConfig:
    """Réglages des agents de référence (fichier configs/baselines.yaml)."""

    heuristic: HeuristicConfig = field(default_factory=HeuristicConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)


AGENT_NAMES = ("random", "heuristic", "oracle")


def make_agent(name: str, world_cfg: AntWorldConfig, baselines: BaselinesConfig | None = None):
    baselines = baselines or BaselinesConfig()
    if name == "random":
        return RandomAgent()
    if name == "heuristic":
        return HeuristicAgent(baselines.heuristic, world_cfg)
    if name == "oracle":
        return OracleAgent(baselines.oracle)
    raise ValueError(f"Agent inconnu : {name!r} (attendu : {AGENT_NAMES})")
