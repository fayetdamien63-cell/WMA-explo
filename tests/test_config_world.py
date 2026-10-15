"""Étape 1 : la config du monde refuse les combinaisons incohérentes."""

import pytest

from antworld.config import AntWorldConfig, ColonyConfig
from common.config import load_config
from helpers import CONFIG_PATH


def test_yaml_config_loads_and_matches_defaults():
    assert load_config(AntWorldConfig, CONFIG_PATH) == AntWorldConfig()


@pytest.mark.parametrize("bad", [dict(initial_food=1000.0), dict(initial_ants=100), dict(max_ants=50),
                                 dict(scout_shares=(0.0, 0.5))])
def test_inconsistent_colony_config_rejected(bad):
    with pytest.raises(ValueError):
        AntWorldConfig(colony=ColonyConfig(**bad))
