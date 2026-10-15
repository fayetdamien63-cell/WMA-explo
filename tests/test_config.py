"""Étape 0 : chargement des configs YAML en dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from common.config import ConfigError, ProjectConfig, load_config, save_config

DEFAULT_YAML = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


@dataclass
class Inner:
    size: int = 32
    scale: float = 1.0
    dims: tuple[int, ...] = (1, 2)


@dataclass
class Outer:
    name: str
    inner: Inner = field(default_factory=Inner)


def write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_default_yaml_loads():
    cfg = load_config(ProjectConfig, DEFAULT_YAML)
    assert cfg.seed == 0 and cfg.device == "auto"


def test_nested_defaults_and_partial_override(tmp_path):
    cfg = load_config(Outer, write(tmp_path, "name: a\ninner:\n  size: 48\n"))
    assert cfg.inner.size == 48 and cfg.inner.scale == 1.0


def test_dotted_overrides_and_int_to_float_and_list_to_tuple(tmp_path):
    cfg = load_config(Outer, write(tmp_path, "name: a\n"),
                      overrides=["inner.scale=2", "inner.dims=[3, 4]", "name=b"])
    assert cfg.inner.scale == 2.0 and isinstance(cfg.inner.scale, float)
    assert cfg.inner.dims == (3, 4)
    assert cfg.name == "b"


def test_unknown_key_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="inconnue"):
        load_config(Outer, write(tmp_path, "name: a\ninner:\n  sizee: 3\n"))


def test_wrong_type_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="inner.size"):
        load_config(Outer, write(tmp_path, "name: a\ninner:\n  size: 'grand'\n"))
    with pytest.raises(ConfigError):  # un bool n'est pas un int
        load_config(Outer, write(tmp_path, "name: a\ninner:\n  size: true\n"))


def test_missing_required_field(tmp_path):
    with pytest.raises(ConfigError, match="obligatoire"):
        load_config(Outer, write(tmp_path, "inner:\n  size: 3\n"))


def test_save_then_reload_roundtrip(tmp_path):
    cfg = load_config(ProjectConfig, DEFAULT_YAML, overrides=["seed=7"])
    save_config(cfg, tmp_path / "run" / "config.yaml")
    assert load_config(ProjectConfig, tmp_path / "run" / "config.yaml") == cfg
