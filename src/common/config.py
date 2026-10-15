"""Configurations : des dataclasses remplies depuis un fichier YAML.

Les valeurs vivent dans `configs/*.yaml`, jamais en dur dans le code. Le fichier YAML
n'a pas besoin de tout lister : les champs absents prennent la valeur par défaut de la
dataclass. En revanche une clé inconnue (faute de frappe) est une ERREUR, pour éviter
qu'un paramètre mal orthographié soit ignoré en silence pendant un entraînement.

    cfg = load_config(ProjectConfig, "configs/default.yaml", overrides=["seed=7"])
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """Configuration invalide (clé inconnue, mauvais type, champ obligatoire absent)."""


@dataclass
class ProjectConfig:
    """Paramètres globaux. Les étapes suivantes ajouteront leurs sections (world, env, ...)."""

    seed: int = 0               # seed maître : toute l'aléa en dérive
    device: str = "auto"        # auto | cuda | mps | cpu
    runs_dir: str = "runs"      # où écrire logs TensorBoard, checkpoints, config sauvegardée


def _convert(tp: Any, value: Any, path: str) -> Any:
    """Convertit/valide `value` selon le type annoté `tp`."""
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)  # "lr: 1" doit être accepté comme 1.0
    if tp in (int, float, str, bool):
        # bool est une sous-classe d'int : on le rejette explicitement pour int/float
        if not isinstance(value, tp) or (tp is not bool and isinstance(value, bool)):
            raise ConfigError(f"'{path}' : attendu {tp.__name__}, reçu {value!r}")
        return value
    if typing.get_origin(tp) is tuple and isinstance(value, list):
        return tuple(value)  # YAML n'a que des listes
    return value  # types plus riches : acceptés tels quels


def _build(cls: type[T], data: Any, path: str) -> T:
    where = path or "<racine>"
    if not isinstance(data, dict):
        raise ConfigError(f"'{where}' : attendu un dictionnaire, reçu {data!r}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"Clé(s) inconnue(s) dans '{where}' : {unknown}. Attendues : {sorted(fields)}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, f in fields.items():
        sub = f"{path}.{name}" if path else name
        if name in data:
            kwargs[name] = _convert(hints[name], data[name], sub)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"Champ obligatoire absent : '{sub}'")
    return cls(**kwargs)


def _apply_override(data: dict, override: str) -> None:
    """Applique "section.cle=valeur" sur le dictionnaire (valeur interprétée comme du YAML)."""
    if "=" not in override:
        raise ConfigError(f"Surcharge invalide (attendu 'a.b=valeur') : {override!r}")
    dotted, raw = override.split("=", 1)
    keys = dotted.strip().split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ConfigError(f"Surcharge '{override}' : '{key}' n'est pas une section")
    node[keys[-1]] = yaml.safe_load(raw)


def load_config(cls: type[T], path: str | Path | None = None, overrides: list[str] | None = None) -> T:
    """Charge `cls` depuis un YAML (optionnel), puis applique les surcharges "a.b=valeur"."""
    data: dict = {}
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    for override in overrides or []:
        _apply_override(data, override)
    return _build(cls, data, "")


def save_config(cfg: Any, path: str | Path) -> None:
    """Écrit la configuration effective (à stocker dans le dossier de chaque run)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dataclasses.asdict(cfg), fh, sort_keys=False, allow_unicode=True)
