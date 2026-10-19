"""Chargement de la configuration d'un entraînement depuis un fichier YAML."""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float
    entropy_cost: float
    discounting: float
    unroll_length: int
    batch_size: int
    num_minibatches: int
    num_updates_per_batch: int
    num_resets_per_eval: int
    reward_scaling: float
    normalize_observations: bool
    max_grad_norm: float
    action_repeat: int
    policy_hidden_layer_sizes: tuple[int, ...]
    value_hidden_layer_sizes: tuple[int, ...]


@dataclass(frozen=True)
class Config:
    run_name: str
    output_dir: str
    env_name: str
    episode_length: int
    impl: str
    seed: int
    num_timesteps: int
    num_envs: int
    num_evals: int
    domain_randomization: bool
    privileged_critic: bool
    ppo: PPOConfig


class ConfigError(ValueError):
    """Configuration invalide : clé inconnue ou manquante, valeur du mauvais type."""


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Lit `path` puis applique les overrides `clé.sous_clé=valeur` dans l'ordre."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    for override in overrides or []:
        _apply_override(raw, override)
    return _build(Config, raw, prefix="")


def _apply_override(raw: dict[str, Any], override: str) -> None:
    key, sep, value = override.partition("=")
    if not sep or not key:
        raise ConfigError(f"Override invalide {override!r} : format attendu clé=valeur")
    *sections, leaf = key.split(".")
    node = raw
    for section in sections:
        node = node.setdefault(section, {})
        if not isinstance(node, dict):
            raise ConfigError(f"Override {override!r} : {section!r} n'est pas une section")
    # La valeur est lue comme du YAML : "3" -> int, "true" -> bool, "[1, 2]" -> liste.
    # Les clés inconnues ajoutées ici sont rejetées ensuite par _build.
    node[leaf] = yaml.safe_load(value)


def _build(cls: type, data: Any, prefix: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{prefix.rstrip('.') or 'racine'} : section attendue, reçu {data!r}")
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(data) - set(names))
    if unknown:
        raise ConfigError("Clés inconnues : " + ", ".join(prefix + k for k in unknown))
    missing = [name for name in names if name not in data]
    if missing:
        raise ConfigError("Clés manquantes : " + ", ".join(prefix + k for k in missing))
    hints = typing.get_type_hints(cls)
    return cls(**{name: _convert(data[name], hints[name], prefix + name) for name in names})


def _convert(value: Any, typ: Any, key: str) -> Any:
    if dataclasses.is_dataclass(typ):
        return _build(typ, value, prefix=key + ".")
    if typing.get_origin(typ) is tuple:
        item_type = typing.get_args(typ)[0]
        if isinstance(value, (list, tuple)):
            return tuple(_convert(v, item_type, f"{key}[{i}]") for i, v in enumerate(value))
    # bool est une sous-classe de int : on l'exclut pour que `seed=true` soit refusé.
    elif typ is bool:
        if isinstance(value, bool):
            return value
    elif typ is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif typ is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        # PyYAML lit "3e-4" (sans point) comme une chaîne.
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
    elif typ is str:
        if isinstance(value, str):
            return value
    raise ConfigError(f"{key} : type attendu {getattr(typ, '__name__', typ)}, reçu {value!r}")
