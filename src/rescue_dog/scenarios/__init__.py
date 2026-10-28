"""Scénarios physiques du projet ; voir `base` pour l'interface commune."""

from typing import Any

from rescue_dog.scenarios import glace, patte, tous

# Scénarios de test, avec leur grille.
SCENARIOS = {"glace": glace, "patte": patte}
# Scénarios d'entraînement : ceux de test, ou tous à la fois.
TRAINING = {**SCENARIOS, "tous": tous}


def parse_condition(scenario: str, pairs: list[str]) -> dict[str, Any]:
    """Condition de test depuis la ligne de commande, ex. ["friction=0.1", "payload=5"].

    Les clés sont celles de la grille du scénario ; toutes sont obligatoires.
    """
    row_key, _, col_key, _ = SCENARIOS[scenario].AXES
    condition: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or key not in (row_key, col_key):
            raise ValueError(f"{pair!r} : attendu {row_key}=… et {col_key}=…")
        try:
            condition[key] = float(value)
        except ValueError:
            condition[key] = value
    missing = {row_key, col_key} - set(condition)
    if missing:
        raise ValueError(
            f"Condition incomplète pour {scenario} : manque {sorted(missing)}"
        )
    return condition


def in_training(scenario: str, condition: dict[str, Any]) -> bool:
    """La condition est-elle dans les plages randomisées à l'entraînement ?"""
    train = SCENARIOS[scenario].TRAIN
    return all(lo <= condition[key] <= hi for key, (lo, hi) in train.items())
