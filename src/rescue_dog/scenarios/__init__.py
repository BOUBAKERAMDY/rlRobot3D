"""Scénarios physiques du projet ; voir `base` pour l'interface commune."""

from typing import Any

from rescue_dog.scenarios import glace, patte, tous

# Scénarios de test, avec leur grille.
SCENARIOS = {"glace": glace, "patte": patte}
# Scénarios d'entraînement : ceux de test, ou tous à la fois.
TRAINING = {**SCENARIOS, "tous": tous}


def in_training(scenario: str, condition: dict[str, Any]) -> bool:
    """La condition est-elle dans les plages randomisées à l'entraînement ?"""
    train = SCENARIOS[scenario].TRAIN
    return all(lo <= condition[key] <= hi for key, (lo, hi) in train.items())
