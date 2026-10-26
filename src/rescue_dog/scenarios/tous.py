"""Entraînement sur tous les scénarios à la fois.

Chaque chien simulé tire ensemble les paramètres ξ des deux scénarios, dans
leurs plages `TRAIN` : friction du sol, charge sur le dos, et une patte au
hasard qui garde 50 à 100 % de sa force. Une seule politique est ainsi
préparée au sol glissant, à la charge lourde et à la patte blessée.

Ce n'est pas un scénario de test : on évalue sur les grilles de `glace` et
de `patte`.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from mujoco import mjx

from rescue_dog.scenarios import glace, patte
from rescue_dog.scenarios.base import randomizer

TRAIN = {**glace.TRAIN, **patte.TRAIN}


def sample(model: mjx.Model, rng: jax.Array) -> dict[str, jax.Array]:
    """Friction, charge et patte affaiblie tirées ensemble."""
    key_glace, key_patte = jax.random.split(rng)
    fields = glace.sample(model, key_glace)
    leg_fields = patte.sample(model, key_patte)
    for name in ("actuator_forcerange", "actuator_gainprm", "actuator_biasprm"):
        fields[name] = leg_fields[name]
    return fields


randomize = randomizer(sample)


def privileged_params(model: mjx.Model, nominal: mjx.Model) -> jax.Array:
    """ξ vu par le critique : friction, charge, force de chaque patte (6)."""
    return jp.concatenate(
        [
            glace.privileged_params(model, nominal),
            patte.privileged_params(model, nominal),
        ]
    )
