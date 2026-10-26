"""Scénario 1, « Mission sur la glace » : sol glissant et charge sur le dos.

Deux paramètres physiques ξ :
- la friction du sol μ (le sol a la priorité de contact dans le XML du Go1 :
  c'est sa friction qui s'applique aux pieds) ;
- la charge Δm ajoutée à la masse du tronc (5,2 kg, robot entier 12,7 kg).

L'entraînement randomise ξ dans `TRAIN` ; le test va au-delà, dans `TEST`.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from mujoco import mjx

from rescue_dog.scenarios.base import FLOOR_GEOM_ID, TORSO_BODY_ID, jitter, randomizer

# Plages (min, max) de ξ.
TRAIN = {"friction": (0.3, 1.0), "payload": (0.0, 3.0)}
TEST = {"friction": (0.1, 1.0), "payload": (0.0, 5.0)}

# Grille de test fixe : 5 frictions × 5 charges = 25 conditions.
TEST_FRICTIONS = (0.1, 0.25, 0.4, 0.7, 1.0)
TEST_PAYLOADS = (0.0, 1.0, 2.0, 3.0, 5.0)
AXES = ("payload", TEST_PAYLOADS, "friction", TEST_FRICTIONS)


def set_condition(model: mjx.Model, friction: float, payload: float) -> mjx.Model:
    """Modèle avec la friction du sol `friction` et `payload` kg sur le dos."""
    return model.tree_replace(
        {
            "geom_friction": model.geom_friction.at[FLOOR_GEOM_ID, 0].set(friction),
            "body_mass": model.body_mass.at[TORSO_BODY_ID].add(payload),
        }
    )


def sample(model: mjx.Model, rng: jax.Array) -> dict[str, jax.Array]:
    """Friction et charge tirées dans `TRAIN`, plus les petites variations."""
    rng, key_f, key_p = jax.random.split(rng, 3)
    friction = jax.random.uniform(
        key_f, minval=TRAIN["friction"][0], maxval=TRAIN["friction"][1]
    )
    payload = jax.random.uniform(
        key_p, minval=TRAIN["payload"][0], maxval=TRAIN["payload"][1]
    )
    fields = jitter(model, rng)
    fields["geom_friction"] = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(friction)
    fields["body_mass"] = fields["body_mass"].at[TORSO_BODY_ID].add(payload)
    return fields


randomize = randomizer(sample)


def privileged_params(model: mjx.Model, nominal: mjx.Model) -> jax.Array:
    """ξ vu par le critique privilégié : (friction du sol, charge en kg)."""
    friction = model.geom_friction[FLOOR_GEOM_ID, 0]
    payload = model.body_mass[TORSO_BODY_ID] - nominal.body_mass[TORSO_BODY_ID]
    return jp.array([friction, payload])


def grid() -> list[dict[str, float]]:
    """Les 25 conditions du test, charge d'abord."""
    return [
        {"friction": f, "payload": p} for p in TEST_PAYLOADS for f in TEST_FRICTIONS
    ]
