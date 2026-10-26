"""Scénario 2, « La patte blessée » : un moteur perd une partie de sa force.

Une patte k (FR, FL, RR, RL) est touchée. Ses trois moteurs (hanche, cuisse,
genou) gardent une fraction η de leur force : le couple maximal et la raideur
de l'asservissement en position sont multipliés par η,

    τ_max ← η · τ_max,    kp ← η · kp.

ξ vu par le critique privilégié : la force restante de chaque patte (4
valeurs, 1 pour une patte saine). L'entraînement tire une patte au hasard et
η dans `TRAIN` ; le test essaie les 4 pattes avec η dans `TEST`.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from mujoco import mjx

from rescue_dog.scenarios.base import jitter, randomizer

LEGS = ("FR", "FL", "RR", "RL")  # ordre des actionneurs du Go1, 3 par patte

# Plage (min, max) de la force restante η.
TRAIN = {"strength": (0.5, 1.0)}
TEST = {"strength": (0.2, 1.0)}

# Grille de test fixe : 4 pattes × 5 forces = 20 conditions.
TEST_STRENGTHS = (0.2, 0.4, 0.6, 0.8, 1.0)
AXES = ("leg", LEGS, "strength", TEST_STRENGTHS)


def _leg_scale(leg: jax.Array, strength: jax.Array) -> jax.Array:
    """Facteur de chacun des 12 actionneurs : `strength` pour la patte `leg`, 1 sinon."""
    per_leg = jp.ones(len(LEGS)).at[leg].set(strength)
    return jp.repeat(per_leg, 3)


def _weaken(model: mjx.Model, scale: jax.Array) -> dict[str, jax.Array]:
    """Couple maximal et raideur (kp = gain = -biais) multipliés par `scale`."""
    return {
        "actuator_forcerange": model.actuator_forcerange * scale[:, None],
        "actuator_gainprm": model.actuator_gainprm.at[:, 0].multiply(scale),
        "actuator_biasprm": model.actuator_biasprm.at[:, 1].multiply(scale),
    }


def set_condition(model: mjx.Model, leg: str, strength: float) -> mjx.Model:
    """Modèle dont la patte `leg` ne garde que `strength` de sa force."""
    scale = _leg_scale(jp.array(LEGS.index(leg)), jp.array(strength))
    return model.tree_replace(_weaken(model, scale))


def sample(model: mjx.Model, rng: jax.Array) -> dict[str, jax.Array]:
    """Une patte au hasard, η tiré dans `TRAIN`, plus les petites variations."""
    rng, key_leg, key_eta = jax.random.split(rng, 3)
    leg = jax.random.randint(key_leg, (), 0, len(LEGS))
    strength = jax.random.uniform(
        key_eta, minval=TRAIN["strength"][0], maxval=TRAIN["strength"][1]
    )
    fields = jitter(model, rng)
    fields.update(_weaken(model, _leg_scale(leg, strength)))
    return fields


randomize = randomizer(sample)


def privileged_params(model: mjx.Model, nominal: mjx.Model) -> jax.Array:
    """ξ vu par le critique privilégié : force restante de chaque patte (4)."""
    ratio = model.actuator_forcerange[:, 1] / nominal.actuator_forcerange[:, 1]
    return ratio[::3]


def grid() -> list[dict[str, object]]:
    """Les 20 conditions du test, patte d'abord."""
    return [{"leg": leg, "strength": s} for leg in LEGS for s in TEST_STRENGTHS]
