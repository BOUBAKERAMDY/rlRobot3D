"""Briques communes aux scénarios.

Chaque scénario (`glace`, `patte`) expose la même interface :
- `sample(model, rng)` : les champs du modèle MJX tirés pour un environnement
  d'entraînement (ses paramètres ξ dans les plages `TRAIN`) ;
- `privileged_params(model, nominal)` : ξ lu dans le modèle courant, ajouté à
  l'état privilégié ;
- `set_condition(model, **condition)` et `grid()` : les conditions de test ;
- `AXES` : (clé des lignes, valeurs, clé des colonnes, valeurs) de la grille.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def jitter(model: mjx.Model, rng: jax.Array) -> dict[str, jax.Array]:
    """Petites variations du randomiseur de Playground, communes aux scénarios.

    Frottements et inerties des articulations, masses des segments, centre de
    masse du tronc et posture de départ.
    """
    keys = jax.random.split(rng, 5)
    return {
        "dof_frictionloss": model.dof_frictionloss.at[6:].multiply(
            jax.random.uniform(keys[0], (12,), minval=0.9, maxval=1.1)
        ),
        "dof_armature": model.dof_armature.at[6:].multiply(
            jax.random.uniform(keys[1], (12,), minval=1.0, maxval=1.05)
        ),
        "body_ipos": model.body_ipos.at[TORSO_BODY_ID].add(
            jax.random.uniform(keys[2], (3,), minval=-0.05, maxval=0.05)
        ),
        "body_mass": model.body_mass
        * jax.random.uniform(keys[3], (model.nbody,), minval=0.9, maxval=1.1),
        "qpos0": model.qpos0.at[7:].add(
            jax.random.uniform(keys[4], (12,), minval=-0.05, maxval=0.05)
        ),
    }


def randomizer(
    sample: Callable[[mjx.Model, jax.Array], dict[str, jax.Array]],
) -> Callable[[mjx.Model, jax.Array], tuple[mjx.Model, mjx.Model]]:
    """Randomiseur au format de Brax à partir de `sample` (un environnement)."""

    def randomize(model: mjx.Model, rng: jax.Array):
        values = jax.vmap(lambda key: sample(model, key))(rng)
        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({field: 0 for field in values})
        return model.tree_replace(values), in_axes

    return randomize
