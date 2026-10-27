"""Évaluation d'une politique figée sur la grille de test d'un scénario.

Pour chaque condition ξ de la grille, on joue `n_episodes` épisodes de
`steps` pas (1000 pas = 20 s) en parallèle, avec une commande joystick fixe.
Un épisode est une chute dès que l'environnement le termine (robot retourné)
ou que le tronc descend sous `FALL_HEIGHT`. Après une chute, les pas suivants
ne comptent plus.

Métriques par condition :
- `success` : part des épisodes sans chute pendant toute la durée ;
- `falls` : nombre moyen de chutes par épisode (0 ou 1 ici) ;
- `velocity_error` : écart moyen (m/s) entre la vitesse du tronc dans le plan
  et la commande, sur les pas avant la chute ;
- `return` : somme moyenne des récompenses avant la chute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

from rescue_dog.envs.go1 import make_eval_env
from rescue_dog.scenarios import SCENARIOS
from rescue_dog.training.ppo import load_policy
from rescue_dog.utils.config import Config

FALL_HEIGHT = 0.15  # m ; le tronc est à ~0.28 m debout


def make_rollout(cfg: Config, checkpoint_dir: Path, steps: int):
    """Fonction jittée `(modèle, rngs, commande) -> métriques par épisode`.

    Le modèle est un argument : une seule compilation sert pour toute la grille.
    On le substitue à celui de l'environnement comme le fait le wrapper de
    randomisation de Brax.
    """
    env, _ = make_eval_env(cfg)
    policy = load_policy(cfg, checkpoint_dir, env.observation_size, env.action_size)
    default_model = env.mjx_model

    def episode(rng, command):
        rng, key = jax.random.split(rng)
        state = env.reset(key)
        state.info["command"] = command

        def body(carry, _):
            state, rng, alive = carry
            rng, key = jax.random.split(rng)
            action, _ = policy(state.obs, key)
            state = env.step(state, action)
            state.info["command"] = command
            fell = (state.done > 0) | (state.data.qpos[2] < FALL_HEIGHT)
            alive_after = alive & ~fell
            vel = env.get_local_linvel(state.data)[:2]
            out = (
                alive * state.reward,
                alive * jp.linalg.norm(vel - command[:2]),
                alive.astype(jp.float32),
            )
            return (state, rng, alive_after), out

        (_, _, alive), (rewards, errors, counted) = jax.lax.scan(
            body, (state, rng, jp.array(True)), None, length=steps
        )
        return {
            "success": alive.astype(jp.float32),
            "return": rewards.sum(),
            "velocity_error": errors.sum() / jp.maximum(counted.sum(), 1.0),
        }

    @jax.jit
    def rollout(model, rngs, command):
        env.unwrapped._mjx_model = model
        try:
            return jax.vmap(episode, in_axes=(0, None))(rngs, command)
        finally:
            env.unwrapped._mjx_model = default_model

    return rollout, default_model


def evaluate(
    cfg: Config,
    checkpoint_dir: Path,
    scenario: str,
    command: tuple[float, float, float] = (0.5, 0.0, 0.0),
    n_episodes: int = 32,
    steps: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Métriques de la politique sur chaque condition de la grille de `scenario`.

    Le scénario de test peut différer de celui de l'entraînement : l'acteur ne
    lit jamais ξ, seul le modèle physique change.
    """
    module = SCENARIOS[scenario]
    rollout, model = make_rollout(cfg, checkpoint_dir, steps)
    rngs = jax.random.split(jax.random.PRNGKey(seed), n_episodes)
    cmd = jp.array(command)
    rows = []
    for condition in module.grid():
        out = rollout(module.set_condition(model, **condition), rngs, cmd)
        success = float(np.asarray(out["success"]).mean())
        rows.append(
            {
                **condition,
                "success": success,
                "falls": 1.0 - success,
                "velocity_error": float(np.asarray(out["velocity_error"]).mean()),
                "return": float(np.asarray(out["return"]).mean()),
            }
        )
    return rows
