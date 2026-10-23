"""Entraînement PPO (Brax) du Go1 à partir de notre config.

Les deux interrupteurs de la config choisissent la politique (A/B/C/D) :
- `domain_randomization` : friction, masses et frottements tirés à chaque
  environnement avec le randomiseur de Playground.
- `privileged_critic` : le critique lit "privileged_state" au lieu de "state".
L'acteur lit toujours "state".
"""

from __future__ import annotations

import functools
from typing import Any

from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry, wrapper

from rescue_dog.utils.config import Config


def make_network_factory(cfg: Config) -> functools.partial:
    """Réseaux acteur et critique ; le critique lit l'état privilégié si demandé."""
    return functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=cfg.ppo.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=cfg.ppo.value_hidden_layer_sizes,
        policy_obs_key="state",
        value_obs_key="privileged_state" if cfg.privileged_critic else "state",
    )


def make_train_kwargs(cfg: Config) -> dict[str, Any]:
    """Arguments de `brax...ppo.train`, sans l'environnement ni les callbacks."""
    p = cfg.ppo
    randomization_fn = (
        registry.get_domain_randomizer(cfg.env_name)
        if cfg.domain_randomization
        else None
    )
    return {
        "num_timesteps": cfg.num_timesteps,
        "num_envs": cfg.num_envs,
        "num_evals": cfg.num_evals,
        "episode_length": cfg.episode_length,
        "seed": cfg.seed,
        "learning_rate": p.learning_rate,
        "entropy_cost": p.entropy_cost,
        "discounting": p.discounting,
        "unroll_length": p.unroll_length,
        "batch_size": p.batch_size,
        "num_minibatches": p.num_minibatches,
        "num_updates_per_batch": p.num_updates_per_batch,
        "num_resets_per_eval": p.num_resets_per_eval,
        "reward_scaling": p.reward_scaling,
        "normalize_observations": p.normalize_observations,
        "max_grad_norm": p.max_grad_norm,
        "action_repeat": p.action_repeat,
        "network_factory": make_network_factory(cfg),
        "randomization_fn": randomization_fn,
        "wrap_env_fn": wrapper.wrap_for_brax_training,
    }
