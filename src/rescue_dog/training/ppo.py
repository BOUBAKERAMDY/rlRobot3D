"""Entraînement PPO (Brax) du Go1 à partir de notre config.

Les deux interrupteurs de la config choisissent la politique (A/B/C/D) :
- `domain_randomization` : friction, masses et frottements tirés à chaque
  environnement, avec le randomiseur du scénario (`scenarios/`) ou, sans
  scénario, celui de Playground.
- `privileged_critic` : le critique lit "privileged_state" au lieu de "state".
L'acteur lit toujours "state".
"""

from __future__ import annotations

import dataclasses
import functools
import json
import time
from pathlib import Path
from typing import Any

from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper

from rescue_dog.envs.go1 import make_env, make_eval_env
from rescue_dog.scenarios import TRAINING
from rescue_dog.utils.config import Config


def run_dir(cfg: Config) -> Path:
    """Dossier du run : `output_dir/run_name`, en absolu (orbax refuse les chemins relatifs)."""
    return (Path(cfg.output_dir) / cfg.run_name).resolve()


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
    if not cfg.domain_randomization:
        randomization_fn = None
    elif cfg.scenario in TRAINING:
        randomization_fn = TRAINING[cfg.scenario].randomize
    else:
        randomization_fn = registry.get_domain_randomizer(cfg.env_name)
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


def train(cfg: Config) -> Path:
    """Entraîne une politique et renvoie le dossier du run.

    Le dossier contient `config.json` (la config utilisée), `metrics.jsonl`
    (une ligne par évaluation) et `checkpoints/<pas>/` (une par évaluation).
    """
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    metrics_path = out / "metrics.jsonl"
    metrics_path.write_text("")

    env, _ = make_env(cfg)
    eval_env, _ = make_eval_env(cfg)

    start = time.monotonic()

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:
        row = {"step": int(num_steps), "time": time.monotonic() - start}
        row.update({k: float(v) for k, v in metrics.items()})
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[{row['time'] / 60:6.1f} min] pas {num_steps:>12,} "
            f"récompense {row.get('eval/episode_reward', float('nan')):8.2f}",
            flush=True,
        )

    ppo.train(
        environment=env,
        eval_env=eval_env,
        progress_fn=progress,
        save_checkpoint_path=out / "checkpoints",
        **make_train_kwargs(cfg),
    )
    return out


def latest_checkpoint(out: Path) -> Path:
    """Checkpoint du dernier pas enregistré dans `out/checkpoints`."""
    steps = sorted((out / "checkpoints").iterdir(), key=lambda p: int(p.name))
    if not steps:
        raise FileNotFoundError(f"Aucun checkpoint dans {out / 'checkpoints'}")
    return steps[-1]


def load_policy(
    cfg: Config, checkpoint_dir: Path, observation_size: Any, action_size: int
):
    """Politique déterministe `(obs, rng) -> (action, extras)` d'un checkpoint.

    On reconstruit le réseau depuis notre config plutôt qu'avec
    `brax...checkpoint.load_policy`, qui plante dans brax 0.14.2 en relisant
    son propre `ppo_network_config.json` (initialiseur `null`).
    """
    preprocess = (
        running_statistics.normalize
        if cfg.ppo.normalize_observations
        else types.identity_observation_preprocessor
    )
    network = make_network_factory(cfg)(
        observation_size, action_size, preprocess_observations_fn=preprocess
    )
    params = checkpoint.load(checkpoint_dir.resolve())
    return ppo_networks.make_inference_fn(network)(params, deterministic=True)
