"""Vidéo d'une politique entraînée qui suit une commande joystick fixe."""

from __future__ import annotations

import shutil
from pathlib import Path

import imageio_ffmpeg
import jax
import jax.numpy as jp
import mediapy
import numpy as np

from rescue_dog.envs.go1 import make_eval_env
from rescue_dog.training.ppo import load_policy
from rescue_dog.utils.config import Config

# Sans ffmpeg installé (ex. sur Mac), on prend celui fourni par imageio-ffmpeg.
if shutil.which("ffmpeg") is None:
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


def record_video(
    cfg: Config,
    checkpoint_dir: Path,
    out_path: Path,
    command: tuple[float, float, float] = (1.0, 0.0, 0.0),
    steps: int = 500,
    render_every: int = 2,
) -> None:
    """Joue la politique de `checkpoint_dir` pendant `steps` pas et écrit un mp4.

    `command` = (avant m/s, côté m/s, rotation rad/s). On la réimpose à chaque pas,
    car l'environnement en tire une nouvelle au hasard en cours d'épisode.
    """
    env, _ = make_eval_env(cfg)
    policy = jax.jit(
        load_policy(cfg, checkpoint_dir, env.observation_size, env.action_size)
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    cmd = jp.array(command)

    rng = jax.random.PRNGKey(cfg.seed)
    state = reset(rng)
    state.info["command"] = cmd
    trajectory = [state]
    for _ in range(steps):
        rng, key = jax.random.split(rng)
        action, _ = policy(state.obs, key)
        state = step(state, action)
        state.info["command"] = cmd
        trajectory.append(state)

    frames = env.render(
        trajectory[::render_every], height=480, width=640, camera="track"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(out_path, np.asarray(frames), fps=1.0 / env.dt / render_every)
