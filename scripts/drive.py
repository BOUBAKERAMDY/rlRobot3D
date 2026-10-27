"""Pilote au clavier le chien d'un run entraîné, en temps réel.

Sur Mac, la visionneuse passive de MuJoCo doit être lancée avec mjpython :

mjpython scripts/drive.py results/go1_mac               (avec `uv run` devant)
uv run mjpython scripts/drive.py results/go1_mac --scenario sol_glissant

Touches (cliquer d'abord dans la fenêtre) :
  flèche haut / bas      accélérer / ralentir (marche arrière en dessous de 0)
  flèche gauche / droite tourner à gauche / à droite
  Page préc. / Page suiv. pas chassé à gauche / à droite
  Entrée                 tout arrêter
Le chien repart de sa position de départ s'il tombe.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np

from rescue_dog.envs.go1 import make_eval_env
from rescue_dog.training.ppo import latest_checkpoint, load_policy
from rescue_dog.utils.config import config_from_dict
from rescue_dog.visualization.scenes import SCENARIOS, scene_model

# Codes GLFW des touches, et effet sur la commande (avant, côté, rotation).
KEYS = {
    265: (0.25, 0.0, 0.0),  # haut
    264: (-0.25, 0.0, 0.0),  # bas
    263: (0.0, 0.0, 0.3),  # gauche
    262: (0.0, 0.0, -0.3),  # droite
    266: (0.0, 0.2, 0.0),  # page précédente
    267: (0.0, -0.2, 0.0),  # page suivante
}
STOP_KEYS = (257, 335)  # Entrée, Entrée du pavé numérique
# Bornes des commandes vues à l'entraînement.
LIMITS = np.array([1.5, 0.8, 1.2])

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--scenario", default="normal", choices=SCENARIOS)
args = parser.parse_args()

cfg = config_from_dict(json.loads((args.run_dir / "config.json").read_text()))
ckpt = latest_checkpoint(args.run_dir.resolve())
env, _ = make_eval_env(cfg)
policy = jax.jit(load_policy(cfg, ckpt, env.observation_size, env.action_size))
reset = jax.jit(env.reset)
step = jax.jit(env.step)

command = np.zeros(3)


def on_key(key: int) -> None:
    if key in KEYS:
        command[:] = np.clip(command + KEYS[key], -LIMITS, LIMITS)
    elif key in STOP_KEYS:
        command[:] = 0.0


model = scene_model(args.scenario)
data = mujoco.MjData(model)
rng = jax.random.PRNGKey(cfg.seed)
state = reset(rng)

print("Compilation de la politique (environ 1 min)...")
action, _ = policy(state.obs, rng)
step(state, action)
print(f"Prêt (checkpoint {ckpt.name}). Clique dans la fenêtre et utilise les flèches.")

with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("trunk").id
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -20
    shown = None
    while viewer.is_running():
        tic = time.monotonic()
        state.info["command"] = jp.array(command)
        rng, key = jax.random.split(rng)
        action, _ = policy(state.obs, key)
        state = step(state, action)
        if state.done:
            rng, key = jax.random.split(rng)
            state = reset(key)

        qpos = np.asarray(state.data.qpos)
        with viewer.lock():
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
        viewer.sync()
        # Hors du verrou : set_texts attend la fenêtre, qui a besoin du verrou.
        if shown is None or not np.array_equal(shown, command):
            shown = command.copy()
            viewer.set_texts(
                (
                    None,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "avant\ncôté\nrotation",
                    (
                        f"{command[0]:+.2f} m/s\n{command[1]:+.2f} m/s\n"
                        f"{command[2]:+.2f} rad/s"
                    ),
                )
            )
        time.sleep(max(0.0, env.dt - (time.monotonic() - tic)))
