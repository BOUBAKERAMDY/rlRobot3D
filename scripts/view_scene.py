"""Affiche le décor d'un scénario, le chien debout dans sa posture "home".

python scripts/view_scene.py sol_glissant                # visionneuse interactive
python scripts/view_scene.py patte_blessee --leg RL
python scripts/view_scene.py --png results/scenes       # images de tous les scénarios
"""

import argparse
from pathlib import Path

import mediapy
import mujoco
import mujoco.viewer

from rescue_dog.visualization.scenes import (
    LEGS,
    SCENARIOS,
    home_qpos,
    render_frames,
    scene_model,
)

CAMERAS = ("cinema", "panorama", "large", "rase")

parser = argparse.ArgumentParser()
parser.add_argument("scenario", nargs="?", default="normal", choices=SCENARIOS)
parser.add_argument("--leg", default="FR", choices=LEGS, help="patte blessée")
parser.add_argument(
    "--png",
    type=Path,
    help="dossier où écrire les images au lieu d'ouvrir la visionneuse",
)
args = parser.parse_args()

if args.png is not None:
    args.png.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        model = scene_model(scenario, injured_leg=args.leg)
        for cam in CAMERAS:
            frame = render_frames(
                model, [home_qpos(model)], camera=cam, height=720, width=1280
            )[0]
            mediapy.write_image(args.png / f"{scenario}_{cam}.png", frame)
    print(f"Images : {args.png}")
else:
    model = scene_model(args.scenario, injured_leg=args.leg)
    data = mujoco.MjData(model)
    key = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = model.key_ctrl[key]  # les moteurs tiennent la posture debout
    mujoco.viewer.launch(model, data)
