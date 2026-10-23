"""Filme la politique du dernier checkpoint d'un run.

python scripts/record_video.py results/go1_baseline
python scripts/record_video.py results/go1_baseline --command 0.5 0 0.8
"""

import argparse
import json
from pathlib import Path

from rescue_dog.training.ppo import latest_checkpoint
from rescue_dog.utils.config import config_from_dict
from rescue_dog.visualization.video import record_video

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument(
    "--command",
    type=float,
    nargs=3,
    default=[1.0, 0.0, 0.0],
    metavar=("AVANT", "COTE", "ROTATION"),
)
parser.add_argument("--steps", type=int, default=500)
args = parser.parse_args()

cfg = config_from_dict(json.loads((args.run_dir / "config.json").read_text()))
ckpt = latest_checkpoint(args.run_dir.resolve())
out = args.run_dir / "walk.mp4"
record_video(cfg, ckpt, out, command=tuple(args.command), steps=args.steps)
print(f"Vidéo : {out} (checkpoint {ckpt.name})")
