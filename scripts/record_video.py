"""Filme la politique du dernier checkpoint d'un run.

python scripts/record_video.py results/go1_baseline
python scripts/record_video.py results/go1_baseline --command 0.5 0 0.8
python scripts/record_video.py results/go1_baseline --scenario sol_glissant --camera cinema

Dans une condition de test physique (le décor suit, avec une légende) :
python scripts/record_video.py results/glace_D --test glace friction=0.1 payload=5
python scripts/record_video.py results/patte_A --test patte leg=FL strength=0.2
"""

import argparse
import json
from pathlib import Path

from rescue_dog.scenarios import SCENARIOS as TESTS
from rescue_dog.scenarios import parse_condition
from rescue_dog.training.ppo import latest_checkpoint
from rescue_dog.utils.config import config_from_dict
from rescue_dog.visualization.scenes import SCENARIOS
from rescue_dog.visualization.video import describe, record_video

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument(
    "--command",
    type=float,
    nargs=3,
    default=[1.0, 0.0, 0.0],
    metavar=("AVANT", "COTE", "ROTATION"),
)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument(
    "--scenario", choices=SCENARIOS, help="décor de la vidéo (scène nue sinon)"
)
parser.add_argument(
    "--test",
    nargs="+",
    metavar=("SCENARIO", "CLE=VALEUR"),
    help=f"condition de test physique : {', '.join(TESTS)} puis ses paramètres",
)
parser.add_argument(
    "--camera", default=None, help="track, cinema, panorama, large, rase..."
)
args = parser.parse_args()

cfg = config_from_dict(json.loads((args.run_dir / "config.json").read_text()))
ckpt = latest_checkpoint(args.run_dir.resolve())
test, condition, caption = None, None, None
if args.test:
    test = args.test[0]
    if test not in TESTS:
        parser.error(f"scénario de test inconnu {test!r}, choisir parmi {list(TESTS)}")
    condition = parse_condition(test, args.test[1:])
    caption = [args.run_dir.name, describe(test, condition)]
    suffix = "_".join(
        f"{k}{v:g}" if isinstance(v, float) else f"{k}{v}" for k, v in condition.items()
    )
    out = args.run_dir / f"test_{test}_{suffix}.mp4"
elif args.scenario:
    out = args.run_dir / f"walk_{args.scenario}.mp4"
else:
    out = args.run_dir / "walk.mp4"
camera = args.camera or ("cinema" if args.test else "track")
run = record_video(
    cfg,
    ckpt,
    out,
    command=tuple(args.command),
    steps=args.steps,
    scenario=args.scenario,
    camera=camera,
    test=test,
    condition=condition,
    caption=caption,
)
fall = (
    "aucune chute"
    if run.fall_step is None
    else f"chute à {run.fall_step * run.dt:.1f} s"
)
print(f"Vidéo : {out} (checkpoint {ckpt.name}, {fall})")
