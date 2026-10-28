"""Filme plusieurs runs côte à côte dans la même condition de test.

python scripts/duel_video.py A=results/glace_A D=results/glace_D --test glace friction=0.1 payload=5
python scripts/duel_video.py A=results/patte_A B=results/patte_B C=results/patte_C D=results/patte_D \\
    --test patte leg=FL strength=0.2 --camera panorama

Même état de départ et même commande pour tous ; chaque vignette affiche son
étiquette, la condition, puis « CHUTE » au moment où le chien tombe.
Par défaut, écrit dans results/comparaison/.
"""

import argparse
import json
from pathlib import Path

from rescue_dog.scenarios import SCENARIOS as TESTS
from rescue_dog.scenarios import parse_condition
from rescue_dog.training.ppo import latest_checkpoint
from rescue_dog.utils.config import config_from_dict
from rescue_dog.visualization.video import record_duel

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+", metavar="ETIQUETTE=DOSSIER")
parser.add_argument(
    "--test", nargs="+", required=True, metavar=("SCENARIO", "CLE=VALEUR")
)
parser.add_argument(
    "--command",
    type=float,
    nargs=3,
    default=[1.0, 0.0, 0.0],
    metavar=("AVANT", "COTE", "ROTATION"),
)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--camera", default="cinema")
parser.add_argument("--out", type=Path)
args = parser.parse_args()

test = args.test[0]
if test not in TESTS:
    parser.error(f"scénario de test inconnu {test!r}, choisir parmi {list(TESTS)}")
condition = parse_condition(test, args.test[1:])

runs = []
for item in args.runs:
    label, sep, path = item.partition("=")
    if not sep:
        parser.error(f"{item!r} : attendu ETIQUETTE=DOSSIER, ex. A=results/glace_A")
    run_dir = Path(path)
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text()))
    runs.append((label, cfg, latest_checkpoint(run_dir.resolve())))

suffix = "_".join(
    f"{k}{v:g}" if isinstance(v, float) else f"{k}{v}" for k, v in condition.items()
)
labels = "".join(label for label, _, _ in runs)
out = args.out or Path("results/comparaison") / f"duel_{labels}_{test}_{suffix}.mp4"
results = record_duel(
    runs, out, test, condition, tuple(args.command), args.steps, camera=args.camera
)
for (label, _, _), run in zip(runs, results):
    fall = (
        "debout" if run.fall_step is None else f"chute à {run.fall_step * run.dt:.1f} s"
    )
    print(f"{label} : {fall}")
print(f"Vidéo : {out}")
