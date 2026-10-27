"""Évalue le dernier checkpoint d'un run sur la grille de test d'un scénario.

python scripts/evaluate.py results/D                        # grilles du run
python scripts/evaluate.py results/go1_mac --scenario patte
python scripts/evaluate.py results/glace_D --episodes 64 --command 1.0 0 0

Sans `--scenario`, un run entraîné sur « tous » (ou sans scénario) est testé
sur toutes les grilles. Écrit `eval_<scénario>.json` (une ligne par condition) et `eval_<scénario>.png`
(heatmap du taux de réussite) dans le dossier du run.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rescue_dog.evaluation.evaluate import evaluate
from rescue_dog.scenarios import SCENARIOS
from rescue_dog.training.ppo import latest_checkpoint
from rescue_dog.utils.config import config_from_dict

LABELS = {
    "friction": "friction du sol μ",
    "payload": "charge ajoutée (kg)",
    "leg": "patte blessée",
    "strength": "force restante η",
}

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument(
    "--scenario",
    choices=SCENARIOS,
    help="par défaut, celui du run, ou toutes les grilles",
)
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument(
    "--command",
    type=float,
    nargs=3,
    default=[0.5, 0.0, 0.0],
    metavar=("AVANT", "COTE", "ROTATION"),
)
args = parser.parse_args()

cfg = config_from_dict(json.loads((args.run_dir / "config.json").read_text()))
ckpt = latest_checkpoint(args.run_dir.resolve())
if args.scenario:
    scenarios = [args.scenario]
elif cfg.scenario in SCENARIOS:
    scenarios = [cfg.scenario]
else:  # entraîné sur « tous » ou sans scénario : toutes les grilles
    scenarios = list(SCENARIOS)


def evaluate_one(scenario: str) -> None:
    rows = evaluate(
        cfg, ckpt, scenario, command=tuple(args.command), n_episodes=args.episodes
    )
    out = {
        "scenario": scenario,
        "checkpoint": ckpt.name,
        "command": args.command,
        "episodes": args.episodes,
        "conditions": rows,
    }
    (args.run_dir / f"eval_{scenario}.json").write_text(json.dumps(out, indent=2))

    row_key, row_values, col_key, col_values = SCENARIOS[scenario].AXES
    success = np.array([r["success"] for r in rows]).reshape(len(row_values), -1) * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(success, cmap="Blues", vmin=0, vmax=100, origin="lower")
    for i in range(success.shape[0]):
        for j in range(success.shape[1]):
            color = "white" if success[i, j] > 60 else "black"
            ax.text(j, i, f"{success[i, j]:.0f}", ha="center", va="center", color=color)
    ax.set_xticks(range(len(col_values)), [str(v) for v in col_values])
    ax.set_yticks(range(len(row_values)), [str(v) for v in row_values])
    ax.set_xlabel(LABELS[col_key])
    ax.set_ylabel(LABELS[row_key])
    ax.set_title(f"{args.run_dir.name} : réussite (%), scénario {scenario}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(args.run_dir / f"eval_{scenario}.png", dpi=150)
    plt.close(fig)

    print(f"Scénario {scenario}")
    print(f"{LABELS[row_key]} \\ {LABELS[col_key]} : " + " ".join(map(str, col_values)))
    for value, line in zip(row_values, success):
        print(f"{value!s:>8} " + " ".join(f"{v:6.0f}%" for v in line))
    print(f"Résultats : {args.run_dir / f'eval_{scenario}.json'}")


for scenario in scenarios:
    evaluate_one(scenario)
