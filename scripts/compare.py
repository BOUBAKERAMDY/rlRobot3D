"""Compare les politiques A/B/C/D évaluées sur un même scénario de test.

python scripts/compare.py glace A=results/glace_A B=results/glace_B \\
    C=results/glace_C D=results/glace_D
python scripts/compare.py patte A=results/patte_A_s0,results/patte_A_s1 ...

Chaque run doit avoir son `eval_<scénario>.json` (scripts/evaluate.py). Avec
plusieurs graines (dossiers séparés par des virgules), les taux sont moyennés
et l'écart-type entre graines est affiché.

Écrit dans results/comparaison/ :
Métrique principale (--metric) : « mission » par défaut, debout pendant 20 s
ET au moins la moitié de la distance demandée parcourue ; « success » =
seulement debout.

- heatmaps_<scénario>.png : la métrique par condition, une carte par politique,
  le domaine d'entraînement encadré en pointillé ;
- classement_<scénario>.png : réussite moyenne, toutes conditions et hors
  entraînement ;
- resume_<scénario>.csv : les mêmes chiffres, plus chutes, erreur de vitesse
  et retour.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from rescue_dog.scenarios import SCENARIOS, in_training

LABELS = {
    "friction": "friction du sol μ",
    "payload": "charge ajoutée (kg)",
    "leg": "patte blessée",
    "strength": "force restante η",
}
METRIC_NAMES = {"mission": "mission réussie", "success": "debout 20 s"}
ALL, OUTSIDE = "#1F78B4", "#E0641E"  # palette validée (daltonisme, contraste)
INK, MUTED = "#13202A", "#5B6670"

parser = argparse.ArgumentParser()
parser.add_argument("scenario", choices=SCENARIOS)
parser.add_argument("policies", nargs="+", metavar="ETIQUETTE=DOSSIER[,DOSSIER...]")
parser.add_argument("--out", type=Path, default=Path("results/comparaison"))
parser.add_argument(
    "--metric",
    choices=("mission", "success"),
    default="mission",
    help="mission : debout 20 s et avance ; success : seulement debout 20 s",
)
args = parser.parse_args()

METRIC_NAME = METRIC_NAMES[args.metric]
module = SCENARIOS[args.scenario]
row_key, row_values, col_key, col_values = module.AXES
conditions = module.grid()
outside = np.array([not in_training(args.scenario, c) for c in conditions])

# Taux par politique : un tableau (graines × conditions) par métrique.
policies = {}
for item in args.policies:
    label, sep, dirs = item.partition("=")
    if not sep:
        parser.error(f"{item!r} : attendu ETIQUETTE=DOSSIER, ex. A=results/glace_A")
    seeds = []
    for d in dirs.split(","):
        path = Path(d) / f"eval_{args.scenario}.json"
        if not path.exists():
            parser.error(f"{path} manquant : lancer scripts/evaluate.py {d}")
        rows = json.loads(path.read_text())["conditions"]
        if "mission" not in rows[0]:
            parser.error(
                f"{path} : évaluation ancienne, relancer scripts/evaluate.py {d}"
            )
        if [{k: r[k] for k in (row_key, col_key)} for r in rows] != conditions:
            parser.error(f"{path} : grille différente de celle du scénario")
        seeds.append(rows)
    policies[label] = {
        m: np.array([[r[m] for r in rows] for rows in seeds])
        for m in ("success", "mission", "progress", "falls", "velocity_error", "return")
    }

args.out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": INK})

# --- Heatmaps côte à côte ---
n = len(policies)
fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 1, 4.2), squeeze=False)
train_cells = np.argwhere(~outside.reshape(len(row_values), len(col_values)))
for ax, (label, metrics) in zip(axes[0], policies.items()):
    grid = metrics[args.metric].mean(axis=0).reshape(len(row_values), -1) * 100
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=100, origin="lower")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            color = "white" if grid[i, j] > 60 else INK
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", color=color)
    (r0, c0), (r1, c1) = train_cells.min(axis=0), train_cells.max(axis=0)
    ax.add_patch(
        Rectangle(
            (c0 - 0.5, r0 - 0.5),
            c1 - c0 + 1,
            r1 - r0 + 1,
            fill=False,
            linestyle="--",
            linewidth=2,
            edgecolor=OUTSIDE,
        )
    )
    ax.set_title(label, fontweight="bold")
    ax.set_xticks(range(len(col_values)), [str(v) for v in col_values])
    ax.set_yticks(range(len(row_values)), [str(v) for v in row_values])
    ax.set_xlabel(LABELS[col_key])
    ax.tick_params(colors=MUTED)
axes[0][0].set_ylabel(LABELS[row_key])
fig.colorbar(im, ax=axes[0].tolist(), label=f"{METRIC_NAME} (%)", shrink=0.85)
fig.suptitle(
    f"Scénario {args.scenario} : {METRIC_NAME} par condition "
    "(cadre pointillé = plages vues à l'entraînement)",
    color=INK,
)
fig.savefig(args.out / f"heatmaps_{args.scenario}.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# --- Classement : réussite moyenne, toutes conditions et hors entraînement ---
summary = []
for label, m in policies.items():
    per_seed_all = m[args.metric].mean(axis=1) * 100
    per_seed_out = m[args.metric][:, outside].mean(axis=1) * 100
    summary.append(
        {
            "politique": label,
            "graines": len(per_seed_all),
            "reussite_%": per_seed_all.mean(),
            "reussite_std": per_seed_all.std(),
            "hors_entrainement_%": per_seed_out.mean(),
            "hors_entrainement_std": per_seed_out.std(),
            "debout_20s_%": m["success"].mean() * 100,
            "progression_%": m["progress"].mean() * 100,
            "chutes": m["falls"].mean(),
            "erreur_vitesse_m_s": m["velocity_error"].mean(),
            "retour": m["return"].mean(),
        }
    )

fig, ax = plt.subplots(figsize=(8, 0.9 * n + 1.4))
y = np.arange(n)[::-1]
h = 0.36
for offset, key, color, name in (
    (h / 2, "reussite_%", ALL, "toutes conditions"),
    (-h / 2, "hors_entrainement_%", OUTSIDE, "hors entraînement"),
):
    values = [s[key] for s in summary]
    errors = [s[key.replace("_%", "_std")] for s in summary]
    has_err = any(s["graines"] > 1 for s in summary)
    ax.barh(
        y + offset,
        values,
        height=h - 0.04,
        color=color,
        label=name,
        xerr=errors if has_err else None,
        error_kw={"ecolor": MUTED, "capsize": 3},
    )
    for yi, v in zip(y + offset, values):
        ax.text(v + 1.2, yi, f"{v:.0f} %", va="center", color=INK, fontsize=10)
ax.set_yticks(y, [s["politique"] for s in summary])
ax.set_xlim(0, 110)
ax.set_xlabel(f"{METRIC_NAME}, moyenne sur la grille (%)")
ax.grid(axis="x", color="#E3E6E8", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
ax.set_title(f"Scénario {args.scenario} : classement des politiques", color=INK)
fig.savefig(args.out / f"classement_{args.scenario}.png", dpi=160, bbox_inches="tight")
plt.close(fig)

with open(args.out / f"resume_{args.scenario}.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

print(f"Métrique : {METRIC_NAME}")
print(
    f"{'politique':<12}{'réussite':>10}{'hors entr.':>12}{'debout':>8}"
    f"{'avancée':>9}{'chutes':>8}{'err. vit.':>11}"
)
for s in summary:
    print(
        f"{s['politique']:<12}{s['reussite_%']:>9.0f}%{s['hors_entrainement_%']:>11.0f}%"
        f"{s['debout_20s_%']:>7.0f}%{s['progression_%']:>8.0f}%"
        f"{s['chutes']:>8.2f}{s['erreur_vitesse_m_s']:>11.3f}"
    )
print(f"Figures et tableau : {args.out}/")
