# Chien de sauvetage RL — robot quadrupède robuste

Ce projet entraîne en simulation un robot quadrupède avec la randomisation de domaine et l'acteur-critique asymétrique (PPO).
L'objectif est qu'il reste fonctionnel, sans réapprentissage, face à un sol glissant, une charge lourde ou une patte blessée.

**Dossiers :** `configs/` (paramètres), `scripts/` (lancement), `src/rescue_dog/` (code : envs, scenarios, training, evaluation, visualization, utils),
`tests/`, `notebooks/`, `results/`, `docs/`.

## Utilisation

```bash
uv sync --extra dev
uv run pytest -q tests

# Entraîner les politiques A (sans randomisation ni critique privilégié) et D (avec les deux)
scripts/train_mac.sh scenario=tous run_name=A domain_randomization=false privileged_critic=false
scripts/train_mac.sh scenario=tous run_name=D domain_randomization=true  privileged_critic=true

# Tester une politique figée sur les grilles glace (friction × charge) et patte (patte × force)
uv run python scripts/evaluate.py results/D

# Comparer les politiques (heatmaps, classement, CSV dans results/comparaison/)
uv run python scripts/compare.py glace A=results/A D=results/D

# Vidéos : une condition de test, ou plusieurs politiques côte à côte
uv run python scripts/record_video.py results/D --test glace friction=0.1 payload=5
uv run python scripts/duel_video.py A=results/A D=results/D --test patte leg=FL strength=0.2

# Décors et pilotage au clavier
uv run python scripts/view_scene.py sol_glissant
uv run mjpython scripts/drive.py results/D
```

Une mission est réussie quand le chien reste debout 20 s **et** parcourt au moins la moitié
de la distance demandée.
