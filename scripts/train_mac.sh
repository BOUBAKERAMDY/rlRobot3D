#!/usr/bin/env bash
# Entraîne le Go1 à marcher sur sol plat (scénario normal) sur un Mac, puis
# enregistre une vidéo du chien dans le décor "normal".
#
# JAX tourne sur le CPU du Mac : on réduit le nombre d'environnements parallèles
# (8192 est prévu pour un GPU). 200M pas prennent environ 3 h sur un M5.
#
# scripts/train_mac.sh                              # run "go1_mac", 200M pas
# scripts/train_mac.sh run_name=essai num_timesteps=50_000_000
#
# Les arguments sont des overrides de la config (clé=valeur), comme train.py.

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME=go1_mac
OUTPUT_DIR=results
for arg in "$@"; do
  case "$arg" in
    run_name=*) RUN_NAME="${arg#run_name=}" ;;
    output_dir=*) OUTPUT_DIR="${arg#output_dir=}" ;;
  esac
done
RUN_DIR="${OUTPUT_DIR%/}/${RUN_NAME}"
LOG="${RUN_DIR}.log"
mkdir -p "$OUTPUT_DIR"

echo "Entraînement : ${RUN_DIR} (journal : ${LOG})"
# caffeinate empêche la mise en veille du Mac pendant l'entraînement.
caffeinate -i uv run python scripts/train.py \
  run_name="${RUN_NAME}" num_envs=1024 "$@" 2>&1 | tee "$LOG"

echo "Vidéo de la politique entraînée dans le décor normal..."
uv run python scripts/record_video.py "$RUN_DIR" \
  --scenario normal --camera cinema
