"""Lance un entraînement PPO du Go1.

python scripts/train.py                       # configs/base.yaml tel quel
python scripts/train.py seed=3 num_timesteps=50_000_000
python scripts/train.py --config configs/autre.yaml run_name=essai
"""

import argparse

from rescue_dog.training.ppo import train
from rescue_dog.utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/base.yaml")
parser.add_argument(
    "overrides", nargs="*", help="clé=valeur, ex. ppo.learning_rate=1e-4"
)
args = parser.parse_args()

cfg = load_config(args.config, args.overrides)
out = train(cfg)
print(f"Terminé : {out}")
