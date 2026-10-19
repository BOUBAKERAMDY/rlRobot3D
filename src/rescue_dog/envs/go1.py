"""Construction de l'environnement Go1 de MuJoCo Playground à partir de notre config.

Observations : un dict de deux vecteurs.
- "state" (48), lu par l'acteur : ce que le vrai robot peut mesurer, avec du bruit.
  Vitesse linéaire du corps (3), gyroscope (3), direction de la gravité (3),
  angles articulaires moins la posture debout (12), vitesses articulaires (12),
  action précédente (12), commande joystick : avant, côté, rotation (3).
- "privileged_state" (123), lu par le critique privilégié : "state" plus les mêmes
  mesures sans bruit, l'accéléromètre, les forces des moteurs, les contacts et
  vitesses des pieds, le temps en l'air des pieds et les poussées externes.

Actions : 12 valeurs dans [-1, 1], une par articulation (hanche, cuisse, genou
pour chacune des 4 pattes). Chaque valeur est un décalage d'angle cible par
rapport à la posture debout : q_cible = q_debout + 0.5 * a.
"""

from __future__ import annotations

from ml_collections import config_dict
from mujoco_playground import MjxEnv, registry

from rescue_dog.utils.config import Config


def make_env(cfg: Config) -> tuple[MjxEnv, config_dict.ConfigDict]:
    """Charge l'environnement `cfg.env_name` avec nos réglages par-dessus ceux de Playground."""
    env_cfg = registry.get_default_config(cfg.env_name)
    env_cfg.episode_length = cfg.episode_length
    env_cfg.impl = cfg.impl
    env = registry.load(cfg.env_name, config=env_cfg)
    return env, env_cfg


def make_eval_env(cfg: Config) -> tuple[MjxEnv, config_dict.ConfigDict]:
    """Même environnement que `make_env`, dans une instance séparée pour l'évaluation."""
    return make_env(cfg)
