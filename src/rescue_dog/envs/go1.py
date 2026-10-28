"""Construction de l'environnement Go1 de MuJoCo Playground à partir de notre config.

Observations : un dict de deux vecteurs.
- "state" (48), lu par l'acteur : ce que le vrai robot peut mesurer, avec du bruit.
  Vitesse linéaire du corps (3), gyroscope (3), direction de la gravité (3),
  angles articulaires moins la posture debout (12), vitesses articulaires (12),
  action précédente (12), commande joystick : avant, côté, rotation (3).
- "privileged_state" (123), lu par le critique privilégié : "state" plus les mêmes
  mesures sans bruit, l'accéléromètre, les forces des moteurs, les contacts et
  vitesses des pieds, le temps en l'air des pieds et les poussées externes.
  Avec un scénario, on y ajoute ses paramètres ξ : glace, friction du sol et
  charge sur le dos (125 valeurs) ; patte, force restante de chaque patte (127) ;
  tous, les deux (129).

Actions : 12 valeurs dans [-1, 1], une par articulation (hanche, cuisse, genou
pour chacune des 4 pattes). Chaque valeur est un décalage d'angle cible par
rapport à la posture debout : q_cible = q_debout + 0.5 * a.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import MjxEnv, registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import joystick

from rescue_dog.scenarios import TRAINING
from rescue_dog.utils.config import Config

TASKS = {"Go1JoystickFlatTerrain": "flat_terrain"}


class RescueJoystick(joystick.Joystick):
    """Joystick de Playground, avec le ξ du scénario ajouté à l'état privilégié.

    `self.mjx_model` est le modèle randomisé de l'environnement courant (le
    wrapper de randomisation de Brax le remplace pendant chaque pas), donc ξ
    se lit directement dans le modèle, par comparaison au modèle nominal.
    """

    def __init__(self, scenario: ModuleType, **kwargs: Any):
        self._scenario = scenario
        super().__init__(**kwargs)

    def _post_init(self) -> None:
        super()._post_init()
        self._nominal_model = self._mjx_model

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        obs = super()._get_obs(data, info)
        xi = self._scenario.privileged_params(self.mjx_model, self._nominal_model)
        obs["privileged_state"] = jax.numpy.hstack([obs["privileged_state"], xi])
        return obs


def make_env(cfg: Config) -> tuple[MjxEnv, config_dict.ConfigDict]:
    """Charge l'environnement `cfg.env_name` avec nos réglages par-dessus ceux de Playground.

    Sans scénario, c'est l'environnement de Playground tel quel.
    """
    env_cfg = registry.get_default_config(cfg.env_name)
    env_cfg.episode_length = cfg.episode_length
    env_cfg.impl = cfg.impl
    if cfg.scenario == "aucun":
        return registry.load(cfg.env_name, config=env_cfg), env_cfg
    if cfg.scenario not in TRAINING:
        raise ValueError(
            f"Scénario inconnu : {cfg.scenario!r}, choisir aucun, "
            + ", ".join(TRAINING)
        )
    if cfg.env_name not in TASKS:
        raise ValueError(
            f"Scénario non géré pour {cfg.env_name}, choisir parmi {list(TASKS)}"
        )
    # registry.load télécharge Menagerie (maillages du Go1) au premier usage ;
    # on construit l'environnement nous-mêmes, donc on le fait ici.
    mjx_env.ensure_menagerie_exists()
    env = RescueJoystick(
        TRAINING[cfg.scenario], task=TASKS[cfg.env_name], config=env_cfg
    )
    return env, env_cfg


def make_eval_env(cfg: Config) -> tuple[MjxEnv, config_dict.ConfigDict]:
    """Même environnement que `make_env`, dans une instance séparée pour l'évaluation."""
    return make_env(cfg)
