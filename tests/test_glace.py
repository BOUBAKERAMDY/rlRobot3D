from pathlib import Path

import jax
import numpy as np
import pytest
from mujoco_playground import wrapper

from rescue_dog.envs.go1 import make_env
from rescue_dog.scenarios import glace
from rescue_dog.training.ppo import make_train_kwargs
from rescue_dog.utils.config import config_from_dict, load_config

BASE = Path(__file__).parent.parent / "configs" / "base.yaml"


def test_randomizer_follows_scenario():
    cfg = load_config(BASE, ["scenario=glace", "domain_randomization=true"])
    assert make_train_kwargs(cfg)["randomization_fn"] is glace.randomize


def test_old_runs_have_no_scenario():
    raw = load_config(BASE).__dict__ | {"ppo": load_config(BASE).ppo.__dict__}
    del raw["scenario"]
    assert config_from_dict(raw).scenario == "aucun"


def test_privileged_state_sees_randomized_xi():
    """Le critique lit la friction et la charge tirées pour chaque environnement."""
    env, _ = make_env(load_config(BASE, ["scenario=glace"]))
    n = 16
    wrapped = wrapper.wrap_for_brax_training(
        env,
        episode_length=10,
        action_repeat=1,
        randomization_fn=lambda m: glace.randomize(
            m, jax.random.split(jax.random.PRNGKey(0), n)
        ),
    )
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(1), n))
    friction, payload = np.asarray(state.obs["privileged_state"][:, -2:]).T
    lo, hi = glace.TRAIN["friction"]
    assert np.all((friction >= lo) & (friction <= hi)) and friction.std() > 0.05
    # La charge inclut les ±10 % de masse du tronc (5,2 kg) de la randomisation.
    lo, hi = glace.TRAIN["payload"]
    assert np.all((payload >= lo - 0.6) & (payload <= hi + 0.6)) and payload.std() > 0.3


def test_set_condition():
    env, _ = make_env(load_config(BASE, ["scenario=glace"]))
    model = glace.set_condition(env.mjx_model, friction=0.1, payload=5.0)
    assert float(model.geom_friction[glace.FLOOR_GEOM_ID, 0]) == np.float32(0.1)
    added = (
        model.body_mass[glace.TORSO_BODY_ID]
        - env.mjx_model.body_mass[glace.TORSO_BODY_ID]
    )
    assert abs(float(added) - 5.0) < 1e-5
    assert len(glace.grid()) == 25
    assert glace.grid()[0] == {"friction": 0.1, "payload": 0.0}


def test_parse_condition_and_training_domain():
    from rescue_dog.scenarios import in_training, parse_condition

    cond = parse_condition("glace", ["friction=0.1", "payload=5"])
    assert cond == {"friction": 0.1, "payload": 5.0}
    assert not in_training("glace", cond)
    assert in_training("glace", {"friction": 0.7, "payload": 2.0})
    with pytest.raises(ValueError):
        parse_condition("glace", ["friction=0.1"])
    with pytest.raises(ValueError):
        parse_condition("glace", ["friction=0.1", "leg=FR"])
