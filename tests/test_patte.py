from pathlib import Path

import jax
import numpy as np
from mujoco_playground import wrapper

from rescue_dog.envs.go1 import make_env
from rescue_dog.scenarios import patte
from rescue_dog.training.ppo import make_train_kwargs
from rescue_dog.utils.config import load_config

BASE = Path(__file__).parent.parent / "configs" / "base.yaml"


def test_randomizer_follows_scenario():
    cfg = load_config(BASE, ["scenario=patte", "domain_randomization=true"])
    assert make_train_kwargs(cfg)["randomization_fn"] is patte.randomize


def test_privileged_state_sees_injured_leg():
    """Le critique lit la force restante de chaque patte : une seule est affaiblie."""
    env, _ = make_env(load_config(BASE, ["scenario=patte"]))
    assert env.observation_size["privileged_state"] == (127,)
    n = 64
    wrapped = wrapper.wrap_for_brax_training(
        env,
        episode_length=10,
        action_repeat=1,
        randomization_fn=lambda m: patte.randomize(
            m, jax.random.split(jax.random.PRNGKey(0), n)
        ),
    )
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(1), n))
    strength = np.asarray(state.obs["privileged_state"][:, -4:])
    weakest = strength.min(axis=1)
    lo, hi = patte.TRAIN["strength"]
    assert np.all((weakest >= lo - 1e-5) & (weakest <= hi + 1e-5))
    assert np.all(np.sort(strength, axis=1)[:, 1:] == 1.0)  # les 3 autres saines
    assert len(set(strength.argmin(axis=1))) == 4  # chaque patte est tirée


def test_set_condition():
    env, _ = make_env(load_config(BASE, ["scenario=patte"]))
    nominal = env.mjx_model
    model = patte.set_condition(nominal, leg="RL", strength=0.2)
    scale = np.asarray(
        model.actuator_forcerange[:, 1] / nominal.actuator_forcerange[:, 1]
    )
    np.testing.assert_allclose(scale, [1] * 9 + [0.2] * 3, rtol=1e-6)
    kp = np.asarray(model.actuator_gainprm[:, 0] / nominal.actuator_gainprm[:, 0])
    np.testing.assert_allclose(kp, scale, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(model.actuator_biasprm[:, 1]),
        -np.asarray(model.actuator_gainprm[:, 0]),
    )
    np.testing.assert_allclose(patte.privileged_params(model, nominal), [1, 1, 1, 0.2])
    assert len(patte.grid()) == 20
