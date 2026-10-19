from pathlib import Path

import jax
import jax.numpy as jp
import pytest

from rescue_dog.envs.go1 import make_env
from rescue_dog.utils.config import load_config

BASE = Path(__file__).parent.parent / "configs" / "base.yaml"


@pytest.fixture(scope="module")
def env():
    env, _ = make_env(load_config(BASE))
    return env


def test_sizes(env):
    assert env.action_size == 12
    assert env.observation_size == {"state": (48,), "privileged_state": (123,)}


def test_reset_and_step(env):
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    for _ in range(10):
        rng, key = jax.random.split(rng)
        action = jax.random.uniform(key, (env.action_size,), minval=-1.0, maxval=1.0)
        state = step(state, action)
    assert state.obs["state"].shape == (48,)
    assert state.obs["privileged_state"].shape == (123,)
    assert jp.all(jp.isfinite(state.obs["state"]))
    assert jp.all(jp.isfinite(state.obs["privileged_state"]))


def test_config_applied():
    _, env_cfg = make_env(load_config(BASE, ["episode_length=500"]))
    assert env_cfg.episode_length == 500
    assert env_cfg.impl == "jax"
