from pathlib import Path

import mujoco
import numpy as np
import pytest

from rescue_dog.envs.go1 import make_env
from rescue_dog.utils.config import load_config
from rescue_dog.visualization.scenes import (
    SCENARIOS,
    home_qpos,
    render_frames,
    scene_model,
)

BASE = Path(__file__).parent.parent / "configs" / "base.yaml"


@pytest.fixture(scope="module")
def env_model():
    env, _ = make_env(load_config(BASE))
    return env.mj_model


def simulate(model: mujoco.MjModel, steps: int = 200) -> np.ndarray:
    data = mujoco.MjData(model)
    key = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = model.key_ctrl[key]
    data.qvel[0] = 0.5  # une poussée pour que les pattes travaillent
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data.qpos.copy()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_decor_does_not_change_physics(env_model, scenario):
    model = scene_model(scenario)
    assert model.nq == env_model.nq
    assert model.nv == env_model.nv
    assert model.nu == env_model.nu
    np.testing.assert_allclose(model.body_mass, env_model.body_mass)
    np.testing.assert_allclose(simulate(model), simulate(env_model), atol=1e-8)


def test_render(env_model):
    model = scene_model("patte_blessee", injured_leg="RL")
    frame = render_frames(
        model, [home_qpos(model)], camera="cinema", height=60, width=80
    )[0]
    assert frame.shape == (60, 80, 3)
    assert frame.std() > 0


def test_unknown_names():
    with pytest.raises(ValueError):
        scene_model("lave")
    with pytest.raises(ValueError):
        scene_model("patte_blessee", injured_leg="XX")
