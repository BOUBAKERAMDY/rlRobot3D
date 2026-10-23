from pathlib import Path

import pytest

from rescue_dog.training.ppo import make_train_kwargs
from rescue_dog.utils.config import load_config

BASE = Path(__file__).parent.parent / "configs" / "base.yaml"


@pytest.mark.parametrize("dr", [True, False])
@pytest.mark.parametrize("critic", [True, False])
def test_switches(dr, critic):
    cfg = load_config(
        BASE, [f"domain_randomization={dr}", f"privileged_critic={critic}"]
    )
    kwargs = make_train_kwargs(cfg)
    assert (kwargs["randomization_fn"] is not None) == dr
    network = kwargs["network_factory"].keywords
    assert network["policy_obs_key"] == "state"
    assert network["value_obs_key"] == ("privileged_state" if critic else "state")


def test_ppo_values_passed():
    kwargs = make_train_kwargs(
        load_config(BASE, ["ppo.learning_rate=1e-4", "num_envs=64"])
    )
    assert kwargs["learning_rate"] == 1e-4
    assert kwargs["num_envs"] == 64
