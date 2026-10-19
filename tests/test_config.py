from pathlib import Path

import pytest

from rescue_dog.utils.config import ConfigError, load_config

BASE = Path(__file__).parent.parent / "configs" / "base.yaml"


def test_load_base():
    cfg = load_config(BASE)
    assert cfg.env_name == "Go1JoystickFlatTerrain"
    assert cfg.seed == 0
    assert cfg.num_envs == 8192
    assert cfg.num_timesteps == 200_000_000
    assert cfg.domain_randomization is True
    assert cfg.privileged_critic is True
    assert cfg.ppo.learning_rate == pytest.approx(3e-4)
    assert cfg.ppo.policy_hidden_layer_sizes == (512, 256, 128)


def test_override_seed():
    cfg = load_config(BASE, ["seed=3"])
    assert cfg.seed == 3


def test_override_nested():
    cfg = load_config(BASE, ["ppo.learning_rate=1e-4", "ppo.policy_hidden_layer_sizes=[64, 64]"])
    assert cfg.ppo.learning_rate == pytest.approx(1e-4)
    assert cfg.ppo.policy_hidden_layer_sizes == (64, 64)


def test_override_switch():
    cfg = load_config(BASE, ["domain_randomization=false"])
    assert cfg.domain_randomization is False


def test_later_override_wins():
    cfg = load_config(BASE, ["seed=1", "seed=2"])
    assert cfg.seed == 2


@pytest.mark.parametrize("override", ["sed=3", "ppo.lr=1e-4", "seed.x=1"])
def test_unknown_key(override):
    with pytest.raises(ConfigError):
        load_config(BASE, [override])


@pytest.mark.parametrize("override", ["seed=abc", "seed=true", "num_envs=1.5", "privileged_critic=1"])
def test_wrong_type(override):
    with pytest.raises(ConfigError, match="type attendu"):
        load_config(BASE, [override])


def test_malformed_override():
    with pytest.raises(ConfigError, match="clé=valeur"):
        load_config(BASE, ["seed"])


def test_missing_key(tmp_path):
    path = tmp_path / "incomplete.yaml"
    path.write_text("seed: 0\n")
    with pytest.raises(ConfigError, match="Clés manquantes"):
        load_config(path)
