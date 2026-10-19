import mujoco
import mujoco.viewer
from mujoco_playground import registry

cfg = registry.get_default_config("Go1JoystickFlatTerrain")
cfg.impl = "jax"
env = registry.load("Go1JoystickFlatTerrain", config=cfg)

model = env.mj_model
data = mujoco.MjData(model)
key = model.key("home").id
mujoco.mj_resetDataKeyframe(model, data, key)
data.ctrl[:] = model.key_ctrl[key]  # motors hold the standing pose
print("viewer opening", flush=True)
mujoco.viewer.launch(model, data)
