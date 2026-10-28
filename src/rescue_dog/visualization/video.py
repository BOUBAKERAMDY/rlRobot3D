"""Vidéos d'une politique entraînée qui suit une commande joystick fixe.

Deux étapes séparées :
- `rollout` joue la politique, éventuellement dans une condition de test
  physique (`scenarios/` : sol glissant, charge, patte affaiblie), et repère
  la première chute ;
- le rendu rejoue la trajectoire dans un décor (`visualization.scenes`) et
  peut y écrire une légende.

`record_video` filme un run ; `record_duel` filme plusieurs runs côte à côte,
dans la même condition et depuis le même état de départ.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import jax
import jax.numpy as jp
import mediapy
import numpy as np
from matplotlib import font_manager
from mujoco import mjx
from PIL import Image, ImageDraw, ImageFont

from rescue_dog.envs.go1 import make_eval_env
from rescue_dog.evaluation.evaluate import FALL_HEIGHT
from rescue_dog.scenarios import SCENARIOS as TESTS
from rescue_dog.training.ppo import load_policy
from rescue_dog.utils.config import Config
from rescue_dog.visualization.scenes import render_frames, scene_model

# Sans ffmpeg installé (ex. sur Mac), on prend celui fourni par imageio-ffmpeg.
if shutil.which("ffmpeg") is None:
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


FREEZE_AFTER_FALL = 0.5  # s


@dataclass
class Rollout:
    states: list[Any]  # états de l'environnement, un par pas, départ compris
    dt: float
    fall_step: int | None  # premier pas où le chien tombe

    @property
    def qpos(self) -> list[np.ndarray]:
        return [np.asarray(s.data.qpos) for s in self.states]


def rollout(
    cfg: Config,
    checkpoint_dir: Path,
    command: tuple[float, float, float] = (1.0, 0.0, 0.0),
    steps: int = 1000,
    test: str | None = None,
    condition: dict[str, Any] | None = None,
    align: bool = True,
    seed: int | None = None,
) -> Rollout:
    """Joue la politique de `checkpoint_dir` pendant `steps` pas.

    `command` = (avant m/s, côté m/s, rotation rad/s). On la réimpose à chaque pas,
    car l'environnement en tire une nouvelle au hasard en cours d'épisode.
    Avec `test` et `condition` (ex. "glace", {"friction": 0.1, "payload": 5}),
    la physique est celle de cette condition de test. Avec `align`, le chien
    part de l'origine tourné vers +x, l'entrée du couloir balisé des décors
    (le reset tire sinon une position et un cap au hasard).
    """
    env, _ = make_eval_env(cfg)
    if test is not None:
        model = TESTS[test].set_condition(env.mjx_model, **(condition or {}))
        env.unwrapped._mjx_model = model
    policy = jax.jit(
        load_policy(cfg, checkpoint_dir, env.observation_size, env.action_size)
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    cmd = jp.array(command)

    rng = jax.random.PRNGKey(cfg.seed if seed is None else seed)
    state = reset(rng)
    if align:
        qpos = state.data.qpos.at[:2].set(0.0).at[3:7].set(jp.array([1.0, 0, 0, 0]))
        data = jax.jit(mjx.forward)(env.mjx_model, state.data.replace(qpos=qpos))
        state = state.replace(data=data)
    state.info["command"] = cmd
    states = [state]
    fall_step = None
    for i in range(steps):
        rng, key = jax.random.split(rng)
        action, _ = policy(state.obs, key)
        state = step(state, action)
        state.info["command"] = cmd
        states.append(state)
        fell = float(state.done) > 0 or float(state.data.qpos[2]) < FALL_HEIGHT
        if fell and fall_step is None:
            fall_step = i + 1
    return Rollout(states, env.dt, fall_step)


def decor_for(test: str | None, condition: dict[str, Any] | None) -> tuple[str, dict]:
    """Décor qui montre une condition de test, et ses options."""
    condition = condition or {}
    if test == "glace":
        return "sol_glissant", {"load": condition.get("payload", 0) > 0}
    if test == "patte":
        return "patte_blessee", {"injured_leg": condition.get("leg", "FR")}
    return "normal", {}


def describe(test: str | None, condition: dict[str, Any] | None) -> str:
    """Condition de test en clair, pour les légendes."""
    condition = condition or {}
    if test == "glace":
        return f"μ = {condition['friction']:g} · charge +{condition['payload']:g} kg"
    if test == "patte":
        return f"patte {condition['leg']} à {condition['strength'] * 100:.0f} % de sa force"
    return "conditions normales"


def _caption(
    frames: list[np.ndarray],
    lines: list[str],
    fall_frame: int | None,
    fall_time: float | None,
) -> list[np.ndarray]:
    """Écrit `lines` en haut à gauche, puis « CHUTE » à partir de la chute."""
    h = frames[0].shape[0]
    size = max(18, h // 24)
    # DejaVu Sans, livrée avec matplotlib : accents et μ, sur Mac comme sur Kaggle.
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    bold = font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    big = ImageFont.truetype(font_manager.findfont(bold), int(size * 1.6))
    out = []
    for k, frame in enumerate(frames):
        img = Image.fromarray(frame).convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        pad = size // 2
        box_h = pad * 2 + len(lines) * int(size * 1.3)
        box_w = pad * 2 + max(int(draw.textlength(t, font=font)) for t in lines)
        draw.rounded_rectangle(
            (pad, pad, pad + box_w, pad + box_h), 10, fill=(10, 18, 26, 170)
        )
        for i, text in enumerate(lines):
            color = (240, 100, 30, 255) if i == 0 else (243, 239, 230, 255)
            draw.text(
                (2 * pad, 2 * pad + i * int(size * 1.3)), text, font=font, fill=color
            )
        if fall_frame is not None and k >= fall_frame:
            text = f"CHUTE à {fall_time:.1f} s"
            w = draw.textlength(text, font=big)
            x, y = (img.width - w) / 2, img.height - 3 * size
            draw.rounded_rectangle(
                (x - pad, y - pad, x + w + pad, y + 1.6 * size + pad),
                10,
                fill=(200, 30, 20, 210),
            )
            draw.text((x, y), text, font=big, fill=(255, 255, 255, 255))
        out.append(np.asarray(Image.alpha_composite(img, layer).convert("RGB")))
    return out


def render_rollout(
    run: Rollout,
    scenario: str | None,
    camera: str,
    height: int,
    width: int,
    render_every: int = 2,
    decor_options: dict | None = None,
    caption: list[str] | None = None,
) -> list[np.ndarray]:
    """Images de la trajectoire dans le décor `scenario`, avec légende.

    Le modèle d'entraînement ne fait toucher le sol qu'aux pieds : après une
    chute, le tronc s'enfonce dans le sol. L'image est donc figée
    `FREEZE_AFTER_FALL` s après la chute.
    """
    model = scene_model(scenario, **(decor_options or {}))
    qpos = run.qpos
    if run.fall_step is not None:
        last = min(len(qpos) - 1, run.fall_step + int(FREEZE_AFTER_FALL / run.dt))
        qpos = qpos[: last + 1] + [qpos[last]] * (len(qpos) - last - 1)
    frames = render_frames(
        model, qpos[::render_every], camera=camera, height=height, width=width
    )
    if caption:
        fall_frame = None if run.fall_step is None else run.fall_step // render_every
        fall_time = None if run.fall_step is None else run.fall_step * run.dt
        frames = _caption(frames, caption, fall_frame, fall_time)
    return frames


def record_video(
    cfg: Config,
    checkpoint_dir: Path,
    out_path: Path,
    command: tuple[float, float, float] = (1.0, 0.0, 0.0),
    steps: int = 1000,
    render_every: int = 2,
    scenario: str | None = None,
    camera: str = "track",
    test: str | None = None,
    condition: dict[str, Any] | None = None,
    caption: list[str] | None = None,
) -> Rollout:
    """Filme un run et écrit un mp4.

    Sans décor ni test, la scène nue de Playground. Avec un `test`, le décor
    qui lui correspond, sauf si `scenario` en impose un autre.
    """
    decorated = scenario is not None or test is not None
    run = rollout(cfg, checkpoint_dir, command, steps, test, condition, align=decorated)
    if not decorated:
        env, _ = make_eval_env(cfg)
        frames = env.render(
            run.states[::render_every], height=480, width=640, camera=camera
        )
    else:
        decor, options = decor_for(test, condition)
        frames = render_rollout(
            run,
            scenario or decor,
            camera,
            720,
            1280,
            render_every,
            options if scenario is None else None,
            caption,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(out_path, np.asarray(frames), fps=1.0 / run.dt / render_every)
    return run


def record_duel(
    runs: list[tuple[str, Config, Path]],
    out_path: Path,
    test: str,
    condition: dict[str, Any],
    command: tuple[float, float, float] = (1.0, 0.0, 0.0),
    steps: int = 1000,
    render_every: int = 2,
    camera: str = "cinema",
) -> list[Rollout]:
    """Plusieurs runs côte à côte, même condition, même état de départ.

    `runs` = [(étiquette, config, checkpoint), ...]. Chaque vignette porte son
    étiquette et la condition, puis « CHUTE » si le chien tombe.
    """
    decor, options = decor_for(test, condition)
    width = 1920 // len(runs)
    height = width * 9 // 16
    results, columns = [], []
    for label, cfg, ckpt in runs:
        run = rollout(cfg, ckpt, command, steps, test, condition, align=True, seed=0)
        caption = [label, describe(test, condition)]
        columns.append(
            render_rollout(
                run, decor, camera, height, width, render_every, options, caption
            )
        )
        results.append(run)
    frames = [np.concatenate(row, axis=1) for row in zip(*columns)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(
        out_path, np.asarray(frames), fps=1.0 / results[0].dt / render_every
    )
    return results
