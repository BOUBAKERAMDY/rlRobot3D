"""Décors des scénarios de sauvetage, pour la visualisation uniquement.

On part de la scène MuJoCo de l'environnement Go1 et on n'y ajoute que du
visuel : textures, ciel, lumières, brouillard, objets de décor et accessoires
du chien. Tous les nouveaux geoms ont `contype = conaffinity = 0` et les corps
du Go1 ont une inertie explicite : la physique ne change pas, et `qpos` garde
la même forme. Une trajectoire de l'environnement d'entraînement se rejoue
donc telle quelle dans n'importe quel décor.

Scénarios :
- "normal" : champ de décombres, chien en gilet de sauvetage.
- "sol_glissant" : lac gelé enneigé, blocs de glace et sapins.
- "charge_lourde" : décombres, caisse de secours sanglée sur le dos.
- "patte_blessee" : décombres, une patte bandée et teintée de rouge.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import mujoco
import numpy as np
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import go1_constants
from PIL import Image

SCENARIOS = ("normal", "sol_glissant", "charge_lourde", "patte_blessee")
LEGS = ("FR", "FL", "RR", "RL")

BOX = mujoco.mjtGeom.mjGEOM_BOX
CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID
SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
MESH = mujoco.mjtGeom.mjGEOM_MESH


# --- Textures procédurales (aucun fichier image à versionner) ---


def _noise(rng: np.random.Generator, size: int, octaves: int = 6) -> np.ndarray:
    """Bruit fractal dans [0, 1], qui se répète sans couture sur les bords."""
    out = np.zeros((size, size))
    amp, total = 1.0, 0.0
    for k in range(octaves):
        cells = 2 ** (k + 2)
        grid = rng.random((cells, cells))
        # On recopie la première ligne/colonne pour que la texture boucle.
        grid = np.pad(grid, ((0, 1), (0, 1)), mode="wrap")
        img = Image.fromarray((grid * 255).astype(np.uint8))
        step = size / cells
        big = img.resize((int(size + step), int(size + step)), Image.Resampling.BICUBIC)
        out += amp * np.asarray(big, dtype=float)[:size, :size] / 255
        total += amp
        amp *= 0.55
    out /= total
    return (out - out.min()) / (out.max() - out.min())


def _mix(t: np.ndarray, *colors: tuple[float, float, float]) -> np.ndarray:
    """Dégradé de couleurs RGB (0-255) suivant `t` dans [0, 1]."""
    stops = np.linspace(0, 1, len(colors))
    cols = np.array(colors, dtype=float)
    return np.stack([np.interp(t, stops, cols[:, c]) for c in range(3)], axis=-1)


def _to_bytes(rgb: np.ndarray) -> bytes:
    return np.clip(rgb, 0, 255).astype(np.uint8).tobytes()


def _dirt(rng: np.random.Generator, size: int = 512) -> np.ndarray:
    """Terre poussiéreuse avec des cailloux sombres et des taches de cendre."""
    base = _mix(_noise(rng, size), (62, 50, 40), (104, 88, 70), (140, 124, 102))
    ash = _noise(rng, size, octaves=4)[..., None]
    base = base * (0.75 + 0.35 * ash)
    # Cailloux : quelques pixels isolés, grossis d'un pixel.
    pebbles = rng.random((size, size)) > 0.993
    pebbles |= np.roll(pebbles, 1, 0) | np.roll(pebbles, 1, 1)
    base[pebbles] *= 0.5
    return base


def _concrete(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Béton gris, piqué de trous et de traces de suie."""
    base = _mix(_noise(rng, size), (90, 90, 88), (150, 148, 142), (182, 180, 172))
    soot = _noise(rng, size, octaves=3)[..., None]
    base = base * (0.7 + 0.3 * soot)
    holes = rng.random((size, size)) > 0.985
    base[holes] *= 0.4
    return base


def _ice(rng: np.random.Generator, size: int = 512) -> np.ndarray:
    """Glace bleutée parcourue de fissures blanches."""
    base = _mix(_noise(rng, size), (70, 125, 165), (115, 170, 205), (160, 205, 228))
    cracks = np.abs(_noise(rng, size, octaves=6) - 0.5) < 0.006
    cracks |= np.abs(_noise(rng, size, octaves=6) - 0.5) < 0.004
    base[cracks] = (235, 245, 252)
    frost = _noise(rng, size, octaves=4)[..., None] ** 3
    return base * (1 - 0.3 * frost) + 255 * 0.3 * frost


def _snow(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    return _mix(_noise(rng, size), (214, 222, 235), (238, 242, 250), (252, 253, 255))


def _rust(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    return _mix(_noise(rng, size), (70, 34, 20), (128, 60, 28), (170, 96, 52))


# --- Petits outils MjSpec ---


def _texture(spec: mujoco.MjSpec, name: str, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    tex = spec.add_texture(
        name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=w, height=h, nchannel=3
    )
    tex.data = _to_bytes(rgb)


def _material(
    spec: mujoco.MjSpec,
    name: str,
    rgba=(1, 1, 1, 1),
    texture: str | None = None,
    texrepeat=(1, 1),
    texuniform: bool = False,
    reflectance: float = 0.0,
    specular: float = 0.3,
    shininess: float = 0.3,
    emission: float = 0.0,
) -> None:
    mat = spec.add_material(
        name=name,
        rgba=list(rgba),
        texrepeat=list(texrepeat),
        texuniform=texuniform,
        reflectance=reflectance,
        specular=specular,
        shininess=shininess,
        emission=emission,
    )
    if texture is not None:
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture


def _deco(body, type_, size, pos, material: str, quat=(1, 0, 0, 0), **kw):
    """Geom purement visuel : aucune collision, aucune masse ajoutée."""
    return body.add_geom(
        type=type_,
        size=list(size) + [0] * (3 - len(size)),
        pos=list(pos),
        quat=list(quat),
        material=material,
        contype=0,
        conaffinity=0,
        group=1,
        **kw,
    )


def _random_quat(rng: np.random.Generator, tilt: float) -> list[float]:
    """Rotation aléatoire autour de z, inclinée d'au plus `tilt` radians."""
    yaw = rng.uniform(0, 2 * np.pi)
    ax = rng.normal(size=2)
    ax /= np.linalg.norm(ax)
    a = rng.uniform(0, tilt)
    q_yaw = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    q_tilt = np.array([np.cos(a / 2), ax[0] * np.sin(a / 2), ax[1] * np.sin(a / 2), 0])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q_yaw, q_tilt)
    return out.tolist()


def _side_spot(
    rng: np.random.Generator, x_range=(-8, 30), y_min=1.4, y_max=9.0
) -> tuple[float, float]:
    """Position hors du couloir où marche le chien (axe x, |y| < y_min)."""
    x = rng.uniform(*x_range)
    y = rng.uniform(y_min, y_max) * rng.choice((-1, 1))
    return x, y


def _cone_mesh(spec: mujoco.MjSpec, name: str, radius: float, height: float, n=24):
    """Cône (enveloppe convexe d'un cercle et d'une pointe)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts = [[radius * np.cos(a), radius * np.sin(a), 0.0] for a in angles]
    verts.append([0.0, 0.0, height])
    spec.add_mesh(name=name, uservert=np.ravel(verts).tolist())


# --- Ambiance commune ---


def _atmosphere(
    spec: mujoco.MjSpec, sky_top, sky_bottom, fog, fog_range, ambient=0.3
) -> None:
    for tex in spec.textures:
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
            tex.rgb1, tex.rgb2 = list(sky_top), list(sky_bottom)
    spec.visual.rgba.fog = list(fog) + [1]
    spec.visual.rgba.haze = list(fog) + [1]
    spec.visual.map.fogstart, spec.visual.map.fogend = fog_range
    spec.visual.map.haze = 0.6
    spec.visual.headlight.ambient = [ambient] * 3
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    spec.visual.quality.shadowsize = 8192
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080


def _floor(spec: mujoco.MjSpec, texture: str, repeat: float, reflectance: float):
    """Remplace l'aspect du sol (le geom "floor" et sa friction ne changent pas)."""
    mat = spec.material("groundplane")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture
    mat.texrepeat = [repeat, repeat]
    mat.texuniform = True
    mat.reflectance = reflectance
    mat.rgba = [1, 1, 1, 1]


def _sun(spec: mujoco.MjSpec, direction, color, intensity=1.0) -> None:
    d = np.asarray(direction, dtype=float)
    spec.worldbody.add_light(
        name="soleil",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        dir=(d / np.linalg.norm(d)).tolist(),
        pos=(-d * 10).tolist(),
        diffuse=[c * intensity for c in color],
        specular=[0.3, 0.3, 0.3],
        castshadow=True,
    )


# --- Décor : champ de décombres ---


def _rubble_field(spec: mujoco.MjSpec, rng: np.random.Generator) -> None:
    _texture(spec, "terre", _dirt(rng))
    _texture(spec, "beton", _concrete(rng))
    _texture(spec, "rouille", _rust(rng))
    _floor(spec, "terre", repeat=0.6, reflectance=0.0)
    _material(spec, "beton", texture="beton", texrepeat=(2, 2), specular=0.1)
    _material(spec, "rouille", texture="rouille", specular=0.4, shininess=0.6)
    _material(spec, "brique", rgba=(0.55, 0.22, 0.16, 1), specular=0.1)
    _material(spec, "orange_cone", rgba=(1.0, 0.38, 0.05, 1), specular=0.5)
    _material(
        spec, "blanc_reflechissant", rgba=(0.95, 0.95, 0.95, 1), specular=1, shininess=1
    )
    _material(
        spec, "baril_jaune", rgba=(0.95, 0.72, 0.08, 1), specular=0.6, shininess=0.7
    )
    _material(
        spec, "baril_rouge", rgba=(0.7, 0.08, 0.06, 1), specular=0.6, shininess=0.7
    )
    _material(spec, "noir", rgba=(0.06, 0.06, 0.06, 1))
    _material(spec, "gyrophare", rgba=(1.0, 0.15, 0.05, 1), emission=1.0)
    _cone_mesh(spec, "cone", radius=0.13, height=0.5)

    _atmosphere(
        spec,
        sky_top=(0.32, 0.36, 0.45),
        sky_bottom=(0.85, 0.62, 0.42),
        fog=(0.62, 0.52, 0.44),
        fog_range=(6, 40),
    )
    _sun(spec, direction=(0.3, 0.7, -0.5), color=(1.0, 0.8, 0.6), intensity=0.95)

    wb = spec.worldbody
    # Gravats de béton et de brique éparpillés de part et d'autre du couloir.
    for _ in range(220):
        x, y = _side_spot(rng)
        s = rng.uniform(0.04, 0.28, size=3) * np.array([1, 1, 0.5])
        mat = "beton" if rng.random() < 0.75 else "brique"
        _deco(wb, BOX, s, (x, y, s[2] * 0.6), mat, _random_quat(rng, 0.5))
    # Dalles effondrées, inclinées contre le sol.
    for _ in range(18):
        x, y = _side_spot(rng, y_min=2.5)
        s = (rng.uniform(0.6, 1.4), rng.uniform(0.4, 0.9), 0.06)
        _deco(wb, BOX, s, (x, y, 0.3), "beton", _random_quat(rng, 0.7))
    # Piliers brisés, debout ou couchés.
    for _ in range(10):
        x, y = _side_spot(rng, y_min=3)
        h = rng.uniform(0.4, 1.6)
        if rng.random() < 0.5:
            _deco(
                wb,
                CYLINDER,
                (0.18, h / 2),
                (x, y, h / 2),
                "beton",
                _random_quat(rng, 0.15),
            )
        else:
            q = _random_quat(rng, 0.0)
            q_flat = np.zeros(4)
            mujoco.mju_mulQuat(
                q_flat,
                np.array(q),
                np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]),
            )
            _deco(wb, CYLINDER, (0.18, h / 2), (x, y, 0.18), "beton", q_flat.tolist())
    # Barres d'armature rouillées qui dépassent.
    for _ in range(40):
        x, y = _side_spot(rng, y_min=2)
        L = rng.uniform(0.3, 0.9)
        _deco(
            wb,
            CAPSULE,
            (0.008, L / 2),
            (x, y, L / 3),
            "rouille",
            _random_quat(rng, 0.9),
        )
    # Plots de chantier qui balisent le couloir.
    for x in np.arange(-4, 30, 3.0):
        for side in (-1, 1):
            y = side * 1.15
            _deco(wb, MESH, (), (x, y, 0.0), "orange_cone", meshname="cone")
            _deco(wb, CYLINDER, (0.085, 0.03), (x, y, 0.28), "blanc_reflechissant")
            _deco(wb, BOX, (0.17, 0.17, 0.012), (x, y, 0.012), "noir")
    # Barils, parfois renversés.
    for _ in range(14):
        x, y = _side_spot(rng, y_min=1.8, y_max=6)
        mat = "baril_jaune" if rng.random() < 0.6 else "baril_rouge"
        if rng.random() < 0.7:
            _deco(wb, CYLINDER, (0.28, 0.42), (x, y, 0.42), mat)
            _deco(wb, CYLINDER, (0.285, 0.015), (x, y, 0.62), "noir")
            _deco(wb, CYLINDER, (0.285, 0.015), (x, y, 0.22), "noir")
        else:
            q = np.zeros(4)
            mujoco.mju_mulQuat(
                q,
                np.array(_random_quat(rng, 0)),
                np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]),
            )
            _deco(wb, CYLINDER, (0.28, 0.42), (x, y, 0.28), mat, q.tolist())
    # Gyrophares rouges sur poteaux, avec une lumière ponctuelle.
    for i, x in enumerate((2.0, 12.0, 22.0)):
        y = 2.6 * (1 if i % 2 else -1)
        _deco(wb, CYLINDER, (0.03, 0.6), (x, y, 0.6), "noir")
        _deco(wb, SPHERE, (0.07,), (x, y, 1.25), "gyrophare")
        wb.add_light(
            type=mujoco.mjtLightType.mjLIGHT_POINT,
            pos=[x, y, 1.3],
            diffuse=[0.9, 0.12, 0.05],
            specular=[0.2, 0.0, 0.0],
            attenuation=[0.4, 0.4, 0.4],
            castshadow=False,
        )


# --- Décor : lac gelé ---


def _frozen_lake(spec: mujoco.MjSpec, rng: np.random.Generator) -> None:
    _texture(spec, "glace", _ice(rng))
    _texture(spec, "neige", _snow(rng))
    _floor(spec, "glace", repeat=0.3, reflectance=0.15)
    _material(spec, "neige", texture="neige", specular=0.05, emission=0.35)
    _material(
        spec,
        "bloc_glace",
        rgba=(0.6, 0.86, 1.0, 1),
        specular=1,
        shininess=1,
        emission=0.35,
    )
    _material(spec, "sapin", rgba=(0.06, 0.2, 0.12, 1), specular=0.05)
    _material(spec, "sapin_neige", rgba=(0.88, 0.92, 0.97, 1), specular=0.1)
    _material(spec, "ecorce", rgba=(0.24, 0.15, 0.09, 1), specular=0.05)
    _material(spec, "balise", rgba=(0.95, 0.15, 0.1, 1), specular=0.5, emission=0.3)

    _atmosphere(
        spec,
        sky_top=(0.35, 0.5, 0.72),
        sky_bottom=(0.86, 0.9, 0.96),
        fog=(0.82, 0.87, 0.94),
        fog_range=(8, 45),
        ambient=0.45,
    )
    _sun(spec, direction=(-0.2, 0.7, -0.55), color=(0.95, 0.97, 1.0), intensity=0.9)

    wb = spec.worldbody
    # Congères de neige sur les rives.
    for _ in range(70):
        x, y = _side_spot(rng, y_min=3.5, y_max=12)
        s = (rng.uniform(0.5, 1.8), rng.uniform(0.4, 1.2), rng.uniform(0.15, 0.5))
        _deco(wb, ELLIPSOID, s, (x, y, 0), "neige", _random_quat(rng, 0))
    # Blocs de glace brillants près du couloir.
    for _ in range(30):
        x, y = _side_spot(rng, y_min=1.6, y_max=5)
        s = rng.uniform(0.1, 0.45, size=3)
        _deco(wb, BOX, s, (x, y, s[2] * 0.5), "bloc_glace", _random_quat(rng, 0.4))
    # Sapins enneigés en lisière : trois étages de cônes et un tronc.
    for i in range(45):
        x, y = _side_spot(rng, y_min=6, y_max=14)
        h = rng.uniform(1.8, 3.6)
        _deco(wb, CYLINDER, (0.04 * h, 0.25 * h), (x, y, 0.25 * h), "ecorce")
        for k, (r, z) in enumerate(((0.42, 0.22), (0.32, 0.45), (0.22, 0.66))):
            name = f"sapin_{i}_{k}"
            _cone_mesh(spec, name, radius=r * h, height=0.42 * h)
            mat = "sapin_neige" if k == 2 else "sapin"
            _deco(wb, MESH, (), (x, y, z * h), mat, meshname=name)
    # Piquets de balisage rouges le long du couloir.
    for x in np.arange(-4, 30, 4.0):
        for side in (-1, 1):
            _deco(wb, CYLINDER, (0.02, 0.4), (x, side * 1.3, 0.4), "balise")


# --- Le chien ---


def _follow_camera(body, name: str, offset, target=(0.0, 0.0, -0.1)) -> None:
    """Caméra qui suit le chien depuis `offset` en visant `target` (repère du tronc)."""
    fwd = np.asarray(target, dtype=float) - np.asarray(offset, dtype=float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, (0, 0, 1))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    cam = body.add_camera(
        name=name, pos=list(offset), mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
    )
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = np.concatenate([right, up]).tolist()
    cam.fovy = 40


def _dog_livery(spec: mujoco.MjSpec) -> None:
    """Robot graphite satiné, gilet de sauvetage orange et lampe frontale."""
    dark = spec.material("dark")
    dark.rgba = [0.16, 0.17, 0.19, 1]
    dark.specular, dark.shininess, dark.reflectance = 0.6, 0.7, 0.05
    _material(spec, "gilet", rgba=(1.0, 0.42, 0.0, 1), specular=0.3)
    _material(
        spec,
        "bande_argent",
        rgba=(0.85, 0.87, 0.9, 1),
        specular=1,
        shininess=1,
        emission=0.15,
    )
    _material(spec, "croix_rouge", rgba=(0.85, 0.05, 0.05, 1), specular=0.3)
    _material(spec, "blanc", rgba=(0.96, 0.96, 0.96, 1), specular=0.3)
    _material(spec, "lampe", rgba=(1.0, 0.97, 0.85, 1), emission=1.0)

    trunk = spec.body(go1_constants.ROOT_BODY)
    _follow_camera(trunk, "cinema", offset=(1.5, -2.4, 0.9))
    _follow_camera(trunk, "large", offset=(-3.0, -4.5, 2.2))
    _follow_camera(trunk, "rase", offset=(0.9, -0.7, -0.12), target=(0.0, 0.0, 0.0))
    # Gilet : plaque dorsale et deux flancs, avec bandes réfléchissantes.
    _deco(trunk, BOX, (0.13, 0.068, 0.008), (-0.01, 0, 0.062), "gilet")
    for side in (-1, 1):
        _deco(trunk, BOX, (0.13, 0.006, 0.035), (-0.01, side * 0.074, 0.03), "gilet")
        _deco(
            trunk,
            BOX,
            (0.131, 0.007, 0.007),
            (-0.01, side * 0.075, 0.025),
            "bande_argent",
        )
    _deco(trunk, BOX, (0.008, 0.069, 0.009), (0.06, 0, 0.063), "bande_argent")
    _deco(trunk, BOX, (0.008, 0.069, 0.009), (-0.08, 0, 0.063), "bande_argent")
    # Croix rouge sur fond blanc.
    _deco(trunk, BOX, (0.03, 0.03, 0.002), (-0.01, 0, 0.071), "blanc")
    _deco(trunk, BOX, (0.022, 0.007, 0.002), (-0.01, 0, 0.073), "croix_rouge")
    _deco(trunk, BOX, (0.007, 0.022, 0.002), (-0.01, 0, 0.073), "croix_rouge")
    # Lampe frontale et son faisceau.
    _deco(
        trunk,
        CYLINDER,
        (0.018, 0.006),
        (0.245, 0, 0.025),
        "lampe",
        quat=(0.7071, 0, 0.7071, 0),
    )
    trunk.add_light(
        name="lampe_frontale",
        type=mujoco.mjtLightType.mjLIGHT_SPOT,
        pos=[0.26, 0, 0.025],
        dir=[1, 0, -0.35],
        diffuse=[0.6, 0.57, 0.48],
        specular=[0.3, 0.3, 0.3],
        cutoff=35,
        exponent=10,
        attenuation=[1, 0.05, 0.02],
        castshadow=False,
    )


def _heavy_load(spec: mujoco.MjSpec) -> None:
    """Caisse de secours sanglée sur le dos, avec deux bouteilles d'eau."""
    _material(spec, "caisse", rgba=(0.9, 0.9, 0.86, 1), specular=0.4)
    _material(spec, "sangle", rgba=(0.08, 0.08, 0.1, 1), specular=0.2)
    _material(
        spec,
        "bouteille",
        rgba=(0.45, 0.75, 1.0, 0.85),
        specular=1,
        shininess=1,
        emission=0.2,
    )
    _material(spec, "bouchon", rgba=(0.1, 0.35, 0.8, 1))
    trunk = spec.body(go1_constants.ROOT_BODY)
    z = 0.07 + 0.055
    _deco(trunk, BOX, (0.1, 0.07, 0.055), (-0.01, 0, z), "caisse")
    # Croix rouges sur le dessus et les flancs.
    _deco(trunk, BOX, (0.035, 0.009, 0.002), (-0.01, 0, z + 0.056), "croix_rouge")
    _deco(trunk, BOX, (0.009, 0.035, 0.002), (-0.01, 0, z + 0.056), "croix_rouge")
    for side in (-1, 1):
        _deco(trunk, BOX, (0.03, 0.002, 0.008), (-0.01, side * 0.071, z), "croix_rouge")
        _deco(trunk, BOX, (0.008, 0.002, 0.03), (-0.01, side * 0.071, z), "croix_rouge")
    # Sangles qui font le tour du dos.
    for x in (-0.085, 0.065):
        _deco(trunk, BOX, (0.008, 0.072, 0.057), (x, 0, z), "sangle")
        _deco(trunk, BOX, (0.008, 0.08, 0.004), (x, 0, -0.06), "sangle")
        for side in (-1, 1):
            _deco(trunk, BOX, (0.008, 0.004, 0.065), (x, side * 0.078, 0.0), "sangle")
    # Bouteilles couchées à l'arrière.
    for side in (-1, 1):
        y = side * 0.035
        _deco(
            trunk,
            CYLINDER,
            (0.022, 0.06),
            (-0.16, y, 0.1),
            "bouteille",
            quat=(0.7071, 0.7071, 0, 0),
        )
        _deco(
            trunk,
            CYLINDER,
            (0.01, 0.008),
            (-0.16, y + side * 0.066, 0.1),
            "bouchon",
            quat=(0.7071, 0.7071, 0, 0),
        )


def _injured_leg(spec: mujoco.MjSpec, leg: str) -> None:
    """Patte `leg` bandée, teintée de rouge, avec un anneau d'alerte lumineux."""
    if leg not in LEGS:
        raise ValueError(f"Patte inconnue {leg!r}, choisir parmi {LEGS}")
    _material(
        spec, "patte_blessee", rgba=(0.45, 0.07, 0.06, 1), specular=0.5, shininess=0.6
    )
    _material(spec, "bandage", rgba=(0.93, 0.91, 0.86, 1), specular=0.05)
    _material(spec, "sang", rgba=(0.55, 0.0, 0.02, 1), specular=0.4)
    _material(spec, "alerte", rgba=(1.0, 0.1, 0.05, 0.6), emission=1.0)
    for part in ("hip", "thigh", "calf"):
        for g in spec.body(f"{leg}_{part}").geoms:
            if g.type == MESH:
                g.material = "patte_blessee"
    calf = spec.body(f"{leg}_calf")
    for k, z in enumerate((-0.05, -0.08, -0.11)):
        _deco(
            calf,
            CYLINDER,
            (0.021, 0.012),
            (0.012, 0, z),
            "bandage",
            quat=(0.998, 0, 0.06, 0),
        )
        if k == 1:
            _deco(
                calf,
                CYLINDER,
                (0.0215, 0.006),
                (0.012, 0, z),
                "sang",
                quat=(0.998, 0, 0.06, 0),
            )
    thigh = spec.body(f"{leg}_thigh")
    _deco(
        thigh,
        CYLINDER,
        (0.034, 0.014),
        (-0.01, 0, -0.1),
        "bandage",
        quat=(0.998, 0, 0.06, 0),
    )
    _deco(
        thigh,
        CYLINDER,
        (0.035, 0.005),
        (-0.01, 0, -0.1),
        "sang",
        quat=(0.998, 0, 0.06, 0),
    )
    _deco(calf, CYLINDER, (0.03, 0.002), (0.0, 0, 0.0), "alerte")


# --- Point d'entrée ---


def scene_spec(
    scenario: str = "normal",
    xml_path: Path | str | None = None,
    injured_leg: str = "FR",
    seed: int = 0,
) -> mujoco.MjSpec:
    """Spec MuJoCo de la scène Go1 habillée pour `scenario`."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Scénario inconnu {scenario!r}, choisir parmi {SCENARIOS}")
    if xml_path is None:
        xml_path = go1_constants.FEET_ONLY_FLAT_TERRAIN_XML
    spec = mujoco.MjSpec.from_file(str(xml_path))
    # Les chemins des maillages du XML de Playground supposent une autre
    # arborescence que celle du paquet installé : on les fait pointer vers la
    # copie de Menagerie que Playground télécharge au premier usage.
    mjx_env.ensure_menagerie_exists()
    meshes = mjx_env.MENAGERIE_PATH / "unitree_go1" / "assets"
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = str(meshes / Path(mesh.file).name)
    rng = np.random.default_rng(seed)

    if scenario == "sol_glissant":
        _frozen_lake(spec, rng)
    else:
        _rubble_field(spec, rng)
    _dog_livery(spec)
    if scenario == "charge_lourde":
        _heavy_load(spec)
    if scenario == "patte_blessee":
        _injured_leg(spec, injured_leg)
    return spec


def scene_model(scenario: str = "normal", **kwargs) -> mujoco.MjModel:
    """Modèle compilé de `scene_spec` ; même `qpos` que l'environnement."""
    return scene_spec(scenario, **kwargs).compile()


def render_frames(
    model: mujoco.MjModel,
    qpos_seq: Iterable[np.ndarray],
    camera: str | int = "track",
    height: int = 480,
    width: int = 640,
) -> list[np.ndarray]:
    """Images de `model` pour chaque `qpos`, avec brouillard, ombres et reflets."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    data = mujoco.MjData(model)
    frames = []
    for qpos in qpos_seq:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        flags = renderer.scene.flags
        flags[mujoco.mjtRndFlag.mjRND_FOG] = True
        flags[mujoco.mjtRndFlag.mjRND_HAZE] = True
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def home_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Posture debout du keyframe "home"."""
    return model.key("home").qpos.copy()
