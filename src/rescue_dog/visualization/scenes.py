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
HFIELD = mujoco.mjtGeom.mjGEOM_HFIELD


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


def _facade(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Façade de béton sale, 4 × 4 fenêtres noires, quelques-unes éclairées."""
    wall = _mix(
        _noise(rng, size, octaves=4), (88, 80, 80), (128, 118, 112), (150, 140, 132)
    )
    cell = size // 4
    for i in range(4):
        for j in range(4):
            r0, c0 = i * cell + cell // 4, j * cell + cell // 4
            lit = rng.random() < 0.07
            color = (225, 125, 50) if lit else (18, 16, 20)
            wall[r0 : r0 + cell // 2, c0 : c0 + cell // 2] = color
    return wall


def _rock(rng: np.random.Generator, *colors, size: int = 512) -> np.ndarray:
    """Roche striée : bruit fractal étiré, pour les montagnes lointaines."""
    strata = _noise(rng, size, octaves=5)
    strata = (
        0.6 * strata
        + 0.4 * np.sin(np.linspace(0, 40 * np.pi, size))[:, None] * 0.5
        + 0.2
    )
    return _mix(np.clip(strata, 0, 1), *colors)


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


# Côté y < 0, les caméras « cinema » et « panorama » suivent le chien jusqu'à
# 4,2 m du couloir : aucun objet haut ne doit s'y trouver.
CAMERA_SIDE_CLEAR = 4.6


def _side_spot(
    rng: np.random.Generator,
    x_range=(-8, 30),
    y_min=1.4,
    y_max=9.0,
    tall: bool = False,
) -> tuple[float, float]:
    """Position hors du couloir où marche le chien (axe x, |y| < y_min).

    Un objet `tall` (plus haut que les caméras) évite aussi la bande où
    passent les caméras, côté y < 0.
    """
    x = rng.uniform(*x_range)
    y = rng.uniform(y_min, y_max) * rng.choice((-1, 1))
    if tall and y < 0:
        y = -max(-y, CAMERA_SIDE_CLEAR + rng.uniform(0, 2))
    return x, y


def _cone_mesh(spec: mujoco.MjSpec, name: str, radius: float, height: float, n=24):
    """Cône (enveloppe convexe d'un cercle et d'une pointe)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts = [[radius * np.cos(a), radius * np.sin(a), 0.0] for a in angles]
    verts.append([0.0, 0.0, height])
    spec.add_mesh(name=name, uservert=np.ravel(verts).tolist())


# --- Ciel peint : dégradé, soleil et nuages dans la texture du skybox ---

SKY_SIZE = 512  # pixels par face du cube


def _sky_directions(size: int) -> np.ndarray:
    """Direction (monde, z vers le haut) de chaque pixel des 6 faces empilées.

    Les faces suivent la convention des cube maps OpenGL (+X, -X, +Y, -Y, +Z,
    -Z, axe Y vers le haut), que MuJoCo affiche avec z vers le haut :
    (x, y, z) monde = (X, -Z, Y) OpenGL. Vérifié par un rendu de test.
    """
    c = (np.arange(size) + 0.5) / size * 2 - 1
    s, t = np.meshgrid(c, c)
    one = np.ones_like(s)
    faces = [
        (one, -t, -s),
        (-one, -t, s),
        (s, one, t),
        (s, -one, -t),
        (s, -t, one),
        (-s, -t, -one),
    ]
    gl = np.concatenate([np.stack(f, -1) for f in faces])
    d = np.stack([gl[..., 0], -gl[..., 2], gl[..., 1]], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _sample(tex: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Lecture bilinéaire d'une texture qui boucle, coordonnées en pixels."""
    n = tex.shape[0]
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - u0, v - v0
    a = tex[v0 % n, u0 % n] * (1 - fu) + tex[v0 % n, (u0 + 1) % n] * fu
    b = tex[(v0 + 1) % n, u0 % n] * (1 - fu) + tex[(v0 + 1) % n, (u0 + 1) % n] * fu
    return a * (1 - fv) + b * fv


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _paint_sky(
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
    zenith,
    horizon,
    sun_dir,
    sun_color,
    cloud_lit,
    cloud_shadow,
    cloud_cover: float,
    glow: float,
) -> None:
    """Remplace le dégradé du skybox par un ciel peint, en RGB 0-1."""
    d = _sky_directions(SKY_SIZE)
    z = d[..., 2]
    sun = np.asarray(sun_dir, dtype=float)
    sun /= np.linalg.norm(sun)
    cos_sun = d @ sun

    # Dégradé vertical, plus clair et plus chaud autour du soleil.
    t = _smoothstep(-0.05, 0.6, z) ** 0.6
    sky = _mix(t, np.multiply(horizon, 255), np.multiply(zenith, 255)) / 255
    halo = np.exp((cos_sun - 1) * 8)[..., None]
    sky = sky + glow * 0.35 * halo * np.asarray(sun_color)

    # Nuages : bruit fractal projeté sur un plafond nuageux plat.
    noise = _noise(rng, 512, octaves=7)
    detail = _noise(rng, 512, octaves=5)
    up = np.maximum(z, 0.02)
    u = d[..., 0] / up * 140 + 1000
    v = d[..., 1] / up * 140 + 1000
    n = 0.75 * _sample(noise, u, v) + 0.25 * _sample(detail, 2.3 * u, 2.3 * v)
    density = _smoothstep(1 - cloud_cover, 1 - cloud_cover + 0.22, n)
    density *= _smoothstep(0.03, 0.22, z)  # les nuages s'effacent à l'horizon
    shade = _smoothstep(0.35, 1.0, n)[..., None]
    toward_sun = np.clip(cos_sun, 0, 1)[..., None] ** 4
    cloud = (
        _mix(shade[..., 0], np.multiply(cloud_shadow, 255), np.multiply(cloud_lit, 255))
        / 255
    )
    cloud = cloud + 0.5 * toward_sun * np.asarray(sun_color)  # liseré doré
    sky = sky * (1 - 0.92 * density[..., None]) + cloud * 0.92 * density[..., None]

    # Disque du soleil et éclat serré, par-dessus les nuages.
    disc = _smoothstep(np.cos(np.deg2rad(1.6)), np.cos(np.deg2rad(1.1)), cos_sun)[
        ..., None
    ]
    sky = sky + glow * np.exp((cos_sun - 1) * 900)[..., None] * np.asarray(sun_color)
    sky = np.maximum(sky, disc * np.array([1.0, 0.98, 0.94]))

    sky = np.where((z < 0)[..., None], np.asarray(horizon), sky)  # sous l'horizon
    for tex in spec.textures:
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_NONE
            tex.width, tex.height, tex.nchannel = SKY_SIZE, 6 * SKY_SIZE, 3
            tex.data = _to_bytes(np.clip(sky, 0, 1) * 255)


# --- Montagnes lointaines : champs de hauteur purement visuels ---


def _ridges(rng: np.random.Generator, size: int) -> np.ndarray:
    """Relief de crêtes dans [0, 1] : bruit « ridged », pics plus pointus."""
    ridged = 1 - np.abs(2 * _noise(rng, size, octaves=7) - 1)
    return 0.55 * ridged**2.2 + 0.45 * _noise(rng, size, octaves=4) * ridged


def _mountains(
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
    height: float,
    material: str,
    inner: float = 110.0,
    outer: float = 200.0,
    half_size: float = 420.0,
    snow_line: float | None = None,
    snow_material: str = "",
) -> None:
    """Chaîne de montagnes en anneau autour de la scène.

    Le relief est nul jusqu'au rayon `inner` (et caché sous le sol), puis monte
    jusqu'à `outer`. Avec `snow_line`, un second champ de hauteur identique
    au-dessus de cette altitude, surélevé de quelques mètres, pose des neiges
    éternelles exactement sur les sommets (MuJoCo normalise les deux reliefs
    par leur maximum, commun).
    """
    n = 320
    xs = np.linspace(-half_size, half_size, n)
    x, y = np.meshgrid(xs, xs)
    ring = _smoothstep(inner, outer, np.hypot(x, y))
    h = ring * (0.15 + 0.85 * _ridges(rng, n))
    h /= h.max()
    # La roche part de 3 m sous le sol ; la neige, 2,8 m plus haut, reste
    # elle aussi sous le sol là où il n'y en a pas (relief nul).
    layers = [("montagnes", h, material, -3.0)]
    if snow_line is not None:
        snow = np.where(h > snow_line, h, 0.0)
        layers.append(("neiges_eternelles", snow, snow_material, -0.2))
    for name, data, mat, z in layers:
        spec.add_hfield(
            name=name,
            size=[half_size, half_size, height, 1.0],
            nrow=n,
            ncol=n,
            userdata=data.ravel().tolist(),
        )
        _deco(spec.worldbody, HFIELD, (), (0, 0, z), mat, hfieldname=name)


# --- Ambiance commune ---


def _atmosphere(spec: mujoco.MjSpec, horizon, ambient=0.3, haze: float = 0.35) -> None:
    """Voile d'horizon, lumière ambiante et plan lointain.

    Pas de brouillard MuJoCo : il s'applique aussi au skybox et noierait le
    ciel peint. Les objets lointains pâlissent d'eux-mêmes (matériaux teintés
    vers l'horizon, voir `_haze_material`). Les distances de MuJoCo sont en
    unités de `extent` du modèle (0,8 m pour le Go1).
    """
    spec.visual.rgba.haze = list(horizon) + [1]
    spec.visual.map.haze = haze
    spec.visual.map.znear = 0.02
    spec.visual.map.zfar = 3000
    spec.visual.headlight.ambient = [ambient] * 3
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    spec.visual.quality.shadowsize = 8192
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080


def _haze_material(
    spec: mujoco.MjSpec, name: str, rgb, horizon, depth: float, **kw
) -> None:
    """Matériau d'un objet lointain : sa couleur fondue dans l'horizon.

    `depth` dans [0, 1] imite la perspective atmosphérique : 0 garde la
    couleur, 1 la remplace par celle de l'horizon, qui brille (émission) pour
    rester claire même à contre-jour.
    """
    color = (1 - depth) * np.asarray(rgb) + depth * np.asarray(horizon)
    _material(spec, name, rgba=list(color) + [1], emission=depth, **kw)


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
        ambient=[0.18, 0.18, 0.2],
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

    _texture(
        spec, "roche_ocre", _rock(rng, (70, 46, 40), (128, 80, 58), (176, 120, 84))
    )

    # Fin d'après-midi poussiéreuse : soleil bas, en face de la caméra « cinema ».
    sun = (-0.25, 1.0, 0.2)
    horizon = (0.97, 0.66, 0.42)
    _paint_sky(
        spec,
        rng,
        zenith=(0.16, 0.2, 0.38),
        horizon=horizon,
        sun_dir=sun,
        sun_color=(1.0, 0.76, 0.46),
        cloud_lit=(1.0, 0.74, 0.52),
        cloud_shadow=(0.42, 0.32, 0.4),
        cloud_cover=0.5,
        glow=1.0,
    )
    _atmosphere(spec, horizon, ambient=0.42, haze=0.45)
    _sun(spec, direction=-np.asarray(sun), color=(1.0, 0.8, 0.6), intensity=1.1)
    _haze_material(
        spec,
        "roche_ocre",
        (0.55, 0.38, 0.32),
        horizon,
        0.45,
        texture="roche_ocre",
        texrepeat=(40, 40),
        specular=0.0,
    )
    _mountains(spec, rng, height=34, material="roche_ocre")
    _ruined_skyline(spec, rng, horizon, sun_azimuth=np.arctan2(sun[1], sun[0]))

    wb = spec.worldbody
    # Gravats de béton et de brique éparpillés de part et d'autre du couloir.
    for _ in range(220):
        x, y = _side_spot(rng)
        s = rng.uniform(0.04, 0.28, size=3) * np.array([1, 1, 0.5])
        mat = "beton" if rng.random() < 0.75 else "brique"
        _deco(wb, BOX, s, (x, y, s[2] * 0.6), mat, _random_quat(rng, 0.5))
    # Dalles effondrées, inclinées contre le sol.
    for _ in range(18):
        x, y = _side_spot(rng, y_min=2.5, tall=True)
        s = (rng.uniform(0.6, 1.4), rng.uniform(0.4, 0.9), 0.06)
        _deco(wb, BOX, s, (x, y, 0.3), "beton", _random_quat(rng, 0.7))
    # Piliers brisés, debout ou couchés.
    for _ in range(10):
        x, y = _side_spot(rng, y_min=3, tall=True)
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
        x, y = _side_spot(rng, y_min=2, tall=True)
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
        x, y = _side_spot(rng, y_min=1.8, y_max=6, tall=True)
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
        y = 2.6 if i % 2 else -5.2
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


def _ruined_skyline(
    spec: mujoco.MjSpec, rng: np.random.Generator, horizon, sun_azimuth: float
) -> None:
    """Ville effondrée à l'horizon : immeubles éventrés, incendies et fumées.

    Aucun immeuble devant le soleil, pour garder le contre-jour.
    """
    _texture(spec, "facade", _facade(rng))
    facade = {"texture": "facade", "texuniform": True, "texrepeat": (0.09, 0.09)}
    _haze_material(
        spec, "immeuble", (0.5, 0.44, 0.46), horizon, 0.2, specular=0.0, **facade
    )
    _haze_material(
        spec, "immeuble_clair", (0.7, 0.62, 0.6), horizon, 0.2, specular=0.0, **facade
    )
    _material(spec, "braise", rgba=(1.0, 0.45, 0.1, 1), emission=1.0)
    _material(spec, "fumee", rgba=(0.4, 0.36, 0.36, 0.18), specular=0, emission=0.3)
    wb = spec.worldbody
    fires = []
    for _ in range(46):
        az = rng.uniform(-np.pi, np.pi)
        if abs(np.angle(np.exp(1j * (az - sun_azimuth)))) < 0.28:
            continue
        r = rng.uniform(85, 150)
        x, y = 8 + r * np.cos(az), r * np.sin(az)
        w, depth, h = rng.uniform(4, 9), rng.uniform(4, 9), rng.uniform(8, 30)
        mat = "immeuble" if rng.random() < 0.7 else "immeuble_clair"
        yaw = _random_quat(rng, 0.0)
        _deco(wb, BOX, (w, depth, h / 2), (x, y, h / 2), mat, yaw)
        if rng.random() < 0.55:  # étage effondré, en biais sur le toit
            _deco(
                wb,
                BOX,
                (w * 0.8, depth * 0.9, 0.5),
                (x, y, h + 0.8),
                mat,
                _random_quat(rng, 0.5),
            )
        if rng.random() < 0.25:
            fires.append((x, y, h))
    # Incendies sur les toits et colonnes de fumée poussées par le vent (+x).
    for x, y, h in fires[:6]:
        _deco(wb, SPHERE, (1.2,), (x, y, h + 0.6), "braise")
        # Bouffées qui se chevauchent, grossissent et s'inclinent avec le vent.
        for k in range(30):
            t = k / 29
            radius = 1.5 + 8 * t
            center = (x + 30 * t**2, y + rng.normal(0, 1.5), h + 2 + 40 * t)
            jitter = rng.normal(0, radius * 0.3, size=3)
            _deco(wb, SPHERE, (radius,), tuple(np.add(center, jitter)), "fumee")


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

    _texture(
        spec, "roche_bleue", _rock(rng, (52, 58, 70), (92, 102, 118), (140, 150, 165))
    )

    # Matin alpin : grand ciel bleu, cumulus et sommets enneigés.
    sun = (0.55, 0.65, 0.5)
    horizon = (0.8, 0.88, 0.97)
    _paint_sky(
        spec,
        rng,
        zenith=(0.12, 0.32, 0.72),
        horizon=horizon,
        sun_dir=sun,
        sun_color=(1.0, 0.97, 0.9),
        cloud_lit=(1.0, 1.0, 1.0),
        cloud_shadow=(0.68, 0.74, 0.86),
        cloud_cover=0.42,
        glow=0.6,
    )
    _atmosphere(spec, horizon, ambient=0.45, haze=0.35)
    _sun(spec, direction=-np.asarray(sun), color=(0.97, 0.98, 1.0), intensity=0.95)
    _haze_material(
        spec,
        "roche_bleue",
        (0.42, 0.47, 0.56),
        horizon,
        0.35,
        texture="roche_bleue",
        texrepeat=(40, 40),
        specular=0.1,
    )
    _haze_material(
        spec,
        "neige_eternelle",
        (0.95, 0.97, 1.0),
        horizon,
        0.15,
        texture="neige",
        texrepeat=(60, 60),
    )
    _mountains(
        spec,
        rng,
        height=80,
        material="roche_bleue",
        inner=150,
        outer=260,
        snow_line=0.38,
        snow_material="neige_eternelle",
    )
    _mountain_hut(spec)

    wb = spec.worldbody
    # Congères de neige sur les rives.
    for _ in range(70):
        x, y = _side_spot(rng, y_min=3.5, y_max=12, tall=True)
        s = (rng.uniform(0.5, 1.8), rng.uniform(0.4, 1.2), rng.uniform(0.15, 0.5))
        _deco(wb, ELLIPSOID, s, (x, y, 0), "neige", _random_quat(rng, 0))
    # Blocs de glace brillants près du couloir.
    for _ in range(30):
        x, y = _side_spot(rng, y_min=1.6, y_max=5, tall=True)
        s = rng.uniform(0.1, 0.45, size=3)
        _deco(wb, BOX, s, (x, y, s[2] * 0.5), "bloc_glace", _random_quat(rng, 0.4))
    # Forêt de sapins enneigés : trois étages de cônes et un tronc. Quelques
    # tailles de maillages, partagées par tous les arbres.
    sizes = (1.8, 2.6, 3.4, 4.4, 5.6)
    for i, h in enumerate(sizes):
        for k, r in enumerate((0.42, 0.32, 0.22)):
            _cone_mesh(spec, f"sapin_{i}_{k}", radius=r * h, height=0.42 * h)
    for x_range, y_min, y_max, count in (
        ((-8, 30), 6, 14, 45),
        ((-30, 60), 14, 45, 140),
    ):
        for _ in range(count):
            x, y = _side_spot(rng, x_range, y_min, y_max, tall=True)
            i = rng.integers(len(sizes))
            h = sizes[i]
            _deco(wb, CYLINDER, (0.04 * h, 0.25 * h), (x, y, 0.25 * h), "ecorce")
            for k, z in enumerate((0.22, 0.45, 0.66)):
                mat = "sapin_neige" if k == 2 else "sapin"
                _deco(wb, MESH, (), (x, y, z * h), mat, meshname=f"sapin_{i}_{k}")
    # Piquets de balisage rouges le long du couloir.
    for x in np.arange(-4, 30, 4.0):
        for side in (-1, 1):
            _deco(wb, CYLINDER, (0.02, 0.4), (x, side * 1.3, 0.4), "balise")


def _mountain_hut(spec: mujoco.MjSpec) -> None:
    """Refuge de montagne au bout du couloir : la destination de la mission."""
    _material(spec, "bois", rgba=(0.42, 0.26, 0.15, 1), specular=0.05)
    _material(spec, "toit_rouge", rgba=(0.62, 0.1, 0.07, 1), specular=0.2)
    _material(spec, "fenetre", rgba=(1.0, 0.78, 0.4, 1), emission=0.9)
    _material(spec, "drapeau", rgba=(1.0, 0.42, 0.0, 1), emission=0.2)
    wb = spec.worldbody
    x, y = 26.0, 5.5
    _deco(wb, BOX, (2.2, 1.6, 1.1), (x, y, 1.1), "bois")
    for side in (-1, 1):  # toit à deux pans couvert de neige
        q = [np.cos(0.32), -side * np.sin(0.32), 0, 0]
        _deco(wb, BOX, (2.5, 1.0, 0.08), (x, y + side * 0.85, 2.55), "toit_rouge", q)
        _deco(wb, BOX, (2.5, 1.0, 0.05), (x, y + side * 0.85, 2.66), "neige", q)
    _deco(wb, BOX, (0.35, 0.02, 0.3), (x - 0.9, y - 1.61, 1.3), "fenetre")
    _deco(wb, BOX, (0.35, 0.02, 0.3), (x + 0.9, y - 1.61, 1.3), "fenetre")
    _deco(wb, BOX, (0.25, 0.25, 0.6), (x + 1.2, y + 0.6, 3.0), "bois")  # cheminée
    for k in range(5):
        r = 0.3 + 0.25 * k
        _deco(wb, SPHERE, (r,), (x + 1.2 + 0.3 * k, y + 0.6, 3.8 + 0.7 * k), "neige")
    # Mât et drapeau orange de l'équipe de secours.
    _deco(wb, CYLINDER, (0.04, 2.5), (x - 3.0, y - 1.5, 2.5), "ecorce")
    _deco(wb, BOX, (0.45, 0.01, 0.28), (x - 2.53, y - 1.5, 4.6), "drapeau")


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
    # « cinema » vise assez haut pour garder l'horizon (montagnes, soleil).
    _follow_camera(trunk, "cinema", offset=(1.6, -2.6, 0.45), target=(0.0, 0.0, 0.18))
    _follow_camera(trunk, "large", offset=(-3.0, -4.5, 2.2))
    _follow_camera(trunk, "panorama", offset=(-3.0, -4.2, 0.8), target=(2.0, 0.0, 0.35))
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
    """Images de `model` pour chaque `qpos`, avec voile d'horizon, ombres et reflets."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    data = mujoco.MjData(model)
    frames = []
    for qpos in qpos_seq:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        flags = renderer.scene.flags
        flags[mujoco.mjtRndFlag.mjRND_FOG] = False
        flags[mujoco.mjtRndFlag.mjRND_HAZE] = True
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def home_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Posture debout du keyframe "home"."""
    return model.key("home").qpos.copy()
