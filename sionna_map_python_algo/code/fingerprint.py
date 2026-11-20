# [원본과 다름] 모듈 docstring 전체 수정 (원본: LOS + 1차 벽 반사 4개 해석식 추적기 설명)
"""Task 1.1: indoor channel fingerprint built with Sionna RT (CPU ray tracing).

This is the only module whose behaviour differs from the pure-Python project.
The stored channel keeps the exact same format, shape (Ny, Nx, 4, 64), grid,
caching and dictionary keys, so every downstream algorithm is unchanged.

Map model
1. Sionna RT traces the handout room (perimeter walls, floor, ceiling and all
   12 obstacles, ITU-R P.2040 materials) from each of the four ULA elements to
   all 720 cell centres.  Paths may contain up to ``config.RT_MAX_DEPTH``
   interactions mixing specular reflection, refraction (slab transmission),
   diffraction and diffuse scattering, i.e. thousands of paths per link
   instead of the handout's LOS + four first-order wall reflections.
2. Small environmental fluctuations (air flow, insects, small moving objects)
   are modelled as Gaussian: each path gets a Gaussian path-length and
   amplitude jitter, and a weak circular complex Gaussian diffuse field is added.
3. The paths are summed coherently per subcarrier,
       H[k] = sum_p a_p g_p exp(-j 2 pi f_k (tau_p + dtau_p)) * lambda_k / lambda_c.

The original analytic tracer (LOS + four reflections) is kept below as
``build_handout_fingerprint`` only to draw a comparison figure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import config

EPS = 1e-12


def _segment_parameter_interval_for_z(p0: np.ndarray, p1: np.ndarray, height_m: float) -> tuple[float, float] | None:
    """Return s in [0,1] for which p0+s*(p1-p0) lies in 0 <= z <= height."""
    dz = float(p1[2] - p0[2])
    if abs(dz) < EPS:
        return (0.0, 1.0) if 0.0 <= p0[2] <= height_m else None
    s0, s1 = (0.0 - p0[2]) / dz, (height_m - p0[2]) / dz
    lo, hi = max(0.0, min(s0, s1)), min(1.0, max(s0, s1))
    return (lo, hi) if hi > lo else None


def line_box_chord_m(p0: np.ndarray, p1: np.ndarray, obstacle: config.Obstacle) -> float:
    """3-D slab-method chord length inside a floor-mounted axis-aligned box."""
    if obstacle.size_xyz_m is None:
        raise ValueError("Box obstacle requires size_xyz_m")
    dx, dy, dz = obstacle.size_xyz_m
    cx, cy = obstacle.center_xy_m
    lo = np.array([cx - dx / 2.0, cy - dy / 2.0, 0.0])
    hi = np.array([cx + dx / 2.0, cy + dy / 2.0, dz])
    direction = p1 - p0
    s_enter, s_exit = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < EPS:
            if p0[axis] < lo[axis] or p0[axis] > hi[axis]:
                return 0.0
            continue
        s_a = (lo[axis] - p0[axis]) / direction[axis]
        s_b = (hi[axis] - p0[axis]) / direction[axis]
        s_enter, s_exit = max(s_enter, min(s_a, s_b)), min(s_exit, max(s_a, s_b))
        if s_enter >= s_exit:
            return 0.0
    return max(0.0, s_exit - s_enter) * float(np.linalg.norm(direction))


def line_cylinder_chord_m(p0: np.ndarray, p1: np.ndarray, obstacle: config.Obstacle) -> float:
    """Chord length inside a vertical finite cylinder, including its z limits."""
    if obstacle.radius_m is None:
        raise ValueError("Cylinder obstacle requires radius_m")
    direction = p1 - p0
    dxy = direction[:2]
    origin = p0[:2] - np.asarray(obstacle.center_xy_m)
    a = float(dxy @ dxy)
    if a < EPS:
        return 0.0
    b = 2.0 * float(origin @ dxy)
    c = float(origin @ origin) - obstacle.radius_m**2
    discriminant = b * b - 4.0 * a * c
    if discriminant <= 0.0:
        return 0.0
    root = np.sqrt(discriminant)
    s0, s1 = (-b - root) / (2.0 * a), (-b + root) / (2.0 * a)
    z_interval = _segment_parameter_interval_for_z(p0, p1, obstacle.height_m)
    if z_interval is None:
        return 0.0
    lo, hi = max(0.0, s0, z_interval[0]), min(1.0, s1, z_interval[1])
    return max(0.0, hi - lo) * float(np.linalg.norm(direction))


def obstacle_penetration_db(p0: np.ndarray, p1: np.ndarray) -> float:
    """Accumulated amplitude loss in dB along one LOS or reflected ray."""
    total_db = 0.0
    for obstacle in config.OBSTACLES:
        chord_m = line_box_chord_m(p0, p1, obstacle) if obstacle.kind == "box" else line_cylinder_chord_m(p0, p1, obstacle)
        total_db += obstacle.alpha_db_per_m * chord_m
    return total_db


def reflected_image_source_m(antenna_position_m: np.ndarray, wall: str) -> np.ndarray:
    """Mirror one antenna across a perimeter wall for the image-source method."""
    image = antenna_position_m.copy()
    if wall == "x=0":
        image[0] *= -1.0
    elif wall == "x=18":
        image[0] = 2.0 * config.ROOM_X_M - image[0]
    elif wall == "y=0":
        image[1] *= -1.0
    elif wall == "y=10":
        image[1] = 2.0 * config.ROOM_Y_M - image[1]
    else:
        raise ValueError(f"Unknown perimeter wall: {wall}")
    return image


def complex_path_gain(p0_m: np.ndarray, p1_m: np.ndarray, reflection_coefficient: float) -> np.ndarray:
    """Return h_path[k] = rho 10^(-Lpen/20) lambda/(4 pi d) exp(-j2pi d/lambda)."""
    distance_m = float(np.linalg.norm(p1_m - p0_m))
    if distance_m < EPS:
        raise ValueError("Ray path length must be nonzero")
    penetration_db = obstacle_penetration_db(p0_m, p1_m)
    amplitude = reflection_coefficient * 10.0 ** (-penetration_db / 20.0)
    lam = config.SUBCARRIER_WAVELENGTH_M
    return amplitude * lam / (4.0 * np.pi * distance_m) * np.exp(-1j * 2.0 * np.pi * distance_m / lam)


def channel_at_position(receiver_position_m: np.ndarray) -> np.ndarray:
    """Return H[antenna, subcarrier] from LOS plus four coherent reflections."""
    h = np.zeros((config.N_ANT_BS, config.N_SC), dtype=np.complex128)
    for antenna_index, antenna_position in enumerate(config.bs_antenna_positions_m()):
        h[antenna_index] += complex_path_gain(antenna_position, receiver_position_m, 1.0)
        for wall in ("x=0", "x=18", "y=0", "y=10"):
            image_source = reflected_image_source_m(antenna_position, wall)
            h[antenna_index] += complex_path_gain(image_source, receiver_position_m, config.CONCRETE_RHO)
    return h


def grid_coordinates_m() -> tuple[np.ndarray, np.ndarray]:
    """Return the 36 x 20 cell centres specified for the 0.5 m grid."""
    x = np.arange(config.GRID_RES_M / 2.0, config.ROOM_X_M, config.GRID_RES_M)
    y = np.arange(config.GRID_RES_M / 2.0, config.ROOM_Y_M, config.GRID_RES_M)
    return x, y


def is_inside_obstacle(x_m: float, y_m: float) -> bool:
    """True for footprint occupancy; planning later adds the 0.2 m safety buffer."""
    for obstacle in config.OBSTACLES:
        cx, cy = obstacle.center_xy_m
        if obstacle.kind == "box":
            assert obstacle.size_xyz_m is not None
            dx, dy, _ = obstacle.size_xyz_m
            if abs(x_m - cx) <= dx / 2.0 and abs(y_m - cy) <= dy / 2.0:
                return True
        else:
            assert obstacle.radius_m is not None
            if (x_m - cx) ** 2 + (y_m - cy) ** 2 <= obstacle.radius_m**2:
                return True
    return False


def _scene_signature() -> str:
    """Invalidate a cached fingerprint whenever scene-defining constants change."""
    payload = {
        "schema": "sionna-rt-map-v1",  # [원본과 다름] 원본: "cell-centre-grid-v2"
        "freq": config.FREQ_HZ, "bandwidth": config.BW_HZ, "n_sc": config.N_SC,
        "bs": config.BS_CENTER_M.tolist(), "grid": config.GRID_RES_M,
        "obstacles": [obstacle.__dict__ for obstacle in config.OBSTACLES],
        # [원본과 다름] 아래 두 줄 추가: RT·노이즈 설정이 바뀌면 캐시 무효화
        "rt": _rt_settings(),
        "noise": [config.NOISE_PATH_LENGTH_STD_M, config.NOISE_AMPLITUDE_STD, config.NOISE_DIFFUSE_K_DB, config.NOISE_SEED],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


# [원본과 다름] 원본의 build_fingerprint()를 이름만 바꿔 남김 (내용은 원본 그대로, 비교 그림용)
def build_handout_fingerprint(verbose: bool = True) -> dict[str, np.ndarray]:
    """Original handout tracer (LOS + four wall reflections); used only for comparison."""
    x_m, y_m = grid_coordinates_m()
    h = np.empty((len(y_m), len(x_m), config.N_ANT_BS, config.N_SC), dtype=np.complex64)
    occupied = np.zeros((len(y_m), len(x_m)), dtype=bool)
    total = len(x_m) * len(y_m)
    for iy, y in enumerate(y_m):
        for ix, x in enumerate(x_m):
            receiver = np.array([x, y, config.RX_HEIGHT_M], dtype=float)
            h[iy, ix] = channel_at_position(receiver)
            occupied[iy, ix] = is_inside_obstacle(x, y)
        if verbose:
            print(f"Ray tracing row {iy + 1:2d}/{len(y_m)} ({(iy + 1) * len(x_m):4d}/{total} cells)")
    return {"H": h, "x_m": x_m, "y_m": y_m, "occupied": occupied, "signature": np.array(_scene_signature())}


# ===== [원본과 다름] Sionna RT 맵 생성 + 가우시안 노이즈 (새로 추가한 부분 전체) =====
# ---------------------------------------------------------------------------
# Sionna RT map
# ---------------------------------------------------------------------------
INTERACTION_NAMES = {1: "specular", 2: "diffuse", 4: "refraction", 8: "diffraction"}


def _rt_settings() -> dict:
    from sionna_scene import SURFACE_MATERIALS
    return {
        "depth": config.RT_MAX_DEPTH, "samples": config.RT_SAMPLES_PER_SRC,
        "diffuse_depth": config.RT_DIFFUSE_MAX_DEPTH, "diffuse_samples": config.RT_DIFFUSE_SAMPLES_PER_SRC,
        "refraction": config.RT_REFRACTION, "diffraction": config.RT_DIFFRACTION,
        "edge_diffraction": config.RT_EDGE_DIFFRACTION, "diffuse": config.RT_DIFFUSE,
        "materials": SURFACE_MATERIALS, "seed": config.RANDOM_SEED,
    }


def _rt_signature() -> str:
    payload = {"rt": _rt_settings(), "bs": config.BS_CENTER_M.tolist(), "f": config.FREQ_HZ,
               "grid": config.GRID_RES_M, "rx_h": config.RX_HEIGHT_M,
               "obstacles": [obstacle.__dict__ for obstacle in config.OBSTACLES]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _solve_batch(solver, scene, rt, mi, receivers: np.ndarray, diffuse_pass: bool) -> list[dict[str, np.ndarray]]:
    """Run one PathSolver call for a batch of receivers; return valid paths per (rx, tx) link."""
    names = []
    for index, position in enumerate(receivers):
        name = f"rx{index:03d}"
        scene.add(rt.Receiver(name=name, position=mi.Point3f(*map(float, position))))
        names.append(name)
    if diffuse_pass:
        # Diffuse scattering creates one path per scattering point and target, so it uses
        # fewer rays; only paths that contain a diffuse interaction are kept from this pass.
        depth, samples = config.RT_DIFFUSE_MAX_DEPTH, config.RT_DIFFUSE_SAMPLES_PER_SRC
    else:
        depth, samples = config.RT_MAX_DEPTH, config.RT_SAMPLES_PER_SRC
    paths = solver(scene, max_depth=depth, los=not diffuse_pass, specular_reflection=config.RT_SPECULAR,
                   diffuse_reflection=diffuse_pass, refraction=config.RT_REFRACTION,
                   diffraction=config.RT_DIFFRACTION, edge_diffraction=config.RT_EDGE_DIFFRACTION,
                   synthetic_array=False, samples_per_src=samples,
                   max_num_paths_per_src=config.RT_MAX_PATHS_PER_SRC, seed=config.RANDOM_SEED)
    a_re, a_im = paths.a
    a = (np.asarray(a_re) + 1j * np.asarray(a_im))[:, 0, :, 0, :]          # (rx, tx, P)
    tau = np.asarray(paths.tau)[:, 0, :, 0, :]
    valid = np.asarray(paths.valid)[:, 0, :, 0, :].astype(bool)
    inter = np.asarray(paths.interactions)[:, :, 0, :, 0, :]              # (depth, rx, tx, P)
    kinds = np.bitwise_or.reduce(inter.astype(np.uint8), axis=0)          # interaction types used by each path
    n_inter = (inter != 0).sum(axis=0).astype(np.uint8)
    if diffuse_pass:
        valid &= (kinds & 2) != 0
    for name in names:
        scene.remove(name)
    out = []
    for m in range(len(receivers)):
        links = []
        for n in range(a.shape[1]):
            sel = valid[m, n] & (np.abs(a[m, n]) > 0)
            links.append({"a": a[m, n, sel].astype(np.complex64), "tau": tau[m, n, sel].astype(np.float64),
                          "kinds": kinds[m, n, sel], "depth": n_inter[m, n, sel]})
        out.append(links)
    return out


def run_sionna_rt(verbose: bool = True) -> dict[str, np.ndarray]:
    """Trace every (BS antenna, cell centre) link with Sionna RT on the CPU (LLVM)."""
    import time

    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("llvm_ad_mono_polarized")
    import sionna.rt as rt
    from sionna_scene import build_scene_files

    rt_xml, _ = build_scene_files()
    scene = rt.load_scene(str(rt_xml), merge_shapes=False)
    scene.frequency = config.FREQ_HZ
    # Each ULA element is its own isotropic transmitter at its exact position.
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    for n, position in enumerate(config.bs_antenna_positions_m()):
        scene.add(rt.Transmitter(name=f"bs_ant{n}", position=mi.Point3f(*map(float, position))))
    x_m, y_m = grid_coordinates_m()
    gx, gy = np.meshgrid(x_m, y_m)
    receivers = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, config.RX_HEIGHT_M)]
    solver = rt.PathSolver()
    if verbose:
        print(f"[Sionna RT] variant={mi.variant()}  {config.N_ANT_BS} Tx x {len(receivers)} Rx  "
              f"depth={config.RT_MAX_DEPTH}  rays/antenna={config.RT_SAMPLES_PER_SRC:,}  "
              f"refraction={config.RT_REFRACTION} diffraction={config.RT_DIFFRACTION} diffuse={config.RT_DIFFUSE}")
    links: list[list[dict[str, np.ndarray]]] = []
    batch = config.RT_RX_BATCH
    t0 = time.time()
    for start in range(0, len(receivers), batch):
        chunk = receivers[start:start + batch]
        result = _solve_batch(solver, scene, rt, mi, chunk, diffuse_pass=False)
        if config.RT_DIFFUSE:
            extra = _solve_batch(solver, scene, rt, mi, chunk, diffuse_pass=True)
            result = [[{key: np.concatenate([l1[key], l2[key]]) for key in l1} for l1, l2 in zip(r1, r2)]
                      for r1, r2 in zip(result, extra)]
        links += result
        if verbose:
            done = start + len(chunk)
            eta = (time.time() - t0) / done * (len(receivers) - done)
            print(f"[Sionna RT] {done:3d}/{len(receivers)} cells  ({time.time() - t0:6.0f} s elapsed, ~{eta:5.0f} s left)")
    # Flatten to (link index, path) arrays; link index = cell * N_ANT_BS + antenna.
    counts = np.array([len(link["a"]) for cell in links for link in cell])
    flat = {key: np.concatenate([link[key] for cell in links for link in cell]) for key in ("a", "tau", "kinds", "depth")}
    flat["link_index"] = np.repeat(np.arange(len(counts)), counts).astype(np.int32)
    flat["counts"] = counts.reshape(len(y_m), len(x_m), config.N_ANT_BS)
    flat["signature"] = np.array(_rt_signature())
    flat["runtime_s"] = np.array(time.time() - t0)
    return flat


def load_or_run_sionna_rt(force: bool = False, verbose: bool = True) -> dict[str, np.ndarray]:
    """Cache the raw (noise-free) Sionna paths so noise settings can change without re-tracing."""
    path = config.RT_PATHS_PATH
    if not force and path.exists():
        cached = dict(np.load(path))
        if str(cached["signature"]) == _rt_signature():
            if verbose:
                print(f"Loading cached Sionna RT paths: {path}")
            return cached
    data = run_sionna_rt(verbose)
    np.savez_compressed(path, **data)
    if verbose:
        print(f"Saved Sionna RT paths: {path}")
    return data


def paths_to_channel(rt_paths: dict[str, np.ndarray], with_noise: bool = True) -> np.ndarray:
    """Coherent per-subcarrier sum of all paths, optionally with Gaussian environment noise."""
    ny, nx, n_ant = rt_paths["counts"].shape
    n_links = ny * nx * n_ant
    a = rt_paths["a"].astype(np.complex128)
    tau = rt_paths["tau"].astype(np.float64)
    link = rt_paths["link_index"]
    rng = np.random.default_rng(config.NOISE_SEED)
    if with_noise:
        a = a * np.maximum(1.0 + config.NOISE_AMPLITUDE_STD * rng.standard_normal(len(a)), 0.0)
        tau = tau + config.NOISE_PATH_LENGTH_STD_M * rng.standard_normal(len(tau)) / config.C_MPS
    h = np.zeros((n_links, config.N_SC), dtype=np.complex128)
    # Sionna's a lacks the carrier phase; exp(-j 2 pi f_k tau) supplies carrier + subcarrier phase.
    for k, f_k in enumerate(config.SUBCARRIER_HZ):
        term = a * np.exp(-2j * np.pi * f_k * tau)
        h[:, k] = np.bincount(link, term.real, n_links) + 1j * np.bincount(link, term.imag, n_links)
    h *= (config.SUBCARRIER_WAVELENGTH_M / config.LAMBDA_C_M)[None, :]
    h = h.reshape(ny, nx, n_ant, config.N_SC)
    if with_noise:
        # Weak diffuse field from many small moving scatterers: circular complex Gaussian,
        # power = (cell's mean deterministic power) / K, independent per antenna and subcarrier.
        cell_power = np.mean(np.abs(h) ** 2, axis=(2, 3), keepdims=True)
        sigma = np.sqrt(cell_power / 10.0 ** (config.NOISE_DIFFUSE_K_DB / 10.0) / 2.0)
        h = h + sigma * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
    return h


def fill_cells_without_paths(h: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Cells deep inside a pillar receive no ray at all; copy the nearest traced cell so the
    downstream dB interpolation stays finite (these cells are never on a planned path)."""
    empty = counts.sum(axis=2) == 0
    if not empty.any():
        return empty
    full = np.argwhere(~empty)
    for iy, ix in np.argwhere(empty):
        jy, jx = full[np.argmin((full[:, 0] - iy) ** 2 + (full[:, 1] - ix) ** 2)]
        h[iy, ix] = h[jy, jx]
    return empty


def build_fingerprint(verbose: bool = True, force_rt: bool = False) -> dict[str, np.ndarray]:
    """Sionna RT paths + Gaussian environment noise -> AMR-agnostic channel fingerprint."""
    x_m, y_m = grid_coordinates_m()
    rt_paths = load_or_run_sionna_rt(force=force_rt, verbose=verbose)
    h = paths_to_channel(rt_paths, with_noise=True)
    no_path = fill_cells_without_paths(h, rt_paths["counts"])
    occupied = np.array([[is_inside_obstacle(x, y) for x in x_m] for y in y_m])
    if verbose:
        counts = rt_paths["counts"]
        print(f"Sionna paths per link: mean {counts.mean():.0f}, max {counts.max()}, total {counts.sum():,}; "
              f"cells without any path (filled from neighbour): {int(no_path.sum())}")
    return {"H": h.astype(np.complex64), "x_m": x_m, "y_m": y_m, "occupied": occupied,
            "no_path": no_path, "signature": np.array(_scene_signature())}

# ===== [원본과 다름 끝] =====


def save_fingerprint(data: dict[str, np.ndarray], path: Path = config.FINGERPRINT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The handout explicitly asks for a dictionary-style .npy fingerprint cache.
    np.save(path, data, allow_pickle=True)


def load_fingerprint(path: Path = config.FINGERPRINT_PATH) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    if str(data["signature"].item()) != _scene_signature():
        raise ValueError("Cached fingerprint does not match the current scene configuration")
    return data


def load_or_build_fingerprint(force: bool = False, verbose: bool = True) -> dict[str, np.ndarray]:
    if not force and config.FINGERPRINT_PATH.exists():
        try:
            if verbose:
                print(f"Loading cached fingerprint: {config.FINGERPRINT_PATH}")
            return load_fingerprint()
        except ValueError:
            if verbose:
                print("Cached fingerprint is stale; rebuilding it.")
    if verbose:
        # [원본과 다름] 메시지와 호출 대상 변경: 이제 build_fingerprint()가 Sionna RT 버전
        print("Building channel fingerprint with Sionna RT …")
    data = build_fingerprint(verbose=verbose)
    save_fingerprint(data)
    if verbose:
        print(f"Saved fingerprint: {config.FINGERPRINT_PATH}")
    return data
