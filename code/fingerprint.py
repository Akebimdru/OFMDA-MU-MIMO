"""Task 1.1: deterministic, complex indoor channel fingerprint ray tracer.

The stored channel is AMR-agnostic and has shape (Ny, Nx, 4, 64).  It is the
coherent sum of one LOS and four first-order perimeter-wall reflection paths.
Obstacle intersections only add material penetration loss; they do not reflect.
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
        "schema": "cell-centre-grid-v2",
        "freq": config.FREQ_HZ, "bandwidth": config.BW_HZ, "n_sc": config.N_SC,
        "bs": config.BS_CENTER_M.tolist(), "grid": config.GRID_RES_M,
        "obstacles": [obstacle.__dict__ for obstacle in config.OBSTACLES],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def build_fingerprint(verbose: bool = True) -> dict[str, np.ndarray]:
    """Trace all grid cells and return an AMR-agnostic channel fingerprint."""
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
        print("Building channel fingerprint from the specified scene …")
    data = build_fingerprint(verbose=verbose)
    save_fingerprint(data)
    if verbose:
        print(f"Saved fingerprint: {config.FINGERPRINT_PATH}")
    return data
