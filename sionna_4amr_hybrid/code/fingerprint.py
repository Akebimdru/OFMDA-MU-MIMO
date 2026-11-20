"""새 맵의 채널 fingerprint: Sionna RT (CPU) + 가우시안 환경 노이즈 3종.

이전 sionna_map_python_algo/code/fingerprint.py와 같은 방식이고, 맵/BS 위치만 새 과제 값이다.

1. Sionna RT: BS 안테나 4개를 각각 송신기로, 셀 중심 512개(높이 0.5 m)를 수신기로 둔다.
   ITU 재질, 경로당 상호작용 최대 6번(경면 반사/굴절/회절), 확산 산란은 별도 패스(깊이 4).
2. 예측할 수 없는 환경 잡음 3종 (기류, 벌레, 작은 움직이는 물체):
   (a) 경로 길이 지터 σ = 2 mm  -> 위상 지터 2π·δl/λ
   (b) 경로 진폭 지터 σ = 0.05 (상대값)
   (c) 약한 원형 복소 가우시안 확산장, K = 20 dB (안테나·부반송파마다 독립)
3. 모든 경로를 부반송파별로 coherent하게 더한다.
   H[k] = Σ a_p·exp(−j2π f_k(τ_p+δτ_p))·λ_k/λ_c
   (Sionna Paths.a에는 반송파 위상이 빠져 있어서 exp(−j2π f_k τ)로 넣는다.)

출력: dict(H (Ny, Nx, 4, 64) complex64, H_clean(노이즈 없음), x_m, y_m, occupied, no_path, signature)
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

import config


def grid_coordinates_m() -> tuple[np.ndarray, np.ndarray]:
    return config.grid_coordinates_m()


def is_inside_obstacle(x_m: float, y_m: float) -> bool:
    for ob in config.OBSTACLES:
        cx, cy = ob.center_xy_m
        if ob.kind == "box":
            dx, dy, _ = ob.size_xyz_m
            if abs(x_m - cx) <= dx / 2.0 and abs(y_m - cy) <= dy / 2.0:
                return True
        elif (x_m - cx) ** 2 + (y_m - cy) ** 2 <= ob.radius_m ** 2:
            return True
    return False


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
    payload = {"schema": "4amr-map-v1", "rt": _rt_settings(),
               "bs": config.bs_antenna_positions_m(rt=True).tolist(), "f": config.FREQ_HZ,
               "room": [config.ROOM_X_M, config.ROOM_Y_M, config.ROOM_Z_M],
               "grid": config.GRID_RES_M, "rx_h": config.RX_HEIGHT_M,
               "obstacles": [ob.__dict__ for ob in config.OBSTACLES]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _scene_signature() -> str:
    payload = {"rt": _rt_signature(), "n_sc": config.N_SC, "bw": config.BW_HZ,
               "noise": [config.NOISE_PATH_LENGTH_STD_M, config.NOISE_AMPLITUDE_STD,
                         config.NOISE_DIFFUSE_K_DB, config.NOISE_SEED]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Sionna RT
# ---------------------------------------------------------------------------
def _solve_batch(solver, scene, rt, mi, receivers: np.ndarray, diffuse_pass: bool) -> list:
    names = []
    for index, position in enumerate(receivers):
        name = f"rx{index:03d}"
        scene.add(rt.Receiver(name=name, position=mi.Point3f(*map(float, position))))
        names.append(name)
    if diffuse_pass:
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
    kinds = np.bitwise_or.reduce(inter.astype(np.uint8), axis=0)
    n_inter = (inter != 0).sum(axis=0).astype(np.uint8)
    if diffuse_pass:
        valid &= (kinds & 2) != 0                                          # 확산 산란이 들어간 경로만
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
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("llvm_ad_mono_polarized")
    import sionna.rt as rt
    from sionna_scene import build_scene_files

    rt_xml, _ = build_scene_files()
    scene = rt.load_scene(str(rt_xml), merge_shapes=False)
    scene.frequency = config.FREQ_HZ
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    for n, position in enumerate(config.bs_antenna_positions_m(rt=True)):
        scene.add(rt.Transmitter(name=f"bs_ant{n}", position=mi.Point3f(*map(float, position))))
    x_m, y_m = grid_coordinates_m()
    gx, gy = np.meshgrid(x_m, y_m)
    receivers = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, config.RX_HEIGHT_M)]
    solver = rt.PathSolver()
    if verbose:
        print(f"[Sionna RT] variant={mi.variant()}  {config.N_ANT_BS} Tx x {len(receivers)} Rx  "
              f"depth={config.RT_MAX_DEPTH}  rays/antenna={config.RT_SAMPLES_PER_SRC:,}", flush=True)
    links: list = []
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
            print(f"[Sionna RT] {done:3d}/{len(receivers)} cells  ({time.time() - t0:6.0f} s, ~{eta:5.0f} s left)", flush=True)
    counts = np.array([len(link["a"]) for cell in links for link in cell])
    flat = {key: np.concatenate([link[key] for cell in links for link in cell]) for key in ("a", "tau", "kinds", "depth")}
    flat["link_index"] = np.repeat(np.arange(len(counts)), counts).astype(np.int32)
    flat["counts"] = counts.reshape(len(y_m), len(x_m), config.N_ANT_BS)
    flat["signature"] = np.array(_rt_signature())
    flat["runtime_s"] = np.array(time.time() - t0)
    return flat


def load_or_run_sionna_rt(force: bool = False, verbose: bool = True) -> dict[str, np.ndarray]:
    """노이즈 없는 원시 경로를 캐시한다. 노이즈 설정만 바꾸면 레이트레이싱을 다시 하지 않는다."""
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


# ---------------------------------------------------------------------------
# 경로 -> 채널 (+ 가우시안 잡음 3종)
# ---------------------------------------------------------------------------
def paths_to_channel(rt_paths: dict[str, np.ndarray], with_noise: bool = True) -> np.ndarray:
    ny, nx, n_ant = rt_paths["counts"].shape
    n_links = ny * nx * n_ant
    a = rt_paths["a"].astype(np.complex128)
    tau = rt_paths["tau"].astype(np.float64)
    link = rt_paths["link_index"]
    rng = np.random.default_rng(config.NOISE_SEED)
    if with_noise:
        # (b) 진폭 지터, (a) 경로 길이 지터
        a = a * np.maximum(1.0 + config.NOISE_AMPLITUDE_STD * rng.standard_normal(len(a)), 0.0)
        tau = tau + config.NOISE_PATH_LENGTH_STD_M * rng.standard_normal(len(tau)) / config.C_MPS
    h = np.zeros((n_links, config.N_SC), dtype=np.complex128)
    for k, f_k in enumerate(config.SUBCARRIER_HZ):
        term = a * np.exp(-2j * np.pi * f_k * tau)
        h[:, k] = np.bincount(link, term.real, n_links) + 1j * np.bincount(link, term.imag, n_links)
    h *= (config.SUBCARRIER_WAVELENGTH_M / config.LAMBDA_C_M)[None, :]
    h = h.reshape(ny, nx, n_ant, config.N_SC)
    if with_noise:
        # (c) 확산장: 셀 평균 전력 / K 크기의 원형 복소 가우시안
        cell_power = np.mean(np.abs(h) ** 2, axis=(2, 3), keepdims=True)
        sigma = np.sqrt(cell_power / 10.0 ** (config.NOISE_DIFFUSE_K_DB / 10.0) / 2.0)
        h = h + sigma * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
    return h


def fill_cells_without_paths(h: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """장애물 안쪽처럼 레이가 하나도 안 닿는 셀은 가장 가까운 셀 값을 복사한다 (경로 계획에선 막힌 칸)."""
    empty = counts.sum(axis=2) == 0
    if empty.any():
        full = np.argwhere(~empty)
        for iy, ix in np.argwhere(empty):
            jy, jx = full[np.argmin((full[:, 0] - iy) ** 2 + (full[:, 1] - ix) ** 2)]
            h[iy, ix] = h[jy, jx]
    return empty


def build_fingerprint(verbose: bool = True, force_rt: bool = False) -> dict[str, np.ndarray]:
    x_m, y_m = grid_coordinates_m()
    rt_paths = load_or_run_sionna_rt(force=force_rt, verbose=verbose)
    h = paths_to_channel(rt_paths, with_noise=True)
    h_clean = paths_to_channel(rt_paths, with_noise=False)
    no_path = fill_cells_without_paths(h, rt_paths["counts"])
    fill_cells_without_paths(h_clean, rt_paths["counts"])
    occupied = np.array([[is_inside_obstacle(x, y) for x in x_m] for y in y_m])
    if verbose:
        c = rt_paths["counts"]
        print(f"Sionna paths per link: mean {c.mean():.0f}, max {c.max()}, total {c.sum():,}; "
              f"cells without any path: {int(no_path.sum())}")
    return {"H": h.astype(np.complex64), "H_clean": h_clean.astype(np.complex64), "x_m": x_m, "y_m": y_m,
            "occupied": occupied, "no_path": no_path, "signature": np.array(_scene_signature())}


def load_or_build_fingerprint(force: bool = False, verbose: bool = True) -> dict[str, np.ndarray]:
    path = config.FINGERPRINT_PATH
    if not force and path.exists():
        data = np.load(path, allow_pickle=True).item()
        if str(data["signature"].item()) == _scene_signature():
            if verbose:
                print(f"Loading cached fingerprint: {path}")
            return data
        if verbose:
            print("Cached fingerprint is stale; rebuilding it.")
    data = build_fingerprint(verbose=verbose)
    np.save(path, data, allow_pickle=True)
    if verbose:
        print(f"Saved fingerprint: {path}")
    return data


def snr_grid_db(h: np.ndarray, mode: str = "mrc") -> np.ndarray:
    """(Ny, Nx, 64) 부반송파별 SNR [dB]. siso = 안테나 0개만, mrc = 안테나 4개 MRC."""
    if mode == "siso":
        power = np.abs(h[:, :, 0, :]) ** 2
    elif mode == "mrc":
        power = np.sum(np.abs(h) ** 2, axis=2)
    else:
        raise ValueError("mode must be 'siso' or 'mrc'")
    scale = config.tx_power_w() / config.noise_power_w()
    return 10.0 * np.log10(np.maximum(scale * power, 1e-30))


if __name__ == "__main__":
    load_or_build_fingerprint(force="--force" in __import__("sys").argv)
