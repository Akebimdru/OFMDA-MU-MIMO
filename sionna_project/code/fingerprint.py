"""Task 1.1: Sionna RT 기반 채널 fingerprint  H(p) ∈ C^{4×64}  (20×36 격자).

흐름
1. scene_builder 로 만든 Mitsuba XML(둘레 콘크리트 벽 4개)을 ``sionna.rt.load_scene`` 으로 읽는다.
2. ULA 4소자를 각각 Transmitter 로 정확한 위치 p_ant^(n)에 놓고, 720개 격자점에 Receiver 를 놓는다.
3. ``PathSolver(max_depth=1, los=True, specular_reflection=True)`` 로 LOS + 1차 경면반사 경로를
   찾는다.  → 셀/안테나마다 5개 경로(LOS 1 + 벽 반사 4)의 지연 τ와 반사점을 얻는다.
4. 각 경로의 복소 이득을 handout 식으로 만든다.
       h_path = ρ · 10^{-L_pen/20} · λ/(4πd) · e^{-j2π d/λ},  d = c·τ
   L_pen = Σ α_m · chord_m 은 경로 선분과 12개 장애물의 교차 길이(slab / 원기둥 2차식)로 계산.
   (handout은 반사 경로의 장애물 검사를 image-BS → AMR 직선에 하라고 했으므로, 방 안에서는
   반사점 → AMR 구간이 검사된다.  ``REFLECTION_BLOCKING="both_legs"`` 로 BS → 반사점 구간도 포함 가능.)
5. ``sionna.phy.channel.cir_to_ofdm_channel`` 로 CIR(a, τ) → 64 부반송파 주파수 응답으로 변환한다.
   a 에 e^{-j2π f_c τ}를 넣고 기저대역 주파수 (k-N/2)Δf 를 쓰면 위상이 정확히 e^{-j2π f_k τ}가 되고,
   진폭의 λ_k/λ_c 비율은 모든 경로에 공통이므로 마지막에 곱해 λ_k 를 정확히 반영한다.
6. 결과를 .npy(dict)로 캐시한다.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np

import config
from scene_builder import build_scene_files

REFLECTION_BLOCKING = "image_line"     # "image_line"(handout 그대로) | "both_legs"
EPS = 1e-12


# ----------------------------------------------------------------------------
# 장애물 교차 길이 (Toolkit p.21-23), 선분 배치 단위 벡터화
# ----------------------------------------------------------------------------
def _z_interval(p0, d, h):
    """z ∈ [0, h] 구간의 선분 파라미터 s 범위."""
    dz = d[:, 2]
    safe = np.where(np.abs(dz) < EPS, 1.0, dz)
    s_a, s_b = (0.0 - p0[:, 2]) / safe, (h - p0[:, 2]) / safe
    lo, hi = np.minimum(s_a, s_b), np.maximum(s_a, s_b)
    inside = (p0[:, 2] >= 0) & (p0[:, 2] <= h)
    lo = np.where(np.abs(dz) < EPS, np.where(inside, -np.inf, np.inf), lo)
    hi = np.where(np.abs(dz) < EPS, np.where(inside, np.inf, -np.inf), hi)
    return lo, hi


def chord_box(p0: np.ndarray, p1: np.ndarray, ob: config.Obstacle) -> np.ndarray:
    """slab method.  p0, p1: (M, 3) → chord (M,)."""
    d = p1 - p0
    lo_b = np.array([ob.cx - ob.dx / 2, ob.cy - ob.dy / 2, 0.0])
    hi_b = np.array([ob.cx + ob.dx / 2, ob.cy + ob.dy / 2, ob.height])
    s_in, s_out = np.zeros(len(p0)), np.ones(len(p0))
    for ax in range(3):
        da = d[:, ax]
        par = np.abs(da) < EPS
        safe = np.where(par, 1.0, da)
        a, b = (lo_b[ax] - p0[:, ax]) / safe, (hi_b[ax] - p0[:, ax]) / safe
        inside = (p0[:, ax] >= lo_b[ax]) & (p0[:, ax] <= hi_b[ax])
        a_ = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(a, b))
        b_ = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(a, b))
        s_in, s_out = np.maximum(s_in, a_), np.minimum(s_out, b_)
    return np.maximum(0.0, s_out - s_in) * np.linalg.norm(d, axis=1)


def chord_cylinder(p0: np.ndarray, p1: np.ndarray, ob: config.Obstacle) -> np.ndarray:
    """수직 원기둥: xy 평면 2차식 A s^2 + B s + C = 0 + 높이 [0, h] 제한."""
    d = p1 - p0
    o = p0[:, :2] - np.array([ob.cx, ob.cy])
    A = np.sum(d[:, :2] ** 2, axis=1)
    B = 2 * np.sum(o * d[:, :2], axis=1)
    C = np.sum(o ** 2, axis=1) - ob.r ** 2
    disc = B ** 2 - 4 * A * C
    ok = (disc > 0) & (A > EPS)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    A_ = np.where(ok, A, 1.0)
    s1, s2 = (-B - sq) / (2 * A_), (-B + sq) / (2 * A_)
    zlo, zhi = _z_interval(p0, d, ob.height)
    lo = np.maximum.reduce([np.zeros_like(s1), s1, zlo])
    hi = np.minimum.reduce([np.ones_like(s2), s2, zhi])
    return np.where(ok, np.maximum(0.0, hi - lo), 0.0) * np.linalg.norm(d, axis=1)


def penetration_db(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """L_pen = Σ α_m · L_inside  [dB, amplitude]."""
    total = np.zeros(len(p0))
    for ob in config.OBSTACLES:
        chord = chord_box(p0, p1, ob) if ob.kind == "box" else chord_cylinder(p0, p1, ob)
        total += ob.alpha * chord
    return total


# ----------------------------------------------------------------------------
# Sionna RT
# ----------------------------------------------------------------------------
def receiver_positions() -> np.ndarray:
    """(Ny*Nx, 3) row-major (y, x).  벽 위(x=0, y=0) 격자점은 메시와 겹치지 않도록 1 mm 안쪽으로."""
    X, Y = np.meshgrid(config.GRID_X_M, config.GRID_Y_M)
    pos = np.c_[X.ravel(), Y.ravel(), np.full(X.size, config.RX_HEIGHT_M)]
    pos[:, 0] = np.clip(pos[:, 0], 1e-3, config.ROOM_X_M - 1e-3)
    pos[:, 1] = np.clip(pos[:, 1], 1e-3, config.ROOM_Y_M - 1e-3)
    return pos


def _to_numpy(x):
    if isinstance(x, tuple):
        return tuple(_to_numpy(v) for v in x)
    if hasattr(x, "numpy"):
        try:
            return np.asarray(x.numpy())
        except TypeError:
            pass
    return np.asarray(x)


def run_sionna_rt(verbose: bool = True) -> dict[str, np.ndarray]:
    """Sionna RT 로 모든 (안테나, 셀) 쌍의 경로를 찾고, 경로별 기하 정보를 반환한다."""
    import sionna.rt as rt
    import mitsuba as mi

    rt_xml, _ = build_scene_files()
    scene = rt.load_scene(str(rt_xml), merge_shapes=False)
    scene.frequency = config.FREQ_HZ
    # 단일 등방성 소자. 4개 ULA 소자는 각각 별도 Transmitter 로 정확한 위치에 둔다.
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    ant = config.bs_antenna_positions()
    for n, p in enumerate(ant):
        scene.add(rt.Transmitter(name=f"bs_ant{n}", position=mi.Point3f(*map(float, p))))
    rx_pos = receiver_positions()
    for m, p in enumerate(rx_pos):
        scene.add(rt.Receiver(name=f"rx{m:03d}", position=mi.Point3f(*map(float, p))))

    if verbose:
        print(f"[Sionna RT] mitsuba variant = {mi.variant()}, "
              f"{len(ant)} Tx × {len(rx_pos)} Rx, max_depth={config.MAX_REFLECTIONS}")
    t0 = time.time()
    solver = rt.PathSolver()
    paths = solver(scene, max_depth=config.MAX_REFLECTIONS, los=True, specular_reflection=True,
                   diffuse_reflection=False, refraction=False, synthetic_array=False,
                   samples_per_src=config.RT_SAMPLES_PER_SRC, seed=config.RANDOM_SEED)
    if verbose:
        print(f"[Sionna RT] PathSolver done in {time.time() - t0:.1f} s")

    # 모양: tau (rx, rx_ant, tx, tx_ant, P), vertices (depth, rx, rx_ant, tx, tx_ant, P, 3)
    tau = _to_numpy(paths.tau)
    a_re, a_im = _to_numpy(paths.a)
    inter = _to_numpy(paths.interactions)
    verts = _to_numpy(paths.vertices)
    valid = _to_numpy(paths.valid).astype(bool)
    tau = tau[:, 0, :, 0, :]                     # (rx, tx, P)
    a = a_re + 1j * a_im
    if a.ndim == 6:                              # 일부 버전은 time-step 축이 있음
        a = a[..., 0]
    a = a[:, 0, :, 0, :]                         # (rx, tx, P)
    # Paths.a 에는 반송파 위상 e^{-j2π f_c τ}가 빠져 있다(Paths.cir()가 붙여 줌). 경로 간 coherent
    # 합을 위해 여기서 붙여 기저대역 계수 a^b = a·e^{-j2π f_c τ} 로 만든다.
    a = a * np.exp(-2j * np.pi * config.FREQ_HZ * tau.astype(np.float64))
    valid = valid[:, 0, :, 0, :]
    inter = inter[:, :, 0, :, 0, :]             # (depth, rx, tx, P)
    verts = verts[:, :, 0, :, 0, :, :]          # (depth, rx, tx, P, 3)
    return {"tau": tau, "a_sionna": a, "valid": valid, "interactions": inter, "vertices": verts,
            "rx_pos": rx_pos, "tx_pos": ant}


def image_sources(p: np.ndarray) -> list[tuple[int, float, np.ndarray]]:
    """둘레 벽 4개에 대한 image-BS (축, 벽 좌표, image 위치)."""
    out = []
    for ax, w in ((0, 0.0), (0, config.ROOM_X_M), (1, 0.0), (1, config.ROOM_Y_M)):
        q = p.copy(); q[ax] = 2 * w - q[ax]
        out.append((ax, w, q))
    return out


def complete_missing_paths(rt_out: dict[str, np.ndarray]) -> int:
    """벽 바로 위 격자점(x=0 또는 y=0)에서는 그 벽의 반사점이 수신기와 겹쳐 Sionna RT가 해당
    반사를 놓친다(반사 경로 = LOS 와 길이가 같음).  image-source 해석해로 그 경로만 채운다."""
    tau, valid = rt_out["tau"], rt_out["valid"]
    n_rx, n_tx, n_p = tau.shape
    extra = 4
    z = lambda arr, ax, val=0: np.concatenate(
        [arr, np.full(arr.shape[:ax] + (extra,) + arr.shape[ax + 1:], val, dtype=arr.dtype)], axis=ax)
    tau, valid = z(tau, 2), z(valid, 2, False)
    a_s = z(rt_out["a_sionna"].astype(complex), 2, np.nan)
    inter, verts = z(rt_out["interactions"], 3), z(rt_out["vertices"], 3)
    filled = 0
    for m in np.flatnonzero(rt_out["valid"].sum(-1).min(-1) < 5):
        r = rt_out["rx_pos"][m]
        for n, p in enumerate(rt_out["tx_pos"]):
            refl = valid[m, n] & (inter[0, m, n] != 0)
            hit = verts[0, m, n][refl]                              # 찾은 반사점들
            slot = n_p
            for ax, w, q in image_sources(p):
                if np.any(np.abs(hit[:, ax] - w) < 1e-3):           # 이 벽 반사는 Sionna가 찾음
                    continue
                s_ = (w - q[ax]) / (r[ax] - q[ax])                 # image→rx 직선과 벽 평면의 교점
                tau[m, n, slot] = np.linalg.norm(r - q) / config.C_MPS
                valid[m, n, slot] = True
                inter[0, m, n, slot] = 1
                verts[0, m, n, slot] = q + s_ * (r - q)
                slot += 1
                filled += 1
    rt_out.update(tau=tau, valid=valid, a_sionna=a_s, interactions=inter, vertices=verts)
    return filled


def path_gains_spec(rt_out: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Sionna RT 경로 → handout 이득식.  반환: a (rx, tx, P) complex (반송파 기준), n_refl (rx, tx, P)."""
    tau, valid = rt_out["tau"], rt_out["valid"]
    inter, verts = rt_out["interactions"], rt_out["vertices"]
    rx, tx = rt_out["rx_pos"], rt_out["tx_pos"]
    n_rx, n_tx, n_p = tau.shape
    d = config.C_MPS * tau
    is_refl = (inter[0] != 0) & valid                          # InteractionType.NONE == 0
    rx_b = np.broadcast_to(rx[:, None, None, :], (n_rx, n_tx, n_p, 3))
    tx_b = np.broadcast_to(tx[None, :, None, :], (n_rx, n_tx, n_p, 3))
    refl_pt = verts[0]

    # 마지막 구간(LOS: tx→rx, 반사: 반사점→rx)
    start = np.where(is_refl[..., None], refl_pt, tx_b).reshape(-1, 3)
    lpen = penetration_db(start, rx_b.reshape(-1, 3)).reshape(n_rx, n_tx, n_p)
    if REFLECTION_BLOCKING == "both_legs":
        first = penetration_db(tx_b.reshape(-1, 3), refl_pt.reshape(-1, 3)).reshape(n_rx, n_tx, n_p)
        lpen = lpen + np.where(is_refl, first, 0.0)

    rho = np.where(is_refl, config.WALL_RHO, 1.0)
    d_safe = np.where(valid, d, 1.0)
    a = rho * 10 ** (-lpen / 20) * config.LAMBDA_C_M / (4 * np.pi * d_safe) \
        * np.exp(-2j * np.pi * config.FREQ_HZ * tau)
    if config.RT_GAIN_MODEL == "sionna":                       # Sionna ITU-concrete Fresnel 반사 + FSPL
        a_s = rt_out["a_sionna"] * 10 ** (-lpen / 20)
        a = np.where(np.isnan(a_s), a, a_s)
    a = np.where(valid, a, 0.0)
    return a, is_refl.astype(int)


def cir_to_H(a: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Sionna PHY cir_to_ofdm_channel 로 (rx, tx, P) CIR → (rx, tx, 64) 주파수 응답."""
    import torch
    from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies

    freqs = subcarrier_frequencies(config.N_SC, config.DELTA_F_HZ)      # (k - N/2)·Δf
    n_rx, n_tx, n_p = a.shape
    # Sionna 형식: a [batch, rx, rx_ant, tx, tx_ant, paths, time], tau [batch, rx, tx, paths]
    a_t = torch.as_tensor(a.reshape(1, n_rx, 1, n_tx, 1, n_p, 1), dtype=torch.complex64)
    tau_t = torch.as_tensor(np.where(np.abs(a) > 0, tau, 0.0).reshape(1, n_rx, n_tx, n_p), dtype=torch.float32)
    h = cir_to_ofdm_channel(freqs, a_t, tau_t, normalize=False)          # [1, rx, 1, tx, 1, 1, 64]
    h = _to_numpy(h).reshape(n_rx, n_tx, config.N_SC)
    return h * (config.LAMBDA_K_M / config.LAMBDA_C_M)[None, None, :]   # λ_c → λ_k 진폭 보정


def analytic_check(rt_out: dict[str, np.ndarray]) -> dict[str, float]:
    """Sionna RT가 찾은 경로 길이를 image-source 해석해와 비교 (검증용)."""
    rx, tx = rt_out["rx_pos"], rt_out["tx_pos"]
    d_rt = np.sort(np.where(rt_out["valid"], config.C_MPS * rt_out["tau"], np.inf), axis=-1)[..., :5]
    rt_out = {**rt_out, "a_sionna": np.nan_to_num(rt_out["a_sionna"])}
    imgs = []
    for p in tx:
        cand = [p.copy()]
        for ax, w in ((0, 0.0), (0, config.ROOM_X_M), (1, 0.0), (1, config.ROOM_Y_M)):
            q = p.copy(); q[ax] = 2 * w - q[ax]; cand.append(q)
        imgs.append(cand)
    imgs = np.array(imgs)                                           # (tx, 5, 3)
    d_an = np.sort(np.linalg.norm(rx[:, None, None, :] - imgs[None], axis=-1), axis=-1)
    n_paths = rt_out["valid"].sum(-1)
    # LOS 진폭: Sionna(iso 패턴) vs Friis λ/(4πd)
    is_los = (rt_out["interactions"][0] == 0) & rt_out["valid"]
    d = config.C_MPS * rt_out["tau"]
    friis = config.LAMBDA_C_M / (4 * np.pi * np.where(is_los, d, 1.0))
    los_ok = is_los & (np.abs(rt_out["a_sionna"]) > 0)
    ratio_db = 20 * np.log10(np.abs(rt_out["a_sionna"][los_ok]) / friis[los_ok])
    return {
        "paths_per_link_min": int(n_paths.min()), "paths_per_link_max": int(n_paths.max()),
        "max_abs_delay_error_vs_image_source_m": float(np.max(np.abs(d_rt - d_an))),
        "sionna_los_amplitude_vs_friis_db_mean": float(ratio_db.mean()),
        "sionna_los_amplitude_vs_friis_db_maxabs": float(np.abs(ratio_db).max()),
    }


def _signature() -> str:
    payload = {"v": 3, "f": config.FREQ_HZ, "bw": config.BW_HZ, "n": config.N_SC,
               "bs": config.BS_CENTER_M.tolist(), "gx": config.GRID_X_M.tolist(), "gy": config.GRID_Y_M.tolist(),
               "obs": [o.__dict__ for o in config.OBSTACLES], "model": config.RT_GAIN_MODEL,
               "blocking": REFLECTION_BLOCKING}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_fingerprint(verbose: bool = True) -> dict:
    rt_out = run_sionna_rt(verbose)
    filled = complete_missing_paths(rt_out)
    check = analytic_check(rt_out)
    check["paths_filled_by_image_source"] = filled
    if verbose:
        print("[Sionna RT] geometry check:", check)
    if check["paths_per_link_min"] < 5:
        print("  경고: 일부 링크에서 5개 미만의 경로가 발견됨 → config.RT_SAMPLES_PER_SRC 를 늘리세요.")
    a, n_refl = path_gains_spec(rt_out)
    H = cir_to_H(a, rt_out["tau"])                                  # (720, 4, 64)
    ny, nx = len(config.GRID_Y_M), len(config.GRID_X_M)
    H = H.reshape(ny, nx, config.N_ANT_BS, config.N_SC).astype(np.complex128)
    return {"H": H, "x_m": config.GRID_X_M, "y_m": config.GRID_Y_M, "rt_check": check,
            "tau": rt_out["tau"].reshape(ny, nx, config.N_ANT_BS, -1),
            "signature": _signature()}


def load_or_build(force: bool = False, verbose: bool = True) -> dict:
    path = config.FINGERPRINT_PATH
    if path.exists() and not force:
        data = np.load(path, allow_pickle=True).item()
        if data.get("signature") == _signature():
            if verbose:
                print(f"[fingerprint] cache loaded: {path.name}")
            return data
        print("[fingerprint] cache is stale → rebuilding")
    data = build_fingerprint(verbose)
    np.save(path, data, allow_pickle=True)
    if verbose:
        print(f"[fingerprint] saved: {path}")
    return data
