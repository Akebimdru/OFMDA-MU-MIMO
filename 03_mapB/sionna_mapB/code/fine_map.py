"""맵 B: 10 cm 간격 Sionna RT 채널 맵 (160 × 80 = 12,800점, 장애물 안쪽 제외). sionna_4amr_hybrid 버전과 같은 코드.

50 cm 맵(fingerprint.py)과 같은 장면·RT 설정·가우시안 잡음 3종을 쓴다. 점은 10 cm 칸의 중심
(0.05, 0.15, …)이라 50 cm 맵의 점(0.25, 0.75, …)도 모두 포함된다.

* 한 번에 수신점 6개씩 푼다 (더 키우면 Sionna 경로 버퍼가 넘쳐 경로가 조용히 빠진다).
* 원시 경로는 너무 커서 저장하지 않고, 배치마다 바로 채널 H[k]로 합친다.
* 360점마다 code/fine_map_parts/part_XXXX.npz로 저장한다. 다시 실행하면 저장된 부분은 건너뛴다.
* 잡음 난수는 배치 번호로 시드를 정해서, 중간에 끊고 다시 돌려도 결과가 같다.

실행:  python code/fine_map.py            (이어서 계산)
       python code/fine_map.py --assemble (부분 파일을 합쳐 fine_H.npy / fine_H_clean.npy 생성)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import config
from fingerprint import _solve_batch, is_inside_obstacle

FINE_RES_M = 0.10
PART_POINTS = 60                        # 10 배치마다 저장 (클라우드 컨테이너가 재시작돼도 손실을 줄이려고 360 -> 60)
PARTS_DIR = config.PROJECT_ROOT / "code" / "fine_map_parts"
FINE_H_PATH = config.PROJECT_ROOT / "code" / "fine_H.npy"
FINE_H_CLEAN_PATH = config.PROJECT_ROOT / "code" / "fine_H_clean.npy"
FINE_META_PATH = config.PROJECT_ROOT / "code" / "fine_meta.npz"


def fine_grid():
    x = np.round(np.arange(FINE_RES_M / 2, config.ROOM_X_M, FINE_RES_M), 4)
    y = np.round(np.arange(FINE_RES_M / 2, config.ROOM_Y_M, FINE_RES_M), 4)
    gx, gy = np.meshgrid(x, y)
    occupied = np.array([[is_inside_obstacle(xx, yy) for xx in x] for yy in y])
    free_idx = np.flatnonzero(~occupied.ravel())
    return x, y, gx.ravel(), gy.ravel(), occupied, free_idx


def links_to_channel(links_per_rx: list, seed: int):
    """수신점별 [안테나 4개 링크] -> (n, 4, 64) 채널 (잡음 有, 無)."""
    rng = np.random.default_rng(seed)
    n = len(links_per_rx)
    h = np.zeros((n, config.N_ANT_BS, config.N_SC), dtype=np.complex128)
    h0 = np.zeros_like(h)
    counts = np.zeros((n, config.N_ANT_BS), dtype=np.int32)
    scale = config.SUBCARRIER_WAVELENGTH_M / config.LAMBDA_C_M
    f = config.SUBCARRIER_HZ[None, :]
    for i, links in enumerate(links_per_rx):
        for m, link in enumerate(links):
            a, tau = link["a"].astype(np.complex128), link["tau"]
            counts[i, m] = len(a)
            if len(a) == 0:
                continue
            h0[i, m] = (a[:, None] * np.exp(-2j * np.pi * f * tau[:, None])).sum(axis=0) * scale
            a_n = a * np.maximum(1.0 + config.NOISE_AMPLITUDE_STD * rng.standard_normal(len(a)), 0.0)
            tau_n = tau + config.NOISE_PATH_LENGTH_STD_M * rng.standard_normal(len(a)) / config.C_MPS
            h[i, m] = (a_n[:, None] * np.exp(-2j * np.pi * f * tau_n[:, None])).sum(axis=0) * scale
    power = np.mean(np.abs(h) ** 2, axis=(1, 2), keepdims=True)
    sigma = np.sqrt(power / 10.0 ** (config.NOISE_DIFFUSE_K_DB / 10.0) / 2.0)
    h = h + sigma * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
    return h.astype(np.complex64), h0.astype(np.complex64), counts


def run():
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("llvm_ad_mono_polarized")
    import sionna.rt as rt
    from sionna_scene import build_scene_files

    x, y, gx, gy, occupied, free_idx = fine_grid()
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(FINE_META_PATH, x_m=x, y_m=y, occupied=occupied, free_idx=free_idx)
    n_parts = int(np.ceil(len(free_idx) / PART_POINTS))
    todo = [p for p in range(n_parts) if not (PARTS_DIR / f"part_{p:04d}.npz").exists()]
    print(f"[fine map] {len(free_idx):,} free points, {n_parts} parts, {len(todo)} to do", flush=True)
    if not todo:
        return

    rt_xml, _ = build_scene_files()
    scene = rt.load_scene(str(rt_xml), merge_shapes=False)
    scene.frequency = config.FREQ_HZ
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    for n, position in enumerate(config.bs_antenna_positions_m(rt=True)):
        scene.add(rt.Transmitter(name=f"bs_ant{n}", position=mi.Point3f(*map(float, position))))
    solver = rt.PathSolver()
    batch = config.RT_RX_BATCH
    t0, done_pts = time.time(), 0
    total_left = sum(len(free_idx[p * PART_POINTS:(p + 1) * PART_POINTS]) for p in todo)
    for p in todo:
        idx = free_idx[p * PART_POINTS:(p + 1) * PART_POINTS]
        hs, h0s, cs = [], [], []
        for b0 in range(0, len(idx), batch):
            sel = idx[b0:b0 + batch]
            rx = np.c_[gx[sel], gy[sel], np.full(len(sel), config.RX_HEIGHT_M)]
            res = _solve_batch(solver, scene, rt, mi, rx, diffuse_pass=False)
            if config.RT_DIFFUSE:
                extra = _solve_batch(solver, scene, rt, mi, rx, diffuse_pass=True)
                res = [[{k: np.concatenate([l1[k], l2[k]]) for k in l1} for l1, l2 in zip(r1, r2)]
                       for r1, r2 in zip(res, extra)]
            h, h0, c = links_to_channel(res, seed=config.NOISE_SEED * 100_000 + int(sel[0]))
            hs.append(h); h0s.append(h0); cs.append(c)
        np.savez(PARTS_DIR / f"part_{p:04d}.npz", idx=idx, H=np.concatenate(hs), H_clean=np.concatenate(h0s),
                 counts=np.concatenate(cs))
        done_pts += len(idx)
        el = time.time() - t0
        print(f"[fine map] part {p + 1}/{n_parts} saved  {done_pts:,}/{total_left:,} pts  "
              f"{el / 3600:.2f} h elapsed, ~{el / done_pts * (total_left - done_pts) / 3600:.2f} h left", flush=True)


def assemble():
    meta = np.load(FINE_META_PATH)
    ny, nx = meta["occupied"].shape
    H = np.zeros((ny * nx, config.N_ANT_BS, config.N_SC), dtype=np.complex64)
    H0 = np.zeros_like(H)
    counts = np.zeros((ny * nx, config.N_ANT_BS), dtype=np.int32)
    for f in sorted(PARTS_DIR.glob("part_*.npz")):
        d = np.load(f)
        H[d["idx"]], H0[d["idx"]], counts[d["idx"]] = d["H"], d["H_clean"], d["counts"]
    missing = int((counts.sum(axis=1) == 0)[meta["free_idx"]].sum())
    np.save(FINE_H_PATH, H.reshape(ny, nx, config.N_ANT_BS, config.N_SC))
    np.save(FINE_H_CLEAN_PATH, H0.reshape(ny, nx, config.N_ANT_BS, config.N_SC))
    np.savez(FINE_META_PATH, x_m=meta["x_m"], y_m=meta["y_m"], occupied=meta["occupied"],
             free_idx=meta["free_idx"], counts=counts.reshape(ny, nx, config.N_ANT_BS))
    print(f"assembled {ny}x{nx}; free points without any path: {missing}")


if __name__ == "__main__":
    assemble() if "--assemble" in sys.argv else run()
