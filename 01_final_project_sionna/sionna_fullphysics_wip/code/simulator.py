"""Task 3.1: 궤적 샘플링, SNR bilinear 보간, Equal-BW / Opt-BW(prefix-sum) 할당."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config


# ----------------------------------------------------------------------------
# SNR
# ----------------------------------------------------------------------------
def snr_grid_db(H: np.ndarray, mode: str, offset_db: float = 0.0) -> np.ndarray:
    """H (Ny, Nx, 4, 64) → 부반송파별 SNR (Ny, Nx, 64) [dB].  Δ_i는 여기서만 더한다."""
    if mode == "siso":
        power = np.abs(H[:, :, 0, :]) ** 2                 # 0번 안테나만
    elif mode == "mrc":
        power = np.sum(np.abs(H) ** 2, axis=2)             # ||h_k||^2
    else:
        raise ValueError(mode)
    snr = power * config.P_REF_W / config.N0_W
    return 10 * np.log10(np.maximum(snr, 1e-30)) + offset_db


def avg_snr_map_db(H: np.ndarray, mode: str) -> np.ndarray:
    """T1.2: 64개 부반송파 평균 SNR (선형 평균 후 dB), shape (20, 36)."""
    lin = 10 ** (snr_grid_db(H, mode) / 10)
    return 10 * np.log10(lin.mean(axis=-1))


def bilinear(values: np.ndarray, xy: np.ndarray, x_m=config.GRID_X_M, y_m=config.GRID_Y_M) -> np.ndarray:
    """values (Ny, Nx, ...) 를 연속 위치 xy (T, 2)에서 bilinear 보간."""
    x = np.clip(xy[:, 0], x_m[0], x_m[-1])
    y = np.clip(xy[:, 1], y_m[0], y_m[-1])
    ix = np.clip(np.searchsorted(x_m, x, side="right") - 1, 0, len(x_m) - 2)
    iy = np.clip(np.searchsorted(y_m, y, side="right") - 1, 0, len(y_m) - 2)
    shape = (-1,) + (1,) * (values.ndim - 2)
    a = ((x - x_m[ix]) / (x_m[ix + 1] - x_m[ix])).reshape(shape)
    b = ((y - y_m[iy]) / (y_m[iy + 1] - y_m[iy])).reshape(shape)
    return ((1 - a) * (1 - b) * values[iy, ix] + a * (1 - b) * values[iy, ix + 1]
            + (1 - a) * b * values[iy + 1, ix] + a * b * values[iy + 1, ix + 1])


# ----------------------------------------------------------------------------
# Trajectories
# ----------------------------------------------------------------------------
def positions_at(path_xy: np.ndarray, t: np.ndarray, speed: float, t0: float = 0.0) -> np.ndarray:
    """등속 이동, 도착 후에는 목표점에 정지."""
    seg = np.diff(path_xy, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.r_[0.0, np.cumsum(seg_len)]
    s = np.clip((np.asarray(t) - t0) * speed, 0.0, cum[-1])
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    frac = (s - cum[i]) / np.maximum(seg_len[i], 1e-12)
    pos = path_xy[i] + frac[:, None] * seg[i]
    pos[s >= cum[-1]] = path_xy[-1]
    return pos


def polyline_length(path_xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum())


def time_axis(T: float, dt: float = config.SIM_DT_S) -> np.ndarray:
    return np.arange(0.0, T + dt / 2, dt)


# ----------------------------------------------------------------------------
# Bandwidth allocation (batched over any leading dims)
# ----------------------------------------------------------------------------
def spectral_eff(snr_db: np.ndarray) -> np.ndarray:
    return np.log2(1 + 10 ** (snr_db / 10))


def equal_bw(e1: np.ndarray, e2: np.ndarray):
    k = config.N_SC // 2
    r1 = config.DELTA_F_HZ * e1[..., :k].sum(-1)
    r2 = config.DELTA_F_HZ * e2[..., k:].sum(-1)
    return np.full(r1.shape, k, dtype=int), r1, r2


def opt_bw(e1: np.ndarray, e2: np.ndarray):
    """k*(t) = argmax_k min(R1(t;k), R2(t;k)), 한 번의 O(N) prefix-sum 스캔.

    c1[..., k] = Σ_{j<=k} e1_j       → R1(t;k) = Δf·c1[..., k-1]
    c2[..., k] = Σ_{j>=k} e2_j       → R2(t;k) = Δf·c2[..., k]
    """
    c1 = np.cumsum(e1, axis=-1)
    c2 = np.flip(np.cumsum(np.flip(e2, -1), axis=-1), -1)
    rmin_k = np.minimum(c1[..., :-1], c2[..., 1:])          # k = 1..N-1
    k_star = np.argmax(rmin_k, axis=-1) + 1
    r1 = config.DELTA_F_HZ * np.take_along_axis(c1, (k_star - 1)[..., None], -1)[..., 0]
    r2 = config.DELTA_F_HZ * np.take_along_axis(c2, k_star[..., None], -1)[..., 0]
    return k_star, r1, r2


def allocate(e1, e2, policy: str):
    return opt_bw(e1, e2) if policy == "opt" else equal_bw(e1, e2)


# ----------------------------------------------------------------------------
# Single pair simulation
# ----------------------------------------------------------------------------
@dataclass
class SimResult:
    t: np.ndarray
    pos1: np.ndarray
    pos2: np.ndarray
    snr1_db: np.ndarray        # (T, 64), Δ 적용됨
    snr2_db: np.ndarray
    k_star: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    rmin_t: np.ndarray
    T: float
    L1: float
    L2: float

    @property
    def rmin(self) -> float:
        return float(self.rmin_t.min())

    @property
    def mean(self) -> float:
        return float(self.rmin_t.mean())


def simulate(path1_xy, path2_xy, snr1_grid, snr2_grid, policy: str) -> SimResult:
    L1, L2 = polyline_length(path1_xy), polyline_length(path2_xy)
    v1, v2 = config.AMR_SPEED_MPS["AMR1"], config.AMR_SPEED_MPS["AMR2"]
    T = max(L1 / v1, config.AMR2_START_TIME_S + L2 / v2)
    t = time_axis(T)
    p1 = positions_at(path1_xy, t, v1)
    p2 = positions_at(path2_xy, t, v2, config.AMR2_START_TIME_S)
    s1, s2 = bilinear(snr1_grid, p1), bilinear(snr2_grid, p2)
    k, r1, r2 = allocate(spectral_eff(s1), spectral_eff(s2), policy)
    return SimResult(t, p1, p2, s1, s2, k, r1, r2, np.minimum(r1, r2), T, L1, L2)
