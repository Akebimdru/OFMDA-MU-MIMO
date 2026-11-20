"""Task 3.1 core: trajectory sampling, SNR interpolation, and vectorised Opt-BW."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config


@dataclass(frozen=True)
class SimulationResult:
    time_s: np.ndarray
    positions_amr1_m: np.ndarray
    positions_amr2_m: np.ndarray
    snr1_db: np.ndarray
    snr2_db: np.ndarray
    k_star: np.ndarray
    rate1_bps: np.ndarray
    rate2_bps: np.ndarray
    rmin_bps: np.ndarray
    completion_time_s: float


def snr_grid_db(h: np.ndarray, mode: str, amr_offset_db: float) -> np.ndarray:
    """Derive AMR-specific (Ny, Nx, 64) SNR dB from an AMR-agnostic fingerprint."""
    if mode == "siso":
        channel_power = np.abs(h[:, :, 0, :]) ** 2
    elif mode == "mrc":
        channel_power = np.sum(np.abs(h) ** 2, axis=2)
    else:
        raise ValueError("mode must be 'siso' or 'mrc'")
    scale = config.tx_power_w() / config.noise_power_w()
    return 10.0 * np.log10(np.maximum(scale * channel_power, np.finfo(float).tiny)) + amr_offset_db


def bilinear_sample_db(values: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, positions_m: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate values[Ny, Nx, ...] in dB at continuous positions."""
    if values.shape[:2] != (len(y_m), len(x_m)):
        raise ValueError("Grid array does not match coordinate dimensions")
    position = np.asarray(positions_m, dtype=float)
    x = np.clip(position[:, 0], x_m[0], x_m[-1])
    y = np.clip(position[:, 1], y_m[0], y_m[-1])
    ix0 = np.clip(np.searchsorted(x_m, x, side="right") - 1, 0, len(x_m) - 2)
    iy0 = np.clip(np.searchsorted(y_m, y, side="right") - 1, 0, len(y_m) - 2)
    trailing_dims = (1,) * (values.ndim - 2)
    alpha = ((x - x_m[ix0]) / (x_m[ix0 + 1] - x_m[ix0])).reshape((-1,) + trailing_dims)
    beta = ((y - y_m[iy0]) / (y_m[iy0 + 1] - y_m[iy0])).reshape((-1,) + trailing_dims)
    v00, v10 = values[iy0, ix0], values[iy0, ix0 + 1]
    v01, v11 = values[iy0 + 1, ix0], values[iy0 + 1, ix0 + 1]
    return (1 - alpha) * (1 - beta) * v00 + alpha * (1 - beta) * v10 + (1 - alpha) * beta * v01 + alpha * beta * v11


def path_positions_at_time(path_xy_m: np.ndarray, time_s: np.ndarray, speed_mps: float = config.AMR_SPEED_MPS) -> tuple[np.ndarray, float]:
    """Move at constant ground speed along a polyline and remain at its goal after arrival."""
    segments = np.diff(path_xy_m, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    total_length = float(cumulative[-1])
    distance = np.clip(np.asarray(time_s) * speed_mps, 0.0, total_length)
    segment_index = np.clip(np.searchsorted(cumulative, distance, side="right") - 1, 0, len(segments) - 1)
    segment_start = cumulative[segment_index]
    fraction = (distance - segment_start) / np.maximum(segment_lengths[segment_index], 1e-12)
    positions = path_xy_m[segment_index] + fraction[:, None] * segments[segment_index]
    positions[distance >= total_length] = path_xy_m[-1]
    return positions, total_length / speed_mps


def allocate_equal_bw(snr1_db: np.ndarray, snr2_db: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the required fixed contiguous 32/32 subcarrier split."""
    e1 = np.log2(1.0 + 10.0 ** (snr1_db / 10.0))
    e2 = np.log2(1.0 + 10.0 ** (snr2_db / 10.0))
    k_star = np.full(len(snr1_db), config.N_SC // 2, dtype=int)
    r1 = config.DELTA_F_HZ * e1[:, : config.N_SC // 2].sum(axis=1)
    r2 = config.DELTA_F_HZ * e2[:, config.N_SC // 2 :].sum(axis=1)
    return k_star, r1, r2, np.minimum(r1, r2)


def allocate_opt_bw(snr1_db: np.ndarray, snr2_db: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Handout's batched prefix-sum solution to the instantaneous max-min split."""
    if snr1_db.shape != snr2_db.shape or snr1_db.ndim != 2 or snr1_db.shape[1] != config.N_SC:
        raise ValueError("Both SNR inputs must have shape (T, N_SC)")
    e1 = np.log2(1.0 + 10.0 ** (snr1_db / 10.0))
    e2 = np.log2(1.0 + 10.0 ** (snr2_db / 10.0))
    c1 = np.cumsum(e1, axis=1)
    c2 = np.cumsum(e2[:, ::-1], axis=1)[:, ::-1]
    rmin_per_k = np.minimum(c1[:, :-1], c2[:, 1:])
    k_star = np.argmax(rmin_per_k, axis=1) + 1
    rows = np.arange(len(k_star))
    r1 = config.DELTA_F_HZ * c1[rows, k_star - 1]
    r2 = config.DELTA_F_HZ * c2[rows, k_star]
    return k_star, r1, r2, np.minimum(r1, r2)


def simulate_pair(
    path1_xy_m: np.ndarray,
    path2_xy_m: np.ndarray,
    h: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    antenna_mode: str,
    bw_policy: str,
) -> SimulationResult:
    """Simulate a path pair on one common 0.1 s time grid and apply a BW policy."""
    snr1_grid = snr_grid_db(h, antenna_mode, config.TX_OFFSET_AMR1_DB)
    snr2_grid = snr_grid_db(h, antenna_mode, config.TX_OFFSET_AMR2_DB)
    return simulate_pair_from_snr_grids(path1_xy_m, path2_xy_m, snr1_grid, snr2_grid, x_m, y_m, bw_policy)


def simulate_pair_from_snr_grids(
    path1_xy_m: np.ndarray,
    path2_xy_m: np.ndarray,
    snr1_grid_db: np.ndarray,
    snr2_grid_db: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    bw_policy: str,
) -> SimulationResult:
    """Simulate a pair using precomputed AMR-specific SNR grids for fast enumeration."""
    length1 = float(np.linalg.norm(np.diff(path1_xy_m, axis=0), axis=1).sum())
    length2 = float(np.linalg.norm(np.diff(path2_xy_m, axis=0), axis=1).sum())
    completion_time = max(length1, length2) / config.AMR_SPEED_MPS
    time_s = np.arange(0.0, completion_time + config.SIM_DT_S * 0.5, config.SIM_DT_S)
    positions1, _ = path_positions_at_time(path1_xy_m, time_s)
    positions2, _ = path_positions_at_time(path2_xy_m, time_s)
    snr1 = bilinear_sample_db(snr1_grid_db, x_m, y_m, positions1)
    snr2 = bilinear_sample_db(snr2_grid_db, x_m, y_m, positions2)
    if bw_policy == "equal":
        k_star, r1, r2, rmin = allocate_equal_bw(snr1, snr2)
    elif bw_policy == "opt":
        k_star, r1, r2, rmin = allocate_opt_bw(snr1, snr2)
    else:
        raise ValueError("bw_policy must be 'equal' or 'opt'")
    return SimulationResult(time_s, positions1, positions2, snr1, snr2, k_star, r1, r2, rmin, completion_time)


def validate_opt_bw(snr1_db: np.ndarray, snr2_db: np.ndarray) -> dict[str, float | bool]:
    """Check every explicit T3.1 algorithm condition against an actual SNR batch."""
    k_opt, r1_opt, r2_opt, rmin_opt = allocate_opt_bw(snr1_db, snr2_db)
    _, _, _, rmin_equal = allocate_equal_bw(snr1_db, snr2_db)
    e1 = np.log2(1.0 + 10.0 ** (snr1_db / 10.0))
    e2 = np.log2(1.0 + 10.0 ** (snr2_db / 10.0))
    forward_monotone = bool(np.all(np.diff(np.cumsum(e1, axis=1), axis=1) >= -1e-12))
    reverse_monotone = bool(np.all(np.diff(np.cumsum(e2[:, ::-1], axis=1)[:, ::-1], axis=1) <= 1e-12))
    return {
        "shape_is_T_by_64": snr1_db.ndim == 2 and snr1_db.shape[1] == 64,
        "k_star_in_1_to_63": bool(np.all((1 <= k_opt) & (k_opt <= 63))),
        "equal_bw_exactly_32": bool(np.all(allocate_equal_bw(snr1_db, snr2_db)[0] == 32)),
        "r1_prefix_monotone": forward_monotone,
        "r2_suffix_monotone": reverse_monotone,
        "opt_not_worse_than_equal_every_step": bool(np.all(rmin_opt >= rmin_equal - 1e-6)),
        "minimum_opt_minus_equal_bps": float(np.min(rmin_opt - rmin_equal)),
        "mean_opt_minus_equal_bps": float(np.mean(rmin_opt - rmin_equal)),
        "k_star_min": int(k_opt.min()),
        "k_star_max": int(k_opt.max()),
        "mean_rate1_bps": float(np.mean(r1_opt)),
        "mean_rate2_bps": float(np.mean(r2_opt)),
    }
