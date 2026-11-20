"""Task 2.3: 900개 경로 쌍 brute-force + QoS 필터 (ε-constraint).

outer: (i, j) ∈ 30×30 전수 열거,  inner: 매 시점 prefix-sum Opt-BW (closed form).
각 AMR 후보의 SNR 궤적을 한 번만 샘플링해 두고, i 하나당 30개 j를 한 번에 numpy로 계산한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from path_planner import Grid
from simulator import (allocate, bilinear, polyline_length, positions_at, snr_grid_db,
                       spectral_eff, time_axis)


@dataclass
class PairScores:
    antenna: str
    policy: str
    T: np.ndarray          # (30, 30) 완료 시간 [s]
    rmin: np.ndarray       # (30, 30) [bps]
    mean: np.ndarray       # (30, 30) [bps]
    L1: np.ndarray         # (30,)
    L2: np.ndarray         # (30,)


def score_all_pairs(grid: Grid, paths: dict, H: np.ndarray, antenna: str, policy: str,
                    offset2_db: float | None = None) -> PairScores:
    off1 = config.TX_OFFSET_DB["AMR1"]
    off2 = config.TX_OFFSET_DB["AMR2"] if offset2_db is None else offset2_db
    g1, g2 = snr_grid_db(H, antenna, off1), snr_grid_db(H, antenna, off2)
    xy1 = [grid.xy_of(p) for p in paths["AMR1"]]
    xy2 = [grid.xy_of(p) for p in paths["AMR2"]]
    v1, v2 = config.AMR_SPEED_MPS["AMR1"], config.AMR_SPEED_MPS["AMR2"]
    L1 = np.array([polyline_length(p) for p in xy1])
    L2 = np.array([polyline_length(p) for p in xy2])
    T1, T2 = L1 / v1, config.AMR2_START_TIME_S + L2 / v2
    t_all = time_axis(max(T1.max(), T2.max()))[:: config.PLAN_TIME_STRIDE]

    # 후보별 (T_all, 64) 스펙트럴 효율 — 도착 후에는 목표점의 SNR 유지
    e1 = np.stack([spectral_eff(bilinear(g1, positions_at(p, t_all, v1))) for p in xy1])
    e2 = np.stack([spectral_eff(bilinear(g2, positions_at(p, t_all, v2, config.AMR2_START_TIME_S))) for p in xy2])

    K1, K2 = len(xy1), len(xy2)
    T = np.maximum(T1[:, None], T2[None, :])
    rmin, mean = np.empty((K1, K2)), np.empty((K1, K2))
    for i in range(K1):
        _, r1, r2 = allocate(np.broadcast_to(e1[i], e2.shape), e2, policy)   # (K2, T_all)
        r = np.minimum(r1, r2)
        valid = t_all[None, :] <= T[i][:, None] + 1e-9                     # 쌍별 미션 구간
        rmin[i] = np.where(valid, r, np.inf).min(axis=1)
        mean[i] = np.where(valid, r, 0.0).sum(axis=1) / valid.sum(axis=1)
    return PairScores(antenna, policy, T, rmin, mean, L1, L2)


@dataclass
class Selection:
    pair: tuple[int, int]
    feasible: bool         # False → max-R_min fallback
    n_feasible: int


def select(scores: PairScores, qos_mbps: float = config.R_QOS_MBPS) -> Selection:
    """feasible 중 min T (동률이면 max R_min), 없으면 argmax R_min."""
    ok = scores.rmin >= qos_mbps * 1e6
    if ok.any():
        idx = np.flatnonzero(ok.ravel())
        order = np.lexsort((-scores.rmin.ravel()[idx], np.round(scores.T.ravel()[idx], 9)))
        flat = int(idx[order[0]])
        feasible = True
    else:
        flat = int(np.argmax(scores.rmin))
        feasible = False
    return Selection(tuple(int(v) for v in np.unravel_index(flat, scores.T.shape)), feasible, int(ok.sum()))
