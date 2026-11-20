"""Task 3.3 (8-scheme 2^3 ablation), Task 4 (Pareto, QoS sweep), Task 5.3 (Rayleigh 재생)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from joint_planner import PairScores, score_all_pairs, select
from path_planner import Grid
from simulator import SimResult, simulate, snr_grid_db

#            name          path         BW       antenna
SCHEMES = (("NAIVE", "Shortest", "equal", "siso"),
           ("+MRC", "Shortest", "equal", "mrc"),
           ("+OPTBW", "Shortest", "opt", "siso"),
           ("+SMARTPATH", "CommAware", "equal", "siso"),
           ("MRC OPTBW", "Shortest", "opt", "mrc"),
           ("MRC SMART", "CommAware", "equal", "mrc"),
           ("OPT SMART", "CommAware", "opt", "siso"),
           ("JOINT", "CommAware", "opt", "mrc"))
MAIN5 = ("NAIVE", "+MRC", "+OPTBW", "+SMARTPATH", "JOINT")


@dataclass
class Scheme:
    name: str
    path_kind: str
    policy: str
    antenna: str
    pair: tuple[int, int]
    feasible: bool | None
    sim: SimResult


def run_ablation(grid: Grid, paths: dict, H: np.ndarray) -> tuple[dict[str, Scheme], dict[tuple, PairScores]]:
    """CommAware 경로는 그 scheme이 쓰는 기술(안테나, BW)로 계획한다."""
    scores = {(ant, pol): score_all_pairs(grid, paths, H, ant, pol)
              for ant in ("siso", "mrc") for pol in ("equal", "opt")}
    out = {}
    for name, kind, pol, ant in SCHEMES:
        if kind == "Shortest":
            pair, feas = (0, 0), None          # 최단 경로는 QoS 계획을 하지 않음
        else:
            s = select(scores[(ant, pol)])
            pair, feas = s.pair, s.feasible
        g1 = snr_grid_db(H, ant, config.TX_OFFSET_DB["AMR1"])
        g2 = snr_grid_db(H, ant, config.TX_OFFSET_DB["AMR2"])
        sim = simulate(grid.xy_of(paths["AMR1"][pair[0]]), grid.xy_of(paths["AMR2"][pair[1]]), g1, g2, pol)
        out[name] = Scheme(name, kind, pol, ant, pair, feas, sim)
    return out, scores


def marginals(res: dict[str, Scheme]) -> dict[str, dict[str, float]]:
    r = {k: v.sim.rmin / 1e6 for k, v in res.items()}
    return {"add": {"MRC": r["+MRC"] - r["NAIVE"], "OptBW": r["+OPTBW"] - r["NAIVE"],
                    "SmartPath": r["+SMARTPATH"] - r["NAIVE"]},
            "drop": {"MRC": r["JOINT"] - r["OPT SMART"], "OptBW": r["JOINT"] - r["MRC SMART"],
                     "SmartPath": r["JOINT"] - r["MRC OPTBW"]},
            "joint_minus_naive": r["JOINT"] - r["NAIVE"]}


# ----------------------------------------------------------------------------
# Task 4
# ----------------------------------------------------------------------------
def pareto_frontier(scores: PairScores, qos_mbps: float = config.R_QOS_MBPS) -> np.ndarray:
    """feasible 점을 T 오름차순 정렬 후 running-max R_min 을 갱신하는 점만 유지 (flat index)."""
    T, R = scores.T.ravel(), scores.rmin.ravel()
    idx = np.flatnonzero(R >= qos_mbps * 1e6)
    idx = idx[np.lexsort((-R[idx], T[idx]))]
    front, best = [], -np.inf
    for i in idx:
        if R[i] > best:
            front.append(int(i)); best = R[i]
    return np.array(front, dtype=int)


def qos_sweep(scores: PairScores) -> list[dict]:
    rows = []
    for q in config.R_QOS_SWEEP_MBPS:
        s = select(scores, q)
        i, j = s.pair
        rows.append({"qos_mbps": q, "feasible": s.feasible, "n_feasible": s.n_feasible, "pair": [i, j],
                     "T_s": float(scores.T[i, j]), "rmin_mbps": float(scores.rmin[i, j] / 1e6),
                     "L1_m": float(scores.L1[i]), "L2_m": float(scores.L2[j])})
    return rows


# ----------------------------------------------------------------------------
# Task 5.3: Rayleigh replay (Sionna PHY 난수 생성기 사용, 오프라인 k* 재사용)
# ----------------------------------------------------------------------------
def rayleigh_power(shape: tuple[int, ...]) -> np.ndarray:
    """|g|^2,  g ~ CN(0, 1).  Sionna PHY 의 complex_normal 과 전역 seed 를 사용."""
    try:
        from sionna.phy.utils import complex_normal
        g = complex_normal(list(shape), var=1.0)
        g = g.numpy() if hasattr(g, "numpy") else np.asarray(g)
    except Exception:                               # Sionna PHY 없이도 동작하도록
        rng = np.random.default_rng(config.RANDOM_SEED)
        g = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
    return np.abs(g) ** 2


def online_replay(sim: SimResult, n_real: int = config.N_REALIZATIONS) -> dict[str, np.ndarray | float]:
    T, N = sim.snr1_db.shape
    # handout 의사코드대로 같은 g(t,k)를 쓴다. AMR1은 [0,k*), AMR2는 [k*,N) 부반송파만 쓰므로
    # 실제로 쓰이는 페이딩 계수는 두 AMR 사이에 서로 겹치지 않는다(독립).
    g = rayleigh_power((n_real, T, N))
    s1 = 10 ** (sim.snr1_db / 10)[None] * g                     # SNR_online = SNR_offline · |g|^2
    s2 = 10 ** (sim.snr2_db / 10)[None] * g
    c1 = np.cumsum(np.log2(1 + s1), axis=-1)
    c2 = np.flip(np.cumsum(np.flip(np.log2(1 + s2), -1), axis=-1), -1)
    k = np.broadcast_to(sim.k_star[None, :, None], (n_real, T, 1))  # 오프라인 k* 그대로 (재계획 없음)
    r1 = config.DELTA_F_HZ * np.take_along_axis(c1, k - 1, -1)[..., 0]
    r2 = config.DELTA_F_HZ * np.take_along_axis(c2, k, -1)[..., 0]
    r = np.minimum(r1, r2)                                         # (n_real, T)
    rmin_real = r.min(axis=1)
    return {"mean_t": r.mean(0), "std_t": r.std(0), "rmin_real": rmin_real,
            "ratio": float(rmin_real.mean() / sim.rmin),
            "mean_ratio": float(r.mean() / sim.rmin_t.mean())}
