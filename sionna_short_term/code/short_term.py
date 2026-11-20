"""Long-term + short-term 채널 (Sionna RT, CPU).

long-term  : 과제 모델 그대로 — 0.5 m 격자 fingerprint (경로 손실·장애물·벽 반사).
             추가로 '국소 평균 전력' = 경로 전력의 비동기(incoherent) 합 Σ|a_i|² 을 같은 RT로 계산해
             대규모 성분만 따로 본다.
short-term : (1) AMR의 실제 위치마다 Sionna RT 를 다시 돌려 경로를 coherent 하게 합성 (λ 단위 다중경로 fading)
             (2) 가우시안 교란: 경로별 길이·진폭 요동 + 작은 산란체의 확산 성분 (Gauss-Markov 시간 상관)

최종 채널 (안테나 n, 부반송파 k, 시각 t):
    h_{n,k}(t) = Σ_i a_i(t)(1 + σ_A x_i(t)) e^{-j2π f_k (τ_i(t) + Δd_i(t)/c)}       ← RT 경로 + 가우시안 요동
               + Σ_s g_s(t) √(P_RT(t) / (K·N_s)) e^{jπ n cos φ_s} e^{-j2π f_k τ_s}  ← 가우시안 확산 성분
    x_i, Δd_i/σ_d : 실수 Gauss-Markov,  g_s : 복소 Gauss-Markov CN(0, 1)
주파수 응답은 Sionna PHY 의 cir_to_ofdm_channel 로 계산한다 (fingerprint.cir_to_H).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from fingerprint import cir_to_H, path_gains_spec, rt_paths_at
from simulator import allocate, spectral_eff

N_PATH_IDS = 5   # 0 = LOS, 1..4 = 벽 x=0, x=18, y=0, y=10 반사
RNG = np.random.default_rng(config.RANDOM_SEED + 1)   # 산란체 기하(출발각·지연)용


# ----------------------------------------------------------------------------
# 난수 (Sionna PHY 난수기 + 전역 seed)
# ----------------------------------------------------------------------------
def _normal(shape) -> np.ndarray:
    try:
        from sionna.phy.utils import normal
        x = normal(list(shape))
        return x.numpy().astype(np.float64) if hasattr(x, "numpy") else np.asarray(x, np.float64)
    except Exception:
        return RNG.standard_normal(shape)


def _complex_normal(shape) -> np.ndarray:
    return (_normal(shape) + 1j * _normal(shape)) / np.sqrt(2.0)


def gauss_markov(n_steps: int, shape: tuple, dt: float, tc: float, complex_: bool = False) -> np.ndarray:
    """AR(1): x_t = ρ x_{t-1} + √(1-ρ²) w_t,  ρ = exp(-dt/T_c).  분산 1, 반환 (n_steps, *shape)."""
    rho = np.exp(-dt / tc)
    w = _complex_normal((n_steps, *shape)) if complex_ else _normal((n_steps, *shape))
    x = np.empty_like(w)
    x[0] = w[0]
    s = np.sqrt(1 - rho ** 2)
    for t in range(1, n_steps):
        x[t] = rho * x[t - 1] + s * w[t]
    return x


# ----------------------------------------------------------------------------
# RT at exact positions
# ----------------------------------------------------------------------------
@dataclass
class RTTrack:
    """한 AMR 궤적(T 시점)에서의 RT 경로.  a, tau: (T, 4, P),  pid: (T, 4, P) 경로 id (0..4, -1 무효)."""
    a: np.ndarray
    tau: np.ndarray
    pid: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def power_lt(self) -> np.ndarray:
        """국소 평균(long-term) 전력: 경로 전력의 비동기 합, 안테나 평균 (T,)."""
        return (np.abs(self.a) ** 2).sum(-1).mean(-1)


def _path_ids(rt: dict) -> np.ndarray:
    inter, verts, valid = rt["interactions"][0], rt["vertices"][0], rt["valid"]
    pid = np.full(valid.shape, -1, dtype=int)
    pid[valid & (inter == 0)] = 0
    refl = valid & (inter != 0)
    for k, (ax, w) in enumerate(((0, 0.0), (0, config.ROOM_X_M), (1, 0.0), (1, config.ROOM_Y_M)), start=1):
        pid[refl & (np.abs(verts[..., ax] - w) < 1e-3)] = k
    return pid


def rt_tracks(points_xy: np.ndarray, verbose: bool = True) -> RTTrack:
    rt = rt_paths_at(points_xy, verbose=verbose)
    a, _ = path_gains_spec(rt)                       # handout 이득식 (ρ=0.7, α·chord) — long-term 모델과 동일
    return RTTrack(a=a, tau=rt["tau"].astype(np.float64), pid=_path_ids(rt))


def take(track: RTTrack, idx: np.ndarray) -> RTTrack:
    return RTTrack(track.a[idx], track.tau[idx], track.pid[idx])


# ----------------------------------------------------------------------------
# Channel synthesis
# ----------------------------------------------------------------------------
def synthesize(track: RTTrack, dt: float, gaussian: bool, k_db: float | None = None,
               sd_m: float | None = None, sa: float | None = None) -> np.ndarray:
    """(T, 4, 64) 주파수 응답.  gaussian=False 면 RT 결정적 short-term 만."""
    a, tau = track.a.copy(), track.tau.copy()
    T, n_ant, _ = a.shape
    if not gaussian:
        return cir_to_H(a, tau)
    k_db = config.ST_DIFFUSE_K_DB if k_db is None else k_db
    sd_m = config.ST_PATH_LEN_JITTER_STD_M if sd_m is None else sd_m
    sa = config.ST_PATH_AMP_JITTER_STD if sa is None else sa

    # (a) 경로별 요동: 같은 물리 경로(pid)는 4개 안테나에서 같은 요동을 겪는다.
    xd = gauss_markov(T, (N_PATH_IDS,), dt, config.ST_PATH_COHERENCE_S)          # 길이 요동 (정규화)
    xa = gauss_markov(T, (N_PATH_IDS,), dt, config.ST_PATH_COHERENCE_S)          # 진폭 요동
    pid = np.clip(track.pid, 0, None)
    t_idx = np.arange(T)[:, None, None]
    dtau = sd_m * xd[t_idx, pid] / config.C_MPS
    amp = np.clip(1 + sa * xa[t_idx, pid], 0, None)
    valid = track.pid >= 0
    a = np.where(valid, a * amp * np.exp(-2j * np.pi * config.FREQ_HZ * dtau), 0)
    tau = np.where(valid, tau + dtau, 0)

    # (b) 확산 성분: N_s 개의 작은 산란체 (지연·출발각은 실현마다 고정, 진폭은 시간에 따라 변함)
    ns = config.ST_DIFFUSE_N
    if np.isfinite(k_db):
        g = gauss_markov(T, (ns,), dt, config.ST_DIFFUSE_COHERENCE_S, complex_=True)  # (T, ns)
        phi = RNG.uniform(0, 2 * np.pi, ns)
        excess = RNG.exponential(config.ST_DIFFUSE_DELAY_SPREAD_S, ns)
        p_rt = track.power_lt                                                        # (T,)
        scale = np.sqrt(p_rt / (10 ** (k_db / 10) * ns))                             # (T,)
        n = np.arange(n_ant)
        steer = np.exp(1j * np.pi * n[:, None] * np.cos(phi)[None, :])               # ULA 위상 (4, ns)
        tau0 = np.where(valid, tau, np.inf).min(axis=(1, 2))                         # (T,) 첫 도달
        tau_d = tau0[:, None] + excess[None, :]                                      # (T, ns)
        a_d = (scale[:, None] * g)[:, None, :] * steer[None] * np.exp(-2j * np.pi * config.FREQ_HZ * tau_d)[:, None, :]
        a = np.concatenate([a, a_d], axis=2)
        tau = np.concatenate([tau, np.broadcast_to(tau_d[:, None, :], a_d.shape)], axis=2)
    return cir_to_H(a, tau)


def snr_db(H: np.ndarray, antenna: str, offset_db: float) -> np.ndarray:
    p = np.abs(H[:, 0, :]) ** 2 if antenna == "siso" else (np.abs(H) ** 2).sum(1)
    return 10 * np.log10(np.maximum(p * config.P_REF_W / config.N0_W, 1e-30)) + offset_db


# ----------------------------------------------------------------------------
# Scheme evaluation
# ----------------------------------------------------------------------------
def rates_with_kstar(s1_db, s2_db, k_star):
    e1, e2 = spectral_eff(s1_db), spectral_eff(s2_db)
    c1 = np.cumsum(e1, -1)
    c2 = np.flip(np.cumsum(np.flip(e2, -1), -1), -1)
    k = k_star[..., None]
    r1 = config.DELTA_F_HZ * np.take_along_axis(c1, k - 1, -1)[..., 0]
    r2 = config.DELTA_F_HZ * np.take_along_axis(c2, k, -1)[..., 0]
    return r1, r2


def evaluate(scheme, tr1: RTTrack, tr2: RTTrack, n_real: int, k_db: float | None = None) -> dict:
    """세 가지 채널에서 같은 경로 쌍을 평가.
    offline : 오프라인(long-term) k*(t) 재사용,  adaptive : 순간 CSI로 k* 재계산 (Opt-BW만 의미 있음)."""
    sim = scheme.sim
    off1, off2 = config.TX_OFFSET_DB["AMR1"], config.TX_OFFSET_DB["AMR2"]
    qos = config.R_QOS_MBPS * 1e6

    def metrics(s1, s2):
        r1, r2 = rates_with_kstar(s1, s2, sim.k_star)
        r_off = np.minimum(r1, r2)
        if scheme.policy == "opt":
            _, a1, a2 = allocate(spectral_eff(s1), spectral_eff(s2), "opt")
            r_ad = np.minimum(a1, a2)
        else:
            r_ad = r_off
        return r_off, r_ad

    out = {"lt_trace": sim.rmin_t}
    H1, H2 = synthesize(tr1, config.SIM_DT_S, False), synthesize(tr2, config.SIM_DT_S, False)
    r_off, r_ad = metrics(snr_db(H1, scheme.antenna, off1), snr_db(H2, scheme.antenna, off2))
    out["rt_trace"], out["rt_adapt_trace"] = r_off, r_ad
    traces_off, traces_ad = [], []
    for _ in range(n_real):
        H1 = synthesize(tr1, config.SIM_DT_S, True, k_db)
        H2 = synthesize(tr2, config.SIM_DT_S, True, k_db)
        r_off, r_ad = metrics(snr_db(H1, scheme.antenna, off1), snr_db(H2, scheme.antenna, off2))
        traces_off.append(r_off); traces_ad.append(r_ad)
    traces_off, traces_ad = np.array(traces_off), np.array(traces_ad)
    out.update(st_off=traces_off, st_ad=traces_ad)

    def summ(tr):   # tr: (R, T) or (T,)
        tr = np.atleast_2d(tr)
        return {"rmin_mbps": float(tr.min(1).mean() / 1e6), "rmin_p10_mbps": float(np.percentile(tr.min(1), 10) / 1e6),
                "mean_mbps": float(tr.mean() / 1e6), "outage_frac": float((tr < qos).mean())}
    out["summary"] = {"long_term_grid": summ(sim.rmin_t), "rt_exact": summ(out["rt_trace"]),
                      "rt_exact_adaptive": summ(out["rt_adapt_trace"]),
                      "lt_plus_st_offline_kstar": summ(traces_off), "lt_plus_st_adaptive": summ(traces_ad)}
    return out
