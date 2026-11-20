"""모든 장면(scene)/PHY/시뮬레이션 상수 (기말 프로젝트 handout p.5-12).

AMR별 링크 오프셋(Δ1 = 0 dB, Δ2 = -7 dB)은 fingerprint에 넣지 않고
SNR 계산 단계(simulator.py)에서만 적용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# PHY
# ----------------------------------------------------------------------------
C_MPS = 299_792_458.0
FREQ_HZ = 5.8e9
BW_HZ = 20e6
N_SC = 64
DELTA_F_HZ = BW_HZ / N_SC                     # 312.5 kHz
TX_POWER_DBM = -5.0                            # P_ref
TX_OFFSET_DB = {"AMR1": 0.0, "AMR2": -7.0}     # Δ_i (downstream)
N_ANT_BS = 4
NF_DB = 7.0
NOISE_DBM = -174.0 + 10.0 * np.log10(DELTA_F_HZ) + NF_DB   # ≈ -112 dBm / subcarrier

LAMBDA_C_M = C_MPS / FREQ_HZ
SUBCARRIER_HZ = FREQ_HZ + (np.arange(N_SC) - N_SC / 2) * DELTA_F_HZ   # f_k
LAMBDA_K_M = C_MPS / SUBCARRIER_HZ                                    # λ_k

# ----------------------------------------------------------------------------
# Room, BS, materials, obstacles
# ----------------------------------------------------------------------------
ROOM_X_M, ROOM_Y_M, ROOM_Z_M = 18.0, 10.0, 3.0
BS_CENTER_M = np.array([3.0, 5.0, 2.7])
RX_HEIGHT_M = 0.5

MATERIALS = {            # α [dB/m], ρ (amplitude reflection coefficient)
    "concrete": (50.0, 0.7),
    "metal": (200.0, 0.95),
    "wood": (8.0, 0.3),
}
WALL_RHO = MATERIALS["concrete"][1]


@dataclass(frozen=True)
class Obstacle:
    name: str
    kind: str                      # "box" | "cylinder"
    cx: float
    cy: float
    material: str
    height: float
    dx: float = 0.0                # box size
    dy: float = 0.0
    r: float = 0.0                 # cylinder radius

    @property
    def alpha(self) -> float:
        return MATERIALS[self.material][0]


OBSTACLES: tuple[Obstacle, ...] = (
    Obstacle("Wall S", "box", 9.0, 2.5, "concrete", 2.7, dx=0.3, dy=5.0),
    Obstacle("Wall N", "box", 9.0, 7.75, "concrete", 2.7, dx=0.3, dy=4.5),
    Obstacle("Pillar BS", "cylinder", 5.0, 5.0, "concrete", 2.7, r=0.3),
    Obstacle("Pillar SW", "cylinder", 3.5, 1.0, "concrete", 3.0, r=0.3),
    Obstacle("Pillar NW", "cylinder", 3.5, 9.0, "concrete", 3.0, r=0.3),
    Obstacle("Pillar SE", "cylinder", 15.0, 1.0, "concrete", 3.0, r=0.3),
    Obstacle("Pillar NE", "cylinder", 15.0, 9.0, "concrete", 3.0, r=0.3),
    Obstacle("Pillar E", "cylinder", 16.0, 5.0, "concrete", 3.0, r=0.4),
    Obstacle("Cab S1", "box", 6.0, 3.0, "wood", 2.0, dx=0.6, dy=0.6),
    Obstacle("Cab N1", "box", 6.0, 7.0, "wood", 2.0, dx=0.6, dy=0.6),
    Obstacle("Cab S2", "box", 13.0, 3.0, "wood", 2.0, dx=0.6, dy=0.6),
    Obstacle("Cab N2", "box", 13.0, 7.0, "wood", 2.0, dx=0.6, dy=0.6),
)


def bs_antenna_positions() -> np.ndarray:
    """p_ant^(n) = (3 + (n-1.5)·λc/2, 5, 2.7), n = 0..3  (x축 ULA)."""
    p = np.repeat(BS_CENTER_M[None, :], N_ANT_BS, axis=0)
    p[:, 0] += (np.arange(N_ANT_BS) - 1.5) * LAMBDA_C_M / 2
    return p


def dbm_to_w(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


P_REF_W = dbm_to_w(TX_POWER_DBM)
N0_W = dbm_to_w(NOISE_DBM)

# ----------------------------------------------------------------------------
# Grid / planning
# ----------------------------------------------------------------------------
GRID_RES_M = 0.5
# 셀 중심 격자. (격자점을 0, 0.5, ... 로 두면 y=5.0, 5.5 점이 모두 중앙 벽 버퍼 안에 들어가
# 0.5 m 통로가 막힌다. 셀 중심 y=5.25 는 벽에서 0.25 m > 0.2 m 이므로 통과 가능.)
GRID_X_M = np.arange(GRID_RES_M / 2, ROOM_X_M, GRID_RES_M)   # 36 columns: 0.25 ... 17.75
GRID_Y_M = np.arange(GRID_RES_M / 2, ROOM_Y_M, GRID_RES_M)   # 20 rows:    0.25 ... 9.75
OBSTACLE_BUFFER_M = 0.2
MAX_REFLECTIONS = 1
N_PATH_CANDIDATES = 30
MAX_PATH_RATIO = 1.5

AMR_START_GOAL = {
    "AMR1": ((1.0, 2.0), (17.0, 8.0)),
    "AMR2": ((17.0, 2.0), (1.0, 8.0)),
}
AMR_SPEED_MPS = {"AMR1": 0.5, "AMR2": 0.5}
AMR2_START_TIME_S = 0.0

# ----------------------------------------------------------------------------
# Simulation / QoS / reproducibility
# ----------------------------------------------------------------------------
SIM_DT_S = 0.1
# handout T2.3는 계획 단계 시간축을 t = 0, 2·SIM_DT, ... 로 적었다.
# 1로 두면 계획/평가가 같은 0.1 s 격자를 써서 선택된 쌍의 R_min이 그대로 재현된다.
PLAN_TIME_STRIDE = 1
R_QOS_MBPS = 70.0
R_QOS_SWEEP_MBPS = (10, 30, 50, 55, 60, 65, 68, 70, 72, 75, 80)
N_REALIZATIONS = 30
RANDOM_SEED = 42

# Sionna RT
RT_SAMPLES_PER_SRC = 10**6
# "spec"  : Sionna RT가 찾은 경로 기하(지연·반사점)에 handout의 이득식(ρ=0.7, α·chord)을 적용 (기본)
# "sionna": 반사 계수까지 Sionna RT의 ITU concrete Fresnel 계수를 그대로 사용 (비교용)
RT_GAIN_MODEL = "spec"

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
SCENE_DIR = PROJECT_DIR / "scene"
FIGURES_DIR = PROJECT_DIR / "figures"
FINGERPRINT_PATH = CODE_DIR / "fingerprint_sionna.npy"

# ----------------------------------------------------------------------------
# Short-term 성분 (교수님 지적 사항 반영)
# ----------------------------------------------------------------------------
# 과제의 채널 모델(경로 손실 + 장애물 투과 + 벽 반사)은 long-term(대규모) 성분이다.
# 여기에 두 가지 short-term(소규모) 성분을 더한다.
#  (1) 결정적 short-term: AMR의 실제 연속 위치에서 Sionna RT로 다중경로를 다시 합성 → λ 단위
#      위치 변화에 따른 다중경로 간섭(fading). 0.5 m 격자 + 보간으로는 보이지 않는 성분.
#  (2) 확률적 short-term (가우시안): 공기 흐름·온도 요동, 벌레·먼지 같은 작은 움직이는 산란체.
#      (a) 경로별 가우시안 요동 — 경로 길이 Δd ~ N(0, σ_d²) (위상), 진폭 (1 + N(0, σ_A²))
#      (b) 확산 성분 — 작은 산란체들의 복소 가우시안 경로 (Rician K-factor로 세기 지정)
#      두 과정 모두 Gauss-Markov(AR(1), ρ = exp(-Δt/T_c))로 시간 상관을 준다.
ST_PATH_LEN_JITTER_STD_M = 2e-3      # σ_d: 경로 길이 요동 2 mm → 위상 σ ≈ 2π·σ_d/λ ≈ 0.24 rad
ST_PATH_AMP_JITTER_STD = 0.05        # σ_A: 진폭 요동 5 % (≈ ±0.4 dB)
ST_PATH_COHERENCE_S = 0.5            # 경로 요동의 상관 시간
ST_DIFFUSE_K_DB = 10.0               # Rician K = (RT 경로 전력) / (가우시안 확산 전력)
ST_DIFFUSE_N = 20                    # 확산 산란체(가상 경로) 수
ST_DIFFUSE_DELAY_SPREAD_S = 15e-9    # 확산 경로 초과 지연 ~ Exp(15 ns)
ST_DIFFUSE_COHERENCE_S = 0.2         # 확산 성분의 상관 시간
ST_N_REALIZATIONS = 30
ST_K_SWEEP_DB = (40.0, 20.0, 15.0, 10.0, 5.0, 0.0)
ST_FINE_SEGMENT_M = 2.0              # 미세 해상도 fading 관찰 구간 길이
ST_FINE_STEP_M = 2e-3                # 2 mm 간격 (λ/25)
ST_DIR = PROJECT_DIR / "figures"
