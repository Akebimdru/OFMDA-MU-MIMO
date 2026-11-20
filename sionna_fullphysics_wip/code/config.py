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

# Sionna RT — 풀 물리 버전
# handout의 단순화(1차 벽 반사, ρ=0.7, α·chord 투과)를 쓰지 않고 Sionna RT가 계산한 경로와
# 복소 이득(ITU 재질 Fresnel 반사/투과 계수, 회절 계수, 확산 산란, 자유공간 손실)을 그대로 쓴다.
RT_SAMPLES_PER_SRC = 10**6
RT_MAX_DEPTH = 3                 # 반사/굴절/회절/산란 상호작용 최대 횟수
RT_LOS = True
RT_SPECULAR = True
RT_REFRACTION = True             # 장애물·벽 투과 (thin-slab, 두께는 scene_builder.MATERIALS_FULL)
RT_DIFFRACTION = True            # 모서리 회절 (UTD)
RT_EDGE_DIFFRACTION = False      # 메시 내부 모서리 회절까지 (매우 느림)
RT_DIFFUSE = True                # 확산 반사 (Lambertian)
FULL_SCATTERING = {"concrete": 0.2, "wood": 0.1}   # 산란 계수 S (에너지 비율 S^2이 확산으로)
RT_GAIN_MODEL = "sionna_full"

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
SCENE_DIR = PROJECT_DIR / "scene"
FIGURES_DIR = PROJECT_DIR / "figures"
FINGERPRINT_PATH = CODE_DIR / "fingerprint_sionna_full.npy"
