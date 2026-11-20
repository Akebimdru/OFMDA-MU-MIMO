"""논문용 새 맵 B 설정 (맵, PHY, AMR).

sionna_4amr_hybrid/code/config.py를 복사해서 맵(장애물), AMR 출발/도착점만 바꿨다 (2026-11-11).
맵 B = 가운데 칸막이(높이 2.5 m) + 출입구 2개 + 높이 5 m 콘크리트 벽 2개 + 철 원기둥 2개.
PHY/OFDM, BS, AMR 크기, 이동 설정은 이전 맵과 같다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# PHY / OFDM (이전 프로젝트와 동일)
# ---------------------------------------------------------------------------
C_MPS = 299_792_458.0
FREQ_HZ = 5.8e9                      # f_c
BW_HZ = 20e6
N_SC = 64                            # K
DELTA_F_HZ = BW_HZ / N_SC
TX_POWER_DBM = -5.0                  # 부반송파 하나당 송신 전력 (사용자 확인 2026-11-02)
N_ANT_BS = 4                         # BS 4소자 ULA (MU-MIMO 4x4 ZF에 사용)
NF_DB = 7.0
NOISE_POWER_DBM = -174.0 + 10.0 * np.log10(DELTA_F_HZ) + NF_DB

SUBCARRIER_HZ = FREQ_HZ + (np.arange(N_SC) - N_SC / 2.0) * DELTA_F_HZ
SUBCARRIER_WAVELENGTH_M = C_MPS / SUBCARRIER_HZ
LAMBDA_C_M = C_MPS / FREQ_HZ


# ---------------------------------------------------------------------------
# 맵 (새 과제)
# ---------------------------------------------------------------------------
ROOM_X_M = 16.0
ROOM_Y_M = 8.0
ROOM_Z_M = 5.0
GRID_RES_M = 0.5                     # 이전 프로젝트와 같은 격자 간격 -> 32 x 16 셀 (셀 중심 0.25, 0.75, ...)

# BS 1개, 안테나 4개(x축 λ/2 간격 ULA). 방 한가운데 천장 높이(5 m).
BS_CENTER_M = np.array([ROOM_X_M / 2, ROOM_Y_M / 2, 5.0], dtype=float)
# 안테나가 천장 면 위에 정확히 놓이면 레이가 천장과 겹치므로, 레이트레이싱 때만 이만큼 아래에 둔다.
BS_RT_CEILING_GAP_M = 0.05


def bs_antenna_positions_m(rt: bool = False) -> np.ndarray:
    """BS 4소자 ULA 위치 (4, 3). rt=True면 천장 간격만큼 내린 위치."""
    positions = np.repeat(BS_CENTER_M[None, :], N_ANT_BS, axis=0)
    positions[:, 0] += (np.arange(N_ANT_BS) - 1.5) * LAMBDA_C_M / 2.0
    if rt:
        positions[:, 2] = min(BS_CENTER_M[2], ROOM_Z_M - BS_RT_CEILING_GAP_M)
    return positions


@dataclass(frozen=True)
class Obstacle:
    name: str
    kind: Literal["box", "cylinder"]
    center_xy_m: tuple[float, float]
    material: Literal["concrete", "wood", "metal"]
    height_m: float
    size_xyz_m: tuple[float, float, float] | None = None   # box: (dx, dy, dz)
    radius_m: float | None = None                          # cylinder


# 맵 B 장애물 (2026-11-11 사용자 확정, 간이 미리보기 new_map_candidates/B_v5_preview.png)
# * 칸막이: y 3.75~4.25 (두께 0.5 m), 높이 2.5 m 콘크리트, 세 조각. 출입구 x 3.5~5.0, x 11.0~12.5 (폭 1.5 m)
# * 세로 벽 2개: 높이 5 m(천장까지) 콘크리트, 두께 0.5 m
# * 원기둥 2개: 철, 반지름 0.4 m, 높이 5 m. 칸막이와 0.85 m 떨어져 AMR 한 대만 지나갈 수 있다.
OBSTACLES: tuple[Obstacle, ...] = (
    # 둘레 벽과 면이 겹치면 레이트레이싱이 불안정해서, 벽에 붙는 쪽은 5 mm 띄운다.
    Obstacle("Partition W", "box", (1.7525, 4.0), "concrete", 2.5, size_xyz_m=(3.495, 0.5, 2.5)),  # x 0.005~3.5
    Obstacle("Partition C", "box", (8.0, 4.0), "concrete", 2.5, size_xyz_m=(6.0, 0.5, 2.5)),     # x 5~11
    Obstacle("Partition E", "box", (14.2475, 4.0), "concrete", 2.5, size_xyz_m=(3.495, 0.5, 2.5)),  # x 12.5~15.995
    Obstacle("Wall S", "box", (6.25, 1.4025), "concrete", 4.995, size_xyz_m=(0.5, 2.795, 4.995)),  # x 6~6.5, y 0.005~2.8
    Obstacle("Wall N", "box", (9.75, 6.5975), "concrete", 4.995, size_xyz_m=(0.5, 2.795, 4.995)),  # x 9.5~10, y 5.2~7.995
    Obstacle("Pillar NW", "cylinder", (7.0, 5.5), "metal", 4.995, radius_m=0.4),
    Obstacle("Pillar SE", "cylinder", (9.0, 2.5), "metal", 4.995, radius_m=0.4),
)


# ---------------------------------------------------------------------------
# AMR (4대, 모두 같은 원통)
# ---------------------------------------------------------------------------
N_AMR = 4
AMR_RADIUS_M = 0.25
AMR_HEIGHT_M = 0.5
RX_HEIGHT_M = AMR_HEIGHT_M           # 수신 안테나는 로봇 윗면(0.5 m)에 있다고 가정
OBSTACLE_BUFFER_M = AMR_RADIUS_M     # 경로 계획 시 장애물에서 로봇 반지름만큼 띄운다
MOVE_CONNECTIVITY = 8                # 0.5 m 격자 위 8방향(가로/세로/대각선) 이동 (2026-11-02 결정)
AMR_SPEED_MPS = 0.5                  # 이전 프로젝트와 동일
SIM_DT_S = 0.1                       # 이전 프로젝트와 동일

# AMR 4대의 출발점과 도착점 (AMR 중심 x, y) [m]  (2026-11-11 사용자 확정)
# AMR2/AMR4는 칸막이 오른쪽 조각(x 12.5~16)의 아래/위에서 출발 (x는 조각 가운데 14.25)
AMR_STARTS_M: dict[str, tuple[float, float]] = {
    "AMR1": (1.0, 1.0),
    "AMR2": (14.25, 3.0),
    "AMR3": (1.0, 7.0),
    "AMR4": (14.25, 5.0),
}
AMR_GOALS_M: dict[str, tuple[float, float]] = {
    "AMR1": (15.0, 7.0),
    "AMR2": (1.0, 7.0),
    "AMR3": (15.0, 1.0),
    "AMR4": (1.0, 1.0),
}
AMR_ENDPOINTS_M: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    amr: (AMR_STARTS_M[amr], AMR_GOALS_M[amr]) for amr in AMR_STARTS_M
}

RANDOM_SEED = 42

# 후보 경로: 최단 1개 + 우회 25개 (경유 셀 1개를 무작위로 골라 A*로 잇는 이전 방식), 길이 상한 최단의 1.5배
N_DETOUR_PATHS = 25
N_PATH_CANDIDATES = 1 + N_DETOUR_PATHS
MAX_PATH_RATIO = 1.5


# ---------------------------------------------------------------------------
# Sionna RT (이전 sionna_map_python_algo 설정을 출발점으로 가져옴)
# ---------------------------------------------------------------------------
RT_MAX_DEPTH = 6
RT_SAMPLES_PER_SRC = 1_000_000
RT_MAX_PATHS_PER_SRC = 4_000_000
RT_SPECULAR = True
RT_REFRACTION = True
RT_DIFFRACTION = True
RT_EDGE_DIFFRACTION = True
RT_DIFFUSE = True
RT_SCATTERING_COEFFICIENT = 0.2
RT_RX_BATCH = 6                     # 한 번에 푸는 수신 셀 수 (경로 버퍼가 넘치지 않게)
RT_DIFFUSE_MAX_DEPTH = 4            # 확산 산란 전용 패스 (이전과 동일)
RT_DIFFUSE_SAMPLES_PER_SRC = 300_000

# 가우시안 환경 노이즈 (이전과 동일)
NOISE_PATH_LENGTH_STD_M = 0.002
NOISE_AMPLITUDE_STD = 0.05
NOISE_DIFFUSE_K_DB = 20.0
NOISE_SEED = RANDOM_SEED


# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = PROJECT_ROOT / "scene"
FIGURES_DIR = PROJECT_ROOT / "figures"
RT_PATHS_PATH = PROJECT_ROOT / "code" / "sionna_paths.npz"     # 원시 경로 캐시 (용량 큼)
FINGERPRINT_PATH = PROJECT_ROOT / "code" / "fingerprint.npy"   # 셀별 채널 H (Ny, Nx, 4, 64)
PATH_CANDIDATES_PATH = PROJECT_ROOT / "code" / "path_candidates.npy"


def tx_power_w() -> float:
    return 10.0 ** ((TX_POWER_DBM - 30.0) / 10.0)


def noise_power_w() -> float:
    return 10.0 ** ((NOISE_POWER_DBM - 30.0) / 10.0)


def grid_coordinates_m() -> tuple[np.ndarray, np.ndarray]:
    """셀 중심 좌표 (x: 32개, y: 16개)."""
    x = np.arange(GRID_RES_M / 2, ROOM_X_M, GRID_RES_M)
    y = np.arange(GRID_RES_M / 2, ROOM_Y_M, GRID_RES_M)
    return x, y
