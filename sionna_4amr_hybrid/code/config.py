"""4-AMR 과제 설정 (맵, PHY, AMR).

맵 값은 사용자가 2026-11-02에 알려준 새 과제 맵이고,
PHY/OFDM 수치는 이전 프로젝트(sionna_map_python_algo/code/config.py)에서 그대로 가져왔다.
아직 정해지지 않은 값(BS 위치, AMR 4대의 시작/도착점)은 None으로 두었다.
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


# 장애물 4개. 모두 바닥(z=0)부터 천장(5 m)까지.
# 원기둥 재질은 말씀이 없어서 콘크리트로 가정했다.
OBSTACLES: tuple[Obstacle, ...] = (
    # 2026-11-02 수정: 두 벽 두께 절반(1 m -> 0.5 m), Wall 2 왼쪽 끝 1 m 연장(x 10 -> 9).
    Obstacle("Wall 1", "box", (3.0, 4.0), "concrete", 5.0, size_xyz_m=(0.5, 5.0, 5.0)),   # x 2.75~3.25, y 1.5~6.5
    Obstacle("Wall 2", "box", (11.5, 3.0), "concrete", 5.0, size_xyz_m=(5.0, 0.5, 5.0)),  # x 9~14, y 2.75~3.25
    Obstacle("Pillar 3", "cylinder", (8.0, 1.0), "concrete", 5.0, radius_m=0.4),
    Obstacle("Pillar 4", "cylinder", (9.0, 6.0), "concrete", 5.0, radius_m=0.4),
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

# AMR 4대의 출발점 (AMR 중심 x, y) [m]
AMR_STARTS_M: dict[str, tuple[float, float]] = {
    "AMR1": (0.5, 0.5),
    "AMR2": (0.5, 7.5),
    "AMR3": (15.5, 0.5),
    "AMR4": (15.5, 7.5),
}
# 목적지 = 대각선 반대편 AMR의 출발점 (1<->4, 2<->3 서로 맞바꿈)
AMR_GOAL_OF: dict[str, str] = {"AMR1": "AMR4", "AMR4": "AMR1", "AMR2": "AMR3", "AMR3": "AMR2"}
AMR_ENDPOINTS_M: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    amr: (start, AMR_STARTS_M[AMR_GOAL_OF[amr]]) for amr, start in AMR_STARTS_M.items()
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
