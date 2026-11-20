"""Single source of truth for the final-project scene and PHY constants.

All values in this file are transcribed from the final-project handout.  The
AMR-specific link offsets intentionally do *not* belong in the fingerprint:
they are applied later while computing each AMR's SNR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


# Physical and OFDM parameters (handout pp. 5, 9, 11-12).
C_MPS = 299_792_458.0
FREQ_HZ = 5.8e9
BW_HZ = 20e6
N_SC = 64
DELTA_F_HZ = BW_HZ / N_SC
TX_POWER_DBM = -5.0
N_ANT_BS = 4
NF_DB = 7.0
NOISE_POWER_DBM = -174.0 + 10.0 * np.log10(DELTA_F_HZ) + NF_DB

# Scene and simulator parameters.
ROOM_X_M = 18.0
ROOM_Y_M = 10.0
ROOM_Z_M = 3.0
BS_CENTER_M = np.array([3.0, 5.0, 2.7], dtype=float)
RX_HEIGHT_M = 0.5
GRID_RES_M = 0.5
OBSTACLE_BUFFER_M = 0.2
MAX_REFLECTIONS = 1
SIM_DT_S = 0.1
AMR_SPEED_MPS = 0.5
TX_OFFSET_AMR1_DB = 0.0
TX_OFFSET_AMR2_DB = -7.0
R_QOS_REQUIRED_MBPS = 70.0
R_QOS_SWEEP_MBPS = (10, 30, 50, 55, 60, 65, 68, 70, 72, 75, 80)
N_PATH_CANDIDATES = 30
MAX_PATH_RATIO = 1.5
RANDOM_SEED = 42
N_REALIZATIONS = 30

CONCRETE_ALPHA_DB_PER_M = 50.0
CONCRETE_RHO = 0.7
WOOD_ALPHA_DB_PER_M = 8.0
WOOD_RHO = 0.3
METAL_ALPHA_DB_PER_M = 200.0
METAL_RHO = 0.95


@dataclass(frozen=True)
class Obstacle:
    """An obstacle that blocks a ray via material penetration loss."""

    name: str
    kind: Literal["box", "cylinder"]
    center_xy_m: tuple[float, float]
    material: Literal["concrete", "wood", "metal"]
    height_m: float
    size_xyz_m: tuple[float, float, float] | None = None
    radius_m: float | None = None

    @property
    def alpha_db_per_m(self) -> float:
        return {
            "concrete": CONCRETE_ALPHA_DB_PER_M,
            "wood": WOOD_ALPHA_DB_PER_M,
            "metal": METAL_ALPHA_DB_PER_M,
        }[self.material]


# Exact 12-item obstacle list (handout p. 7).  Boxes sit on z=0.
OBSTACLES: tuple[Obstacle, ...] = (
    Obstacle("Wall S", "box", (9.0, 2.5), "concrete", 2.7, (0.3, 5.0, 2.7)),
    Obstacle("Wall N", "box", (9.0, 7.75), "concrete", 2.7, (0.3, 4.5, 2.7)),
    Obstacle("Pillar BS", "cylinder", (5.0, 5.0), "concrete", 2.7, radius_m=0.3),
    Obstacle("Pillar SW", "cylinder", (3.5, 1.0), "concrete", 3.0, radius_m=0.3),
    Obstacle("Pillar NW", "cylinder", (3.5, 9.0), "concrete", 3.0, radius_m=0.3),
    Obstacle("Pillar SE", "cylinder", (15.0, 1.0), "concrete", 3.0, radius_m=0.3),
    Obstacle("Pillar NE", "cylinder", (15.0, 9.0), "concrete", 3.0, radius_m=0.3),
    Obstacle("Pillar E", "cylinder", (16.0, 5.0), "concrete", 3.0, radius_m=0.4),
    Obstacle("Cab S1", "box", (6.0, 3.0), "wood", 2.0, (0.6, 0.6, 2.0)),
    Obstacle("Cab N1", "box", (6.0, 7.0), "wood", 2.0, (0.6, 0.6, 2.0)),
    Obstacle("Cab S2", "box", (13.0, 3.0), "wood", 2.0, (0.6, 0.6, 2.0)),
    Obstacle("Cab N2", "box", (13.0, 7.0), "wood", 2.0, (0.6, 0.6, 2.0)),
)


def wavelength_m(frequency_hz: np.ndarray | float) -> np.ndarray | float:
    return C_MPS / frequency_hz


SUBCARRIER_HZ = FREQ_HZ + (np.arange(N_SC) - N_SC / 2.0) * DELTA_F_HZ
SUBCARRIER_WAVELENGTH_M = wavelength_m(SUBCARRIER_HZ)
LAMBDA_C_M = C_MPS / FREQ_HZ


def bs_antenna_positions_m() -> np.ndarray:
    """Return the specified 4-element x-axis half-wavelength ULA positions."""
    positions = np.repeat(BS_CENTER_M[None, :], N_ANT_BS, axis=0)
    positions[:, 0] += (np.arange(N_ANT_BS) - 1.5) * LAMBDA_C_M / 2.0
    return positions


def tx_power_w() -> float:
    return 10.0 ** ((TX_POWER_DBM - 30.0) / 10.0)


def noise_power_w() -> float:
    return 10.0 ** ((NOISE_POWER_DBM - 30.0) / 10.0)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_PATH = PROJECT_ROOT / "code" / "fingerprint.npy"
FIGURES_DIR = PROJECT_ROOT / "figures"


# ===== [원본과 다름] 여기부터 파일 끝까지 전부 새로 추가 (Sionna RT·가우시안 노이즈 설정) =====
# ---------------------------------------------------------------------------
# Sionna RT map step (the only stage that differs from the pure-Python project).
# ---------------------------------------------------------------------------
SCENE_DIR = PROJECT_ROOT / "scene"
RT_MAX_DEPTH = 6                    # up to 6 interactions (reflection/refraction/diffraction/scattering) per path
RT_SAMPLES_PER_SRC = 1_000_000      # rays launched per BS antenna
RT_MAX_PATHS_PER_SRC = 4_000_000   # candidate buffer; ~5.5 GB RAM peak with RT_RX_BATCH=6
RT_SPECULAR = True
RT_REFRACTION = True                # transmission through ITU slabs (walls, pillars, cabinets)
RT_DIFFRACTION = True
RT_EDGE_DIFFRACTION = True
RT_DIFFUSE = True
RT_SCATTERING_COEFFICIENT = 0.2     # share of reflected field amplitude scattered diffusely (Lambertian)

# Gaussian environmental fluctuations (air flow, insects, small moving objects).
# Applied per ray path before the coherent sum, plus a weak diffuse Gaussian field.
NOISE_PATH_LENGTH_STD_M = 0.002     # path length jitter -> phase jitter of 2*pi*dl/lambda
NOISE_AMPLITUDE_STD = 0.05          # relative amplitude jitter per path
NOISE_DIFFUSE_K_DB = 20.0           # power ratio deterministic / diffuse complex Gaussian field
NOISE_SEED = RANDOM_SEED
RT_DIFFUSE_MAX_DEPTH = 4            # diffuse pass: paths with at least one diffuse scattering point
RT_DIFFUSE_SAMPLES_PER_SRC = 300_000
RT_RX_BATCH = 6                     # receivers per PathSolver call (keeps the path buffer from overflowing)
RT_PATHS_PATH = PROJECT_ROOT / "code" / "sionna_paths.npz"
# ===== [원본과 다름 끝] =====
