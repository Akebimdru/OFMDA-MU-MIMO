"""Task 2.3: QoS-constrained enumeration over 30 x 30 candidate path pairs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

import config
from path_planner import Grid, build_path_sets, path_xy_m
from simulator import SimulationResult, simulate_pair_from_snr_grids, snr_grid_db, validate_opt_bw


@dataclass(frozen=True)
class PlanSelection:
    name: str
    antenna_mode: str
    bw_policy: str
    selected_pair: tuple[int, int]
    selected_result: SimulationResult
    pair_i: np.ndarray
    pair_j: np.ndarray
    completion_time_s: np.ndarray
    rmin_bps: np.ndarray
    mean_throughput_bps: np.ndarray
    feasible: np.ndarray
    used_fallback: bool


def enumerate_path_pairs(
    name: str,
    antenna_mode: str,
    bw_policy: str,
    grid: Grid,
    paths: dict[str, list[list[tuple[int, int]]]],
    h: np.ndarray,
) -> PlanSelection:
    """Vectorise each pair's inner BW scan, then filter 900 scalar evaluations by QoS."""
    path1_xy = [path_xy_m(grid, path) for path in paths["AMR1"]]
    path2_xy = [path_xy_m(grid, path) for path in paths["AMR2"]]
    snr1_grid = snr_grid_db(h, antenna_mode, config.TX_OFFSET_AMR1_DB)
    snr2_grid = snr_grid_db(h, antenna_mode, config.TX_OFFSET_AMR2_DB)
    pair_i: list[int] = []
    pair_j: list[int] = []
    completion_times: list[float] = []
    rmins: list[float] = []
    means: list[float] = []

    for i, candidate1 in enumerate(path1_xy):
        for j, candidate2 in enumerate(path2_xy):
            result = simulate_pair_from_snr_grids(candidate1, candidate2, snr1_grid, snr2_grid, grid.x_m, grid.y_m, bw_policy)
            pair_i.append(i)
            pair_j.append(j)
            completion_times.append(result.completion_time_s)
            rmins.append(float(result.rmin_bps.min()))
            means.append(float(result.rmin_bps.mean()))

    pair_i_arr = np.asarray(pair_i, dtype=int)
    pair_j_arr = np.asarray(pair_j, dtype=int)
    time_arr = np.asarray(completion_times)
    rmin_arr = np.asarray(rmins)
    mean_arr = np.asarray(means)
    feasible = rmin_arr >= config.R_QOS_REQUIRED_MBPS * 1e6
    if feasible.any():
        candidate_indices = np.flatnonzero(feasible)
        # Primary objective min T; handout's requested tie-break is max Rmin.
        selected_index = min(candidate_indices, key=lambda idx: (time_arr[idx], -rmin_arr[idx]))
        used_fallback = False
    else:
        selected_index = int(np.argmax(rmin_arr))
        used_fallback = True
    selected_pair = int(pair_i_arr[selected_index]), int(pair_j_arr[selected_index])
    selected_result = simulate_pair_from_snr_grids(
        path1_xy[selected_pair[0]], path2_xy[selected_pair[1]], snr1_grid, snr2_grid, grid.x_m, grid.y_m, bw_policy
    )
    return PlanSelection(
        name, antenna_mode, bw_policy, selected_pair, selected_result,
        pair_i_arr, pair_j_arr, time_arr, rmin_arr, mean_arr, feasible, used_fallback,
    )


def build_task23_plans(h: np.ndarray) -> tuple[Grid, dict[str, list[list[tuple[int, int]]]], PlanSelection, PlanSelection]:
    """Return the required SmartPath (SISO/equal) and JOINT (MRC/Opt-BW) plans."""
    grid, paths = build_path_sets()
    smart = enumerate_path_pairs("SMARTPATH", "siso", "equal", grid, paths, h)
    joint = enumerate_path_pairs("JOINT", "mrc", "opt", grid, paths, h)
    return grid, paths, smart, joint


def validate_selection(selection: PlanSelection, h: np.ndarray) -> dict[str, bool | int | float]:
    """Verify every T2.3 condition and the selected-plan objective exactly."""
    r_qos_bps = config.R_QOS_REQUIRED_MBPS * 1e6
    selection_index = np.flatnonzero(
        (selection.pair_i == selection.selected_pair[0]) & (selection.pair_j == selection.selected_pair[1])
    )
    assert len(selection_index) == 1
    idx = int(selection_index[0])
    validation: dict[str, bool | int | float] = {
        "candidate_pair_count_is_900": len(selection.rmin_bps) == config.N_PATH_CANDIDATES**2,
        "qos_threshold_mbps": config.R_QOS_REQUIRED_MBPS,
        "amr_offset_difference_is_7db": bool(np.allclose(
            snr_grid_db(h, selection.antenna_mode, config.TX_OFFSET_AMR1_DB)
            - snr_grid_db(h, selection.antenna_mode, config.TX_OFFSET_AMR2_DB), 7.0
        )),
        "selected_pair_index": idx,
        "feasible_pair_count": int(selection.feasible.sum()),
        "used_infeasible_fallback": selection.used_fallback,
        "selected_rmin_mbps": float(selection.rmin_bps[idx] / 1e6),
        "selected_completion_time_s": float(selection.completion_time_s[idx]),
    }
    if selection.feasible.any():
        feasible_times = selection.completion_time_s[selection.feasible]
        validation["selected_satisfies_qos"] = bool(selection.rmin_bps[idx] >= r_qos_bps)
        validation["selected_has_minimum_feasible_time"] = bool(np.isclose(selection.completion_time_s[idx], feasible_times.min()))
    else:
        validation["selected_is_max_rmin_fallback"] = bool(np.isclose(selection.rmin_bps[idx], selection.rmin_bps.max()))
    if selection.bw_policy == "opt":
        validation.update({f"opt_bw_{key}": value for key, value in validate_opt_bw(selection.selected_result.snr1_db, selection.selected_result.snr2_db).items() if isinstance(value, bool)})
    return validation


def save_task23_outputs(smart: PlanSelection, joint: PlanSelection, output_dir: Path = config.FIGURES_DIR) -> None:
    """Persist 900 JOINT scores for later Pareto analysis and report-ready plan metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "joint_pair_scores.npz",
        pair_i=joint.pair_i, pair_j=joint.pair_j, completion_time_s=joint.completion_time_s,
        rmin_bps=joint.rmin_bps, mean_throughput_bps=joint.mean_throughput_bps, feasible=joint.feasible,
    )
    summary = {
        selection.name: {
            "antenna_mode": selection.antenna_mode, "bw_policy": selection.bw_policy,
            "selected_pair": list(selection.selected_pair), "used_fallback": selection.used_fallback,
            "rmin_mbps": float(selection.selected_result.rmin_bps.min() / 1e6),
            "mean_mbps": float(selection.selected_result.rmin_bps.mean() / 1e6),
            "completion_time_s": selection.selected_result.completion_time_s,
        }
        for selection in (smart, joint)
    }
    with (output_dir / "task2_3_plan_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
