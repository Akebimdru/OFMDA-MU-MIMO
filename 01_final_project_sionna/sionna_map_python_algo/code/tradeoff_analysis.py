"""Task 4.1-4.2: Pareto frontier and QoS sweep for the JOINT technology stack."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from joint_planner import PlanSelection, enumerate_path_pairs
from path_planner import Grid, build_path_sets, path_length_m


@dataclass(frozen=True)
class SweepPoint:
    qos_mbps: float
    selected_index: int
    path_pair: tuple[int, int]
    completion_time_s: float
    rmin_mbps: float
    path1_length_m: float
    path2_length_m: float
    feasible: bool


def select_at_qos(plan: PlanSelection, qos_mbps: float, path_lengths1_m: np.ndarray, path_lengths2_m: np.ndarray) -> SweepPoint:
    """Apply the handout's feasible-min-T / infeasible-max-Rmin decision rule."""
    feasible = plan.rmin_bps >= qos_mbps * 1e6
    if feasible.any():
        indices = np.flatnonzero(feasible)
        selected = min(indices, key=lambda idx: (plan.completion_time_s[idx], -plan.rmin_bps[idx]))
        is_feasible = True
    else:
        selected = int(np.argmax(plan.rmin_bps))
        is_feasible = False
    i, j = int(plan.pair_i[selected]), int(plan.pair_j[selected])
    return SweepPoint(
        qos_mbps, int(selected), (i, j), float(plan.completion_time_s[selected]), float(plan.rmin_bps[selected] / 1e6),
        float(path_lengths1_m[i]), float(path_lengths2_m[j]), is_feasible,
    )


def pareto_frontier(plan: PlanSelection, qos_mbps: float = config.R_QOS_REQUIRED_MBPS) -> np.ndarray:
    """Return indices of non-dominated feasible points, sorted by completion time."""
    feasible_indices = np.flatnonzero(plan.rmin_bps >= qos_mbps * 1e6)
    order = feasible_indices[np.argsort(plan.completion_time_s[feasible_indices], kind="stable")]
    frontier: list[int] = []
    running_best = -np.inf
    for index in order:
        if plan.rmin_bps[index] > running_best:
            frontier.append(int(index))
            running_best = plan.rmin_bps[index]
    return np.asarray(frontier, dtype=int)


def build_tradeoff_analysis(h: np.ndarray) -> tuple[PlanSelection, np.ndarray, list[SweepPoint]]:
    """Evaluate once, then reuse the exact MRC+Opt-BW scores across every QoS value."""
    grid, paths = build_path_sets()
    plan = enumerate_path_pairs("JOINT", "mrc", "opt", grid, paths, h)
    length1 = np.asarray([path_length_m(path) for path in paths["AMR1"]])
    length2 = np.asarray([path_length_m(path) for path in paths["AMR2"]])
    frontier = pareto_frontier(plan)
    sweep = [select_at_qos(plan, float(qos), length1, length2) for qos in config.R_QOS_SWEEP_MBPS]
    return plan, frontier, sweep


def validate_tradeoff_analysis(plan: PlanSelection, frontier: np.ndarray, sweep: list[SweepPoint]) -> dict[str, bool | int]:
    """Check all stated T4.1/T4.2 algorithmic requirements."""
    default = select_at_qos(plan, config.R_QOS_REQUIRED_MBPS,
                            np.zeros(config.N_PATH_CANDIDATES), np.zeros(config.N_PATH_CANDIDATES))
    frontier_set = set(frontier.tolist())
    validation: dict[str, bool | int] = {
        "has_all_900_candidate_pairs": len(plan.rmin_bps) == config.N_PATH_CANDIDATES**2,
        "frontier_contains_only_feasible_points": bool(np.all(plan.rmin_bps[frontier] >= config.R_QOS_REQUIRED_MBPS * 1e6)),
        "frontier_is_strictly_improving_in_rmin": bool(np.all(np.diff(plan.rmin_bps[frontier]) > 0)) if len(frontier) > 1 else True,
        "joint_choice_is_on_frontier": default.selected_index in frontier_set,
        "qos_sweep_uses_exact_11_values": tuple(point.qos_mbps for point in sweep) == config.R_QOS_SWEEP_MBPS,
        "qos_sweep_has_11_points": len(sweep) == 11,
        "feasible_sweep_choices_satisfy_qos": all((not point.feasible) or point.rmin_mbps >= point.qos_mbps for point in sweep),
        "infeasible_sweep_choices_use_max_rmin": all(
            point.feasible or np.isclose(point.rmin_mbps, plan.rmin_bps.max() / 1e6) for point in sweep
        ),
    }
    return validation


def save_tradeoff_outputs(plan: PlanSelection, frontier: np.ndarray, sweep: list[SweepPoint], output_dir: Path = config.FIGURES_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pareto_frontier_indices": frontier.tolist(),
        "feasibility_counts": {
            str(qos): int((plan.rmin_bps >= qos * 1e6).sum()) for qos in (50, 70, 80)
        },
        "qos_sweep": [point.__dict__ | {"path_pair": list(point.path_pair)} for point in sweep],
    }
    with (output_dir / "task4_tradeoff_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
