"""Task 3.3: full 2^3 Path/BW/Antenna ablation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from joint_planner import PlanSelection, enumerate_path_pairs
from path_planner import Grid, build_path_sets, path_xy_m
from simulator import SimulationResult, simulate_pair


@dataclass(frozen=True)
class SchemeResult:
    name: str
    path_kind: str
    bw_policy: str
    antenna_mode: str
    path_pair: tuple[int, int]
    simulation: SimulationResult
    used_fallback: bool

    @property
    def rmin_mbps(self) -> float:
        return float(self.simulation.rmin_bps.min() / 1e6)

    @property
    def mean_mbps(self) -> float:
        return float(self.simulation.rmin_bps.mean() / 1e6)

    @property
    def completion_time_s(self) -> float:
        return self.simulation.completion_time_s


SCHEME_DEFINITIONS = (
    ("NAIVE", "Shortest", "equal", "siso"),
    ("+MRC", "Shortest", "equal", "mrc"),
    ("+OPTBW", "Shortest", "opt", "siso"),
    ("+SMARTPATH", "CommAware", "equal", "siso"),
    ("MRC OPTBW", "Shortest", "opt", "mrc"),
    ("MRC SMART", "CommAware", "equal", "mrc"),
    ("OPT SMART", "CommAware", "opt", "siso"),
    ("JOINT", "CommAware", "opt", "mrc"),
)


def build_ablation(h: np.ndarray) -> tuple[Grid, dict[str, SchemeResult], dict[tuple[str, str], PlanSelection]]:
    """Evaluate all eight handout schemes with planning matched to active technologies."""
    grid, paths = build_path_sets()
    commaware_plans: dict[tuple[str, str], PlanSelection] = {}
    for antenna_mode, bw_policy in (("siso", "equal"), ("mrc", "equal"), ("siso", "opt"), ("mrc", "opt")):
        key = antenna_mode, bw_policy
        commaware_plans[key] = enumerate_path_pairs(
            f"CommAware-{antenna_mode}-{bw_policy}", antenna_mode, bw_policy, grid, paths, h
        )

    results: dict[str, SchemeResult] = {}
    for name, path_kind, bw_policy, antenna_mode in SCHEME_DEFINITIONS:
        if path_kind == "Shortest":
            pair, used_fallback = (0, 0), False
        else:
            plan = commaware_plans[(antenna_mode, bw_policy)]
            pair, used_fallback = plan.selected_pair, plan.used_fallback
        simulation = simulate_pair(
            path_xy_m(grid, paths["AMR1"][pair[0]]), path_xy_m(grid, paths["AMR2"][pair[1]]),
            h, grid.x_m, grid.y_m, antenna_mode, bw_policy,
        )
        results[name] = SchemeResult(name, path_kind, bw_policy, antenna_mode, pair, simulation, used_fallback)
    return grid, results, commaware_plans


def marginal_contributions(results: dict[str, SchemeResult]) -> dict[str, dict[str, float]]:
    """Compute the handout's add-X and JOINT-without-X deltas in Rmin Mbps."""
    baseline = results["NAIVE"].rmin_mbps
    joint = results["JOINT"].rmin_mbps
    return {
        "add": {
            "MRC": results["+MRC"].rmin_mbps - baseline,
            "OptBW": results["+OPTBW"].rmin_mbps - baseline,
            "SmartPath": results["+SMARTPATH"].rmin_mbps - baseline,
        },
        "drop": {
            "MRC": joint - results["OPT SMART"].rmin_mbps,
            "OptBW": joint - results["MRC SMART"].rmin_mbps,
            "SmartPath": joint - results["MRC OPTBW"].rmin_mbps,
        },
    }


def validate_ablation(results: dict[str, SchemeResult]) -> dict[str, bool | int]:
    """Confirm scheme coverage, exact axes, and reported metric consistency."""
    expected = {definition[0] for definition in SCHEME_DEFINITIONS}
    validation: dict[str, bool | int] = {
        "has_exactly_8_schemes": len(results) == 8 and set(results) == expected,
        "unique_path_bw_antenna_combinations": len({(r.path_kind, r.bw_policy, r.antenna_mode) for r in results.values()}) == 8,
        "all_rates_are_finite": all(np.isfinite(r.simulation.rmin_bps).all() for r in results.values()),
        "all_completion_times_positive": all(r.completion_time_s > 0 for r in results.values()),
        "rmin_is_minimum_time_trace": all(np.isclose(r.rmin_mbps, r.simulation.rmin_bps.min() / 1e6) for r in results.values()),
        "mean_is_mean_time_trace": all(np.isclose(r.mean_mbps, r.simulation.rmin_bps.mean() / 1e6) for r in results.values()),
        "shortest_schemes_use_pair_0_0": all(results[name].path_pair == (0, 0) for name in ("NAIVE", "+MRC", "+OPTBW", "MRC OPTBW")),
        "commaware_schemes_record_selection": all(results[name].path_pair is not None for name in ("+SMARTPATH", "MRC SMART", "OPT SMART", "JOINT")),
    }
    return validation


def save_ablation_outputs(results: dict[str, SchemeResult], output_dir: Path = config.FIGURES_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    deltas = marginal_contributions(results)
    rows = {
        name: {
            "path": result.path_kind, "bw": result.bw_policy, "antenna": result.antenna_mode,
            "path_pair": list(result.path_pair), "used_fallback": result.used_fallback,
            "rmin_mbps": result.rmin_mbps, "mean_mbps": result.mean_mbps,
            "completion_time_s": result.completion_time_s,
        }
        for name, result in results.items()
    }
    with (output_dir / "task3_3_ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"schemes": rows, "marginal_rmin_mbps": deltas}, handle, indent=2)
