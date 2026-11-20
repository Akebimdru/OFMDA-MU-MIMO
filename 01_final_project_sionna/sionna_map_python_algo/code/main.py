# [원본과 다름] 아래 docstring 수정 (원본: "Project driver.  Currently implements Stage 0 and Task 1.1 only.")
"""Project driver.  Identical to the pure-Python project except that the Task 1.1 map
is built with Sionna RT (see fingerprint.py) and ``--sionna-figures`` draws the map figures."""

from __future__ import annotations

import argparse
import json

import numpy as np

import config
from fingerprint import load_or_build_fingerprint


def validate_task1(data: dict[str, np.ndarray]) -> None:
    """Small, repeatable sanity checks required before downstream stages."""
    h = data["H"]
    assert h.shape == (20, 36, 4, 64), f"Unexpected fingerprint shape: {h.shape}"
    assert np.iscomplexobj(h)
    assert np.isfinite(h.real).all() and np.isfinite(h.imag).all()
    assert data["occupied"].shape == (20, 36)
    expected_occupied = 20  # Cell centres falling in the exact obstacle footprints.
    assert int(data["occupied"].sum()) == expected_occupied, "Obstacle grid occupancy changed unexpectedly"

    # MRC must use all four antenna powers and therefore cannot be lower than SISO.
    center_h = h[10, 17]  # (x, y) = (8.75, 5.25), a cell in the central gap.
    siso_power = np.abs(center_h[0]) ** 2
    mrc_power = np.sum(np.abs(center_h) ** 2, axis=0)
    assert np.all(mrc_power >= siso_power)

    print("Task 1 sanity checks passed")
    tx_w, noise_w = config.tx_power_w(), config.noise_power_w()
    snr_siso_db = 10.0 * np.log10(tx_w * np.mean(np.abs(h[:, :, 0, :]) ** 2, axis=2) / noise_w)
    snr_mimo_db = 10.0 * np.log10(tx_w * np.mean(np.sum(np.abs(h) ** 2, axis=2), axis=2) / noise_w)
    near_bs = snr_siso_db[5, 5]       # Cell centre (2.75, 2.75), near stated (3, 3).
    east_shadow = snr_siso_db[9, 25]  # Cell centre (12.75, 4.75), near stated (13, 5).
    array_gain_db = snr_mimo_db - snr_siso_db

    print(f"  grid: {h.shape[1]} x {h.shape[0]} cells, 4 antennas, 64 subcarriers")
    print(f"  obstacle-footprint cells: {int(data['occupied'].sum())}")
    print(f"  noise per subcarrier: {config.NOISE_POWER_DBM:.2f} dBm")
    print(f"  central-gap channel power: SISO={np.mean(siso_power):.3e}, MRC={np.mean(mrc_power):.3e}")
    print(f"  SISO SNR near BS: {near_bs:.2f} dB (target about 50 dB)")
    print(f"  SISO SNR east shadow: {east_shadow:.2f} dB (target <= 25 dB)")
    print(f"  mean MIMO-SISO gain: {np.mean(array_gain_db):.2f} dB (target about 6 dB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Indoor multi-AMR final-project simulator")
    parser.add_argument("--force-fingerprint", action="store_true", help="ignore the cached Task-1 fingerprint")
    parser.add_argument("--channel-maps", action="store_true", help="also save Task 1.2-1.3 SNR maps and validation")
    # [원본과 다름] --sionna-figures 옵션 추가
    parser.add_argument("--sionna-figures", action="store_true", help="save the Sionna RT map figures (00_*)")
    parser.add_argument("--paths", action="store_true", help="build and save Task 2.1-2.2 A* candidate paths")
    parser.add_argument("--bandwidth", action="store_true", help="run Task 3.1 Equal-BW and Opt-BW on shortest paths")
    parser.add_argument("--joint-plan", action="store_true", help="run Task 2.3 QoS-constrained 900-pair planning")
    parser.add_argument("--ablation", action="store_true", help="run the Task 3.3 eight-scheme ablation")
    parser.add_argument("--tradeoffs", action="store_true", help="run Task 4.1 Pareto and Task 4.2 QoS sweep")
    parser.add_argument("--task5", action="store_true", help="run Task 5 animations and Rayleigh robustness")
    args = parser.parse_args()
    np.random.seed(config.RANDOM_SEED)
    config.FIGURES_DIR.mkdir(exist_ok=True)
    data = load_or_build_fingerprint(force=args.force_fingerprint)
    validate_task1(data)
    if args.channel_maps:
        from visualizer import create_task1_outputs
        report = create_task1_outputs()
        print(f"Saved Task 1 channel-map outputs; SISO contrast: {report['siso_contrast_db']:.2f} dB")
    # ===== [원본과 다름] Sionna 맵 그림(00_*) 생성 블록 추가 =====
    if args.sionna_figures:
        from sionna_figures import create_sionna_figures
        summary = create_sionna_figures()
        print("Saved Sionna map figures:", json.dumps(summary, indent=2))
    # ===== [원본과 다름 끝] =====
    if args.paths:
        from path_planner import build_path_sets, path_length_m, save_path_sets
        from visualizer import create_candidate_path_plot
        grid, paths = build_path_sets()
        saved_path = save_path_sets(grid, paths)
        print(f"Saved {len(paths['AMR1'])} candidates per AMR: {saved_path}")
        for amr, candidates in paths.items():
            print(f"  {amr} shortest path: {path_length_m(candidates[0]):.3f} m")
        print(f"Saved candidate-path plot: {create_candidate_path_plot()}")
    if args.bandwidth:
        from path_planner import build_path_sets, path_xy_m
        from simulator import simulate_pair, validate_opt_bw
        from visualizer import create_bw_allocation_plot
        grid, paths = build_path_sets()
        path1, path2 = path_xy_m(grid, paths["AMR1"][0]), path_xy_m(grid, paths["AMR2"][0])
        results = {
            "SISO equal": simulate_pair(path1, path2, data["H"], data["x_m"], data["y_m"], "siso", "equal"),
            "SISO opt": simulate_pair(path1, path2, data["H"], data["x_m"], data["y_m"], "siso", "opt"),
            "MRC equal": simulate_pair(path1, path2, data["H"], data["x_m"], data["y_m"], "mrc", "equal"),
            "MRC opt": simulate_pair(path1, path2, data["H"], data["x_m"], data["y_m"], "mrc", "opt"),
        }
        validation = validate_opt_bw(results["SISO opt"].snr1_db, results["SISO opt"].snr2_db)
        assert all(value is not False for value in validation.values() if isinstance(value, bool)), validation
        validation.update({
            "sim_dt_s": config.SIM_DT_S,
            "shortest_path_completion_time_s": results["SISO opt"].completion_time_s,
            "time_steps": int(len(results["SISO opt"].time_s)),
        })
        with (config.FIGURES_DIR / "task3_1_validation.json").open("w", encoding="utf-8") as handle:
            json.dump(validation, handle, indent=2)
        print("Task 3.1 conditions passed:", validation)
        print(f"Saved bandwidth plot: {create_bw_allocation_plot(results)}")
    if args.joint_plan:
        from joint_planner import build_task23_plans, save_task23_outputs, validate_selection
        from path_planner import path_length_m, path_xy_m
        from visualizer import create_paths_by_scheme_plot
        grid, paths, smart, joint = build_task23_plans(data["H"])
        smart_validation, joint_validation = validate_selection(smart, data["H"]), validate_selection(joint, data["H"])
        joint_amr2_y = path_xy_m(grid, paths["AMR2"][joint.selected_pair[1]])[:, 1]
        amr1_shortest_length = path_length_m(paths["AMR1"][0])
        amr1_joint_length = path_length_m(paths["AMR1"][joint.selected_pair[0]])
        joint_validation.update({
            # Handout p. 42 calls these the expected qualitative result at 70 Mbps.
            # They are reported separately from algorithmic correctness checks.
            # Multiple 8-connected paths can tie exactly; use metres, not candidate index.
            "expected_amr1_remains_shortest": bool(np.isclose(amr1_joint_length, amr1_shortest_length)),
            "amr1_joint_minus_shortest_m": float(amr1_joint_length - amr1_shortest_length),
            "expected_amr2_takes_north_detour": bool(joint_amr2_y.max() > 5.5),
        })
        joint_validation["expected_qualitative_pattern_matches"] = bool(
            joint_validation["expected_amr1_remains_shortest"]
            and joint_validation["expected_amr2_takes_north_detour"]
        )
        for validation in (smart_validation, joint_validation):
            required_checks = {
                key: value for key, value in validation.items()
                if isinstance(value, bool) and key != "used_infeasible_fallback" and not key.startswith("expected_")
            }
            assert all(required_checks.values()), validation
        save_task23_outputs(smart, joint)
        with (config.FIGURES_DIR / "task2_3_validation.json").open("w", encoding="utf-8") as handle:
            json.dump({"SMARTPATH": smart_validation, "JOINT": joint_validation}, handle, indent=2)
        print("Task 2.3 conditions passed:", {"SMARTPATH": smart_validation, "JOINT": joint_validation})
        print(f"Saved path-comparison plot: {create_paths_by_scheme_plot(grid, paths, smart, joint)}")
    if args.ablation:
        from ablation import build_ablation, marginal_contributions, save_ablation_outputs, validate_ablation
        from visualizer import create_ablation_bars, create_bottleneck_plots, create_bw_allocation_plot, create_throughput_plot
        _, results, _ = build_ablation(data["H"])
        validation = validate_ablation(results)
        assert all(validation.values()), validation
        deltas = marginal_contributions(results)
        save_ablation_outputs(results)
        with (config.FIGURES_DIR / "task3_3_validation.json").open("w", encoding="utf-8") as handle:
            json.dump(validation, handle, indent=2)
        print("Task 3.3 conditions passed:", validation)
        print(f"Saved ablation plot: {create_ablation_bars(results, deltas)}")
        main_schemes = {name: results[name] for name in ("NAIVE", "+MRC", "+OPTBW", "+SMARTPATH", "JOINT")}
        print(f"Saved five-scheme bandwidth plot: {create_bw_allocation_plot(main_schemes)}")
        print(f"Saved five-scheme throughput plot: {create_throughput_plot(main_schemes)}")
        print("Saved bottleneck plots:", [str(path) for path in create_bottleneck_plots(main_schemes)])
    if args.tradeoffs:
        from tradeoff_analysis import build_tradeoff_analysis, save_tradeoff_outputs, validate_tradeoff_analysis
        from visualizer import create_pareto_plot, create_qos_sweep_plot
        plan, frontier, sweep = build_tradeoff_analysis(data["H"])
        validation = validate_tradeoff_analysis(plan, frontier, sweep)
        assert all(validation.values()), validation
        save_tradeoff_outputs(plan, frontier, sweep)
        with (config.FIGURES_DIR / "task4_validation.json").open("w", encoding="utf-8") as handle:
            json.dump(validation, handle, indent=2)
        print("Task 4 conditions passed:", validation)
        print(f"Saved Pareto plot: {create_pareto_plot(plan, frontier)}")
        print(f"Saved QoS-sweep plot: {create_qos_sweep_plot(sweep)}")
        # [원본과 다름] T4.2 질문 3("AMR2 링크가 −7 dB 대신 −3 dB라면?")용 추가 스윕.
        # AMR2 오프셋만 −3 dB로 잠시 바꿔 같은 알고리즘을 다시 돌리고, 끝나면 원래 값으로 되돌린다.
        original_offset = config.TX_OFFSET_AMR2_DB
        config.TX_OFFSET_AMR2_DB = -3.0
        try:
            plan3, _, sweep3 = build_tradeoff_analysis(data["H"])
        finally:
            config.TX_OFFSET_AMR2_DB = original_offset
        with (config.FIGURES_DIR / "task4_qos_sweep_amr2_minus3db.json").open("w", encoding="utf-8") as handle:
            json.dump({"feasibility_counts": {str(q): int((plan3.rmin_bps >= q * 1e6).sum()) for q in (50, 70, 80)},
                       "qos_sweep": [p.__dict__ | {"path_pair": list(p.path_pair)} for p in sweep3]}, handle, indent=2)
        print("Saved QoS-sweep plot (AMR2 -3 dB):",
              create_qos_sweep_plot(sweep3, filename="12_qos_sweep_amr2_minus3db.png",
                                    title="Task 4.2: QoS sweep (AMR2 offset -3 dB)"))
    if args.task5:
        from task5 import run_task5, save_task5_summary, validate_task5
        robust, _, outputs = run_task5(data["H"], data["x_m"], data["y_m"])
        validation = validate_task5(robust, outputs)
        assert all(validation.values()), validation
        save_task5_summary(robust)
        with (config.FIGURES_DIR / "task5_validation.json").open("w", encoding="utf-8") as handle:
            json.dump(validation, handle, indent=2)
        print("Task 5 conditions passed:", validation)
        print("Task 5 outputs:", [str(path) for path in outputs])


if __name__ == "__main__":
    main()
