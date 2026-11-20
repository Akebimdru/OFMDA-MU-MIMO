"""기말 프로젝트 전체 실행 (Sionna 버전).

    python main.py                 # Task 1~5 전부 (fingerprint는 캐시 재사용)
    python main.py --force-rt      # Sionna RT fingerprint 다시 계산
    python main.py --no-video      # 애니메이션 생략 (빠른 확인용)
    python main.py --render        # Sionna RT 3D 장면 렌더 이미지도 저장

결과: ../figures/*.png, *.mp4, results_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Visual Studio 에서 다른 작업 디렉터리로 실행해도 모듈을 찾도록
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import config


def setup_sionna(verbose: bool = True) -> None:
    """CPU 강제 + 전역 seed.  sionna.rt 를 먼저 import 하면 Mitsuba variant 가 자동 선택된다
    (GPU 없으면 llvm_ad_mono_polarized)."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import sionna.rt  # noqa: F401
    import mitsuba as mi
    import sionna.phy
    sionna.phy.config.device = "cpu"
    sionna.phy.config.seed = config.RANDOM_SEED
    if verbose:
        import sionna
        print(f"Sionna {sionna.__version__} | Mitsuba variant {mi.variant()} | device {sionna.phy.config.device}")


def report_task1(data, grid) -> dict:
    from simulator import avg_snr_map_db, bilinear
    H = data["H"]
    siso, mimo = avg_snr_map_db(H, "siso"), avg_snr_map_db(H, "mrc")
    at = lambda m, x, y: float(bilinear(m, np.array([[x, y]]))[0])     # 셀 중심 격자 → bilinear
    gain = mimo - siso
    free = ~grid.blocked
    bright = np.unravel_index(np.argmax(np.where(free, siso, -np.inf)), siso.shape)
    dark = np.unravel_index(np.argmin(np.where(free, siso, np.inf)), siso.shape)
    out = {
        "H_shape": list(H.shape),
        "noise_dbm_per_sc": float(config.NOISE_DBM),
        "siso_snr_at_3_3_db": at(siso, 3, 3),
        "siso_snr_at_13_5_db": at(siso, 13, 5),
        "mimo_minus_siso_mean_db": float(gain.mean()),
        "mimo_minus_siso_std_db": float(gain.std()),
        "mimo_minus_siso_min_max_db": [float(gain.min()), float(gain.max())],
        "brightest_free_cell": {"x": float(config.GRID_X_M[bright[1]]), "y": float(config.GRID_Y_M[bright[0]]),
                                "snr_db": float(siso[bright])},
        "darkest_free_cell": {"x": float(config.GRID_X_M[dark[1]]), "y": float(config.GRID_Y_M[dark[0]]),
                              "snr_db": float(siso[dark])},
        "east_half_mean_siso_db": float(siso[:, config.GRID_X_M > 9].mean()),
        "west_half_mean_siso_db": float(siso[:, config.GRID_X_M < 9].mean()),
        "sionna_rt_check": data["rt_check"],
    }
    out["siso_contrast_db"] = out["brightest_free_cell"]["snr_db"] - out["darkest_free_cell"]["snr_db"]
    print("\n=== Task 1 sanity (target: (3,3)≈50 dB, (13,5)≲25 dB, MIMO−SISO≈6 dB) ===")
    print(f"  SISO SNR (3,3)  = {out['siso_snr_at_3_3_db']:.2f} dB")
    print(f"  SISO SNR (13,5) = {out['siso_snr_at_13_5_db']:.2f} dB")
    print(f"  MIMO−SISO mean  = {out['mimo_minus_siso_mean_db']:.2f} dB (std {out['mimo_minus_siso_std_db']:.2f})")
    print(f"  SISO contrast   = {out['siso_contrast_db']:.1f} dB")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-rt", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    np.random.seed(config.RANDOM_SEED)
    setup_sionna()

    import visualizer as viz
    from fingerprint import load_or_build
    from joint_planner import score_all_pairs, select
    from path_planner import build_path_sets, path_length_m
    from schemes import MAIN5, SCHEMES, marginals, online_replay, pareto_frontier, qos_sweep, run_ablation

    summary: dict = {}
    # ---------------- Task 1 ----------------
    data = load_or_build(force=args.force_rt)
    H = data["H"]
    from simulator import avg_snr_map_db
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.FIGURES_DIR / "snr_siso_db.npy", avg_snr_map_db(H, "siso"))   # T1.2 (20, 36)
    np.save(config.FIGURES_DIR / "snr_mimo_db.npy", avg_snr_map_db(H, "mrc"))

    # ---------------- Task 2.1 / 2.2 ----------------
    grid, paths = build_path_sets()
    summary["task1"] = report_task1(data, grid)
    summary["task2"] = {
        "n_candidates": {k: len(v) for k, v in paths.items()},
        "shortest_length_m": {k: path_length_m(v[0]) for k, v in paths.items()},
        "shortest_through_gap": {k: bool(np.any((np.abs(grid.xy_of(v[0])[:, 0] - 9) < .3)
                                                & (grid.xy_of(v[0])[:, 1] >= 5.0) & (grid.xy_of(v[0])[:, 1] <= 5.5)))
                                 for k, v in paths.items()},
        "shortest_edge_detour": {k: bool(np.any((grid.xy_of(v[0])[:, 1] < 1) | (grid.xy_of(v[0])[:, 1] > 9))
                                         & np.any(np.abs(grid.xy_of(v[0])[:, 0] - 9) < .3)) for k, v in paths.items()},
        "blocked_cells": int(grid.blocked.sum()),
    }
    print("\n=== Task 2.1/2.2 ===", summary["task2"])
    viz.plot_snr_maps(H, grid, paths)
    _, along = viz.plot_along_path(H, grid, paths)
    summary["task1"]["along_path_amr1_shortest"] = along

    # ---------------- Task 2.3 + 3 ----------------
    print("\n=== Task 2.3 / 3: 900-pair enumeration × 4 technology stacks ===")
    t0 = time.time()
    res, scores = run_ablation(grid, paths, H)
    print(f"  enumeration done in {time.time() - t0:.1f} s")
    deltas = marginals(res)
    summary["task3"] = {
        "schemes": {n: {"path": res[n].path_kind, "bw": res[n].policy, "ant": res[n].antenna,
                        "pair": list(res[n].pair), "qos_feasible": res[n].feasible,
                        "rmin_mbps": res[n].sim.rmin / 1e6, "mean_mbps": res[n].sim.mean / 1e6,
                        "T_s": res[n].sim.T, "L1_m": res[n].sim.L1, "L2_m": res[n].sim.L2}
                    for n, *_ in SCHEMES},
        "marginal_rmin_mbps": deltas,
    }
    for n, *_ in SCHEMES:
        s = summary["task3"]["schemes"][n]
        print(f"  {n:<11s} pair={tuple(s['pair'])!s:<9s} Rmin={s['rmin_mbps']:6.2f}  mean={s['mean_mbps']:6.2f}  "
              f"T={s['T_s']:5.1f}s  L1={s['L1_m']:.2f} L2={s['L2_m']:.2f}")
    print("  Δadd :", {k: round(v, 2) for k, v in deltas["add"].items()})
    print("  Δdrop:", {k: round(v, 2) for k, v in deltas["drop"].items()}, " JOINT−NAIVE =", round(deltas["joint_minus_naive"], 2))
    j = res["JOINT"].sim
    summary["task3"]["joint_bottleneck_fraction_amr2"] = float(np.mean(j.r2 <= j.r1))
    naive, optbw = res["NAIVE"].sim, res["+OPTBW"].sim
    gap = optbw.rmin_t - naive.rmin_t
    summary["task3"]["naive_vs_optbw_max_gap"] = {"t_s": float(naive.t[np.argmax(gap)]), "gap_mbps": float(gap.max() / 1e6)}
    summary["task3"]["naive_rmin_time_s"] = float(naive.t[np.argmin(naive.rmin_t)])
    summary["task3"]["naive_rmin_pos"] = {"AMR1": naive.pos1[np.argmin(naive.rmin_t)].tolist(),
                                          "AMR2": naive.pos2[np.argmin(naive.rmin_t)].tolist()}

    viz.plot_candidates(H, grid, paths, res["JOINT"].pair)
    viz.plot_paths_by_scheme(H, grid, paths, res)
    viz.plot_bw_share(res, MAIN5)
    viz.plot_throughput(res, MAIN5)
    viz.plot_bottleneck(res, MAIN5)
    viz.plot_ablation(res, deltas)

    # ---------------- Task 4 ----------------
    js = scores[("mrc", "opt")]
    front = pareto_frontier(js)
    sweep = qos_sweep(js)
    jflat = int(np.ravel_multi_index(res["JOINT"].pair, js.T.shape))
    summary["task4"] = {
        "feasible_count": {q: int((js.rmin >= q * 1e6).sum()) for q in (50, 70, 80)},
        "frontier_size": int(len(front)), "joint_on_frontier": bool(jflat in set(front.tolist())),
        "frontier_points": [{"T_s": float(js.T.ravel()[i]), "rmin_mbps": float(js.rmin.ravel()[i] / 1e6)} for i in front],
        "qos_sweep": sweep,
    }
    viz.plot_pareto(js, front, res["JOINT"].pair)
    viz.plot_qos_sweep(sweep)
    js3 = score_all_pairs(grid, paths, H, "mrc", "opt", offset2_db=-3.0)
    sweep3 = qos_sweep(js3)
    summary["task4"]["qos_sweep_amr2_minus3db"] = sweep3
    viz.plot_qos_sweep(sweep3, "12_qos_sweep_amr2_minus3db.png", "  —  AMR2 offset −3 dB")
    print("\n=== Task 4 ===")
    print("  feasible pairs @50/70/80 Mbps:", summary["task4"]["feasible_count"], "| JOINT on frontier:",
          summary["task4"]["joint_on_frontier"])
    for r in sweep:
        print(f"  QoS {r['qos_mbps']:>2} → {'OK ' if r['feasible'] else 'INF'} T={r['T_s']:5.1f}  Rmin={r['rmin_mbps']:6.2f}  "
              f"L1={r['L1_m']:.2f}  L2={r['L2_m']:.2f}")

    # ---------------- Task 5 ----------------
    robust = {n: online_replay(res[n].sim) for n in MAIN5}
    summary["task5"] = {n: {"offline_rmin_mbps": res[n].sim.rmin / 1e6,
                            "online_rmin_mean_mbps": float(robust[n]["rmin_real"].mean() / 1e6),
                            "ratio_rmin": robust[n]["ratio"], "ratio_mean": robust[n]["mean_ratio"]} for n in MAIN5}
    viz.plot_online(res, robust, MAIN5)
    print("\n=== Task 5.3 (offline k* reused, 30 Rayleigh realisations) ===")
    for n in MAIN5:
        print(f"  {n:<11s} offline Rmin={res[n].sim.rmin/1e6:6.2f}  online E[Rmin]={summary['task5'][n]['online_rmin_mean_mbps']:6.2f}"
              f"  ratio={robust[n]['ratio']:.3f}")
    if not args.no_video:
        print("  writing animations ...")
        summary["task5_videos"] = [str(viz.animate(res["NAIVE"], H, grid, beam=False).name),
                                   str(viz.animate(res["JOINT"], H, grid, beam=True).name)]
    if args.render:
        viz.render_sionna_scene(H, grid, paths)

    with open(config.FIGURES_DIR / "results_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=float)
    print(f"\nDone in {time.time() - t_start:.1f} s → {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
