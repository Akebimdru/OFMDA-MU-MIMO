"""Task 1.2-1.3 channel-map derivation, validation, and visualisation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# This project writes figures to disk; a GUI/Tk display is neither needed nor
# available in the reproducible command-line execution environment.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.patches import Patch
import numpy as np

import config
from fingerprint import load_or_build_fingerprint


AMR_START_GOALS = {
    "AMR1": ((1.0, 2.0), (17.0, 8.0), "tab:blue"),
    "AMR2": ((17.0, 2.0), (1.0, 8.0), "tab:orange"),
}


def snr_maps_from_fingerprint(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the handout's AMR-agnostic, subcarrier-averaged SISO and MRC maps."""
    scale = config.tx_power_w() / config.noise_power_w()
    siso_linear = scale * np.mean(np.abs(h[:, :, 0, :]) ** 2, axis=2)
    mimo_linear = scale * np.mean(np.sum(np.abs(h) ** 2, axis=2), axis=2)
    return 10.0 * np.log10(siso_linear), 10.0 * np.log10(mimo_linear)


def _draw_scene(ax: plt.Axes) -> None:
    """Overlay the exact handout geometry and AMR endpoint conventions."""
    ax.scatter(*config.BS_CENTER_M[:2], marker="*", s=180, color="gold", edgecolor="black", zorder=5, label="BS")
    for obstacle in config.OBSTACLES:
        cx, cy = obstacle.center_xy_m
        if obstacle.kind == "box":
            assert obstacle.size_xyz_m is not None
            width, height, _ = obstacle.size_xyz_m
            patch = Rectangle((cx - width / 2, cy - height / 2), width, height, facecolor="none", edgecolor="black", linewidth=1.1)
        else:
            assert obstacle.radius_m is not None
            patch = Circle((cx, cy), obstacle.radius_m, facecolor="none", edgecolor="black", linewidth=1.1)
        ax.add_patch(patch)
        ax.annotate(obstacle.name, (cx, cy), xytext=(3, 3), textcoords="offset points", fontsize=6, color="black")
    for name, (start, goal, color) in AMR_START_GOALS.items():
        ax.scatter(*start, marker="s", s=40, color=color, edgecolor="white", linewidth=0.6, zorder=5)
        ax.scatter(*goal, marker="^", s=48, color=color, edgecolor="white", linewidth=0.6, zorder=5)
        ax.annotate(name, start, xytext=(4, -11), textcoords="offset points", fontsize=7, color=color, weight="bold")
    ax.set(xlim=(0, config.ROOM_X_M), ylim=(0, config.ROOM_Y_M), xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal", adjustable="box")


def _cell_value_near(values: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, point: tuple[float, float]) -> float:
    ix = int(np.argmin(np.abs(x_m - point[0])))
    iy = int(np.argmin(np.abs(y_m - point[1])))
    return float(values[iy, ix])


def create_task1_outputs(output_dir: Path = config.FIGURES_DIR) -> dict[str, float]:
    """Save required heatmaps, raw map arrays, and a concise numerical validation record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_or_build_fingerprint(verbose=False)
    h, x_m, y_m = data["H"], data["x_m"], data["y_m"]
    siso_db, mimo_db = snr_maps_from_fingerprint(h)
    np.save(output_dir / "snr_siso_db.npy", siso_db)
    np.save(output_dir / "snr_mimo_db.npy", mimo_db)

    # Common colour scale makes the MRC benefit visually interpretable.
    vmin = float(min(siso_db.min(), mimo_db.min()))
    vmax = float(max(siso_db.max(), mimo_db.max()))
    x_edges = np.r_[0.0, x_m + config.GRID_RES_M / 2.0]
    y_edges = np.r_[0.0, y_m + config.GRID_RES_M / 2.0]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True, sharex=True, sharey=True)
    mesh = None
    for ax, values, title in zip(axes, (siso_db, mimo_db), ("SISO average SNR", "4-antenna MRC average SNR")):
        mesh = ax.pcolormesh(x_edges, y_edges, values, shading="flat", cmap="viridis", vmin=vmin, vmax=vmax)
        _draw_scene(ax)
        ax.set_title(title)
    figure.colorbar(mesh, ax=axes, shrink=0.86, label="Average SNR [dB]")
    figure.suptitle("Task 1: AMR-agnostic channel fingerprint maps", fontsize=14)
    figure.savefig(output_dir / "02_snr_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    gain_db = mimo_db - siso_db
    near_bs_db = _cell_value_near(siso_db, x_m, y_m, (3.0, 3.0))
    east_shadow_db = _cell_value_near(siso_db, x_m, y_m, (13.0, 5.0))
    brightest = float(siso_db.max())
    darkest = float(siso_db.min())
    report = {
        "snr_siso_shape": list(siso_db.shape),
        "snr_mimo_shape": list(mimo_db.shape),
        "siso_near_bs_db": near_bs_db,
        "siso_east_shadow_db": east_shadow_db,
        "brightest_siso_db": brightest,
        "darkest_siso_db": darkest,
        "siso_contrast_db": brightest - darkest,
        "mimo_minus_siso_mean_db": float(gain_db.mean()),
        "mimo_minus_siso_std_db": float(gain_db.std()),
    }
    with (output_dir / "task1_channel_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def create_candidate_path_plot(output_dir: Path = config.FIGURES_DIR) -> Path:
    """Save the Task 2.2 candidate-path overlay on the AMR-agnostic MRC map."""
    from path_planner import build_path_sets, path_xy_m

    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_or_build_fingerprint(verbose=False)
    _, mimo_db = snr_maps_from_fingerprint(data["H"])
    x_m, y_m = data["x_m"], data["y_m"]
    grid, path_sets = build_path_sets()
    x_edges = np.r_[0.0, x_m + config.GRID_RES_M / 2.0]
    y_edges = np.r_[0.0, y_m + config.GRID_RES_M / 2.0]
    figure, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    mesh = ax.pcolormesh(x_edges, y_edges, mimo_db, shading="flat", cmap="viridis")
    _draw_scene(ax)
    styles = {"AMR1": ("tab:blue", "-"), "AMR2": ("tab:orange", "--")}
    for amr, candidates in path_sets.items():
        color, line_style = styles[amr]
        for candidate in candidates[1:]:
            xy = path_xy_m(grid, candidate)
            ax.plot(xy[:, 0], xy[:, 1], color=color, linestyle=line_style, linewidth=0.8, alpha=0.20)
        shortest_xy = path_xy_m(grid, candidates[0])
        ax.plot(shortest_xy[:, 0], shortest_xy[:, 1], color=color, linestyle=line_style, linewidth=2.5, label=f"{amr} shortest")
    ax.legend(loc="lower center", ncol=2, frameon=True)
    ax.set_title("Task 2.2: 30 reproducible A* candidates per AMR (MRC map)")
    figure.colorbar(mesh, ax=ax, label="MRC average SNR [dB]")
    output_path = output_dir / "04_candidate_paths.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def create_bw_allocation_plot(results: dict[str, object], output_dir: Path = config.FIGURES_DIR) -> Path:
    """Create the required stacked-area BW-share diagnostic for the five main schemes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(results), 1, figsize=(10, 2.1 * len(results)), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (name, result) in zip(axes, results.items()):
        result = getattr(result, "simulation", result)
        share = result.k_star / config.N_SC
        ax.fill_between(result.time_s, 0, share, color="tab:blue", alpha=0.80, label="AMR1 share")
        ax.fill_between(result.time_s, share, 1, color="tab:orange", alpha=0.80, label="AMR2 share")
        ax.set_ylim(0, 1)
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("time [s]")
    figure.suptitle("Task 3.1: contiguous bandwidth allocation", fontsize=13)
    output_path = output_dir / "06_bw_allocation_shortest_paths.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def create_throughput_plot(results: dict[str, object], output_dir: Path = config.FIGURES_DIR) -> Path:
    """Create Task 3.2's five-scheme Rmin-versus-time diagnostic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ("#777777", "#4c78a8", "#59a14f", "#f28e2b", "#e15759")
    figure, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    summary_lines = []
    for color, (name, item) in zip(colors, results.items()):
        result = getattr(item, "simulation", item)
        ax.plot(result.time_s, result.rmin_bps / 1e6, color=color, linewidth=1.35, label=name)
        ax.axvline(result.completion_time_s, color=color, linestyle=":", linewidth=1.0, alpha=0.9)
        summary_lines.append(
            f"{name:>10}: Rmin={result.rmin_bps.min()/1e6:5.1f}, mean={result.rmin_bps.mean()/1e6:5.1f}, T={result.completion_time_s:4.1f}"
        )
    ax.axhline(config.R_QOS_REQUIRED_MBPS, color="red", linestyle="--", linewidth=1.3, label="QoS = 70 Mbps")
    ax.set(xlabel="Time [s]", ylabel="Rmin(t) [Mbps]", title="Task 3.2: five-scheme throughput versus time")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    ax.text(0.99, 0.03, "\n".join(summary_lines), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.4, family="monospace", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.90})
    output_path = output_dir / "01_throughput.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def create_bottleneck_plots(results: dict[str, object], output_dir: Path = config.FIGURES_DIR) -> list[Path]:
    """Save one per-scheme time plot identifying the instantaneous bottleneck AMR."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "NAIVE": "03_bottleneck_naive.png",
        "+MRC": "03_bottleneck_mrc.png",
        "+OPTBW": "03_bottleneck_optbw.png",
        "+SMARTPATH": "03_bottleneck_smartpath.png",
        "JOINT": "03_bottleneck_joint.png",
    }
    output_paths: list[Path] = []
    for name, item in results.items():
        result = getattr(item, "simulation", item)
        time = result.time_s
        rate1 = result.rate1_bps / 1e6
        rate2 = result.rate2_bps / 1e6
        amr1_bottleneck = rate1 <= rate2
        figure, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
        ax.fill_between(time, 0, 1, where=amr1_bottleneck, transform=ax.get_xaxis_transform(),
                        color="tab:blue", alpha=0.13)
        ax.fill_between(time, 0, 1, where=~amr1_bottleneck, transform=ax.get_xaxis_transform(),
                        color="tab:orange", alpha=0.13)
        ax.plot(time, rate1, color="tab:blue", linewidth=1.65, label="AMR1 rate")
        ax.plot(time, rate2, color="tab:orange", linewidth=1.65, label="AMR2 rate")
        ax.plot(time, np.minimum(rate1, rate2), color="black", linewidth=1.15, linestyle="--", label="Rmin")
        ax.axhline(config.R_QOS_REQUIRED_MBPS, color="red", linewidth=1.1, linestyle=":", label="QoS = 70 Mbps")
        ax.axvline(result.completion_time_s, color="#555555", linewidth=1.0, linestyle=":")
        ax.set(xlabel="Time [s]", ylabel="Throughput [Mbps]", title=f"{name}: instantaneous bottleneck AMR")
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        handles += [Patch(facecolor="tab:blue", alpha=0.13, label="AMR1 bottleneck"),
                    Patch(facecolor="tab:orange", alpha=0.13, label="AMR2 bottleneck")]
        ax.legend(handles=handles, loc="best", ncol=2, fontsize=8)
        path = output_dir / filenames[name]
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(path)
    return output_paths


def create_paths_by_scheme_plot(grid, paths, smart, joint, output_dir: Path = config.FIGURES_DIR) -> Path:
    """Create the required T2.3 shortest/SmartPath/JOINT comparison overlay."""
    from path_planner import path_xy_m

    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_or_build_fingerprint(verbose=False)
    _, mimo_db = snr_maps_from_fingerprint(data["H"])
    x_m, y_m = data["x_m"], data["y_m"]
    x_edges = np.r_[0.0, x_m + config.GRID_RES_M / 2.0]
    y_edges = np.r_[0.0, y_m + config.GRID_RES_M / 2.0]
    figure, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    mesh = ax.pcolormesh(x_edges, y_edges, mimo_db, shading="flat", cmap="viridis")
    _draw_scene(ax)
    styles = {"AMR1": "-", "AMR2": "--"}
    plans = (("Shortest", (0, 0), "#777777", 1.8), ("SmartPath", smart.selected_pair, "#ff8c00", 2.4), ("JOINT", joint.selected_pair, "#d62728", 3.0))
    for plan_name, (idx1, idx2), color, width in plans:
        for amr, index in (("AMR1", idx1), ("AMR2", idx2)):
            xy = path_xy_m(grid, paths[amr][index])
            ax.plot(xy[:, 0], xy[:, 1], color=color, linestyle=styles[amr], linewidth=width, label=f"{plan_name} {amr}")
    ax.set_title("Task 2.3: paths by scheme on MRC heatmap")
    ax.legend(loc="lower center", ncol=3, fontsize=8, frameon=True)
    figure.colorbar(mesh, ax=ax, label="MRC average SNR [dB]")
    output_path = output_dir / "05_paths_by_scheme.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def create_ablation_bars(results: dict[str, object], deltas: dict[str, dict[str, float]], output_dir: Path = config.FIGURES_DIR) -> Path:
    """Create the required three-panel all-eight-scheme Task 3.3 bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(results)
    x = np.arange(len(names))
    metrics = (
        ("Rmin [Mbps]", [results[name].rmin_mbps for name in names]),
        ("Mean throughput [Mbps]", [results[name].mean_mbps for name in names]),
        ("Completion time [s]", [results[name].completion_time_s for name in names]),
    )
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    colors = ["#777777", "#4c78a8", "#59a14f", "#f28e2b", "#b279a2", "#76b7b2", "#edc948", "#e15759"]
    for ax, (label, values) in zip(axes, metrics):
        bars = ax.bar(x, values, color=colors, edgecolor="white")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        offset = max(values) * 0.015
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
    axes[-1].set_xticks(x, names, rotation=25, ha="right")
    delta_text = "Δadd Rmin [Mbps]\n" + "  ".join(f"{key}: {value:+.1f}" for key, value in deltas["add"].items())
    delta_text += "\nΔdrop Rmin [Mbps]\n" + "  ".join(f"{key}: {value:+.1f}" for key, value in deltas["drop"].items())
    axes[0].text(0.99, 0.96, delta_text, transform=axes[0].transAxes, ha="right", va="top", fontsize=8, family="monospace", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88})
    figure.suptitle("Task 3.3: full 2³ ablation", fontsize=14)
    output_path = output_dir / "10_ablation_bars.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def create_pareto_plot(plan, frontier: np.ndarray, output_dir: Path = config.FIGURES_DIR) -> Path:
    """Create the required 900-point feasible/infeasible T4.1 Pareto scatter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    qos_bps = config.R_QOS_REQUIRED_MBPS * 1e6
    feasible = plan.rmin_bps >= qos_bps
    figure, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.scatter(plan.completion_time_s[~feasible], plan.rmin_bps[~feasible] / 1e6, s=18, color="#9e9e9e", alpha=0.4, label="Infeasible")
    ax.scatter(plan.completion_time_s[feasible], plan.rmin_bps[feasible] / 1e6, s=20, color="tab:blue", alpha=0.7, label="Feasible")
    ax.plot(plan.completion_time_s[frontier], plan.rmin_bps[frontier] / 1e6, color="black", linewidth=1.6, label="Pareto frontier")
    selected = min(np.flatnonzero(feasible), key=lambda idx: (plan.completion_time_s[idx], -plan.rmin_bps[idx]))
    ax.scatter(plan.completion_time_s[selected], plan.rmin_bps[selected] / 1e6, marker="*", s=250, color="#ffd21f", edgecolor="black", linewidth=0.8, zorder=5, label="JOINT selected")
    ax.axhline(config.R_QOS_REQUIRED_MBPS, color="red", linestyle="--", linewidth=1.2, label="QoS = 70 Mbps")
    ax.set(xlabel="Completion time T [s]", ylabel="Rmin [Mbps]", title="Task 4.1: MRC + Opt-BW path-pair Pareto scatter")
    ax.legend(loc="best")
    ax.grid(alpha=0.22)
    output_path = output_dir / "11_pareto.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def create_qos_sweep_plot(
    sweep: list,
    output_dir: Path = config.FIGURES_DIR,
    filename: str = "12_qos_sweep.png",
    title: str = "Task 4.2: QoS sweep",
) -> Path:
    """Create an 11-point QoS sweep with separate feasible/infeasible markers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    qos = np.asarray([point.qos_mbps for point in sweep])
    t = np.asarray([point.completion_time_s for point in sweep])
    rmin = np.asarray([point.rmin_mbps for point in sweep])
    length1 = np.asarray([point.path1_length_m for point in sweep])
    length2 = np.asarray([point.path2_length_m for point in sweep])
    feasible = np.asarray([point.feasible for point in sweep])
    figure, axes = plt.subplots(3, 1, figsize=(9.5, 10), sharex=True, constrained_layout=True)
    for ax, values, label in zip(axes[:2], (t, rmin), ("Completion time T [s]", "Achieved Rmin [Mbps]")):
        ax.plot(qos[feasible], values[feasible], "o-", color="tab:green", label="Feasible")
        ax.plot(qos[~feasible], values[~feasible], "x", color="tab:red", markersize=7, mew=1.8, label="Infeasible fallback")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[1].plot(qos, qos, "r--", linewidth=1.1, label="Rmin = QoS")
    axes[2].plot(qos[feasible], length1[feasible], "o-", color="tab:green", label="AMR1 length (feasible)")
    axes[2].plot(qos[feasible], length2[feasible], "s-", color="tab:blue", label="AMR2 length (feasible)")
    axes[2].plot(qos[~feasible], length1[~feasible], "x", color="tab:red", markersize=7, mew=1.8, label="AMR1 infeasible")
    axes[2].plot(qos[~feasible], length2[~feasible], "x", color="tab:red", markersize=7, mew=1.8, label="AMR2 infeasible")
    axes[2].set_ylabel("Path length [m]")
    axes[2].grid(alpha=0.25)
    for q, time_value in zip(qos, t):
        axes[0].annotate(f"{time_value:.1f}", (q, time_value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    axes[2].legend(loc="best")
    axes[2].set_xlabel("QoS requirement [Mbps]")
    figure.suptitle(title, fontsize=14)
    output_path = output_dir / filename
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    report = create_task1_outputs()
    print("Task 1.2-1.3 channel-map validation complete")
    for label, value in report.items():
        print(f"  {label}: {value}")
    print("  Interpretation: the central concrete wall pair creates the dominant east-side shadow.")
    print("  Note: shortest-path SNR samples are added after the A* stage is implemented.")


if __name__ == "__main__":
    main()
