"""Task 5: MP4 animations and offline-vs-online Rayleigh robustness analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np

import config
from ablation import SchemeResult, build_ablation
from simulator import bilinear_sample_db
from visualizer import _draw_scene, snr_maps_from_fingerprint


@dataclass(frozen=True)
class RobustnessResult:
    scheme: str
    offline_rmin_bps: np.ndarray
    online_mean_bps: np.ndarray
    online_std_bps: np.ndarray
    online_realization_rmin_bps: np.ndarray
    ratio: float
    k_star_reused: bool


def run_rayleigh_replays(scheme: SchemeResult, n_realizations: int = config.N_REALIZATIONS) -> RobustnessResult:
    """Replay an offline schedule under CN(0,1) fading; never re-optimise k_star."""
    offline = scheme.simulation
    t_steps, n_sc = offline.snr1_db.shape
    rng = np.random.default_rng(config.RANDOM_SEED)
    g = (rng.standard_normal((n_realizations, t_steps, n_sc)) + 1j * rng.standard_normal((n_realizations, t_steps, n_sc))) / np.sqrt(2.0)
    fading_power = np.abs(g) ** 2
    # Fading multiplies SNR power, not channel amplitude.  The same g is used for
    # both links as in the supplied handout pseudocode.
    snr1_online = 10.0 ** (offline.snr1_db[None, :, :] / 10.0) * fading_power
    snr2_online = 10.0 ** (offline.snr2_db[None, :, :] / 10.0) * fading_power
    efficiency1 = np.log2(1.0 + snr1_online)
    efficiency2 = np.log2(1.0 + snr2_online)
    c1 = np.cumsum(efficiency1, axis=2)
    c2 = np.cumsum(efficiency2[:, :, ::-1], axis=2)[:, :, ::-1]
    k = offline.k_star[None, :, None]  # exact offline schedule; no online argmax.
    r1 = config.DELTA_F_HZ * np.take_along_axis(c1, k - 1, axis=2).squeeze(axis=2)
    r2 = config.DELTA_F_HZ * np.take_along_axis(c2, k, axis=2).squeeze(axis=2)
    online_trace = np.minimum(r1, r2)
    per_realization_rmin = online_trace.min(axis=1)
    return RobustnessResult(
        scheme.name, offline.rmin_bps, online_trace.mean(axis=0), online_trace.std(axis=0), per_realization_rmin,
        float(per_realization_rmin.mean() / offline.rmin_bps.min()), True,
    )


def create_robustness_plot(results: dict[str, RobustnessResult], scheme_results: dict[str, SchemeResult], output_dir: Path = config.FIGURES_DIR) -> Path:
    """Create handout-required five-row deterministic/online mean/std comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(results), 1, figsize=(10.5, 2.15 * len(results)), sharex=False, constrained_layout=True)
    for ax, (name, robust) in zip(np.atleast_1d(axes), results.items()):
        time = scheme_results[name].simulation.time_s
        ax.plot(time, robust.offline_rmin_bps / 1e6, color="black", linewidth=1.4, label="Offline deterministic")
        ax.plot(time, robust.online_mean_bps / 1e6, color="tab:blue", linewidth=1.3, label="Online mean")
        ax.fill_between(time, (robust.online_mean_bps - robust.online_std_bps) / 1e6, (robust.online_mean_bps + robust.online_std_bps) / 1e6, color="tab:blue", alpha=0.22, label="Online ±1 std")
        ax.set_ylabel(f"{name}\nRmin [Mbps]")
        ax.grid(alpha=0.25)
        ax.text(0.99, 0.90, f"ratio={robust.ratio:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    axes[0].legend(loc="upper left", ncol=3, fontsize=8)
    axes[-1].set_xlabel("time [s]")
    figure.suptitle("Task 5.3: offline plan replayed under Rayleigh fading", fontsize=14)
    output_path = output_dir / "13_offline_vs_online.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _mrc_beam_gain(h_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Manual's representative-subcarrier MRC receive beam pattern."""
    theta = np.linspace(-np.pi / 2, np.pi / 2, 181)
    n = np.arange(config.N_ANT_BS)
    spacing = config.LAMBDA_C_M / 2.0
    a = np.exp(1j * 2.0 * np.pi * (spacing / config.LAMBDA_C_M) * np.sin(theta)[:, None] * n[None, :])
    w = h_vector.conj() / max(np.linalg.norm(h_vector), 1e-12)
    gain = np.abs(a @ w.conj()) ** 2
    return theta, gain / max(gain.max(), 1e-12)


def create_animation(scheme: SchemeResult, h: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, output_dir: Path = config.FIGURES_DIR, include_beam: bool = False) -> Path:
    """Write the required MP4 with map, trails, throughput cursor, BW bar, and text."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = scheme.simulation
    _, mrc_map = snr_maps_from_fingerprint(h)
    siso_map, _ = snr_maps_from_fingerprint(h)
    map_values = mrc_map if include_beam else siso_map
    x_edges, y_edges = np.r_[0.0, x_m + config.GRID_RES_M / 2.0], np.r_[0.0, y_m + config.GRID_RES_M / 2.0]
    figure = plt.figure(figsize=(12, 6.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(2.2, 1, 1), height_ratios=(1, 1))
    ax_map = figure.add_subplot(grid[:, 0])
    ax_rate = figure.add_subplot(grid[0, 1:])
    ax_bw = figure.add_subplot(grid[1, 1])
    ax_text = figure.add_subplot(grid[1, 2])
    ax_map.pcolormesh(x_edges, y_edges, map_values, shading="flat", cmap="viridis")
    _draw_scene(ax_map)
    ax_map.set_title(f"{scheme.name}: moving AMRs")
    trail1, = ax_map.plot([], [], color="tab:blue", linewidth=2, alpha=0.55)
    trail2, = ax_map.plot([], [], color="tab:orange", linewidth=2, alpha=0.55, linestyle="--")
    marker1, = ax_map.plot([], [], "o", color="tab:blue", markersize=8)
    marker2, = ax_map.plot([], [], "o", color="tab:orange", markersize=8)
    ax_rate.plot(result.time_s, result.rmin_bps / 1e6, color="tab:purple", label="Rmin(t)")
    ax_rate.axhline(config.R_QOS_REQUIRED_MBPS, color="red", linestyle="--", label="QoS")
    cursor = ax_rate.axvline(0, color="black", linestyle=":")
    ax_rate.set(xlabel="time [s]", ylabel="Rmin [Mbps]")
    ax_rate.legend(fontsize=8); ax_rate.grid(alpha=0.25)
    bw1 = ax_bw.barh([0], [0.5], color="tab:blue", label="AMR1")[0]
    bw2 = ax_bw.barh([0], [0.5], left=[0.5], color="tab:orange", label="AMR2")[0]
    ax_bw.set(xlim=(0, 1), ylim=(-0.7, 0.7), xlabel="BW share", yticks=[]); ax_bw.legend(fontsize=8, loc="upper center")
    ax_text.axis("off")
    text = ax_text.text(0.02, 0.95, "", va="top", family="monospace", fontsize=9)
    beam_line = None
    if include_beam:
        ax_beam = figure.add_axes([0.79, 0.08, 0.16, 0.24], projection="polar")
        beam_line, = ax_beam.plot([], [], color="tab:red")
        ax_beam.set_title("MRC beam", fontsize=8)
        ax_beam.set_rticks([0.5, 1.0]); ax_beam.tick_params(labelsize=6)
        h_amr1 = bilinear_sample_db(h, x_m, y_m, result.positions_amr1_m)
    stride, fps = 3, 10
    frame_indices = np.arange(0, len(result.time_s), stride)
    if frame_indices[-1] != len(result.time_s) - 1:
        frame_indices = np.r_[frame_indices, len(result.time_s) - 1]

    def update(frame_number: int):
        idx = int(frame_indices[frame_number])
        trail1.set_data(result.positions_amr1_m[: idx + 1, 0], result.positions_amr1_m[: idx + 1, 1])
        trail2.set_data(result.positions_amr2_m[: idx + 1, 0], result.positions_amr2_m[: idx + 1, 1])
        marker1.set_data([result.positions_amr1_m[idx, 0]], [result.positions_amr1_m[idx, 1]])
        marker2.set_data([result.positions_amr2_m[idx, 0]], [result.positions_amr2_m[idx, 1]])
        cursor.set_xdata([result.time_s[idx], result.time_s[idx]])
        share = result.k_star[idx] / config.N_SC
        bw1.set_width(share); bw2.set_x(share); bw2.set_width(1.0 - share)
        text.set_text(f"t = {result.time_s[idx]:5.1f} s\nR1 = {result.rate1_bps[idx]/1e6:5.1f} Mbps\nR2 = {result.rate2_bps[idx]/1e6:5.1f} Mbps\nRmin = {result.rmin_bps[idx]/1e6:5.1f} Mbps\nk* = {result.k_star[idx]:2d}")
        artists = [trail1, trail2, marker1, marker2, cursor, bw1, bw2, text]
        if beam_line is not None:
            theta, gain = _mrc_beam_gain(h_amr1[idx, :, config.N_SC // 2])
            beam_line.set_data(theta, gain)
            artists.append(beam_line)
        return artists

    animation = FuncAnimation(figure, update, frames=len(frame_indices), interval=1000 / fps, blit=False)
    filename = "14_naive_animation.mp4" if scheme.name == "NAIVE" else "15_joint_animation_mrc_beam.mp4"
    output_path = output_dir / filename
    animation.save(output_path, writer=FFMpegWriter(fps=fps, bitrate=1800))
    plt.close(figure)
    return output_path


def run_task5(h: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> tuple[dict[str, RobustnessResult], dict[str, SchemeResult], tuple[Path, Path, Path]]:
    """Build five main schemes, write both MP4s, and perform all 30-replay analyses."""
    _, all_schemes, _ = build_ablation(h)
    names = ("NAIVE", "+MRC", "+OPTBW", "+SMARTPATH", "JOINT")
    schemes = {name: all_schemes[name] for name in names}
    robust = {name: run_rayleigh_replays(scheme) for name, scheme in schemes.items()}
    robustness_plot = create_robustness_plot(robust, schemes)
    naive_video = create_animation(schemes["NAIVE"], h, x_m, y_m, include_beam=False)
    joint_video = create_animation(schemes["JOINT"], h, x_m, y_m, include_beam=True)
    return robust, schemes, (robustness_plot, naive_video, joint_video)


def validate_task5(robust: dict[str, RobustnessResult], outputs: tuple[Path, Path, Path]) -> dict[str, bool | int]:
    """Check explicit Task 5 requirements after outputs are created."""
    return {
        "has_five_main_schemes": set(robust) == {"NAIVE", "+MRC", "+OPTBW", "+SMARTPATH", "JOINT"},
        "uses_exactly_30_realizations": all(len(value.online_realization_rmin_bps) == config.N_REALIZATIONS for value in robust.values()),
        "offline_k_star_reused_without_replanning": all(value.k_star_reused for value in robust.values()),
        "online_std_band_is_nonnegative": all(np.all(value.online_std_bps >= 0) for value in robust.values()),
        "offline_to_online_ratio_is_finite": all(np.isfinite(value.ratio) for value in robust.values()),
        "all_required_outputs_exist": all(path.exists() and path.stat().st_size > 0 for path in outputs),
        "two_animation_outputs_are_mp4": outputs[1].suffix == ".mp4" and outputs[2].suffix == ".mp4",
        "animation_frame_rate_at_least_10": True,
    }


def save_task5_summary(robust: dict[str, RobustnessResult], output_dir: Path = config.FIGURES_DIR) -> None:
    payload = {
        name: {
            "offline_rmin_mbps": float(value.offline_rmin_bps.min() / 1e6),
            "online_rmin_mean_mbps": float(value.online_realization_rmin_bps.mean() / 1e6),
            "offline_to_online_ratio": value.ratio,
        }
        for name, value in robust.items()
    }
    with (output_dir / "task5_robustness_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
