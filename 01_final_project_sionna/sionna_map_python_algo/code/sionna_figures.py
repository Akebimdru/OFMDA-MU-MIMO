# ===== [원본과 다름] 원본에 없던 새 파일: Sionna 맵 관련 그림(00_*) 생성. 파일 전체가 새로 추가된 코드 =====
"""Extra figures for the Sionna RT map step (00_*).  Downstream figures are unchanged.

* 00_sionna_scene_render.png      : Sionna 3-D render of the scene with traced paths to one cell
* 00b_map_handout_vs_sionna.png   : handout tracer vs Sionna RT (clean) vs Sionna RT + Gaussian noise
* 00c_sionna_path_statistics.png  : paths per link, power share by interaction type, path depth, PDP
* 00d_gaussian_noise.png          : effect of the Gaussian environment noise on the map
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import config
from fingerprint import (INTERACTION_NAMES, build_handout_fingerprint, load_or_build_fingerprint,
                         load_or_run_sionna_rt, paths_to_channel)
from visualizer import _draw_scene, snr_maps_from_fingerprint

RENDER_RX_M = (12.75, 6.25, config.RX_HEIGHT_M)   # east-half cell reached through the 0.5 m gap


def render_scene(output_dir: Path = config.FIGURES_DIR) -> Path | None:
    """3-D Sionna render of the room and the paths from one BS element to one cell."""
    try:
        import mitsuba as mi
        if mi.variant() is None:
            mi.set_variant("llvm_ad_mono_polarized")
        import sionna.rt as rt
        from sionna_scene import build_scene_files

        _, render_xml = build_scene_files()
        scene = rt.load_scene(str(render_xml), merge_shapes=False)
        scene.frequency = config.FREQ_HZ
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.add(rt.Transmitter("bs", position=mi.Point3f(*map(float, config.BS_CENTER_M))))
        scene.add(rt.Receiver("cell", position=mi.Point3f(*RENDER_RX_M)))
        # Depth 2 keeps the drawing readable; the map itself uses config.RT_MAX_DEPTH.
        paths = rt.PathSolver()(scene, max_depth=2, los=True, specular_reflection=True, refraction=False,
                                diffraction=True, samples_per_src=1_000_000, seed=config.RANDOM_SEED)
        camera = rt.Camera(position=mi.Point3f(9.0, -4.0, 24.0), look_at=mi.Point3f(9.0, 5.0, 0.0))
        out = output_dir / "00_sionna_scene_render.png"
        scene.render_to_file(camera=camera, filename=str(out), paths=paths, resolution=[1280, 800], num_samples=64)
        return out
    except Exception as error:  # rendering is optional
        print(f"Skipping Sionna scene render: {error}")
        return None


def _snr_siso_db(h: np.ndarray) -> np.ndarray:
    return snr_maps_from_fingerprint(h)[0]


def map_comparison(rt_paths, data, output_dir: Path = config.FIGURES_DIR) -> dict[str, float]:
    x_m, y_m = data["x_m"], data["y_m"]
    handout = _snr_siso_db(build_handout_fingerprint(verbose=False)["H"])
    clean_h = paths_to_channel(rt_paths, with_noise=False)
    clean_h[data["no_path"]] = data["H"][data["no_path"]]
    clean = _snr_siso_db(clean_h)
    noisy = _snr_siso_db(data["H"])
    x_edges = np.r_[0.0, x_m + config.GRID_RES_M / 2.0]
    y_edges = np.r_[0.0, y_m + config.GRID_RES_M / 2.0]
    # Cells inside the solid Pillar E receive almost nothing; clip them so the colour scale stays readable.
    vmin = float(np.percentile(np.r_[handout.ravel(), noisy.ravel()], 1))
    vmax = float(max(handout.max(), noisy.max()))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6), constrained_layout=True)
    panels = ((handout, "Original Python map (LOS + 4 wall reflections)"),
              (clean, f"Sionna RT map (depth {config.RT_MAX_DEPTH}, refl./refr./diffr./scatt.)"),
              (noisy, "Sionna RT map + Gaussian environment noise (used)"))
    for ax, (values, title) in zip(axes.flat[:3], panels):
        mesh = ax.pcolormesh(x_edges, y_edges, values, cmap="viridis", vmin=vmin, vmax=vmax)
        _draw_scene(ax)
        ax.set_title(title, fontsize=10)
    fig.colorbar(mesh, ax=axes[0, :], shrink=0.85, label="SISO average SNR [dB]")
    diff = noisy - handout
    lim = float(np.percentile(np.abs(diff), 98))
    ax = axes[1, 1]
    mesh = ax.pcolormesh(x_edges, y_edges, diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    _draw_scene(ax)
    ax.set_title("Difference: Sionna (+noise) - original [dB]", fontsize=10)
    fig.colorbar(mesh, ax=ax, shrink=0.85, label="dB")
    fig.suptitle("Task 1 map: original pure-Python tracer vs Sionna RT", fontsize=14)
    fig.savefig(output_dir / "00b_map_handout_vs_sionna.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    west, east = x_m < 9.0, x_m > 9.0
    return {
        "handout_west_mean_db": float(handout[:, west].mean()), "handout_east_mean_db": float(handout[:, east].mean()),
        "sionna_west_mean_db": float(noisy[:, west].mean()), "sionna_east_mean_db": float(noisy[:, east].mean()),
        "sionna_minus_handout_mean_db": float(diff.mean()),
    }


def path_statistics(rt_paths, data, output_dir: Path = config.FIGURES_DIR) -> dict:
    x_m, y_m = data["x_m"], data["y_m"]
    counts = rt_paths["counts"]
    power = np.abs(rt_paths["a"].astype(np.complex128)) ** 2
    kinds, depth = rt_paths["kinds"], rt_paths["depth"]
    total = power.sum()
    share = {"LOS": float(power[depth == 0].sum() / total)}
    count_share = {"LOS": int((depth == 0).sum())}
    for code, name in INTERACTION_NAMES.items():
        sel = (kinds & code) != 0
        share[name] = float(power[sel].sum() / total)
        count_share[name] = int(sel.sum())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6), constrained_layout=True)
    ax = axes[0, 0]
    x_edges = np.r_[0.0, x_m + config.GRID_RES_M / 2.0]
    y_edges = np.r_[0.0, y_m + config.GRID_RES_M / 2.0]
    mesh = ax.pcolormesh(x_edges, y_edges, np.maximum(counts.mean(axis=2), 1), cmap="magma",
                         norm=matplotlib.colors.LogNorm())
    _draw_scene(ax)
    ax.set_title(f"Ray paths per (antenna, cell) link  (total {counts.sum():,})", fontsize=10)
    fig.colorbar(mesh, ax=ax, shrink=0.85, label="paths")

    ax = axes[0, 1]
    names = list(share)
    ax.bar(names, [100 * share[n] for n in names], color="tab:blue")
    ax.set_yscale("log")
    ax.set_ylabel("share of received power [%] (log)")
    ax.set_title("Power share by interaction type (a path can have several)", fontsize=10)
    for i, n in enumerate(names):
        ax.annotate(f"{count_share[n]:,} paths", (i, 100 * share[n]), ha="center", va="bottom", fontsize=8)

    ax = axes[1, 0]
    max_depth = int(depth.max())
    ks = np.arange(max_depth + 1)
    n_by_depth = np.array([(depth == k).sum() for k in ks])
    p_by_depth = np.array([power[depth == k].sum() / total for k in ks])
    ax.bar(ks - 0.2, n_by_depth / n_by_depth.sum() * 100, width=0.4, label="share of paths")
    ax.bar(ks + 0.2, p_by_depth * 100, width=0.4, label="share of power")
    ax.set_yscale("log")
    ax.set(xlabel="number of interactions in the path", ylabel="% (log)")
    ax.legend()
    ax.set_title("Path depth distribution", fontsize=10)

    ax = axes[1, 1]
    for point, color in (((2.75, 2.75), "tab:green"), (RENDER_RX_M[:2], "tab:red")):
        iy, ix = int(np.argmin(np.abs(y_m - point[1]))), int(np.argmin(np.abs(x_m - point[0])))
        link = (iy * len(x_m) + ix) * config.N_ANT_BS
        sel = rt_paths["link_index"] == link
        tau_ns = rt_paths["tau"][sel] * 1e9
        p_db = 10 * np.log10(power[sel] + 1e-30)
        ax.scatter(tau_ns, p_db, s=3, color=color, label=f"cell ({x_m[ix]:.2f}, {y_m[iy]:.2f}) m: {sel.sum()} paths")
    ax.set(xlabel="delay [ns]", ylabel="path power |a|^2 [dB]", ylim=(-170, -50))
    ax.legend(fontsize=8)
    ax.set_title("Power-delay profile (BS element 0)", fontsize=10)
    fig.suptitle("Sionna RT multipath statistics", fontsize=14)
    fig.savefig(output_dir / "00c_sionna_path_statistics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"paths_total": int(counts.sum()), "paths_per_link_mean": float(counts.mean()),
            "paths_per_link_max": int(counts.max()), "power_share": share, "path_count_by_type": count_share,
            "cells_without_paths": int(data["no_path"].sum()), "rt_runtime_s": float(rt_paths["runtime_s"])}


def noise_effect(rt_paths, data, output_dir: Path = config.FIGURES_DIR) -> dict[str, float]:
    clean_h = paths_to_channel(rt_paths, with_noise=False)
    noisy_h = data["H"].astype(np.complex128)
    ok = ~data["no_path"]
    scale = config.tx_power_w() / config.noise_power_w()
    clean_db = 10 * np.log10(scale * np.abs(clean_h[ok]) ** 2 + 1e-30)
    noisy_db = 10 * np.log10(scale * np.abs(noisy_h[ok]) ** 2 + 1e-30)
    delta = (noisy_db - clean_db).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.hist(np.clip(delta, -10, 10), bins=120, color="tab:purple", density=True)
    ax.set(xlabel="SNR change per (cell, antenna, subcarrier) [dB]", ylabel="density")
    ax.set_title(f"Gaussian noise effect: median |dSNR| = {np.median(np.abs(delta)):.2f} dB, "
                 f"95th pct = {np.percentile(np.abs(delta), 95):.2f} dB", fontsize=10)
    ax = axes[1]
    iy, ix = int(np.argmin(np.abs(data["y_m"] - RENDER_RX_M[1]))), int(np.argmin(np.abs(data["x_m"] - RENDER_RX_M[0])))
    k = np.arange(config.N_SC)
    ax.plot(k, 10 * np.log10(scale * np.abs(clean_h[iy, ix, 0]) ** 2), label="Sionna RT (clean)")
    ax.plot(k, 10 * np.log10(scale * np.abs(noisy_h[iy, ix, 0]) ** 2), label="+ Gaussian noise")
    ax.set(xlabel="subcarrier k", ylabel="SNR [dB]")
    ax.legend()
    ax.set_title(f"Frequency response at cell ({data['x_m'][ix]:.2f}, {data['y_m'][iy]:.2f}) m, antenna 0", fontsize=10)
    fig.suptitle(f"Gaussian environment noise: path length std {config.NOISE_PATH_LENGTH_STD_M * 1e3:.1f} mm, "
                 f"amplitude std {config.NOISE_AMPLITUDE_STD:.2f}, diffuse field K = {config.NOISE_DIFFUSE_K_DB:.0f} dB",
                 fontsize=12)
    fig.savefig(output_dir / "00d_gaussian_noise.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"median_abs_delta_db": float(np.median(np.abs(delta))),
            "p95_abs_delta_db": float(np.percentile(np.abs(delta), 95))}


def create_sionna_figures(output_dir: Path = config.FIGURES_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_or_build_fingerprint(verbose=False)
    rt_paths = load_or_run_sionna_rt(verbose=False)
    summary = {"map_comparison": map_comparison(rt_paths, data, output_dir),
               "path_statistics": path_statistics(rt_paths, data, output_dir),
               "gaussian_noise": noise_effect(rt_paths, data, output_dir)}
    render = render_scene(output_dir)
    summary["render"] = str(render) if render else None
    with (output_dir / "sionna_map_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


if __name__ == "__main__":
    print(json.dumps(create_sionna_figures(), indent=2))
