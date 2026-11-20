"""SNR 히트맵 (01_snr_heatmaps.png).

AMR 한 대가 각 셀에 있을 때 BS가 받는 SNR (64개 부반송파 평균, dB).
(1) 안테나 1개 (SISO), (2) 안테나 4개 MRC, (3) 가우시안 잡음 3종이 바꾼 양 ΔSNR (MRC, 노이즈 有 − 無).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

import config
from fingerprint import load_or_build_fingerprint, snr_grid_db


def _overlay(ax):
    for ob in config.OBSTACLES:
        cx, cy = ob.center_xy_m
        if ob.kind == "box":
            dx, dy, _ = ob.size_xyz_m
            ax.add_patch(Rectangle((cx - dx / 2, cy - dy / 2), dx, dy, fc="dimgray", ec="k", lw=0.8))
        else:
            ax.add_patch(Circle((cx, cy), ob.radius_m, fc="dimgray", ec="k", lw=0.8))
    ax.plot(*config.BS_CENTER_M[:2], "w^", ms=10, mec="k", label="BS")
    for i, (amr, (start, _)) in enumerate(config.AMR_ENDPOINTS_M.items()):
        ax.add_patch(Circle(start, config.AMR_RADIUS_M, fill=False, ec="w", lw=1.5))
        ax.text(start[0], start[1] + (0.45 if start[1] < config.ROOM_Y_M / 2 else -0.75), amr, color="w", ha="center", fontsize=8, fontweight="bold")
    ax.set_xlim(0, config.ROOM_X_M)
    ax.set_ylim(0, config.ROOM_Y_M)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def main():
    fp = load_or_build_fingerprint()
    occ = fp["occupied"]
    extent = (0, config.ROOM_X_M, 0, config.ROOM_Y_M)
    siso = snr_grid_db(fp["H"], "siso").mean(axis=2)
    mrc = snr_grid_db(fp["H"], "mrc").mean(axis=2)
    mrc_clean = snr_grid_db(fp["H_clean"], "mrc").mean(axis=2)
    delta = mrc - mrc_clean
    for arr in (siso, mrc, delta):
        arr[occ] = np.nan

    vmin = np.floor(np.nanpercentile(siso, 1) / 5) * 5
    vmax = np.ceil(np.nanmax(mrc) / 5) * 5
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    for ax, data, title in ((axes[0], siso, "SNR, 1 BS antenna (SISO)"),
                            (axes[1], mrc, "SNR, 4 BS antennas (MRC)")):
        im = ax.imshow(data, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label="mean SNR over 64 subcarriers [dB]", shrink=0.85)
        _overlay(ax)
        ax.set_title(f"{title}: median {np.nanmedian(data):.1f} dB, min {np.nanmin(data):.1f} dB")
    lim = np.ceil(np.nanmax(np.abs(delta)))
    im = axes[2].imshow(delta, origin="lower", extent=extent, cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=axes[2], label="ΔSNR [dB]", shrink=0.85)
    _overlay(axes[2])
    axes[2].set_title(f"Gaussian noise effect (MRC, with − without): median |Δ| {np.nanmedian(np.abs(delta)):.2f} dB")
    fig.suptitle("Sionna RT map + 3 Gaussian environment noises (path length 2 mm, amplitude 0.05, diffuse K=20 dB)")
    fig.tight_layout()
    out = config.FIGURES_DIR / "01_snr_heatmaps.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "siso_db": {"median": float(np.nanmedian(siso)), "min": float(np.nanmin(siso)), "max": float(np.nanmax(siso))},
        "mrc_db": {"median": float(np.nanmedian(mrc)), "min": float(np.nanmin(mrc)), "max": float(np.nanmax(mrc))},
        "noise_delta_mrc_db": {"median_abs": float(np.nanmedian(np.abs(delta))), "max_abs": float(np.nanmax(np.abs(delta)))},
        "cells_without_path": int(fp["no_path"].sum()),
        "amr_start_mrc_db": {amr: float(mrc[np.argmin(np.abs(fp["y_m"] - s[1])), np.argmin(np.abs(fp["x_m"] - s[0]))])
                             for amr, (s, _) in config.AMR_ENDPOINTS_M.items()},
    }
    (config.FIGURES_DIR / "01_snr_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return out


if __name__ == "__main__":
    main()
