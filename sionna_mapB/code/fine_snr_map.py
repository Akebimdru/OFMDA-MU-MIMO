"""맵 B 10 cm SNR 맵 그림과 요약 (fine_map.py --assemble 다음에 실행).

출력: figures/01_fine_snr_map.png, figures/01_fine_snr_summary.json
* 왼쪽 위: 4안테나 MRC SNR (부반송파 평균, 잡음 3종 포함)
* 오른쪽 위: 잡음 없는 채널과의 차이 |ΔSNR|
* 아래: SNR 분포 (AMR이 갈 수 있는 점 / 전체 빈 점)
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

import config
from fine_map import FINE_H_CLEAN_PATH, FINE_H_PATH, FINE_META_PATH


def mrc_snr_db(H: np.ndarray) -> np.ndarray:
    p = np.mean(np.sum(np.abs(H) ** 2, axis=2), axis=-1)
    return 10 * np.log10(np.maximum(p * config.tx_power_w() / config.noise_power_w(), 1e-30))


def robot_free(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """로봇 중심이 놓일 수 있는 점 (장애물/벽에서 반지름 0.25 m 초과)."""
    X, Y = np.meshgrid(x, y)
    r = config.AMR_RADIUS_M
    free = (X > r) & (X < config.ROOM_X_M - r) & (Y > r) & (Y < config.ROOM_Y_M - r)
    for ob in config.OBSTACLES:
        cx, cy = ob.center_xy_m
        if ob.kind == "box":
            dx, dy, _ = ob.size_xyz_m
            d = np.hypot(np.maximum(np.abs(X - cx) - dx / 2, 0), np.maximum(np.abs(Y - cy) - dy / 2, 0))
        else:
            d = np.maximum(np.hypot(X - cx, Y - cy) - ob.radius_m, 0)
        free &= d > r
    return free


def draw_scene(ax):
    for ob in config.OBSTACLES:
        cx, cy = ob.center_xy_m
        fc = "lightsteelblue" if ob.material == "metal" else "dimgray"
        if ob.kind == "box":
            dx, dy, _ = ob.size_xyz_m
            ax.add_patch(Rectangle((cx - dx / 2, cy - dy / 2), dx, dy, fc=fc, ec="k", lw=0.6))
            ax.text(cx, cy, f"{ob.height_m:.1f} m", ha="center", va="center", fontsize=7, color="w",
                    rotation=90 if dy > 2 * dx else 0)
        else:
            ax.add_patch(Circle((cx, cy), ob.radius_m, fc=fc, ec="k", lw=0.6))
    ax.plot(*config.BS_CENTER_M[:2], "r^", ms=12, mec="w")
    for i, (amr, (s, g)) in enumerate(config.AMR_ENDPOINTS_M.items()):
        ax.add_patch(Circle(s, config.AMR_RADIUS_M, fill=False, ec=f"C{i}", lw=2))
        ax.plot(*g, "*", color=f"C{i}", ms=13, mec="k", mew=0.5)
        ax.text(s[0], s[1] + 0.38, amr, color=f"C{i}", ha="center", fontsize=8, fontweight="bold",
                bbox=dict(fc="w", alpha=0.75, lw=0, pad=0.5))
    ax.set_xlim(0, config.ROOM_X_M); ax.set_ylim(0, config.ROOM_Y_M); ax.set_aspect("equal")
    ax.set_xticks(range(int(config.ROOM_X_M) + 1)); ax.set_yticks(range(int(config.ROOM_Y_M) + 1))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


def main():
    H = np.load(FINE_H_PATH, mmap_mode="r")
    H0 = np.load(FINE_H_CLEAN_PATH, mmap_mode="r")
    meta = np.load(FINE_META_PATH)
    x, y, occ = meta["x_m"], meta["y_m"], meta["occupied"]
    no_path = (meta["counts"].sum(axis=2) == 0) & ~occ
    snr, snr0 = mrc_snr_db(np.asarray(H)), mrc_snr_db(np.asarray(H0))
    free = robot_free(x, y)
    valid = ~occ & ~no_path

    def at(p):
        return float(snr[np.argmin(np.abs(y - p[1])), np.argmin(np.abs(x - p[0]))])

    summary = {
        "grid": f"{len(x)} x {len(y)} (10 cm)", "free_points": int((~occ).sum()),
        "points_without_any_path": int(no_path.sum()),
        "snr_db_robot_reachable": {k: round(float(f(snr[free & valid])), 1) for k, f in
                                   (("min", np.min), ("p5", lambda a: np.percentile(a, 5)),
                                    ("median", np.median), ("max", np.max))},
        "noise_abs_delta_snr_db_median": round(float(np.median(np.abs(snr - snr0)[valid])), 2),
        "start_snr_db": {a: round(at(se[0]), 1) for a, se in config.AMR_ENDPOINTS_M.items()},
        "goal_snr_db": {a: round(at(se[1]), 1) for a, se in config.AMR_ENDPOINTS_M.items()},
    }
    (config.FIGURES_DIR / "01_fine_snr_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.55])
    ax = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[1, :])
    ext = (0, config.ROOM_X_M, 0, config.ROOM_Y_M)
    v = snr[free & valid]
    im = ax.imshow(np.ma.masked_where(~valid, snr), origin="lower", extent=ext, cmap="viridis",
                   vmin=np.percentile(v, 1), vmax=np.percentile(v, 99.5), interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.025, label="MRC SNR [dB] (mean over 64 SC)")
    draw_scene(ax)
    ax.set_title("Map B: Sionna RT 10 cm SNR map (4-antenna MRC, with Gaussian noise)")
    d = np.abs(snr - snr0)
    im2 = ax2.imshow(np.ma.masked_where(~valid, d), origin="lower", extent=ext, cmap="magma", vmin=0,
                     vmax=max(np.percentile(d[valid], 99), 0.5), interpolation="nearest")
    plt.colorbar(im2, ax=ax2, fraction=0.025, label="|SNR(noisy) - SNR(clean)| [dB]")
    draw_scene(ax2)
    ax2.set_title("Effect of the 3 Gaussian impairments")
    bins = np.arange(np.floor(snr[valid].min()), np.ceil(snr[valid].max()) + 1, 1.0)
    ax3.hist(snr[valid], bins=bins, alpha=0.5, label="all free points")
    ax3.hist(v, bins=bins, alpha=0.7, label="robot-reachable points (>0.25 m from obstacles)")
    for i, (a, s) in enumerate(summary["start_snr_db"].items()):
        ax3.axvline(s, color=f"C{i}", ls="--", lw=1.2, label=f"{a} start {s:.1f} dB")
    ax3.set_xlabel("MRC SNR [dB]"); ax3.set_ylabel("points"); ax3.legend(fontsize=8, ncol=2); ax3.grid(alpha=0.3)
    s = summary["snr_db_robot_reachable"]
    ax3.set_title(f"SNR distribution: reachable min {s['min']} / p5 {s['p5']} / median {s['median']} / max {s['max']} dB")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "01_fine_snr_map.png", dpi=140, bbox_inches="tight")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
