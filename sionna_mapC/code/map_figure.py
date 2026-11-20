"""새 맵 평면도(00_map_layout.png): 방, 장애물 4개, 로봇 반지름만큼 막히는 셀."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

import config


def blocked_cells() -> np.ndarray:
    """셀 중심이 장애물(+로봇 반지름) 안에 있으면 True. shape (Ny, Nx)."""
    xs, ys = config.grid_coordinates_m()
    X, Y = np.meshgrid(xs, ys)
    r = config.OBSTACLE_BUFFER_M
    blocked = (X < r) | (X > config.ROOM_X_M - r) | (Y < r) | (Y > config.ROOM_Y_M - r)
    for ob in config.OBSTACLES:
        cx, cy = ob.center_xy_m
        if ob.kind == "box":
            dx, dy, _ = ob.size_xyz_m
            ddx = np.maximum(np.abs(X - cx) - dx / 2, 0.0)
            ddy = np.maximum(np.abs(Y - cy) - dy / 2, 0.0)
            blocked |= np.hypot(ddx, ddy) < r
        else:
            blocked |= np.hypot(X - cx, Y - cy) < ob.radius_m + r
    return blocked


def draw_map(path=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    xs, ys = config.grid_coordinates_m()
    res = config.GRID_RES_M
    ax.imshow(blocked_cells(), origin="lower", cmap="Greys", alpha=0.25,
              extent=(0, config.ROOM_X_M, 0, config.ROOM_Y_M))
    for ob in config.OBSTACLES:
        cx, cy = ob.center_xy_m
        if ob.kind == "box":
            dx, dy, _ = ob.size_xyz_m
            patch = Rectangle((cx - dx / 2, cy - dy / 2), dx, dy, color="dimgray")
        else:
            patch = Circle((cx, cy), ob.radius_m, color="dimgray")
        ax.add_patch(patch)
        ax.text(cx, cy, ob.name, ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    if config.BS_CENTER_M is not None:
        ax.plot(*config.BS_CENTER_M[:2], "r^", ms=12, label=f"BS ({config.BS_CENTER_M[0]:g}, {config.BS_CENTER_M[1]:g}, {config.BS_CENTER_M[2]:g} m), 4-antenna ULA")
    for i, (amr, ends) in enumerate(config.AMR_ENDPOINTS_M.items()):
        if ends is None:
            continue
        (sx, sy), (gx, gy) = ends
        c = f"C{i}"
        ax.add_patch(Circle((sx, sy), config.AMR_RADIUS_M, fill=False, ec=c, lw=2))
        ax.plot(gx, gy, "*", color=c, ms=12)
        ax.annotate("", (gx, gy), (sx, sy), arrowprops=dict(arrowstyle="->", color=c, ls="--"))
        ax.text(sx, sy + 0.4, amr, color=c, ha="center", fontsize=9)
    ax.set_xticks(np.arange(0, config.ROOM_X_M + 1e-9, 1.0))
    ax.set_yticks(np.arange(0, config.ROOM_Y_M + 1e-9, 1.0))
    ax.set_xticks(np.arange(0, config.ROOM_X_M + 1e-9, res), minor=True)
    ax.set_yticks(np.arange(0, config.ROOM_Y_M + 1e-9, res), minor=True)
    ax.grid(which="minor", lw=0.3, alpha=0.4)
    ax.grid(which="major", lw=0.6, alpha=0.6)
    ax.set_xlim(0, config.ROOM_X_M)
    ax.set_ylim(0, config.ROOM_Y_M)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Map {config.ROOM_X_M:g} x {config.ROOM_Y_M:g} x {config.ROOM_Z_M:g} m "
                 f"(gray cells: blocked for AMR radius {config.AMR_RADIUS_M} m)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=False)
    fig.tight_layout()
    path = path or config.FIGURES_DIR / "00_map_layout.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(draw_map())
